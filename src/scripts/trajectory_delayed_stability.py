from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.dlm_interp.load import CKPT, load_model, nano_forward_fn
from src.scripts.archive.counterfactual_bundle_ceiling import (
    baseline_bundle,
    commit,
    forward_state,
    warm_state,
)


COMPATIBLE_V1_SHA256 = "c1fd3a8bc9ce003dc388e0cb895b4259292a68bb0e34416f52ea58e0e7e05eaf"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_no_override(arrays):
    for name in ("states", "bundles", "predictions", "commit_h"):
        natural, treated = arrays[name + "_natural"], arrays[name + "_treated"]
        if name == "bundles":
            # Positions within one simultaneous commit form an unordered set.
            # Preserve step order; only canonicalize positions INSIDE each step.
            natural, treated = np.sort(natural, axis=1), np.sort(treated, axis=1)
        if not np.array_equal(natural, treated):
            raise AssertionError(f"no-override control failed: {name}")


def resume_metadata(saved, current_protocol, completed_seeds):
    previous = saved["protocol"]
    if previous == current_protocol:
        return saved
    expected_v1 = dict(current_protocol, version=1, script_sha256=COMPATIBLE_V1_SHA256)
    if current_protocol["version"] != 2 or previous != expected_v1:
        raise ValueError("resume protocol differs; only the exact original v1 checkpoint is compatible with this fix")
    # This fix changes only a validation of simultaneous bundle position order.
    # It does not change generation, features, outcomes, or any saved v1 result.
    migrated = dict(saved)
    migrated["protocol"] = current_protocol
    migrated["resume_migrations"] = list(saved.get("resume_migrations", [])) + [{
        "from_version": 1, "from_script_sha256": COMPATIBLE_V1_SHA256,
        "to_version": 2, "to_script_sha256": current_protocol["script_sha256"],
        "retained_completed_seeds": list(completed_seeds),
        "reason": "compare simultaneous bundle positions as sets; preserve all exact state/prediction/commit-time controls",
    }]
    return migrated


def delayed_events(masked, logits, other, other_commit, chosen, h, side, mask_id):
    """Scores are taken ONLY from the rollout in which the position is masked."""
    rows = (other[masked] != mask_id).nonzero().flatten()
    if not len(rows):
        return []
    positions = masked[rows]
    anchors = other[positions]
    values = logits[rows]
    top_values, top_ids = values.topk(2, dim=-1)
    anchor_logits = values.gather(1, anchors[:, None]).squeeze(1)
    competitors = torch.where(top_ids[:, 0] == anchors, top_values[:, 1], top_values[:, 0])
    margins = anchor_logits - competitors
    probs = (anchor_logits - values.logsumexp(-1)).exp()
    predictions = values.argmax(-1)  # Same tie convention as commit().
    chosen_positions = set(masked[chosen].tolist())
    return [{"h": h, "decision_step": h + 1, "delayed_rollout": side,
             "position": int(pos), "anchor_token": int(anchor), "prediction": int(pred),
             "anchor_margin": float(margin), "anchor_probability": float(prob),
             "top1_differs_from_anchor": bool(pred != anchor),
             "other_commit_h": int(other_commit[pos]),
             "steps_since_other_commit": h - int(other_commit[pos]),
             "commits_this_decision": int(pos) in chosen_positions}
            for pos, anchor, pred, margin, prob in zip(positions, anchors, predictions, margins, probs)]


@torch.no_grad()
def trace_intervention(fwd, initial, imposed_positions, mask_id, k, gap, origin):
    n, t = initial.clone(), initial.clone()
    steps = math.ceil(int((initial == mask_id).sum()) / k)
    if steps < 2:
        raise ValueError("need at least two continuation steps for the h=1 score")
    imposed = torch.as_tensor(imposed_positions, dtype=torch.long)
    if len(imposed) != min(k, int((initial == mask_id).sum())) or len(set(imposed.tolist())) != len(imposed):
        raise ValueError("imposed bundle must have the usual size and unique positions")
    if bool(((imposed < 0) | (imposed >= initial.numel())).any()) or not bool((initial[imposed] == mask_id).all()):
        raise ValueError("imposed positions must be valid and masked")
    states = {"natural": [n.numpy().copy()], "treated": [t.numpy().copy()]}
    bundles = {side: [] for side in states}
    predictions = {side: [] for side in states}
    commit_times = {side: torch.where(initial == mask_id, -1, 0) for side in states}
    events, rows = [], []
    initial_features = None
    active = None
    outside = torch.arange(initial.numel()) != origin
    for h in range(steps):
        mn, ln, cn = forward_state(fwd, n, mask_id)
        # h=0 is exactly the same state; this forward can be shared.
        mt, lt, ct = (mn, ln, cn) if h == 0 else forward_state(fwd, t, mask_id)
        sn = baseline_bundle(cn, mn, k, gap)
        adaptive_t = baseline_bundle(ct, mt, k, gap)
        st = torch.searchsorted(mt, imposed) if h == 0 else adaptive_t
        if h == 0:
            if not torch.equal(mt[st], imposed):
                raise AssertionError("imposed bundle lookup failed")
            active = set(mn[sn].tolist()) != set(mt[st].tolist())
        visible = (n != mask_id) & (t != mask_id) & outside
        written = int(((n != t) & visible).sum())
        visibility = int(((n == mask_id) != (t == mask_id)).sum())
        pending = delayed_events(mn, ln, t, commit_times["treated"], sn, h, "natural", mask_id)
        pending += delayed_events(mt, lt, n, commit_times["natural"], st, h, "treated", mask_id)
        events.extend(pending)
        rows.append({"h": h, "written_disagreements": written,
                     "visibility_disagreements": visibility, "identical_state": bool(torch.equal(n, t)),
                     "pending_positions": len(pending),
                     "pending_top1_mismatches": sum(e["top1_differs_from_anchor"] for e in pending),
                     "bundle_disagrees": set(mn[sn].tolist()) != set(mt[st].tolist())})
        if h == 1:
            if written != 0:
                raise AssertionError("single-step same-context intervention wrote conflicting common tokens")
            if active != bool(pending):
                raise AssertionError("active intervention and release queue disagree")
            # This object is set once, using only pre-commit h=1 information.
            initial_features = {
                "h": 1, "pending_positions": len(pending),
                "min_anchor_margin": min(e["anchor_margin"] for e in pending) if pending else None,
                "any_top1_mismatch": any(e["top1_differs_from_anchor"] for e in pending),
                "mean_anchor_probability": float(np.mean([e["anchor_probability"] for e in pending])) if pending else None,
            }
            initial_features["risk_score"] = -initial_features["min_anchor_margin"] if pending else None
        for side, masked, logits, selected in (("natural", mn, ln, sn), ("treated", mt, lt, st)):
            pred = np.full(initial.numel(), -1, dtype=np.int64)
            pred[masked.numpy()] = logits.argmax(-1).numpy()
            predictions[side].append(pred)
            bs = np.full(k, -1, dtype=np.int64)
            bs[:len(selected)] = masked[selected].numpy()
            bundles[side].append(bs)
            commit_times[side][masked[selected]] = h + 1
        n = commit(n, mn, ln, sn, sample=False)
        t = commit(t, mt, lt, st, sample=False)
        states["natural"].append(n.numpy().copy())
        states["treated"].append(t.numpy().copy())
    visible_initial = initial != mask_id
    for last in (n, t):
        if bool((last == mask_id).any()) or not torch.equal(last[visible_initial], initial[visible_initial]):
            raise AssertionError("unfinished rollout or rewritten initial context")
    final_diff = int(((n != t) & outside).sum())
    rows.append({"h": steps, "written_disagreements": final_diff,
                 "visibility_disagreements": 0, "identical_state": bool(torch.equal(n, t)),
                 "pending_positions": 0, "pending_top1_mismatches": 0, "bundle_disagrees": False})
    arrays = {}
    for side in states:
        arrays["states_" + side] = np.stack(states[side])
        arrays["bundles_" + side] = np.stack(bundles[side])
        arrays["predictions_" + side] = np.stack(predictions[side])
        arrays["commit_h_" + side] = commit_times[side].numpy()
    written_counts = [row["written_disagreements"] for row in rows]
    if any(b < a for a, b in zip(written_counts, written_counts[1:])):
        raise AssertionError("a committed disagreement was erased")
    reunion = next((row["h"] for row in rows if row["h"] > 1 and row["identical_state"]), None) if active else None
    if reunion is not None and not np.array_equal(arrays["states_natural"][reunion:], arrays["states_treated"][reunion:]):
        raise AssertionError("deterministic rollouts diverged after an identical state")
    if not active:
        check_no_override(arrays)
    first_flip = next((row["h"] for row in rows if row["pending_top1_mismatches"]), None)
    return {"active_override": bool(active), "release_features": initial_features,
            "final_different_tokens_outside_origin": final_diff,
            "persistent_final_difference": bool(final_diff),
            "first_exact_reunion_h": reunion,
            "first_common_written_disagreement_h": next((row["h"] for row in rows if row["written_disagreements"]), None),
            "first_pending_top1_mismatch_h": first_flip,
            "ever_pending_top1_mismatch": first_flip is not None,
            "trace": rows, "delayed_events": events,
            "forward_calls": 2 * steps - 1}, arrays


@torch.no_grad()
def make_contexts(fwd, mask_id, length, k, gap, warmup, probe, seed):
    x = warm_state(fwd, mask_id, length, k, gap, warmup, seed)
    for _ in range(probe):
        m, l, c = forward_state(fwd, x, mask_id)
        x = commit(x, m, l, baseline_bundle(c, m, k, gap), sample=False)
    m, l, c = forward_state(fwd, x, mask_id)
    chosen = baseline_bundle(c, m, k, gap)
    ordered = chosen[torch.argsort(m[chosen])]
    rng = torch.Generator("cpu").manual_seed(1_700_000_000 + seed)
    row = int(ordered[int(torch.randint(len(ordered), (1,), generator=rng))])
    origin = int(m[row])
    token_a = int(l[row].argmax())
    second = l[row].clone()
    second[token_a] = -torch.inf
    token_b = int(second.argmax())
    if token_b == mask_id or not bool(torch.isfinite(second[token_b])):
        raise ValueError("no valid second token")
    a = commit(x, m, l, chosen, sample=False)
    b = a.clone()
    b[origin] = token_b
    probs = l[row].softmax(-1)
    return {"a": a, "b": b}, {"position": origin, "token_a": token_a, "token_b": token_b,
             "p_top1": float(probs[token_a]), "p_top2": float(probs[token_b]),
             "logit_gap": float(l[row, token_a] - l[row, token_b])}


def auc(labels, scores):
    labels, scores = np.asarray(labels, dtype=bool), np.asarray(scores, dtype=float)
    positive, negative = scores[labels], scores[~labels]
    if not len(positive) or not len(negative):
        return None
    delta = positive[:, None] - negative[None, :]
    return float(np.mean((delta > 0) + 0.5 * (delta == 0)))


def analysis(records, bootstrap):
    clusters = [[d for d in r["directions"].values() if d["active_override"]] for r in records]
    active = [d for cluster in clusters for d in cluster]
    labels = [d["persistent_final_difference"] for d in active]
    risks = [d["release_features"]["risk_score"] for d in active]
    predicted = [d["release_features"]["any_top1_mismatch"] for d in active]
    primary = auc(labels, risks)
    draws = []
    if primary is not None and bootstrap:
        rng = np.random.default_rng(90210)
        for _ in range(bootstrap):
            sample = [d for i in rng.integers(0, len(clusters), len(clusters)) for d in clusters[i]]
            value = auc([d["persistent_final_difference"] for d in sample],
                        [d["release_features"]["risk_score"] for d in sample])
            if value is not None:
                draws.append(value)
    def count(actual, prediction):
        return sum(y == actual and p == prediction for y, p in zip(labels, predicted))
    return {
        "seeds": len(records), "directions_total": 2 * len(records),
        "active_directions": len(active), "inactive_directions": 2 * len(records) - len(active),
        "seeds_with_active_directions": sum(bool(c) for c in clusters),
        "active_persistent_directions": sum(labels),
        "active_absorbed_directions": len(labels) - sum(labels),
        "primary_release_margin_auc": primary,
        "primary_auc_seed_bootstrap_ci95": np.quantile(draws, [0.025, 0.975]).tolist() if draws else None,
        "bootstrap_requested": bootstrap, "bootstrap_valid_draws": len(draws),
        "auc_note": "requires both absorbed and persistent active cases; higher risk = smaller h=1 anchor margin",
        "comparison_release_queue_size_auc": auc(labels, [d["release_features"]["pending_positions"] for d in active]),
        "secondary_any_release_top1_mismatch_confusion": {
            "true_positive": count(True, True), "false_positive": count(False, True),
            "false_negative": count(True, False), "true_negative": count(False, False)},
        "later_traces_descriptive_only": {
            "absorbed_with_any_transient_pending_flip": sum(not d["persistent_final_difference"] and d["ever_pending_top1_mismatch"] for d in active),
            "persistent_with_no_pending_flip_ever": sum(d["persistent_final_difference"] and not d["ever_pending_top1_mismatch"] for d in active),
            "active_exact_reunions": sum(d["first_exact_reunion_h"] is not None for d in active)},
        "scope": "prospective association on paired trajectories; no fitted predictor, quality, or semantic claim",
    }


def save(path, records, arrays, metadata, bootstrap=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as f:
        np.savez_compressed(f, records_json=np.asarray(json.dumps(records, allow_nan=False)),
                            metadata_json=np.asarray(json.dumps(metadata, allow_nan=False)), **arrays)
    os.replace(temp, path)
    summary = analysis(records, bootstrap)
    summary["metadata"] = metadata
    summary["per_seed"] = [{**r, "directions": {c: {key: value for key, value in d.items()
                            if key not in ("trace", "delayed_events")} for c, d in r["directions"].items()}}
                           for r in records]
    target = path.with_name(path.stem + "_summary.json")
    temp = target.with_name(target.name + ".tmp")
    temp.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    os.replace(temp, target)
    return summary


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--first-seed", type=int, default=700)
    p.add_argument("--seeds", type=int, default=40)
    p.add_argument("--probe", type=int, default=8)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--gap", type=int, default=16)
    p.add_argument("--warmup-steps", type=int, default=4)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--out", type=Path, default=Path("results/trajectory_delayed_stability_s40.npz"))
    a = p.parse_args()
    if min(a.seeds, a.k, a.threads) < 1 or min(a.first_seed, a.probe, a.gap, a.warmup_steps, a.bootstrap) < 0:
        p.error("invalid negative/zero argument")
    return a


@torch.no_grad()
def main():
    a = parse_args()
    torch.set_num_threads(a.threads)
    protocol = {key: value for key, value in vars(a).items() if key not in ("resume", "out")}
    protocol.update(version=2, checkpoint_sha256=digest(CKPT), script_sha256=digest(__file__),
                    torch_version=str(torch.__version__), imposed_steps=1,
                    primary_score="negative minimum delayed-anchor logit margin at h=1 before decision 2",
                    outcome="any final token-ID difference, same initial context",
                    population="actual policy overrides only; both directions clustered by seed",
                    early_secondary_rule="any delayed top1 differs from anchor at h=1",
                    comparison_score="number of delayed positions at h=1",
                    bootstrap_seed=90210)
    metadata = {"protocol": protocol, "identical_control": None}
    records, arrays = [], {}
    if a.out.exists():
        if not a.resume:
            raise FileExistsError(f"{a.out} exists; use --resume or another output")
        with np.load(a.out, allow_pickle=False) as z:
            metadata = json.loads(z["metadata_json"].item())
            records = json.loads(z["records_json"].item())
            arrays = {key: z[key] for key in z.files if not key.endswith("_json")}
        previous_version = metadata["protocol"]["version"]
        metadata = resume_metadata(metadata, protocol, [r["seed"] for r in records])
        if previous_version == 1:
            print(f"Compatible v1 checkpoint: retaining {len(records)} completed seeds; "
                  "bundle-order control corrected", flush=True)
    model, cfg = load_model()
    model.eval()
    fwd = nano_forward_fn(model)
    mask_id, length = int(cfg["mask_id"]), int(cfg["seq_len"])
    if length - (a.warmup_steps + a.probe + 1) * a.k <= a.k:
        raise ValueError("probe must leave at least two continuation steps")
    done = {r["seed"] for r in records}
    print(f"Delayed-position stability: seeds {a.first_seed}..{a.first_seed + a.seeds - 1}, "
          f"probe={a.probe}, one imposed step, both contexts; primary score fixed at h=1", flush=True)
    for seed in range(a.first_seed, a.first_seed + a.seeds):
        if seed in done:
            continue
        start = time.monotonic()
        contexts, intervention = make_contexts(fwd, mask_id, length, a.k, a.gap, a.warmup_steps, a.probe, seed)
        first = {}
        for c in ("a", "b"):
            m, l, conf = forward_state(fwd, contexts[c], mask_id)
            first[c] = m[baseline_bundle(conf, m, a.k, a.gap)]
        calls = a.warmup_steps + a.probe + 3
        if metadata["identical_control"] is None:
            control, _ = trace_intervention(fwd, contexts["a"], first["a"], mask_id, a.k, a.gap, intervention["position"])
            if control["active_override"] or control["final_different_tokens_outside_origin"] or control["delayed_events"]:
                raise AssertionError("identical control failed")
            calls += control["forward_calls"]
            metadata["identical_control"] = {"seed": seed, "passed": True}
        result = {"seed": seed, "intervention": intervention, "directions": {}}
        for c, opposite in (("a", "b"), ("b", "a")):
            diagnostic, pair_arrays = trace_intervention(fwd, contexts[c], first[opposite], mask_id, a.k, a.gap, intervention["position"])
            result["directions"][c] = diagnostic
            calls += diagnostic["forward_calls"]
            arrays.update({f"seed_{seed}_{c}_{key}": value for key, value in pair_arrays.items()})
        result.update(seconds=time.monotonic() - start, forward_calls=calls)
        records.append(result)
        save(a.out, records, arrays, metadata)
        parts = []
        for c, d in result["directions"].items():
            margin = d["release_features"]["min_anchor_margin"]
            score = "inactive" if margin is None else f"margin={margin:+.3f}"
            parts.append(f"{c.upper()} {score}, final={d['final_different_tokens_outside_origin']}, reunion={d['first_exact_reunion_h']}")
    summary = save(a.out, records, arrays, metadata, bootstrap=a.bootstrap)
    print(json.dumps({key: value for key, value in summary.items() if key not in ("metadata", "per_seed")}, indent=2))
    print(f"saved {a.out.with_name(a.out.stem + '_summary.json')}")


if __name__ == "__main__":
    main()
