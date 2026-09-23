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
from src.scripts.archive.counterfactual_bundle_ceiling import baseline_bundle, commit, forward_state

SIDES = ("natural", "treated")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def disagreements(a, b, mask_id, exclude=()):
    common = (a != mask_id) & (b != mask_id)
    common = common.copy()
    if len(exclude):
        common[..., list(exclude)] = False
    return ((a != b) & common).sum(axis=-1)


def validate_source_pair(data, diagnostic, mask_id, k, length):
    states = {s: data["states_" + s] for s in SIDES}
    n, t = states["natural"], states["treated"]
    if n.shape != t.shape or n.ndim != 2 or n.shape[1] != length or len(n) < 3:
        raise ValueError("invalid paired source state shapes")
    if not np.array_equal(n[0], t[0]):
        raise ValueError("within-context initial states differ")
    for side, st in states.items():
        bundles, predictions = data["bundles_" + side], data["predictions_" + side]
        if bundles.shape != (len(st) - 1, k) or predictions.shape != st[:-1].shape:
            raise ValueError("invalid source bundle/prediction shape")
        for h, bundle in enumerate(bundles):
            pos = bundle[bundle >= 0]
            expected_count = min(k, int((st[h] == mask_id).sum()))
            if len(pos) != expected_count or not np.array_equal(np.sort(pos), np.flatnonzero(st[h] != st[h + 1])):
                raise ValueError("source transition differs from its commit bundle")
            if not np.all(st[h, pos] == mask_id) or not np.array_equal(st[h + 1, pos], predictions[h, pos]):
                raise ValueError("source rewrites visible tokens or commits other than its predictions")
        if np.any(st[-1] == mask_id):
            raise ValueError("unfinished source rollout")
    written = disagreements(n, t, mask_id)
    if np.any(np.diff(written) < 0):
        raise ValueError("source erased a committed disagreement")
    hits = np.flatnonzero(written)
    first = int(hits[0]) if len(hits) else None
    if first != diagnostic["first_common_written_disagreement_h"]:
        raise ValueError("cached first-conflict time disagrees with raw states")
    if int(written[-1]) != diagnostic["final_different_tokens_outside_origin"]:
        raise ValueError("cached final count disagrees with raw states")
    if first is not None and (first < 2 or not diagnostic["active_override"]):
        raise ValueError("conflict must follow the imposed step in an active pair")
    return first


@torch.no_grad()
def finish(fwd, initial, mask_id, k, gap, expected=None, expected_bundles=None):
    x = torch.as_tensor(initial.copy(), dtype=torch.long)
    steps = math.ceil(int((x == mask_id).sum()) / k)
    states, bundles = [x.numpy().copy()], []
    for h in range(steps):
        m, l, c = forward_state(fwd, x, mask_id)
        chosen = baseline_bundle(c, m, k, gap)
        pos = m[chosen].numpy()
        if expected_bundles is not None:
            bs = expected_bundles[h]
            if not np.array_equal(np.sort(pos), np.sort(bs[bs >= 0])):
                raise AssertionError(f"no-edit replay bundle mismatch at suffix step {h + 1}")
        x = commit(x, m, l, chosen, sample=False)
        st = x.numpy().copy()
        if expected is not None and not np.array_equal(st, expected[h + 1]):
            raise AssertionError(f"no-edit replay state mismatch at suffix step {h + 1}")
        bs = np.full(k, -1, dtype=np.int64)
        bs[:len(pos)] = pos
        states.append(st); bundles.append(bs)
    visible = initial != mask_id
    out = np.stack(states)
    if np.any(out[:, visible] != initial[visible]) or np.any(out[-1] == mask_id):
        raise AssertionError("suffix rewrote committed tokens or did not finish")
    return out, np.stack(bundles) if bundles else np.empty((0, k), dtype=np.int64)


@torch.no_grad()
def trigger_forward(fwd, data, w, mask_id, k, gap):
    snapshots = {}
    for side in SIDES:
        before = torch.as_tensor(data["states_" + side][w - 1].copy(), dtype=torch.long)
        m, l, c = forward_state(fwd, before, mask_id)
        chosen = baseline_bundle(c, m, k, gap)
        bs = data["bundles_" + side][w - 1]
        if not np.array_equal(np.sort(m[chosen].numpy()), np.sort(bs[bs >= 0])):
            raise AssertionError(f"trigger bundle mismatch: {side}")
        if not np.array_equal(l.argmax(-1).numpy(), data["predictions_" + side][w - 1, m.numpy()]):
            raise AssertionError(f"trigger prediction mismatch: {side}")
        after = commit(before, m, l, chosen, sample=False).numpy()
        if not np.array_equal(after, data["states_" + side][w]):
            raise AssertionError(f"trigger next-state mismatch: {side}")
        snapshots[side] = {"before": before.numpy(), "after": after,
                           "masked": m, "logits": l, "positions": m[chosen].numpy()}
    return snapshots


def construct_edits(snapshots, mask_id):
    n, t = (snapshots[s] for s in SIDES)
    if disagreements(n["before"], t["before"], mask_id) != 0:
        raise ValueError("the trigger is not the first written conflict")
    positions = np.flatnonzero((n["after"] != t["after"]) &
                              (n["after"] != mask_id) & (t["after"] != mask_id))
    if not len(positions):
        raise ValueError("no imminent conflict")
    repair = {s: snapshots[s]["after"].copy() for s in SIDES}
    control = {s: snapshots[s]["after"].copy() for s in SIDES}
    edits = []
    for p in positions:
        n_new, t_new = n["before"][p] == mask_id, t["before"][p] == mask_id
        if t_new:
            side, other = "treated", "natural"  # Also the fixed simultaneous tie rule.
        elif n_new:
            side, other = "natural", "treated"
        else:
            raise AssertionError("repair would rewrite a previously visible token")
        snap = snapshots[side]
        if p not in snap["positions"]:
            raise AssertionError("repair position not selected for this commit")
        row = int(torch.searchsorted(snap["masked"], int(p)))
        logits = snap["logits"][row]
        original, target = int(snap["after"][p]), int(snapshots[other]["after"][p])
        if original == target or target == mask_id or not bool(torch.isfinite(logits[target])):
            raise ValueError("invalid alignment target")
        distance = (logits - logits[target]).abs()
        distance[original] = distance[target] = distance[mask_id] = torch.inf
        distance[~torch.isfinite(logits)] = torch.inf
        alternate = int(distance.argmin())
        if not bool(torch.isfinite(distance[alternate])):
            raise ValueError("no nonaligned control token available")
        repair[side][p], control[side][p] = target, alternate
        edits.append({"position": int(p), "edited_rollout": side,
                      "conflict_type": "simultaneous" if n_new and t_new else "delayed",
                      "original_token": original, "repair_token": target, "control_token": alternate,
                      "repair_logit_penalty_from_argmax": float(logits[original] - logits[target]),
                      "control_logit_penalty_from_argmax": float(logits[original] - logits[alternate]),
                      "control_vs_repair_abs_logit_gap": float(distance[alternate])})
    if disagreements(repair["natural"], repair["treated"], mask_id) != 0:
        raise AssertionError("repair did not remove every imminent conflict")
    if disagreements(control["natural"], control["treated"], mask_id) != len(edits):
        raise AssertionError("nonaligned control unexpectedly erased a target conflict")
    for arm in (repair, control):
        for side in SIDES:
            visible_before = snapshots[side]["before"] != mask_id
            if not np.array_equal(arm[side][visible_before], snapshots[side]["before"][visible_before]):
                raise AssertionError("an intervention rewrote a previously visible token")
    return edits, repair, control


def arm_metrics(states, mask_id, exclude, origin, w):
    n, t = (states[s] for s in SIDES)
    outside = disagreements(n, t, mask_id, exclude)
    all_except_origin = disagreements(n, t, mask_id, [origin])
    equal = np.all(n == t, axis=1)
    first_equal = np.flatnonzero(equal)
    if len(first_equal) and not np.array_equal(n[first_equal[0]:], t[first_equal[0]:]):
        raise AssertionError("adaptive rollouts separated after an exact reunion")
    conflicting = np.flatnonzero(all_except_origin)
    return {"final_differences_outside_edited_positions": int(outside[-1]),
            "final_differences_outside_origin": int(all_except_origin[-1]),
            "first_exact_reunion_h": w + int(first_equal[0]) if len(first_equal) else None,
            "first_common_written_conflict_h": w + int(conflicting[0]) if len(conflicting) else None,
            "written_differences_outside_edits_by_suffix_h": outside.tolist(),
            "visibility_difference_by_suffix_h": ((n == mask_id) != (t == mask_id)).sum(1).tolist()}


def summarize(records, metadata, bootstrap=0):
    eligible = [r for r in records if r["triggered"]]
    groups = {}
    for r in eligible:
        groups.setdefault(r["seed"], []).append(r)
    def endpoint(key):
        seed_values = np.asarray([np.mean([r[key] for r in group]) for group in groups.values()])
        ci = None
        if len(seed_values) >= 2 and bootstrap:
            rng = np.random.default_rng(19317)
            samples = seed_values[rng.integers(0, len(seed_values), (bootstrap, len(seed_values)))].mean(1)
            ci = np.quantile(samples, [0.025, 0.975]).tolist()
        return {"mean_over_seeds": float(seed_values.mean()) if len(seed_values) else None,
                "seed_bootstrap_ci95": ci}
    arm_means = {}
    for arm in ("no_edit", "repair", "nonaligned_control"):
        values = [np.mean([r["arms"][arm]["final_differences_outside_edited_positions"] for r in group])
                  for group in groups.values()]
        arm_means[arm] = float(np.mean(values)) if values else None
    edits = [e for r in eligible for e in r["edits"]]
    return {"source_pairs_processed": len(records), "triggered_directions": len(eligible),
            "triggered_seeds": len(groups), "untriggered_directions": len(records) - len(eligible),
            "primary_no_edit_minus_repair": endpoint("no_edit_minus_repair"),
            "secondary_control_minus_repair": endpoint("control_minus_repair"),
            "mean_final_differences_outside_edited_positions_by_arm": arm_means,
            "repair_exact_reunions": sum(r["arms"]["repair"]["first_exact_reunion_h"] is not None for r in eligible),
            "repair_no_later_written_conflict": sum(r["arms"]["repair"]["first_common_written_conflict_h"] is None for r in eligible),
            "edited_tokens": len(edits),
            "mean_control_repair_abs_logit_gap": float(np.mean([e["control_vs_repair_abs_logit_gap"] for e in edits])) if edits else None,
            "max_control_repair_abs_logit_gap": max((e["control_vs_repair_abs_logit_gap"] for e in edits), default=None),
            "scope": "exploratory paired first-conflict intervention on reused data; no quality, single-branch, or speed claim",
            "control_note": "control preserves target conflicts by construction; exact reunion is not a fair control endpoint",
            "per_pair": records, "metadata": metadata}


def save(path, records, arrays, metadata, bootstrap=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as f:
        np.savez_compressed(f, records_json=np.asarray(json.dumps(records, allow_nan=False)),
                            metadata_json=np.asarray(json.dumps(metadata, allow_nan=False)), **arrays)
    os.replace(temp, path)
    summary = summarize(records, metadata, bootstrap)
    target = path.with_name(path.stem + "_summary.json")
    temp = target.with_name(target.name + ".tmp")
    temp.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    os.replace(temp, target)
    return summary


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--out", type=Path, default=Path("results/trajectory_first_conflict_repair_s40.npz"))
    a = p.parse_args()
    if a.bootstrap < 0 or (a.threads is not None and a.threads < 1):
        p.error("invalid threads/bootstrap")
    if a.source.resolve() == a.out.resolve():
        p.error("source and output must differ")
    return a


@torch.no_grad()
def main():
    a = parse_args()
    with np.load(a.source, allow_pickle=False) as z:
        source_metadata = json.loads(z["metadata_json"].item())
        source_records = json.loads(z["records_json"].item())
        source_arrays = {key: z[key] for key in z.files if not key.endswith("_json")}
    sp = source_metadata["protocol"]
    if not source_records or len({r["seed"] for r in source_records}) != len(source_records):
        raise ValueError("source must contain distinct, nonempty seed records")
    if not (source_metadata.get("identical_control") or {}).get("passed"):
        raise ValueError("source identical-branch control did not pass")
    if digest(CKPT) != sp["checkpoint_sha256"] or str(torch.__version__) != sp["torch_version"]:
        raise ValueError("use the same checkpoint and PyTorch version as the source")
    threads = sp["threads"] if a.threads is None else a.threads
    torch.set_num_threads(threads)
    protocol = {"version": 1, "script_sha256": digest(__file__), "source_sha256": digest(a.source),
                "checkpoint_sha256": sp["checkpoint_sha256"], "torch_version": str(torch.__version__),
                "threads": threads, "bootstrap": a.bootstrap,
                "repair": "first imminent conflict only; later writer copies earlier writer; simultaneous treated copies natural",
                "control": "same sites/time/branches; closest-logit token excluding original and repair target",
                "primary": "no-edit minus repair final pair Hamming excluding origin and edited sites; mean within seed then across seeds"}
    metadata = {"protocol": protocol, "source_metadata": source_metadata, "no_edit_suffix_replay_control": None}
    records, arrays = [], {}
    if a.out.exists():
        if not a.resume:
            raise FileExistsError(f"{a.out} exists; use --resume or another path")
        with np.load(a.out, allow_pickle=False) as z:
            metadata = json.loads(z["metadata_json"].item())
            if metadata["protocol"] != protocol:
                raise ValueError("resume protocol differs")
            records = json.loads(z["records_json"].item())
            arrays = {key: z[key] for key in z.files if not key.endswith("_json")}
    done = {(r["seed"], r["context"]) for r in records}
    model, cfg = load_model()
    model.eval(); fwd = nano_forward_fn(model)
    mask_id, k, gap = int(cfg["mask_id"]), int(sp["k"]), int(sp["gap"])
    print(f"First-conflict repair: {len(source_records)} source seeds; one correction; "
          "primary excludes directly edited positions", flush=True)
    for source in source_records:
        seed = source["seed"]
        for context in ("a", "b"):
            if (seed, context) in done:
                continue
            start = time.monotonic()
            prefix = f"seed_{seed}_{context}_"
            data = {name: source_arrays[prefix + name] for side in SIDES
                    for name in ("states_" + side, "bundles_" + side, "predictions_" + side)}
            diagnostic = source["directions"][context]
            w = validate_source_pair(data, diagnostic, mask_id, k, int(cfg["seq_len"]))
            record = {"seed": seed, "context": context, "triggered": w is not None, "trigger_commit_h": w}
            if w is None:
                records.append(record)
                save(a.out, records, arrays, metadata)
                continue
            origin = int(source["intervention"]["position"])
            snapshots = trigger_forward(fwd, data, w, mask_id, k, gap)
            edits, repair, control = construct_edits(snapshots, mask_id)
            baseline = {side: data["states_" + side][w:].copy() for side in SIDES}
            if metadata["no_edit_suffix_replay_control"] is None:
                for side in SIDES:
                    replayed, _ = finish(fwd, baseline[side][0], mask_id, k, gap,
                                         expected=baseline[side], expected_bundles=data["bundles_" + side][w:])
                    if not np.array_equal(replayed, baseline[side]):
                        raise AssertionError("no-edit suffix control failed")
                metadata["no_edit_suffix_replay_control"] = {"seed": seed, "context": context, "passed": True}
                print("  cached no-edit suffix replay PASS (both branches, all remaining steps)", flush=True)
            excluded = sorted({origin} | {e["position"] for e in edits})
            arms = {"no_edit": arm_metrics(baseline, mask_id, excluded, origin, w)}
            for arm, initial in (("repair", repair), ("nonaligned_control", control)):
                states = {}
                for side in SIDES:
                    st, bs = finish(fwd, initial[side], mask_id, k, gap)
                    states[side] = st
                    arrays[prefix + arm + "_states_" + side] = st
                    arrays[prefix + arm + "_bundles_" + side] = bs
                arms[arm] = arm_metrics(states, mask_id, excluded, origin, w)
            for side in SIDES:
                arrays[prefix + "no_edit_states_" + side] = baseline[side]
            record.update(edits=edits, excluded_positions=excluded, original_perturbation_position=origin,
                          arms=arms, trigger_forward_verified=True,
                          no_edit_minus_repair=arms["no_edit"]["final_differences_outside_edited_positions"] - arms["repair"]["final_differences_outside_edited_positions"],
                          control_minus_repair=arms["nonaligned_control"]["final_differences_outside_edited_positions"] - arms["repair"]["final_differences_outside_edited_positions"],
                          seconds=time.monotonic() - start)
            records.append(record)
            save(a.out, records, arrays, metadata)
            counts = [arms[arm]["final_differences_outside_edited_positions"] for arm in ("no_edit", "repair", "nonaligned_control")]
            print(f"seed {seed}-{context.upper()}: h={w}, edits={len(edits)}; "
                  f"outside edits no-edit/repair/control={counts[0]}/{counts[1]}/{counts[2]}; "
                  f"{record['seconds']:.1f}s; saved {a.out}", flush=True)
    summary = save(a.out, records, arrays, metadata, bootstrap=a.bootstrap)
    print(json.dumps({key: value for key, value in summary.items() if key not in ("per_pair", "metadata")}, indent=2))
    print(f"saved {a.out.with_name(a.out.stem + '_summary.json')}")


if __name__ == "__main__":
    main()
