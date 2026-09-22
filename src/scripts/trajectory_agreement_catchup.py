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
from scripts.archive.counterfactual_bundle_ceiling import baseline_bundle, commit, forward_state
def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


SIDES = ("natural", "treated")


def bundle_geometry(positions):
    pos = np.sort(np.asarray(positions))
    if len(pos) < 2:
        return {"min_gap": None, "mean_pair_distance": None}
    distances = pos[None, :] - pos[:, None]
    distances = distances[np.triu_indices(len(pos), 1)]
    return {"min_gap": int(distances.min()), "mean_pair_distance": float(distances.mean())}


def validate_cached(states, bundles, mask_id, k, length):
    if states.ndim != 2 or states.shape[1] != length or bundles.shape != (len(states) - 1, k):
        raise ValueError("invalid cached shape")
    for h, bundle in enumerate(bundles):
        positions = bundle[bundle >= 0]
        if len(positions) != min(k, int((states[h] == mask_id).sum())):
            raise ValueError("cached commit budget differs")
        if not np.array_equal(np.flatnonzero(states[h] != states[h + 1]), np.sort(positions)):
            raise ValueError("cached transition differs from bundle")
        if not np.all(states[h, positions] == mask_id):
            raise ValueError("cached rollout rewrites visible tokens")
    if np.any(states[-1] == mask_id):
        raise ValueError("unfinished cached rollout")


@torch.no_grad()
def verify_boundary(fwd, states, bundles, mask_id, k, gap):
    if len(states) == 1:
        return
    x = torch.as_tensor(states[0].copy(), dtype=torch.long)
    m, l, c = forward_state(fwd, x, mask_id)
    chosen = baseline_bundle(c, m, k, gap)
    target = bundles[0]
    if not np.array_equal(np.sort(m[chosen].numpy()), np.sort(target[target >= 0])):
        raise AssertionError("initial cached bundle cannot be reproduced")
    after = commit(x, m, l, chosen, sample=False).numpy()
    if not np.array_equal(after, states[1]):
        raise AssertionError("initial cached commit cannot be reproduced")


@torch.no_grad()
def rollout_pair(fwd, initial, mask_id, k, gap, window, mode,
                 expected=None, expected_bundles=None):
    if mode not in ("baseline", "unfiltered", "gated"):
        raise ValueError("unknown mode")
    x = {s: torch.as_tensor(initial[s].copy(), dtype=torch.long) for s in SIDES}
    remaining = [int((x[s] == mask_id).sum()) for s in SIDES]
    if remaining[0] != remaining[1]:
        raise ValueError("branches have different remaining budgets")
    steps = math.ceil(remaining[0] / k)
    if window < 0 or window > steps:
        raise ValueError("window exceeds remaining steps")
    states = {s: [x[s].numpy().copy()] for s in SIDES}
    bundles = {s: [] for s in SIDES}
    window_rows, counts, manipulated = [], [], set()
    for h in range(steps):
        plans, after, count_row = {}, {}, {}
        # Both plans read the pre-step states. Commit neither until both exist.
        for side, other in (("natural", "treated"), ("treated", "natural")):
            m, l, confidence = forward_state(fwd, x[side], mask_id)
            native = baseline_bundle(confidence, m, k, gap)
            pending = x[other][m] != mask_id
            agrees = pending & (l.argmax(dim=-1) == x[other][m])
            priority = torch.zeros(len(m), dtype=torch.bool)
            if h < window and mode == "unfiltered":
                priority = pending.clone()
            elif h < window and mode == "gated":
                priority = agrees.clone()
            chosen = baseline_bundle(confidence + 2.0 * priority.float(), m, k, gap) if bool(priority.any()) else native
            positions = m[chosen].numpy()
            native_positions = m[native].numpy()
            changed = sorted(set(positions.tolist()) ^ set(native_positions.tolist()))
            if h >= window and changed:
                raise AssertionError("policy still overridden after the fixed window")
            manipulated.update(changed)
            if expected_bundles is not None:
                bs = expected_bundles[side][h]
                if not np.array_equal(np.sort(positions), np.sort(bs[bs >= 0])):
                    raise AssertionError(f"baseline replay bundle mismatch at suffix step {h + 1}")
            if h < window:
                count_row[side] = int(priority.sum())
                window_rows.append({"suffix_h": h, "rollout": side,
                                    "pending_positions": m[pending].tolist(),
                                    "agreeing_pending_positions": m[agrees].tolist(),
                                    "selected_agreeing_pending": int(agrees[chosen].sum()),
                                    "selected_incompatible_pending": int((pending & ~agrees)[chosen].sum()),
                                    "priority_positions": m[priority].tolist(),
                                    "selected_positions": positions.tolist(),
                                    "native_positions": native_positions.tolist(),
                                    "overridden_positions": changed,
                                    "selected_pending_count": int(pending[chosen].sum()),
                                    "selected_priority_count": int(priority[chosen].sum()),
                                    "mean_selected_confidence": float(confidence[chosen].mean()),
                                    **bundle_geometry(positions)})
            bs = np.full(k, -1, dtype=np.int64)
            bs[:len(positions)] = positions
            plans[side] = bs
            after[side] = commit(x[side], m, l, chosen, sample=False)
        if h < window:
            counts.append(count_row)
        for side in SIDES:
            x[side] = after[side]
            state = x[side].numpy().copy()
            if expected is not None and not np.array_equal(state, expected[side][h + 1]):
                raise AssertionError(f"baseline replay state mismatch at suffix step {h + 1}")
            states[side].append(state); bundles[side].append(plans[side])
    for side in SIDES:
        states[side] = np.stack(states[side])
        bundles[side] = np.stack(bundles[side]) if bundles[side] else np.empty((0, k), dtype=np.int64)
        visible = initial[side] != mask_id
        if np.any(states[side][:, visible] != initial[side][visible]) or np.any(states[side][-1] == mask_id):
            raise AssertionError("rollout rewrote a visible token or failed to finish")
    return {"states": states, "bundles": bundles, "window_rows": window_rows,
            "priority_counts": counts, "manipulated_positions": sorted(manipulated)}


def metrics(states, mask_id, excluded, origin, window, absolute_start):
    n, t = states["natural"], states["treated"]
    common = (n != mask_id) & (t != mask_id)
    common[:, origin] = False
    written = ((n != t) & common).sum(1)
    common[:, excluded] = False
    outside = ((n != t) & common).sum(1)
    visibility = ((n == mask_id) != (t == mask_id)).sum(1)
    equal = np.all(n == t, axis=1)
    hits = np.flatnonzero(equal)
    after_release = np.flatnonzero(equal & (np.arange(len(equal)) >= window))
    if len(after_release) and not np.array_equal(n[after_release[0]:], t[after_release[0]:]):
        raise AssertionError("adaptive trajectories separated after an identical state")
    if len(hits) and not np.array_equal(n[hits[0]:], t[hits[0]:]):
        raise AssertionError("identical states separated under a deterministic paired policy")
    durable = np.flatnonzero(np.logical_and.accumulate(equal[::-1])[::-1])
    first_written = np.flatnonzero(written)
    return {"final_differences_outside_manipulated_positions": int(outside[-1]),
            "final_differences_outside_origin": int(written[-1]),
            "mask_disagreement_at_release": int(visibility[window]),
            "written_disagreement_at_release": int(written[window]),
            "identical_state_at_release": bool(equal[window]),
            "first_identical_state_h": absolute_start + int(hits[0]) if len(hits) else None,
            "first_exact_reunion_h": absolute_start + int(durable[0]) if len(durable) else None,
            "first_written_conflict_h": absolute_start + int(first_written[0]) if len(first_written) else None,
            "visibility_disagreement_by_suffix_h": visibility.tolist(),
            "written_disagreement_by_suffix_h": written.tolist(),
            "written_disagreement_outside_manipulated_by_suffix_h": outside.tolist()}


def summarize(records, metadata, bootstrap=0):
    eligible = [r for r in records if r["eligible"]]
    groups = {}
    for r in eligible:
        groups.setdefault(r["seed"], []).append(r)
    def endpoint(key):
        values = np.asarray([np.mean([r[key] for r in group]) for group in groups.values()])
        ci = None
        if len(values) > 1 and bootstrap:
            rng = np.random.default_rng(19423)
            means = values[rng.integers(0, len(values), (bootstrap, len(values)))].mean(1)
            ci = np.quantile(means, [0.025, 0.975]).tolist()
        return {"mean_over_seeds": float(values.mean()) if len(values) else None, "seed_bootstrap_ci95": ci}
    arms = {}
    for name in ("baseline", "unfiltered", "gated"):
        def mean_metric(key):
            v = [np.mean([r["arms"][name][key] for r in group]) for group in groups.values()]
            return float(np.mean(v)) if v else None
        arms[name] = {"mean_final_differences_outside_manipulated_positions": mean_metric("final_differences_outside_manipulated_positions"),
                      "mean_final_differences_outside_origin": mean_metric("final_differences_outside_origin"),
                      "mean_mask_disagreement_at_release": mean_metric("mask_disagreement_at_release"),
                      "mean_written_disagreement_at_release": mean_metric("written_disagreement_at_release"),
                      "exact_reunions": sum(r["arms"][name]["first_exact_reunion_h"] is not None for r in eligible)}
        if name != "baseline":
            events = [e for r in eligible for e in r["scheduling"][name]]
            arms[name]["overridden_branch_steps"] = sum(bool(e["overridden_positions"]) for e in events)
            arms[name]["prioritized_positions_across_branch_steps"] = sum(len(e["priority_positions"]) for e in events)
            arms[name]["selected_agreeing_pending"] = sum(e["selected_agreeing_pending"] for e in events)
            arms[name]["selected_incompatible_pending"] = sum(e["selected_incompatible_pending"] for e in events)
            arms[name]["mean_realized_min_gap"] = float(np.mean([e["min_gap"] for e in events if e["min_gap"] is not None])) if any(e["min_gap"] is not None for e in events) else None
            arms[name]["mean_selected_confidence"] = float(np.mean([e["mean_selected_confidence"] for e in events])) if events else None
    return {"repaired_directions_processed": len(records), "eligible_desynchronized_directions": len(eligible),
            "eligible_seeds": len(groups), "already_identical_directions": len(records) - len(eligible),
            "catchup_steps": metadata["protocol"]["window"],
            "primary_unfiltered_minus_gated": endpoint("unfiltered_minus_gated"),
            "secondary_baseline_minus_gated": endpoint("baseline_minus_gated"),
            "arms": arms,
            "mean_excluded_positions_per_eligible_direction": float(np.mean([len(r["excluded_positions"]) for r in eligible])) if eligible else None,
            "scope": "exploratory reused-data calendar intervention after identical initial token repair; no quality or speed claim",
            "matching": "same k/gap rule and window; gate changes priority counts and realized overrides; shared new exclusion set",
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
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--out", type=Path, default=Path("results/trajectory_agreement_catchup_s40.npz"))
    a = p.parse_args()
    if a.steps < 1 or a.bootstrap < 0 or (a.threads is not None and a.threads < 1):
        p.error("invalid steps, threads or bootstrap")
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
    source_records = [r for r in source_records if r["triggered"]]
    if not source_records or len({(r["seed"], r["context"]) for r in source_records}) != len(source_records):
        raise ValueError("source must contain distinct corrected pairs")
    sp = source_metadata["protocol"]
    original = source_metadata["source_metadata"]["protocol"]
    if not (source_metadata.get("no_edit_suffix_replay_control") or {}).get("passed"):
        raise ValueError("source no-edit replay control did not pass")
    if digest(CKPT) != sp["checkpoint_sha256"] or str(torch.__version__) != sp["torch_version"]:
        raise ValueError("use the same checkpoint and PyTorch version as the source")
    threads = sp["threads"] if a.threads is None else a.threads
    torch.set_num_threads(threads)
    protocol = {"version": 1, "script_sha256": digest(__file__), "source_sha256": digest(a.source),
                "checkpoint_sha256": sp["checkpoint_sha256"], "torch_version": str(torch.__version__),
                "threads": threads, "window": a.steps, "bootstrap": a.bootstrap,
                "k": int(original["k"]), "gap": int(original["gap"]), "priority_bonus": 2.0,
                "gate": "own current argmax equals other branch visible token at pre-step state",
                "primary": "unfiltered minus gated Hamming outside shared union of manipulated sites; average within seed then across seeds"}
    metadata = {"protocol": protocol, "source_metadata": source_metadata, "baseline_replay_control": None}
    records, arrays = [], {}
    if a.out.exists():
        if not a.resume:
            raise FileExistsError(f"{a.out} exists; use --resume or another output")
        with np.load(a.out, allow_pickle=False) as z:
            metadata = json.loads(z["metadata_json"].item())
            if metadata["protocol"] != protocol:
                raise ValueError("resume protocol differs")
            records = json.loads(z["records_json"].item())
            arrays = {key: z[key] for key in z.files if not key.endswith("_json")}
    done = {(r["seed"], r["context"]) for r in records}
    model, cfg = load_model()
    model.eval(); fwd = nano_forward_fn(model)
    mask_id, k, gap = int(cfg["mask_id"]), protocol["k"], protocol["gap"]
    print(f"Agreement catch-up: {len(source_records)} repaired directions; window={a.steps}; "
          "argmax tokens only; baseline / unfiltered / gated", flush=True)
    for source in source_records:
        seed, context = source["seed"], source["context"]
        if (seed, context) in done:
            continue
        start = time.monotonic()
        prefix = f"seed_{seed}_{context}_"
        base = {s: source_arrays[prefix + "repair_states_" + s] for s in SIDES}
        base_bundles = {s: source_arrays[prefix + "repair_bundles_" + s] for s in SIDES}
        for s in SIDES:
            validate_cached(base[s], base_bundles[s], mask_id, k, int(cfg["seq_len"]))
        initial = {s: base[s][0].copy() for s in SIDES}
        common = (initial["natural"] != mask_id) & (initial["treated"] != mask_id)
        if np.any(initial["natural"][common] != initial["treated"][common]):
            raise ValueError("source correction did not erase the initial common-token conflict")
        if np.array_equal(initial["natural"], initial["treated"]):
            if not np.array_equal(base["natural"], base["treated"]):
                raise ValueError("identical source states did not stay identical")
            records.append({"seed": seed, "context": context, "eligible": False})
            save(a.out, records, arrays, metadata)
            continue
        if a.steps > len(base["natural"]) - 1:
            raise ValueError("window exceeds remaining trajectory; refusing to change it silently")
        for s in SIDES:
            verify_boundary(fwd, base[s], base_bundles[s], mask_id, k, gap)
        if metadata["baseline_replay_control"] is None:
            rollout_pair(fwd, initial, mask_id, k, gap, 0, "baseline", expected=base, expected_bundles=base_bundles)
            metadata["baseline_replay_control"] = {"seed": seed, "context": context, "passed": True}
        unfiltered = rollout_pair(fwd, initial, mask_id, k, gap, a.steps, "unfiltered")
        witness = rollout_pair(fwd, initial, mask_id, k, gap, a.steps, "gated")
        excluded = sorted(set(source["excluded_positions"]) | set(unfiltered["manipulated_positions"]) | set(witness["manipulated_positions"]))
        w, origin = source["trigger_commit_h"], source["original_perturbation_position"]
        arms = {"baseline": metrics(base, mask_id, excluded, origin, a.steps, w)}
        for name, run in (("unfiltered", unfiltered), ("gated", witness)):
            arms[name] = metrics(run["states"], mask_id, excluded, origin, a.steps, w)
            for s in SIDES:
                arrays[prefix + name + "_states_" + s] = run["states"][s]
                arrays[prefix + name + "_bundles_" + s] = run["bundles"][s]
        for s in SIDES:
            arrays[prefix + "baseline_states_" + s] = base[s]
            arrays[prefix + "baseline_bundles_" + s] = base_bundles[s]
        result = {"seed": seed, "context": context, "eligible": True, "absolute_start_h": w,
                  "excluded_positions": excluded, "original_perturbation_position": origin,
                  "priority_counts": {"unfiltered": unfiltered["priority_counts"], "gated": witness["priority_counts"]},
                  "scheduling": {"unfiltered": unfiltered["window_rows"], "gated": witness["window_rows"]},
                  "arms": arms,
                  "baseline_minus_gated": arms["baseline"]["final_differences_outside_manipulated_positions"] - arms["gated"]["final_differences_outside_manipulated_positions"],
                  "unfiltered_minus_gated": arms["unfiltered"]["final_differences_outside_manipulated_positions"] - arms["gated"]["final_differences_outside_manipulated_positions"],
                  "seconds": time.monotonic() - start}
        records.append(result)
        save(a.out, records, arrays, metadata)
        names = ("baseline", "unfiltered", "gated")
        final = [arms[n]["final_differences_outside_manipulated_positions"] for n in names]
        masks = [arms[n]["mask_disagreement_at_release"] for n in names]
    summary = save(a.out, records, arrays, metadata, a.bootstrap)
    print(json.dumps({key: value for key, value in summary.items() if key not in ("per_pair", "metadata")}, indent=2))
    print(f"saved {a.out.with_name(a.out.stem + '_summary.json')}")


if __name__ == "__main__":
    main()
