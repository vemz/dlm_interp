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
                 expected=None, expected_bundles=None, random_seed=0):
    if mode not in ("baseline", "unfiltered", "gated", "random_pending"):
        raise ValueError("unknown mode")
    rng = np.random.default_rng(random_seed)
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
            elif h < window and mode == "random_pending":
                # Match the gate's count at THIS state; never prioritize a nonpending site.
                candidates = torch.nonzero(pending, as_tuple=False).flatten().numpy()
                count = int(agrees.sum())
                if count:
                    chosen_priority = rng.choice(candidates, count, replace=False)
                    priority[torch.as_tensor(chosen_priority)] = True
                if int(priority.sum()) != count or bool((priority & ~pending).any()):
                    raise AssertionError("local priority-count matching failed")
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
        values = np.asarray([np.mean([r[key] for r in g]) for g in groups.values()])
        ci = None
        if len(values) > 1 and bootstrap:
            rng = np.random.default_rng(19423)
            ci = np.quantile(values[rng.integers(0, len(values), (bootstrap, len(values)))].mean(1), [.025, .975]).tolist()
        return {"mean_over_seeds": float(values.mean()) if len(values) else None, "seed_bootstrap_ci95": ci}
    def arm_mean(arm, key):
        return float(np.mean([np.mean([r["arms"][arm][key] for r in g]) for g in groups.values()])) if groups else None
    arms = {}
    for arm in ("baseline", "gated", "random_mean"):
        arms[arm] = {key: arm_mean(arm,key) for key in (
            "final_differences_outside_manipulated_positions", "final_differences_outside_origin",
            "mask_disagreement_at_release", "written_disagreement_at_release")}
    events = [e for r in eligible for run in r["random_scheduling"] for e in run]
    gate_events = [e for r in eligible for e in r["gated_scheduling"]]
    diagnostics = {
        "random_repetitions": metadata["protocol"]["repetitions"],
        "first_step_count_mismatches": sum(e["suffix_h"] == 0 and e["count_delta_vs_cached_gate"] != 0 for e in events),
        "later_step_count_mismatches": sum(e["suffix_h"] > 0 and e["count_delta_vs_cached_gate"] != 0 for e in events),
        "later_step_events": sum(e["suffix_h"] > 0 for e in events),
        "mean_abs_count_delta_vs_cached_gate": float(np.mean([abs(e["count_delta_vs_cached_gate"]) for e in events])) if events else None,
        "random_overridden_branch_steps_per_repetition": sum(bool(e["overridden_positions"]) for e in events) / metadata["protocol"]["repetitions"],
        "gated_overridden_branch_steps": sum(bool(e["overridden_positions"]) for e in gate_events),
        "random_selected_incompatible_pending_per_repetition": sum(e["selected_incompatible_pending"] for e in events) / metadata["protocol"]["repetitions"],
        "gated_selected_incompatible_pending": sum(e["selected_incompatible_pending"] for e in gate_events),
    }
    return {"pairs_processed":len(records), "eligible_directions":len(eligible),"eligible_seeds":len(groups),
            "primary_random_minus_gated":endpoint("random_minus_gated"),
            "secondary_baseline_minus_gated":endpoint("baseline_minus_gated"),
            "arms":arms,"count_matching":diagnostics,
            "scope":"exploratory local-count matched random-pending control; second-step states and realized overrides can differ",
            "per_pair":records,"metadata":metadata}

def save(path, records, arrays, metadata, bootstrap=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as f:
        np.savez_compressed(f, records_json=json.dumps(records, allow_nan=False),
                            metadata_json=json.dumps(metadata, allow_nan=False), **arrays)
    os.replace(temp,path)
    summary=summarize(records,metadata,bootstrap)
    target=path.with_name(path.stem+"_summary.json")
    temp=target.with_name(target.name+".tmp")
    temp.write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    os.replace(temp,target)
    return summary

@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path)
    p.add_argument("--repetitions",type=int,default=4)
    p.add_argument("--bootstrap",type=int,default=2000)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--out",type=Path,default=Path("results/trajectory_agreement_random_control_s80.npz"))
    a=p.parse_args()
    if a.repetitions < 1 or a.bootstrap < 0 or a.source.resolve()==a.out.resolve():
        p.error("invalid repetition count, bootstrap or output")
    with np.load(a.source,allow_pickle=False) as z:
        sm=json.loads(z["metadata_json"].item())
        sr=json.loads(z["records_json"].item())
        sa={k:z[k] for k in z.files if not k.endswith("_json")}
    sp=sm["protocol"]
    if not (sm.get("baseline_replay_control") or {}).get("passed"):
        raise ValueError("source baseline replay control missing")
    if sp.get("gate") != "own current argmax equals other branch visible token at pre-step state":
        raise ValueError("unrecognized source gate")
    if digest(CKPT)!=sp["checkpoint_sha256"] or str(torch.__version__)!=sp["torch_version"]:
        raise ValueError("use the source checkpoint and PyTorch version")
    if len({(r["seed"],r["context"]) for r in sr})!=len(sr):
        raise ValueError("duplicate source pairs")
    torch.set_num_threads(sp["threads"])
    protocol={"version":1,"script_sha256":digest(__file__),"source_sha256":digest(a.source),
              "checkpoint_sha256":digest(CKPT),"torch_version":str(torch.__version__),
              "threads":sp["threads"],"k":sp["k"],"gap":sp["gap"],"window":sp["window"],
              "repetitions":a.repetitions,"bootstrap":a.bootstrap,
              "rng_rule":"SeedSequence([915731, seed, context_b, repetition])",
              "matching":"count agreeing pending in own current state; uniform subset of all own pending"}
    metadata={"protocol":protocol,"source_metadata":sm,"gated_replay_control":None}
    records=[];arrays={}
    if a.out.exists():
        if not a.resume: raise FileExistsError("output exists; use --resume")
        with np.load(a.out,allow_pickle=False) as z:
            metadata=json.loads(z["metadata_json"].item())
            if metadata["protocol"]!=protocol: raise ValueError("resume protocol differs")
            records=json.loads(z["records_json"].item())
            arrays={k:z[k] for k in z.files if not k.endswith("_json")}
    done={(r["seed"],r["context"]) for r in records}
    model,cfg=load_model();model.eval();fwd=nano_forward_fn(model)
    mask,k,gap,window=int(cfg["mask_id"]),sp["k"],sp["gap"],sp["window"]
    # Retain the source exclusion superset, then add all new manipulated sites.
    print(f"Random-pending control: {len(sr)} pairs; window={window}; repetitions={a.repetitions}; local count matching",flush=True)
    for source in sr:
        seed,context=source["seed"],source["context"]
        if (seed,context) in done: continue
        if not source["eligible"]:
            records.append({"seed":seed,"context":context,"eligible":False})
            save(a.out,records,arrays,metadata);continue
        start=time.monotonic();prefix=f"seed_{seed}_{context}_"
        cached={arm:{s:sa[prefix+arm+"_states_"+s] for s in SIDES} for arm in ("baseline","gated")}
        bundles={arm:{s:sa[prefix+arm+"_bundles_"+s] for s in SIDES} for arm in cached}
        for arm in cached:
            for side in SIDES: validate_cached(cached[arm][side],bundles[arm][side],mask,k,int(cfg["seq_len"]))
        initial={s:cached["baseline"][s][0].copy() for s in SIDES}
        if any(not np.array_equal(initial[s],cached["gated"][s][0]) for s in SIDES):
            raise ValueError("cached arms have different starting states")
        for side in SIDES: verify_boundary(fwd,cached["baseline"][side],bundles["baseline"][side],mask,k,gap)
        if metadata["gated_replay_control"] is None:
            rollout_pair(fwd,initial,mask,k,gap,window,"gated",expected=cached["gated"],expected_bundles=bundles["gated"])
            metadata["gated_replay_control"]={"seed":seed,"context":context,"passed":True}
        runs=[]
        for rep in range(a.repetitions):
            rng_seed=np.random.SeedSequence([915731,int(seed),int(context=="b"),rep])
            run=rollout_pair(fwd,initial,mask,k,gap,window,"random_pending",random_seed=rng_seed)
            for e in run["window_rows"]:
                cached_count=source["priority_counts"]["gated"][e["suffix_h"]][e["rollout"]]
                e["count_delta_vs_cached_gate"]=len(e["priority_positions"])-cached_count
                if e["suffix_h"]==0 and e["count_delta_vs_cached_gate"]!=0:
                    raise AssertionError("first-step count mismatch")
            runs.append(run)
        excluded=sorted(set(source["excluded_positions"]).union(*(set(run["manipulated_positions"]) for run in runs)))
        origin=source["original_perturbation_position"];w=source["absolute_start_h"]
        arms={arm:metrics(states,mask,excluded,origin,window,w) for arm,states in cached.items()}
        rms=[metrics(run["states"],mask,excluded,origin,window,w) for run in runs]
        numeric=("final_differences_outside_manipulated_positions","final_differences_outside_origin",
                 "mask_disagreement_at_release","written_disagreement_at_release")
        arms["random_mean"]={key:float(np.mean([r[key] for r in rms])) for key in numeric}
        for arm in cached:
            for side in SIDES:
                arrays[prefix+arm+"_states_"+side]=cached[arm][side]
                arrays[prefix+arm+"_bundles_"+side]=bundles[arm][side]
        for rep,run in enumerate(runs):
            for side in SIDES:
                arrays[prefix+f"random_{rep}_states_"+side]=run["states"][side]
                arrays[prefix+f"random_{rep}_bundles_"+side]=run["bundles"][side]
        key="final_differences_outside_manipulated_positions"
        result={"seed":seed,"context":context,"eligible":True,"excluded_positions":excluded,
                "original_perturbation_position":origin,"absolute_start_h":w,"arms":arms,
                "random_metrics":rms,"random_scheduling":[run["window_rows"] for run in runs],
                "gated_scheduling":source["scheduling"]["gated"],
                "random_minus_gated":arms["random_mean"][key]-arms["gated"][key],
                "baseline_minus_gated":arms["baseline"][key]-arms["gated"][key]}
        records.append(result);save(a.out,records,arrays,metadata)
    summary=save(a.out,records,arrays,metadata,a.bootstrap)
    print(json.dumps({k:v for k,v in summary.items() if k not in ("per_pair","metadata")},indent=2))

if __name__=="__main__":
    main()
