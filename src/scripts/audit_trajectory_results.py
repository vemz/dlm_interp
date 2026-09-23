from __future__ import annotations
import argparse
from collections import defaultdict
import json
import hashlib
from pathlib import Path
import numpy as np

def paired(rows, key, draws, rng_seed):
    groups=defaultdict(list)
    for r in rows:
        value=float(r[key])
        # Recompute contrasts whenever compact arm values are available.
        pairs = r.get(key + "_arms")
        if pairs is None:
            pairs = (("no_edit", "repair") if key=="primary" else ("control", "repair")) if "repair" in r else (
                ("unfiltered", "gated") if key=="primary" else ("baseline", "gated"))
        if all(name in r for name in pairs):
            computed=float(r[pairs[0]])-float(r[pairs[1]])
            if not np.isclose(value,computed,rtol=0,atol=1e-10):
                raise ValueError("stored contrast disagrees with arm values")
            value=computed
        groups[int(r["seed"])].append(value)
    if not groups: raise ValueError("no eligible rows")
    seeds=list(groups)
    values=np.array([np.mean(groups[s]) for s in seeds])
    if not np.all(np.isfinite(values)): raise ValueError("nonfinite observations")
    rng=np.random.default_rng(rng_seed)
    ci=np.quantile(values[rng.integers(len(values),size=(draws,len(values)))].mean(1),[.025,.975]).tolist() if len(values)>1 else None
    return {"directions":len(rows),"seeds":len(seeds),"mean":float(values.mean()),"ci95_audit":ci,
            "seed_signs":{"positive":int((values>0).sum()),"zero":int((values==0).sum()),"negative":int((values<0).sum())},
            "leave_one_seed_out_mean_range":[float(min((values.sum()-v)/(len(values)-1) for v in values)),
                                            float(max((values.sum()-v)/(len(values)-1) for v in values))] if len(values)>1 else None}

def auc_value(rows):
    pos=np.array([r["risk"] for r in rows if r["persistent"]])
    neg=np.array([r["risk"] for r in rows if not r["persistent"]])
    if not len(pos) or not len(neg):return None
    return float(((pos[:,None]>neg).astype(float)+.5*(pos[:,None]==neg)).mean())

def stability(rows,draws):
    groups=defaultdict(list)
    for r in rows:
        if not np.isfinite(r["risk"]):raise ValueError("nonfinite risk")
        groups[int(r["seed"])].append(r)
    group_list=list(groups.values());rng=np.random.default_rng(90210);values=[]
    for _ in range(draws):
        sample=[r for i in rng.integers(len(group_list),size=len(group_list)) for r in group_list[i]]
        a=auc_value(sample)
        if a is not None:values.append(a)
    return {"active_directions":len(rows),"active_seeds":len(groups),"persistent":sum(bool(r["persistent"]) for r in rows),
            "absorbed":sum(not r["persistent"] for r in rows),"auc":auc_value(rows),
            "ci95_audit":np.quantile(values,[.025,.975]).tolist() if values else None,
            "valid_bootstrap_draws":len(values)}

def file_sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def from_npz(path):
    with np.load(path, allow_pickle=False) as z:
        rs = json.loads(z["records_json"].item())
        metadata = json.loads(z["metadata_json"].item())
    summary_path = path.with_name(path.stem + "_summary.json")
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    # Keep the historical aggregate estimates, not duplicate full trajectories.
    reported = {k: v for k, v in summary.items()
                if k.startswith(("primary_", "secondary_"))}
    source = {"kind": "local_npz", "name": path.name, "sha256": file_sha256(path)}
    if summary_path.exists():
        source.update(summary_name=summary_path.name, summary_sha256=file_sha256(summary_path))
    common = dict(id=path.stem, source=source, metadata=metadata, reported=reported)
    if rs and "directions" in rs[0]:
        rows = [{"seed": r["seed"], "context": c,
                 "risk": d["release_features"]["risk_score"],
                 "min_anchor_margin": d["release_features"]["min_anchor_margin"],
                 "persistent": d["persistent_final_difference"]}
                for r in rs for c, d in r["directions"].items() if d["active_override"]]
        return dict(common, kind="stability", rows=rows)
    choices = [("control_minus_delay", "no_edit_minus_delay"),
               ("random_minus_gated", "baseline_minus_gated"),
               ("unfiltered_minus_gated", "baseline_minus_gated"),
               ("no_edit_minus_repair", "control_minus_repair")]
    active = [r for r in rs if r.get("eligible", r.get("triggered", False))]
    if not active:
        raise ValueError("no eligible records in NPZ")
    keys = next(((p, s) for p, s in choices if p in active[0]), None)
    if keys is None:
        raise ValueError("unsupported NPZ record schema")
    primary, secondary = keys
    contrasts = {
        "control_minus_delay": ("control", "delay"),
        "no_edit_minus_delay": ("no_edit", "delay"),
        "random_minus_gated": ("random_mean", "gated"),
        "baseline_minus_gated": ("baseline", "gated"),
        "unfiltered_minus_gated": ("unfiltered", "gated"),
        "no_edit_minus_repair": ("no_edit", "repair"),
        "control_minus_repair": ("nonaligned_control", "repair"),
    }
    metric = ("final_differences_outside_edited_positions" if "repair" in primary
              else "final_differences_outside_manipulated_positions")
    rows = []
    for r in active:
        row = {"seed": r["seed"], "context": r["context"]}
        for label, contrast in (("primary", primary), ("secondary", secondary)):
            left, right = contrasts[contrast]
            row[label] = r[contrast]
            row[label + "_arms"] = [left, right]
            for arm in (left, right):
                row[arm] = r["arms"][arm][metric]
            if not np.isclose(row[label], row[left] - row[right], rtol=0, atol=1e-10):
                raise ValueError(f"{path.name}: stored contrast disagrees with arms")
        rows.append(row)
    return dict(common, kind="paired", primary=primary, secondary=secondary, rows=rows)


def refresh_evidence(data, directory):
    sources = {
        "stability_p8": "trajectory_delayed_stability_confirm_s80",
        "repair_p8": "trajectory_first_conflict_repair_confirm_s80",
        "agreement_p8": "trajectory_agreement_catchup_confirm_s80",
        "stability_p16": "trajectory_delayed_stability_p16_s80",
        "repair_p16": "trajectory_first_conflict_repair_p16_s80",
        "agreement_p16": "trajectory_agreement_catchup_p16_s80",
        "random_p16": "trajectory_agreement_random_control_p16_s80",
        "random_p8": "trajectory_agreement_random_control_s80",
        "delay_p16": "trajectory_first_conflict_delay_p16_s80",
    }
    for model in (1, 2):
        for label, stem in (("stability", "delayed_stability"),
                            ("repair", "first_conflict_repair"),
                            ("agreement", "agreement_catchup"),
                            ("random", "agreement_random_control")):
            sources[f"{label}_model{model}_p8"] = f"trajectory_{stem}_model{model}_s80"
    previous = {r["id"]: r for r in data["runs"]}
    runs = []
    for name, stem in sources.items():
        run = from_npz(directory / (stem + ".npz"))
        run["id"] = name
        old = previous.get(name, {})
        if "source" in old and old["source"] != run["source"]:
            run["previous_source"] = old["source"]
        elif "previous_source" in old:
            run["previous_source"] = old["previous_source"]
        runs.append(run)
    runs.extend(r for name, r in previous.items() if name not in sources)
    data = dict(data, runs=runs)
    data.pop("checkpoint_sha256", None)  # Each run records its own checkpoint.
    return data

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence",type=Path,default=Path(__file__).resolve().parents[2]/"results/trajectory_evidence.json")
    p.add_argument("--npz",type=Path,nargs="+",help="audit original NPZ files instead of compact evidence")
    p.add_argument("--draws",type=int,default=2000)
    p.add_argument("--check",action="store_true",help="fail if recomputed primary/secondary point estimates disagree with recorded summaries")
    p.add_argument("--json-out",type=Path)
    p.add_argument("--refresh-from-npz", type=Path, help="rebuild compact evidence from the fixed local result series; write only after successful audit")
    a=p.parse_args()
    if a.npz and a.refresh_from_npz:p.error("choose --npz or --refresh-from-npz")
    if a.refresh_from_npz:a.check=True
    if a.draws<1:p.error("--draws must be positive")
    if a.npz:runs=[from_npz(path) for path in a.npz]
    else:
        data=json.loads(a.evidence.read_text())
        if data["schema_version"]!=1:raise ValueError("unsupported evidence schema")
        if a.refresh_from_npz:data=refresh_evidence(data,a.refresh_from_npz)
        runs=data["runs"]
    outputs=[]
    print("Statistical audit only; no model replay. Positive differences favor the tested intervention.")
    print("| Run | Directions / seeds | Primary estimate | Audit 95% CI |")
    print("|---|---:|---:|---|")
    for run in runs:
        rows=run["rows"];report=run.get("reported",{});out={"id":run["id"],"reported":report}
        if not rows:
            out["status"]="summary_only_not_recomputed"
            print(f"| {run['id']} | — | summary only | not recomputed |");outputs.append(out);continue
        ids=[(r["seed"],r["context"]) for r in rows]
        if len(set(ids))!=len(ids):raise ValueError("duplicate seed-direction")
        if run["kind"]=="stability":
            result=stability(rows,a.draws);value=result["auc"]
            expected=report.get("primary_release_margin_auc")
            if a.check and expected is not None and not np.isclose(value,expected,rtol=0,atol=1e-10):
                raise AssertionError(f"{run['id']} AUC mismatch")
            if rows and all("min_anchor_margin" in r for r in rows):
                subset = [r for r in rows if r["min_anchor_margin"] > 0]
                out["posthoc_strictly_positive_margins_h1"] = stability(subset, a.draws) if subset else {"active_directions": 0, "auc": None}
            size=f"{result['active_directions']} / {result['active_seeds']}"
        else:
            seed=917513 if run["primary"]=="control_minus_delay" else 19423
            result=paired(rows,"primary",a.draws,seed);value=result["mean"]
            expected=report.get("primary_"+run["primary"],{}).get("mean_over_seeds")
            if a.check and expected is not None and not np.isclose(value,expected,rtol=0,atol=1e-10):
                raise AssertionError(f"{run['id']} primary mismatch: {value} != {expected}")
            if all("secondary" in r for r in rows):
                out["secondary"]=paired(rows,"secondary",a.draws,seed)
                exp=report.get("secondary_"+run["secondary"],{}).get("mean_over_seeds")
                if a.check and exp is not None and not np.isclose(out["secondary"]["mean"],exp,rtol=0,atol=1e-10):
                    raise AssertionError(f"{run['id']} secondary mismatch")
            size=f"{result['directions']} / {result['seeds']}"
            if run["id"]=="agreement_p8":
                # Fixed descriptive sensitivity analysis, explicitly outcome-selected.
                subset=[r for r in rows if r["seed"] not in {817,828,829}]
                out["posthoc_excluding_817_828_829"]=paired(subset,"primary",a.draws,seed)
        ci=result["ci95_audit"];citext=f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "undefined"
        out.update(status="recomputed",primary=result);outputs.append(out)
        value_text = "undefined" if value is None else f"{value:.6f}"
        print(f"| {run['id']} | {size} | {value_text} | {citext} |")
    if a.check:print("PASS: available point estimates agree; summary-only runs are not independently verified.")
    if a.refresh_from_npz:
        a.evidence.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
    if a.json_out:
        a.json_out.parent.mkdir(parents=True,exist_ok=True)
        a.json_out.write_text(json.dumps({"bootstrap_draws":a.draws,"runs":outputs},indent=2,allow_nan=False)+"\n")

if __name__=="__main__":main()
