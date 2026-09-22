from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

try:
    from src.scripts.trajectory_agreement_catchup import (
        digest, validate_cached, verify_boundary, rollout_pair, metrics, SIDES,
        CKPT, load_model, nano_forward_fn, forward_state, baseline_bundle, commit,
        bundle_geometry)
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError("Install trajectory_agreement_catchup.py in src/scripts/ alongside this script") from exc

def conflicts_at(states, bundles, h, mask):
    before={s:states[s][h-1] for s in SIDES}
    after={s:states[s][h] for s in SIDES}
    common=(after["natural"]!=mask)&(after["treated"]!=mask)
    positions=np.flatnonzero(common&(after["natural"]!=after["treated"]))
    out=[]
    for p in positions:
        n=before["natural"][p]==mask;t=before["treated"][p]==mask
        if not n and not t: raise ValueError("conflict was already written before trigger")
        side="treated" if t else "natural"
        anchor=int(after["natural" if side=="treated" else "treated"][p])
        out.append({"position":int(p),"side":side,"anchor":anchor,
                    "kind":"simultaneous" if n and t else "delayed"})
    return out

def feasible_pairs(masked, native, target, conflicting, gap):
    native=list(map(int,native));masked=list(map(int,masked))
    candidates=[]
    for control in native:
        if control in conflicting: continue
        kept_target=[p for p in native if p!=target]
        kept_control=[p for p in native if p!=control]
        replacements=[]
        for q in masked:
            if q in native:continue
            def valid(ps):
                ps=sorted(ps)
                return all(b-a>=gap for a,b in zip(ps,ps[1:]))
            if valid(kept_target+[q]) and valid(kept_control+[q]):
                replacements.append(q)
        if replacements:candidates.append((control,replacements))
    return candidates

def summarize(records, bootstrap=0):
    eligible=[r for r in records if r["eligible"]];groups={}
    for r in eligible:groups.setdefault(r["seed"],[]).append(r)
    def endpoint(key):
        v=np.array([np.mean([r[key] for r in g]) for g in groups.values()])
        ci=None
        if len(v)>1 and bootstrap:
            rng=np.random.default_rng(917513)
            ci=np.quantile(v[rng.integers(len(v),size=(bootstrap,len(v)))].mean(1),[.025,.975]).tolist()
        return {"mean_over_seeds":float(v.mean()) if len(v) else None,"seed_bootstrap_ci95":ci}
    reasons={}
    for r in records:
        if not r["eligible"]:reasons[r["reason"]]=reasons.get(r["reason"],0)+1
    return {"directions_processed":len(records),"eligible_directions":len(eligible),
            "eligible_seeds":len(groups),"excluded_reasons":reasons,
            "primary_control_minus_delay":endpoint("control_minus_delay"),
            "secondary_no_edit_minus_delay":endpoint("no_edit_minus_delay"),
            "target_eventually_matches_anchor":sum(r["target_outcome"]["matches_anchor"] for r in eligible),
            "target_prediction_matches_after_one_step":sum(r["target_outcome"]["prediction_matches_at_release"] for r in eligible),
            "multiple_imminent_conflicts":sum(r["n_conflicts"]>1 for r in eligible),
            "scope":"exploratory paired one-swap delay; same branch/replacement/k, strict gap; one random feasible control; shared exclusions",
            "per_pair":records}

def save(path,records,arrays,metadata,bootstrap=0):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    with tmp.open("wb") as f:np.savez_compressed(f,records_json=json.dumps(records,allow_nan=False),
        metadata_json=json.dumps(metadata,allow_nan=False),**arrays)
    os.replace(tmp,path)
    summary=summarize(records,bootstrap)
    target=path.with_name(path.stem+"_summary.json");tmp=target.with_name(target.name+".tmp")
    tmp.write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n");os.replace(tmp,target)
    return summary

@torch.no_grad()
def changed_rollout(fwd,initial,native_positions,side,drop,replacement,mask,k,gap):
    after={};first={}
    for s in SIDES:
        x=torch.tensor(initial[s],dtype=torch.long);m,l,c=forward_state(fwd,x,mask)
        positions=list(native_positions[s])
        if s==side:
            positions.remove(drop);positions.append(replacement)
        rows=torch.tensor([int((m==p).nonzero().item()) for p in positions],dtype=torch.long)
        after[s]=commit(x,m,l,rows,sample=False).numpy()
        bs=np.full(k,-1,dtype=np.int64);bs[:len(positions)]=positions;first[s]=bs
    if after[side][drop]!=mask:raise AssertionError("target was not deferred")
    suffix=rollout_pair(fwd,after,mask,k,gap,0,"baseline")
    return {"states":{s:np.concatenate([initial[s][None],suffix["states"][s]]) for s in SIDES},
            "bundles":{s:np.concatenate([first[s][None],suffix["bundles"][s]]) for s in SIDES}}

@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path)
    p.add_argument("--out",type=Path,default=Path("results/trajectory_first_conflict_delay_p16_s80.npz"))
    p.add_argument("--bootstrap",type=int,default=2000)
    p.add_argument("--resume",action="store_true")
    a=p.parse_args()
    if a.source.resolve()==a.out.resolve() or a.bootstrap<0:p.error("invalid output or bootstrap")
    with np.load(a.source,allow_pickle=False) as z:
        sm=json.loads(z["metadata_json"].item());sr=json.loads(z["records_json"].item())
        sa={k:z[k] for k in z.files if not k.endswith("_json")}
    sp=sm["protocol"]
    if not sm.get("identical_control",{}).get("passed"):raise ValueError("source identical control missing")
    if digest(CKPT)!=sp["checkpoint_sha256"] or str(torch.__version__)!=sp["torch_version"]:
        raise ValueError("use same checkpoint and PyTorch version as source")
    protocol={"version":1,"script_sha256":digest(__file__),"source_sha256":digest(a.source),
              "checkpoint_sha256":digest(CKPT),"torch_version":str(torch.__version__),
              "threads":sp["threads"],"k":sp["k"],"gap":sp["gap"],"bootstrap":a.bootstrap,
              "target":"lowest-position first conflict, later writer or treated if simultaneous",
              "control":"uniform feasible nonconflict position same branch; common highest-confidence replacement",
              "rng":"SeedSequence([620913,seed,context_b])"}
    metadata={"protocol":protocol,"source_metadata":sm,"baseline_replay_control":None}
    records=[];arrays={}
    if a.out.exists():
        if not a.resume:raise FileExistsError("output exists; use --resume")
        with np.load(a.out,allow_pickle=False) as z:
            metadata=json.loads(z["metadata_json"].item())
            if metadata["protocol"]!=protocol:raise ValueError("resume protocol differs")
            records=json.loads(z["records_json"].item());arrays={k:z[k] for k in z.files if not k.endswith("_json")}
    torch.set_num_threads(sp["threads"]);model,cfg=load_model();model.eval();fwd=nano_forward_fn(model)
    mask,k,gap=int(cfg["mask_id"]),sp["k"],sp["gap"]
    done={(r["seed"],r["context"]) for r in records}
    print(f"First-conflict delay: {len(sr)} seeds; one swap; same replacement control; strict gap={gap}",flush=True)
    for source in sr:
        for context,d in source["directions"].items():
            seed=source["seed"]
            if (seed,context) in done:continue
            row={"seed":seed,"context":context,"eligible":False};start=time.monotonic()
            h=d["first_common_written_disagreement_h"]
            if not d["active_override"] or h is None:
                row["reason"]="no_imminent_written_conflict";records.append(row);save(a.out,records,arrays,metadata);continue
            if h<2:raise ValueError("trigger overlaps imposed initial step")
            prefix=f"seed_{seed}_{context}_"
            states={s:sa[prefix+"states_"+s] for s in SIDES}
            bundles={s:sa[prefix+"bundles_"+s] for s in SIDES}
            for s in SIDES:validate_cached(states[s],bundles[s],mask,k,int(cfg["seq_len"]))
            conflict=conflicts_at(states,bundles,h,mask)
            if not conflict:raise ValueError("missing cached first conflict")
            target=conflict[0];side=target["side"];position=target["position"]
            initial={s:states[s][h-1].copy() for s in SIDES}
            native={s:bundles[s][h-1][bundles[s][h-1]>=0].tolist() for s in SIDES}
            options=feasible_pairs(np.flatnonzero(initial[side]==mask),native[side],position,
                                   {v["position"] for v in conflict},gap)
            if not options:
                row["reason"]="no_common_gap_feasible_swap";records.append(row);save(a.out,records,arrays,metadata);continue
            base={s:states[s][h-1:] for s in SIDES};bb={s:bundles[s][h-1:] for s in SIDES}
            for s in SIDES:verify_boundary(fwd,base[s],bb[s],mask,k,gap)
            if metadata["baseline_replay_control"] is None:
                rollout_pair(fwd,initial,mask,k,gap,0,"baseline",expected=base,expected_bundles=bb)
                metadata["baseline_replay_control"]={"seed":seed,"context":context,"passed":True}
            rng=np.random.default_rng(np.random.SeedSequence([620913,int(seed),int(context=="b")]))
            control,replacements=options[int(rng.integers(len(options)))]
            m,l,c=forward_state(fwd,torch.tensor(initial[side]),mask)
            conf={int(pos):float(c[i]) for i,pos in enumerate(m)}
            replacement=max(replacements,key=lambda pos:(conf[pos],-pos))
            runs={"delay":changed_rollout(fwd,initial,native,side,position,replacement,mask,k,gap),
                  "control":changed_rollout(fwd,initial,native,side,control,replacement,mask,k,gap)}
            origin=source["intervention"]["position"];excluded=sorted({origin,position,control,replacement})
            arms={"no_edit":metrics(base,mask,excluded,origin,1,h-1)}
            for arm,run in runs.items():
                for s in SIDES:validate_cached(run["states"][s],run["bundles"][s],mask,k,int(cfg["seq_len"]))
                arms[arm]=metrics(run["states"],mask,excluded,origin,1,h-1)
            delayed=runs["delay"]["states"][side]
            when=int(np.flatnonzero(delayed[:,position]!=mask)[0])
            mm,ll,cc=forward_state(fwd,torch.tensor(delayed[1]),mask)
            prediction=int(ll[int((mm==position).nonzero().item())].argmax())
            key="final_differences_outside_manipulated_positions"
            row.update(eligible=True,trigger_h=h,target=target,control_position=control,replacement=replacement,
                       n_conflicts=len(conflict),excluded_positions=excluded,arms=arms,
                       geometry={arm:bundle_geometry(run["bundles"][side][0]) for arm,run in runs.items()},
                       selected_confidence={arm:float(np.mean([conf[int(p)] for p in run["bundles"][side][0] if p>=0])) for arm,run in runs.items()},
                       target_outcome={"commit_h":h-1+when,"token":int(delayed[when,position]),
                                       "matches_anchor":int(delayed[when,position])==target["anchor"],
                                       "prediction_after_one_step":prediction,"prediction_matches_at_release":prediction==target["anchor"]},
                       control_minus_delay=arms["control"][key]-arms["delay"][key],
                       no_edit_minus_delay=arms["no_edit"][key]-arms["delay"][key])
            for arm,run in runs.items():
                for s in SIDES:
                    arrays[prefix+arm+"_states_"+s]=run["states"][s];arrays[prefix+arm+"_bundles_"+s]=run["bundles"][s]
            records.append(row);save(a.out,records,arrays,metadata)
    summary=save(a.out,records,arrays,metadata,a.bootstrap)
    print(json.dumps({k:v for k,v in summary.items() if k!="per_pair"},indent=2))
if __name__=="__main__":main()
