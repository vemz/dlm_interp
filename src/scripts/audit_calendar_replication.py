"""Validate a complete replication and describe its saved temporal traces."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

from calendar_swap import digest, load_rows, analyse


def trajectory_features(baseline, treated, mask, swap_step):
    if len(baseline) != len(treated):
        raise ValueError('Mismatched histories')
    metrics = []
    for step, (a,b) in enumerate(zip(baseline, treated)):
        if len(a) != len(b):
            raise ValueError('Mismatched states')
        metrics.append(dict(decisions_completed=step,
            state_hamming=sum(x!=y for x,y in zip(a,b)),
            committed_token_conflicts=sum(x!=y and x!=mask and y!=mask for x,y in zip(a,b))))
    onset = next((m['decisions_completed'] for m in metrics if m['committed_token_conflicts']), None)
    merge = next((m['decisions_completed'] for m in metrics if m['decisions_completed']>swap_step and not m['state_hamming']), None)
    return metrics, dict(first_committed_conflict=onset, first_reconvergence=merge,
                         final_hamming=metrics[-1]['state_hamming'])


def run(directory, plan_path, discovery_path, out):
    plan = json.loads(plan_path.read_text())
    manifest = json.loads((directory/'manifest.json').read_text())
    discovery = json.loads(discovery_path.read_text())
    if manifest['phase'] != 'main' or manifest['plan'] != plan or manifest['plan_sha256'] != digest(plan):
        raise ValueError('Wrong phase or frozen plan')
    if digest(discovery) != plan['discovery_manifest_sha256']:
        raise ValueError('Wrong discovery manifest')
    if set(discovery['question_ids']) & set(manifest['question_ids']):
        raise ValueError('Discovery overlap')
    if manifest['question_ids'] != plan['question_ids']:
        raise ValueError('Sample differs from plan')
    for name, expected in plan['code_sha256'].items():
        if hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen code changed: '+name)
    rows = load_rows(directory, manifest['question_ids'])
    lookup = {r['question_id']:r for r in rows}
    rows = [lookup[i] for i in manifest['question_ids']]
    protected = [*directory.glob('*.json'), plan_path, discovery_path]
    hashes = {str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    cfg = plan['protocol']
    steps = cfg['gen_len']//cfg['block_len']*cfg['steps_per_block']
    k = cfg['block_len']//cfg['steps_per_block']
    mask = cfg['mask_id']
    result_rows = []
    no_eos = Counter()
    for row in rows:
        assert row['nfe_actual_pair']+row['nfe_controls']==plan['expected_forward_calls_per_pair']
        assert row['checks'].get('no_swap_replay') and row['checks'].get('independent_baseline_replay')
        histories = [row['trace'][arm+'_states'] for arm in ('baseline','treated')]
        for arm, history in zip(('baseline','treated'),histories):
            assert len(history)==steps+1
            assert all(len(state)==cfg['gen_len'] for state in history)
            assert history[0]==[mask]*cfg['gen_len']
            assert history[-1]==row[arm+'_ids'] and mask not in history[-1]
            for step,(before,after) in enumerate(zip(history,history[1:])):
                changed = [i for i,(a,b) in enumerate(zip(before,after)) if a!=b]
                assert len(changed)==k
                assert all(before[i]==mask and after[i]!=mask for i in changed)
                assert all(i//cfg['block_len']==step//cfg['steps_per_block'] for i in changed)
            no_eos[arm] += not any(t in row['eos_ids'] for t in history[-1])
        common = row['common_state_ids'][len(row['prompt_ids']):]
        assert histories[0][cfg['swap_step']]==histories[1][cfg['swap_step']]==common
        for arm,history in zip(('baseline','treated'),histories):
            changed = [i for i,(a,b) in enumerate(zip(history[cfg['swap_step']],history[cfg['swap_step']+1])) if a!=b]
            assert set(changed)==set(row['swap'][arm+'_bundle'])
        metrics, features = trajectory_features(*histories, mask, cfg['swap_step'])
        assert metrics==row['trace']['divergence']
        assert features['final_hamming']==row['hamming']
        assert metrics[cfg['swap_step']]['state_hamming']==0
        assert metrics[cfg['swap_step']+1]['state_hamming']==2
        # Absorbing commitments make content conflicts permanent; mask-only changes can merge.
        conflicts=[m['committed_token_conflicts'] for m in metrics]
        assert all(b>=a for a,b in zip(conflicts,conflicts[1:]))
        if features['first_reconvergence'] is not None:
            assert all(m['state_hamming']==0 for m in metrics[features['first_reconvergence']:])
        eos = row['eos_ids']
        visible_end=min(next((i for i,t in enumerate(row[arm+'_ids']) if t in eos),cfg['gen_len']) for arm in ('baseline','treated'))
        visible_onset=next((step for step,(a,b) in enumerate(zip(*histories))
            if any(x!=y and x!=mask and y!=mask for x,y in zip(a[:visible_end],b[:visible_end]))),None)
        result_rows.append(dict(question_id=row['question_id'],text_change=row['baseline']['text']!=row['treated']['text'],
            first_conflict_before_both_final_eos=visible_onset, **features))
    raw_summary=json.loads((directory/'summary.json').read_text())
    recomputed=analyse(rows, manifest['question_ids'])
    assert all(raw_summary[k]==v for k,v in recomputed.items())
    assert raw_summary['complete'] and len(rows)==500
    temporal = dict(first_committed_conflict_histogram=dict(Counter(r['first_committed_conflict'] for r in result_rows if r['first_committed_conflict'] is not None)),
        reconvergence_histogram=dict(Counter(r['first_reconvergence'] for r in result_rows if r['first_reconvergence'] is not None)),
        ever_committed_conflict=sum(r['first_committed_conflict'] is not None for r in result_rows),
        conflicts_before_both_final_eos=sum(r['first_conflict_before_both_final_eos'] is not None for r in result_rows),
        ever_reconverged=sum(r['first_reconvergence'] is not None for r in result_rows),
        note='Counts index completed decisions: intervention completes decision 4. Conflicts cannot disappear under absorbing commitments. EOS restriction uses final EOS retrospectively.')
    result=dict(validation=dict(complete=True,questions=len(rows),overlap=0,all_replays_pass=True,
        source_and_code_hashes_match=True,all_step_invariants_pass=True,stored_summary_matches=True),
        no_eos=dict(no_eos), temporal=temporal, rows=result_rows,
        source_sha256=hashes, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out.mkdir(parents=True,exist_ok=True)
    (out/'REPLICATION_VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n')
    assert hashes=={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    print(json.dumps({k:result[k] for k in ('validation','no_eos','temporal')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--discovery-manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    run(a.directory,a.plan,a.discovery_manifest,a.out)
