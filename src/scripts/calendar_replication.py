"""Independent sample and instrumented replication of the frozen calendar swap.

Local preparation is CPU-only. collect() is invoked explicitly by Modal.
"""
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

from calendar_swap import (Protocol, MODEL, DATASET, PROMPT_SUFFIX, digest,
                           generate_pair, evaluate_output, parse_answer_strict,
                           save_json, load_rows, analyse, wilson)

HERE = Path(__file__).resolve().parent
FILES = ('calendar_swap.py', 'answer_audit_v4.py', 'calendar_replication.py', 'modal_calendar_replication.py')


def code_hashes():
    return {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in FILES}


def prepare(source, destination, dataset_size=1319, n=500):
    manifest = json.loads(source.read_text())
    excluded = set(manifest['question_ids'])
    if not excluded <= set(range(dataset_size)) or n > dataset_size-len(excluded) or n < 1:
        raise ValueError('invalid sample size or dataset universe')
    ids = sorted(set(range(dataset_size))-excluded,
                 key=lambda i: hashlib.sha256(f'calendar-replication-v1:{i}'.encode()).hexdigest())[:n]
    plan = dict(version=1, name='calendar-replication-v1',
        scope='prospective protocol file; not externally preregistered; held-out questions from same dataset',
        discovery_manifest_sha256=digest(manifest), dataset_size=dataset_size,
        model_revision=manifest['model_revision'], dataset_revision=manifest['dataset_revision'],
        discovery_ids=sorted(excluded), question_ids=ids, pilot_ids=list(range(10)),
        protocol=manifest['protocol'], code_sha256=code_hashes(),
        primary='rate of decoded text differences before first EOS on all planned questions',
        secondary='AUC of lower advanced-position confidence for text divergence; 5000 percentile question bootstraps, seed 20260925',
        secondary_success_rule='Single prespecified association: lower bound of two-sided 95% bootstrap CI exceeds 0.5; parser-free',
        exploratory='Gap, deferred confidence, distance, prompt length and numeric changes; no separate confirmatory claims',
        numeric='frozen v4, secondary descriptive, report abstentions separately',
        controls='sham continuation and independent baseline replay on every question; exact equality required',
        stopping='fixed planned sample; no outcome-based early stopping or sample expansion',
        traces='full generated-token states before/after every decision, both arms; no extra model forwards',
        expected_forward_calls_per_pair=248,
        notes=['Dataset size checked on remote before generation.',
               'Independent questions, same dataset/model: no claim of external-domain replication.',
               'Runtime extrapolation ~90 minutes main plus startup/pilot; tracing/control overhead uncertain.'])
    if destination.exists():
        if json.loads(destination.read_text()) != plan:
            raise ValueError('Refusing to overwrite a different frozen plan; use a new filename/version')
        return plan
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_json(destination, plan)
    return plan


def generate_pair_traced(forward, prompt_ids, cfg=Protocol(), controls=True):
    """Observe existing forwards; verify writes without changing model inputs."""
    p = prompt_ids.numel()
    states = []
    def observed(x):
        states.append(x[0, p:].tolist())
        return forward(x)
    row = generate_pair(observed, prompt_ids, cfg, controls=controls)
    shared = cfg.swap_step + 1
    suffix = cfg.steps - shared
    baseline = states[:shared] + states[shared:shared+suffix] + [row['baseline_ids']]
    treated = states[:shared] + states[shared+suffix:shared+2*suffix] + [row['treated_ids']]
    assert len(states) == row['nfe_actual_pair'] + row['nfe_controls']
    for history in (baseline, treated):
        assert len(history) == cfg.steps+1
        for step, (before, after) in enumerate(zip(history, history[1:])):
            positions = [i for i, (a,b) in enumerate(zip(before, after)) if a != b]
            assert len(positions) == cfg.k
            assert all(before[i] == cfg.mask_id and after[i] != cfg.mask_id for i in positions)
            assert all(i//cfg.block_len == step//cfg.steps_per_block for i in positions)
    divergence = []
    for step, (a,b) in enumerate(zip(baseline, treated)):
        divergence.append(dict(decisions_completed=step,
            state_hamming=sum(x != y for x,y in zip(a,b)),
            committed_token_conflicts=sum(x != y and x != cfg.mask_id and y != cfg.mask_id for x,y in zip(a,b))))
    row['trace'] = dict(baseline_states=baseline, treated_states=treated, divergence=divergence,
                       state_index='number of completed decisions; index 0 is all masked',
                       scope='all 256 positions, including positions beyond eventual EOS')
    return row


def summary(rows, expected):
    result = analyse(rows, expected)
    result['text_divergence'] = wilson(sum(r['baseline']['text'] != r['treated']['text'] for r in rows), len(rows))
    result['all_replays_pass'] = all(r['checks'].get('no_swap_replay') and r['checks'].get('independent_baseline_replay') for r in rows)
    return result


def collect(root, phase, plan, checkpoint=lambda: None, launcher_sha256=None):
    import torch
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer
    from answer_audit_v4 import extract
    if phase not in ('pilot', 'main'):
        raise ValueError('phase must be pilot/main')
    if code_hashes() != plan['code_sha256']:
        raise ValueError('Code differs from frozen plan')
    if launcher_sha256 != plan['code_sha256']['modal_calendar_replication.py']:
        raise ValueError('Launcher differs from frozen plan')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for collection')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.manual_seed(20260923)
    cfg = Protocol(**plan['protocol'])
    split = 'train' if phase == 'pilot' else 'test'
    dataset = load_dataset(DATASET, 'main', split=split, revision=plan['dataset_revision'])
    if phase == 'main' and len(dataset) != plan['dataset_size']:
        raise ValueError('Dataset universe changed')
    ids = plan['pilot_ids'] if phase == 'pilot' else plan['question_ids']
    if phase == 'main' and set(ids) & set(plan['discovery_ids']):
        raise ValueError('Discovery/replication overlap')
    directory = Path(root)/phase
    directory.mkdir(parents=True, exist_ok=True)
    manifest = dict(version=1, phase=phase, protocol=asdict(cfg), plan_sha256=digest(plan),
        plan=plan, question_ids=ids, split=split, model=MODEL, dataset=DATASET,
        model_revision=plan['model_revision'], dataset_revision=plan['dataset_revision'],
        gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda,
        packages={p:importlib.metadata.version(p) for p in ('torch','transformers','datasets','huggingface-hub','numpy')},
        questions_sha256=digest([dict(question_id=i, question=dataset[i]['question'], gold_text=dataset[i]['answer']) for i in ids]),
        parser='conclusion-units-v4', launcher_sha256=launcher_sha256)
    if phase == 'main':
        pilot_dir = Path(root)/'pilot'
        pilot = json.loads((pilot_dir/'manifest.json').read_text())
        pilot_rows = load_rows(pilot_dir, plan['pilot_ids'])
        stats = summary(pilot_rows, plan['pilot_ids'])
        if not stats['complete'] or not stats['all_replays_pass']:
            raise ValueError('Complete pilot with passing replays required')
        if any(pilot[k] != manifest[k] for k in ('plan_sha256','gpu','cuda','packages','launcher_sha256')):
            raise ValueError('Configuration differs from pilot')
    mp = directory/'manifest.json'
    if mp.exists():
        if json.loads(mp.read_text()) != manifest:
            raise ValueError('Incompatible resume; use a new run ID')
    elif list(directory.glob('question_*.json')):
        raise ValueError('Orphaned question files')
    else:
        save_json(mp, manifest)
        checkpoint()
    rows = load_rows(directory, ids)
    completed = {r['question_id'] for r in rows}
    if set(ids) != completed:
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=plan['model_revision'], trust_remote_code=True)
        model = AutoModel.from_pretrained(MODEL, revision=plan['model_revision'], code_revision=plan['model_revision'],
            trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to('cuda').eval()
        eos = tokenizer.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        if not eos or None in eos or cfg.mask_id in eos:
            raise ValueError('Invalid EOS')
        for i in ids:
            if i in completed:
                continue
            q = dataset[i]
            prompt = tokenizer.apply_chat_template([{'role':'user','content':q['question']+PROMPT_SUFFIX}], add_generation_prompt=True, return_tensors='pt')[0].to('cuda')
            limit = getattr(model.config, 'max_sequence_length', None) or getattr(model.config, 'max_position_embeddings', None)
            if limit is not None and len(prompt)+cfg.gen_len > limit:
                raise ValueError('Context limit exceeded')
            torch.cuda.synchronize()
            started = time.monotonic()
            row = generate_pair_traced(lambda x:model(x,use_cache=False).logits, prompt, cfg, controls=True)
            torch.cuda.synchronize()
            gold = parse_answer_strict(q['answer'])
            if gold is None:
                raise ValueError('Invalid gold')
            row.update(question_id=i, question=q['question'], gold_text=q['answer'], gold=gold,
                       prompt_ids=prompt.tolist(), eos_ids=eos, seconds=time.monotonic()-started, manifest_sha256=digest(manifest))
            for arm in ('baseline','treated'):
                value = evaluate_output(row[arm+'_ids'], tokenizer, eos, gold)
                decision = {'answer':None,'reason':'length_limit'} if value['status']=='length_limit' else extract(value['text'],q['question'])
                answer = decision['answer']
                row[arm] = dict(value, answer=answer, status='valid' if answer is not None else decision['reason'], correct=answer==gold, extraction=decision)
            row['hamming'] = sum(a!=b for a,b in zip(row['baseline_ids'],row['treated_ids']))
            save_json(directory/f'question_{i:05d}.json',row)
            rows.append(row)
            save_json(directory/'summary.json',summary(rows,ids))
            checkpoint()
            print(f'{phase} {len(rows)}/{len(ids)} id={i} seconds={row["seconds"]:.1f}',flush=True)
    result = summary(rows, ids)
    save_json(directory/'summary.json', result)
    checkpoint()
    return result


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--discovery-manifest', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--dataset-size', type=int, default=1319)
    args = p.parse_args()
    result = prepare(args.discovery_manifest, args.out, args.dataset_size)
    print(json.dumps(dict(plan_sha256=digest(result), planned=len(result['question_ids']), overlap=0),indent=2))
