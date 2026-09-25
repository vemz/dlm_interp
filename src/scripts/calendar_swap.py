"""One fixed calendar swap on GSM8K. Collect on CUDA; analyse saved rows on CPU."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
from decimal import Decimal
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import time

MODEL = 'GSAI-ML/LLaDA-8B-Instruct'
DATASET = 'openai/gsm8k'
PROMPT_SUFFIX = "\n\nReason step by step and end with '#### <answer>'."
NUMBER = r'[+-]?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]+)?'
ANSWER = re.compile(r'#### (' + NUMBER + r')')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def parse_answer_strict(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    match = ANSWER.fullmatch(lines[-1]) if lines else None
    if match is None:
        return None
    value = Decimal(match[1].replace(',', ''))
    if value == 0:
        return '0'
    normalized = format(value, 'f')
    return normalized.rstrip('0').rstrip('.') if '.' in normalized else normalized


PARSER_VERSION = 'explicit-conclusion-v2'
UNITS = r'(?:pounds?|dollars?|cents?|pages?|clips?|flowers?|slices?|pieces?|hours?|minutes?)'


def parse_answer(text):
    """Read an explicit final answer, never search the reasoning for a number.

    Accept a final #### section or a standalone final boxed number. Only
    supported wrappers, numeric syntax and units are stripped. Incomplete
    boxes, extra numbers, expressions and arbitrary trailing prose fail closed.
    """
    markers = list(re.finditer(r'(?m)^\s*####[ \t]*', text))
    if markers:
        conclusion = text[markers[-1].end():].strip()
        conclusion = re.sub(r'^(?:<answer>|<:>|answer\b)[ \t]*:?[ \t]*', '', conclusion,
                            count=1, flags=re.IGNORECASE).strip()
    else:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        conclusion = lines[-1] if lines else ''
        if r'\boxed{' not in conclusion:
            return None
    # Permit paired outer display/inline math and bold wrappers, never unmatched ones.
    for _ in range(3):
        changed = False
        for left, right in ((r'\[', r'\]'), (r'\(', r'\)'), ('**', '**'), ('$$', '$$')):
            if conclusion.startswith(left) and conclusion.endswith(right):
                conclusion = conclusion[len(left):-len(right)].strip()
                changed = True
                break
        if not changed:
            break
    if conclusion.startswith(r'\boxed{') and conclusion.endswith('}'):
        conclusion = conclusion[len(r'\boxed{'):-1].strip()
    # A short declarative conclusion such as "Weng earned $6." is allowed;
    # negation, alternatives and additional numeric material are not.
    prefix = (r'(?:the (?:final )?(?:answer|result|total) (?:is|equals) '
              r'|[A-Za-z]+ (?:earned|owes|needs|paid|has) )?')
    if re.search(r'\b(?:not|never|no|or|maybe|approximately)\b', conclusion, re.IGNORECASE):
        return None
    pattern = prefix + r'[$£€]?\s*(' + NUMBER + r')(?:\s+' + UNITS + r')?[.]?'
    match = re.fullmatch(pattern, conclusion, re.IGNORECASE)
    return parse_answer_strict('#### ' + match[1]) if match else None


def reparse_saved(directory):
    """Return an auditable sidecar, leaving original records and summary untouched."""
    manifest = json.loads((directory / 'manifest.json').read_text())
    original = load_rows(directory, manifest['question_ids'])
    revised = copy.deepcopy(original)
    changes = []
    for row in revised:
        for arm in ('baseline', 'treated'):
            old = dict(row[arm])
            answer = None if old['status'] == 'length_limit' else parse_answer(old['text'])
            row[arm].update(answer=answer,
                            status='length_limit' if old['status'] == 'length_limit' else 'valid' if answer is not None else 'invalid_format',
                            correct=answer is not None and answer == row['gold'])
            changes.append({'question_id': row['question_id'], 'arm': arm,
                            'old_answer': old['answer'], 'new_answer': answer,
                            'old_status': old['status'], 'new_status': row[arm]['status'],
                            'old_correct': old['correct'], 'new_correct': row[arm]['correct']})
    return {'parser_version': PARSER_VERSION, 'source_manifest_sha256': digest(manifest),
            'source_records_sha256': digest(original),
            'analysis_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope': 'post-hoc pilot parser revision; no new generation',
            'original_summary': analyse(original, manifest['question_ids']),
            'revised_summary': analyse(revised, manifest['question_ids']), 'answers': changes}


@dataclass(frozen=True)
class Protocol:
    gen_len: int = 256
    block_len: int = 32
    steps_per_block: int = 8
    swap_step: int = 3  # Zero-based, first block only.
    mask_id: int = 126336

    def validate(self):
        if not (self.gen_len > 0 and self.block_len > 0 and self.steps_per_block > 0):
            raise ValueError('lengths and step counts must be positive')
        if self.gen_len % self.block_len or self.block_len % self.steps_per_block:
            raise ValueError('equal blocks and equal numbers of writes are required')
        if not 0 <= self.swap_step < self.steps_per_block - 1:
            raise ValueError('swap needs an unselected masked position in the first block')

    @property
    def k(self):
        return self.block_len // self.steps_per_block

    @property
    def steps(self):
        return self.gen_len // self.block_len * self.steps_per_block


def selected_predictions(forward, x, prompt_len, step, cfg):
    import torch
    lo = prompt_len + (step // cfg.steps_per_block) * cfg.block_len
    positions = (x[0, lo:lo + cfg.block_len] == cfg.mask_id).nonzero().flatten() + lo
    if positions.numel() < cfg.k:
        raise ValueError('unexpected masked population')
    logits = forward(x)[0, positions].float().clone()
    if not torch.isfinite(logits).all():
        raise ValueError('nonfinite model logits')
    logits[:, cfg.mask_id] = -torch.inf
    tokens = logits.argmax(-1)  # First token ID wins exact ties.
    confidence = (logits.max(-1).values - logits.logsumexp(-1)).exp()
    # positions is increasing, stable sort therefore breaks ties by position.
    order = torch.argsort(confidence, descending=True, stable=True)
    return positions, tokens, confidence, order


def commit(x, positions, tokens, indices, cfg):
    if len(indices) != cfg.k or len(set(indices.tolist())) != cfg.k:
        raise ValueError('invalid bundle')
    pos = positions[indices]
    if not (x[0, pos] == cfg.mask_id).all() or (tokens[indices] == cfg.mask_id).any():
        raise ValueError('invalid commit')
    x[0, pos] = tokens[indices]


def continue_from(forward, x, p, start, cfg):
    x = x.clone()
    for step in range(start, cfg.steps):
        positions, tokens, _, order = selected_predictions(forward, x, p, step, cfg)
        commit(x, positions, tokens, order[:cfg.k], cfg)
    return x


def generate_pair(forward, prompt_ids, cfg=Protocol(), controls=False):
    """Counted forwards; shared prefix and swap forward; independent suffixes."""
    import torch
    cfg.validate()
    calls = 0

    def counted(x):
        nonlocal calls
        calls += 1
        return forward(x)

    with torch.inference_mode():
        p = prompt_ids.numel()
        initial = torch.full((1, p + cfg.gen_len), cfg.mask_id, device=prompt_ids.device, dtype=torch.long)
        initial[0, :p] = prompt_ids
        common = initial.clone()
        for step in range(cfg.swap_step):
            positions, tokens, _, order = selected_predictions(counted, common, p, step, cfg)
            commit(common, positions, tokens, order[:cfg.k], cfg)
        positions, tokens, conf, order = selected_predictions(counted, common, p, cfg.swap_step, cfg)
        if len(order) <= cfg.k:
            raise ValueError('no replacement candidate')
        natural_indices = order[:cfg.k]
        treated_indices = torch.cat((order[:cfg.k - 1], order[cfg.k:cfg.k + 1]))
        natural, treated = common.clone(), common.clone()
        commit(natural, positions, tokens, natural_indices, cfg)
        commit(treated, positions, tokens, treated_indices, cfg)
        natural_start = natural.clone()
        natural = continue_from(counted, natural, p, cfg.swap_step + 1, cfg)
        treated = continue_from(counted, treated, p, cfg.swap_step + 1, cfg)
        actual_nfe = calls
        checks = {'prompt_unchanged': True, 'fully_unmasked': True, 'single_swap': True}
        if controls:
            sham = continue_from(counted, natural_start, p, cfg.swap_step + 1, cfg)
            replay = continue_from(counted, initial, p, 0, cfg)
            if not torch.equal(sham, natural) or not torch.equal(replay, natural):
                raise ValueError('no-swap / independent baseline replay control failed')
            checks.update(no_swap_replay=True, independent_baseline_replay=True)
        for x in (natural, treated):
            if not torch.equal(x[0, :p], prompt_ids) or (x[0, p:] == cfg.mask_id).any():
                raise ValueError('prompt changed or masks survived')
        return {
            'baseline_ids': natural[0, p:].tolist(), 'treated_ids': treated[0, p:].tolist(),
            'common_state_ids': common[0].tolist(),
            'swap': {'step_zero_based': cfg.swap_step,
                     'deferred_position': int(positions[order[cfg.k - 1]]) - p,
                     'advanced_position': int(positions[order[cfg.k]]) - p,
                     'deferred_confidence': float(conf[order[cfg.k - 1]]),
                     'advanced_confidence': float(conf[order[cfg.k]]),
                     'baseline_bundle': (positions[natural_indices] - p).tolist(),
                     'treated_bundle': (positions[treated_indices] - p).tolist()},
            'nfe_per_arm_nominal': cfg.steps, 'nfe_actual_pair': actual_nfe,
            'nfe_controls': calls - actual_nfe, 'checks': checks,
        }


def evaluate_output(ids, tokenizer, eos_ids, gold):
    # Decode only through the first EOS; all 256 tokens remain in the saved row.
    end = next((i for i, token in enumerate(ids) if token in eos_ids), None)
    text = tokenizer.decode(ids if end is None else ids[:end], skip_special_tokens=True)
    pred = parse_answer(text) if end is not None else None
    status = 'length_limit' if end is None else 'valid' if pred is not None else 'invalid_format'
    return {'text': text, 'answer': pred, 'status': status, 'correct': pred is not None and pred == gold}


def wilson(count, n):
    if n == 0:
        return {'count': count, 'n': n, 'rate': None, 'ci95': None}
    z = 1.959963984540054
    p = count / n
    den = 1 + z*z/n
    centre = (p + z*z/(2*n)) / den
    radius = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return {'count': count, 'n': n, 'rate': p, 'ci95': [max(0., centre-radius), min(1., centre+radius)]}


def analyse(rows, expected):
    import numpy as np
    ids = [r['question_id'] for r in rows]
    if len(set(ids)) != len(ids) or not set(ids) <= set(expected):
        raise ValueError('duplicate or unexpected question IDs')
    n = len(rows)
    valid = [r for r in rows if r['baseline']['answer'] is not None and r['treated']['answer'] is not None]
    changes = sum(r['baseline']['answer'] != r['treated']['answer'] for r in valid)
    losses = sum(r['baseline']['correct'] and not r['treated']['correct'] for r in rows)
    gains = sum(not r['baseline']['correct'] and r['treated']['correct'] for r in rows)
    deltas = np.array([int(r['treated']['correct']) - int(r['baseline']['correct']) for r in rows])
    rng = np.random.default_rng(20260923)
    ci = np.quantile(deltas[rng.integers(n, size=(10000, n))].mean(1), [.025, .975]).tolist() if n else None
    return {
        'complete': set(ids) == set(expected), 'questions_processed': n, 'questions_planned': len(expected),
        'scope': 'partial descriptive results' if n != len(expected) else 'fixed-sample paired calendar perturbation',
        'numeric_answer_changes_all': wilson(changes, n),
        'numeric_answer_changes_both_valid': wilson(changes, len(valid)),
        'correct_to_incorrect': wilson(losses, n), 'incorrect_to_correct': wilson(gains, n),
        'accuracy_baseline': wilson(sum(r['baseline']['correct'] for r in rows), n),
        'accuracy_treated': wilson(sum(r['treated']['correct'] for r in rows), n),
        'invalid_baseline': wilson(sum(r['baseline']['answer'] is None for r in rows), n),
        'invalid_treated': wilson(sum(r['treated']['answer'] is None for r in rows), n),
        'valid_to_invalid': wilson(sum(r['baseline']['answer'] is not None and r['treated']['answer'] is None for r in rows), n),
        'invalid_to_valid': wilson(sum(r['baseline']['answer'] is None and r['treated']['answer'] is not None for r in rows), n),
        'accuracy_delta': {'mean': (gains-losses)/n if n else None, 'paired_bootstrap_ci95': ci,
                           'note': 'A degenerate bootstrap CI does not establish zero population risk; consult transition-rate Wilson bounds.'},
        'mean_pair_seconds': float(np.mean([r['seconds'] for r in rows])) if n else None,
        'actual_forward_calls': sum(r['nfe_actual_pair'] + r['nfe_controls'] for r in rows),
    }


def save_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def load_rows(directory, expected):
    rows = [json.loads(p.read_text()) for p in sorted(directory.glob('question_*.json'))]
    manifest_path = directory / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if any(r.get('manifest_sha256') != digest(manifest) for r in rows):
            raise ValueError('question records do not match the manifest')
    analyse(rows, expected)  # Reject duplicates/unexpected IDs before resuming.
    return rows


def collect(root, phase, checkpoint=lambda: None, launcher_sha256=None):
    """Called by Modal. Checkpoint after every atomically saved question."""
    import torch
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoModel, AutoTokenizer
    if phase not in ('pilot', 'main'):
        raise ValueError('phase must be pilot or main')
    if not torch.cuda.is_available():
        raise RuntimeError('collection requires CUDA; use Modal')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20260923)
    # The custom model may use SDPA. Force the deterministic math backend.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    cfg = Protocol()
    directory = Path(root) / phase
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / 'manifest.json'
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    pilot = None
    if phase == 'main':
        pilot_path = Path(root) / 'pilot'
        pilot = json.loads((pilot_path / 'manifest.json').read_text())
        pilot_rows = load_rows(pilot_path, pilot['question_ids'])
        if not analyse(pilot_rows, pilot['question_ids'])['complete'] or not all(
                r['checks'].get('no_swap_replay') and r['checks'].get('independent_baseline_replay') for r in pilot_rows):
            raise ValueError('complete pilot with passing controls required')
    inherited = previous or pilot
    api = HfApi()
    model_revision = inherited['model_revision'] if inherited else api.model_info(MODEL).sha
    dataset_revision = inherited['dataset_revision'] if inherited else api.dataset_info(DATASET).sha
    split = 'train' if phase == 'pilot' else 'test'
    dataset = load_dataset(DATASET, 'main', split=split, revision=dataset_revision)
    indices = list(range(10)) if phase == 'pilot' else sorted(range(len(dataset)), key=lambda i: hashlib.sha256(f'calendar-swap-v1:{i}'.encode()).hexdigest())[:500]
    if len(indices) != (10 if phase == 'pilot' else 500):
        raise ValueError('unexpected dataset size')
    questions = [{'question_id': i, 'question': dataset[i]['question'], 'gold_text': dataset[i]['answer']} for i in indices]
    manifest = {
        'version': 1, 'phase': phase, 'protocol': asdict(cfg), 'model': MODEL,
        'model_revision': model_revision, 'dataset': DATASET, 'dataset_revision': dataset_revision,
        'split': split, 'question_ids': indices, 'questions_sha256': digest(questions),
        'prompt_suffix': PROMPT_SUFFIX, 'dtype': 'bfloat16', 'gpu': torch.cuda.get_device_name(),
        'cuda': torch.version.cuda, 'packages': {p: importlib.metadata.version(p) for p in ('torch','transformers','datasets','huggingface-hub','numpy')},
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'launcher_sha256': launcher_sha256,
        'deterministic': True, 'sdpa_backend': 'math', 'tf32': False,
        'parser': PARSER_VERSION + '; explicit final conclusion; no EOS = invalid length_limit',
    }
    if previous is not None and previous != manifest:
        raise ValueError('incompatible resume manifest; use a new run ID, never overwrite')
    if pilot and any(pilot[k] != manifest[k] for k in ('script_sha256','launcher_sha256','packages','gpu','cuda','protocol')):
        raise ValueError('main configuration differs from validated pilot')
    if previous is None and list(directory.glob('question_*.json')):
        raise ValueError('orphaned question records without manifest')
    save_json(manifest_path, manifest)
    checkpoint()
    rows = load_rows(directory, indices)
    if len(rows) == len(indices):
        summary = analyse(rows, indices)
        save_json(directory / 'summary.json', summary)
        checkpoint()
        return summary
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=model_revision, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL, revision=model_revision, code_revision=model_revision,
                                     trust_remote_code=True, torch_dtype=torch.bfloat16,
                                     low_cpu_mem_usage=True).to('cuda').eval()
    eos = tokenizer.eos_token_id
    eos_ids = eos if isinstance(eos, list) else [eos]
    if not eos_ids or None in eos_ids or cfg.mask_id in eos_ids:
        raise ValueError('invalid EOS configuration')
    torch.cuda.reset_peak_memory_stats()
    completed = {r['question_id'] for r in rows}
    for question in questions:
        if question['question_id'] in completed:
            continue
        prompt_ids = tokenizer.apply_chat_template([{'role':'user','content':question['question'] + PROMPT_SUFFIX}],
                                                   add_generation_prompt=True, return_tensors='pt')[0].to('cuda')
        limit = getattr(model.config, 'max_sequence_length', None) or getattr(model.config, 'max_position_embeddings', None)
        if limit is not None and prompt_ids.numel() + cfg.gen_len > limit:
            raise ValueError('prompt plus generation exceeds model context length')
        torch.cuda.synchronize()
        started = time.monotonic()
        forward = lambda x: model(x, use_cache=False).logits
        row = generate_pair(forward, prompt_ids, cfg, controls=(phase == 'pilot' or not rows))
        torch.cuda.synchronize()
        gold = parse_answer(question['gold_text'])
        if gold is None:
            raise ValueError('invalid gold answer')
        row.update(question, gold=gold, prompt_ids=prompt_ids.tolist(), eos_ids=eos_ids,
                   seconds=time.monotonic()-started, manifest_sha256=digest(manifest))
        row['baseline'] = evaluate_output(row['baseline_ids'], tokenizer, eos_ids, gold)
        row['treated'] = evaluate_output(row['treated_ids'], tokenizer, eos_ids, gold)
        row['hamming'] = sum(a != b for a,b in zip(row['baseline_ids'],row['treated_ids']))
        row['peak_cuda_memory_bytes'] = torch.cuda.max_memory_allocated()
        save_json(directory / f"question_{question['question_id']:05d}.json", row)
        rows.append(row)
        save_json(directory / 'summary.json', analyse(rows, indices))
        checkpoint()
        print(f"{phase} {len(rows)}/{len(indices)} id={question['question_id']} seconds={row['seconds']:.1f} baseline={row['baseline']['status']} treated={row['treated']['status']}", flush=True)
    return analyse(rows, indices)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, help='saved pilot or main directory to analyse; no GPU needed')
    parser.add_argument('--reparse-out', type=Path, help='write a separate parser-v2 analysis; never rewrite original rows')
    args = parser.parse_args()
    if args.reparse_out:
        source = args.directory.resolve()
        target = args.reparse_out.resolve()
        if target == source / 'manifest.json' or target == source / 'summary.json' or (target.parent == source and target.name.startswith('question_')):
            parser.error('reparse output must not replace original data')
        save_json(args.reparse_out, reparse_saved(args.directory))
        print(f'Saved parser revision to {args.reparse_out}')
        raise SystemExit(0)
    manifest = json.loads((args.directory / 'manifest.json').read_text())
    print(json.dumps(analyse(load_rows(args.directory, manifest['question_ids']), manifest['question_ids']), indent=2))
