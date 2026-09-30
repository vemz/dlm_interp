"""A single rank-4/rank-5 calendar substitution on LLaDA, plus shared experiment code.
10 technical pilot questions, then the fixed 500 GSM1k questions used for Dream.
Each branch uses 64 forwards; the shared prefix and exact controls cost 248 total.
This compact reproduction has its own run namespace. Historical collectors are archived.
"""

from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import time

from print_results import extract, summarize

HERE = Path(os.environ.get('CALENDAR_CODE_DIR', Path(__file__).resolve().parent))
PROMPT_SUFFIX = "\n\nReason step by step and end with '#### <answer>'."
SETTINGS = dict(name='llada', model='GSAI-ML/LLaDA-8B-Instruct',
                revision='08b83a6feb34df1a6011b80c3c00c7563e963b07',
                mask_id=126336, end_ids=[126081, 126348], model_kwargs={})
DATA_REVISION = 'bc09569d09a614b9b530edc7f076fb214ac10493'
GSM8K_REVISION = '740312add88f781978c0658806c59bc2815b9866'
EXAMPLE_HASHES = dict(pilot='e85f7196552e13728fd488abddaacf509e998ff2388a92941282961432a9a315',
                      main='aa46d87371972612bd404836b295e1bf127c6441ee6f2199f353d65a2d776c90')
CHECKS = ('prompt_unchanged', 'fully_unmasked', 'single_swap', 'no_swap_replay', 'independent_baseline_replay')


# Decoding: same forward at the intervention, then independent continuations.
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


# Data and provenance: frozen examples, atomic checkpoints, strict resumption.
def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def examples(phase):
    if phase == 'pilot':
        from datasets import load_dataset
        data = load_dataset('openai/gsm8k', 'main', split='train', revision=GSM8K_REVISION)
        items = [dict(id=i, question=data[i]['question'],
                      answer=str(Fraction(data[i]['answer'].split('####')[-1].strip().replace(',', ''))))
                 for i in range(10)]
    else:
        from huggingface_hub import hf_hub_download
        import pyarrow.ipc as ipc
        path = hf_hub_download('ScaleAI/gsm1k', 'data/test/data-00000-of-00001.arrow',
                               repo_type='dataset', revision=DATA_REVISION)
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != 'bf7b848f2badbdcb138d6fbc1f246a1902337ce4c00cc079e4f35baf8ed10764':
            raise ValueError('Dataset file changed')
        with open(path, 'rb') as stream:
            data = ipc.open_stream(stream).read_all().to_pylist()
        ids = sorted(sorted(range(1205), key=lambda i:
            hashlib.sha256(f'calendar-dream-replication-v1:{i}'.encode()).hexdigest())[:500])
        items = [dict(id=i, question=data[i]['question'], answer=str(Fraction(data[i]['answer']))) for i in ids]
    if digest(items) != EXAMPLE_HASHES[phase]:
        raise ValueError('Frozen sample changed')
    return items


def output(ids, tokenizer, question, gold, settings):
    stop = next((i for i, token in enumerate(ids) if token in settings['end_ids']), None)
    text = tokenizer.decode(ids if stop is None else ids[:stop], skip_special_tokens=True)
    parsed = extract(text, question) if stop is not None else dict(answer=None, reason='length_limit')
    answer = parsed['answer']
    return dict(text=text, answer=answer, extraction=parsed, first_terminator_position=stop,
                correct=answer is not None and Fraction(answer) == Fraction(gold))


def load_saved(directory, manifest):
    rows = []
    questions = {q['id']: q for q in manifest['examples']}
    mask, ends = manifest['settings']['mask_id'], manifest['settings']['end_ids']
    for path in sorted(Path(directory).glob('question_*.json')):
        row = read(path)
        checksum = row.pop('sha256')
        if digest(row) != checksum or row['manifest_sha256'] != digest(manifest):
            raise ValueError(f'Corrupt checkpoint: {path}')
        q = questions.get(row['question_id'])
        if q is None or row['question'] != q['question'] or row['gold'] != q['answer']:
            raise ValueError('Record differs from frozen sample')
        if path.name != f"question_{q['id']:05d}.json" or any(row['checks'].get(k) is not True for k in CHECKS):
            raise ValueError('Wrong filename or failed controls')
        if (row['nfe_actual_pair'], row['nfe_controls'], row['nfe_per_arm_nominal']) != (124, 124, 64):
            raise ValueError('Forward budget mismatch')
        p, common, swap = len(row['prompt_ids']), row['common_state_ids'], row['swap']
        b, t = swap['baseline_bundle'], swap['treated_bundle']
        if (len(common) != p+256 or common[:p] != row['prompt_ids'] or
                sum(token != mask for token in common[p:]) != 12 or swap['step_zero_based'] != 3 or
                len(set(b)) != 4 or len(set(t)) != 4 or
                set(b)-set(t) != {swap['deferred_position']} or set(t)-set(b) != {swap['advanced_position']} or
                any(not 0 <= i < 32 or common[p+i] != mask for i in b+t)):
            raise ValueError('Intervention mismatch')
        for arm in ('baseline', 'treated'):
            ids, out = row[arm+'_ids'], row[arm]
            if len(ids) != 256 or any(type(token) is not int or token < 0 or token == mask for token in ids):
                raise ValueError('Invalid generated token IDs')
            stop = next((i for i, token in enumerate(ids) if token in ends), None)
            parsed = extract(out['text'], q['question']) if stop is not None else dict(answer=None, reason='length_limit')
            correct = parsed['answer'] is not None and Fraction(parsed['answer']) == Fraction(q['answer'])
            if (out['first_terminator_position'] != stop or out['extraction'] != parsed or
                    out['answer'] != parsed['answer'] or out['correct'] != correct):
                raise ValueError('Extraction mismatch')
        rows.append(row)
    return rows


def aligned_logits(model, x):
    return model(x, use_cache=False).logits


def collect(root, phase, settings, forward_adapter, checkpoint=lambda: None, code_dir=HERE):
    import torch
    from transformers import AutoModel, AutoTokenizer
    if phase not in ('pilot', 'main'):
        raise ValueError('phase must be pilot or main')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20260929)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    cfg, items = Protocol(mask_id=settings['mask_id']), examples(phase)
    manifest = dict(format='compact-calendar-v1', phase=phase, settings=settings, protocol=asdict(cfg),
                    prompt_suffix=PROMPT_SUFFIX, examples=items, example_hashes=EXAMPLE_HASHES,
                    code={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (code_dir/'llada.py', code_dir/'dream.py', code_dir/'print_results.py')},
                    gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda,
                    packages={p: importlib.metadata.version(p) for p in
                              ('torch', 'transformers', 'huggingface-hub', 'numpy', 'tokenizers', 'safetensors', 'accelerate')})
    root = Path(root)
    if phase == 'main':
        pilot = read(root/'pilot/manifest.json')
        if any(pilot[k] != manifest[k] for k in manifest if k not in ('phase', 'examples')):
            raise ValueError('Pilot code/protocol/runtime differs; use a new run ID')
        if pilot['phase'] != 'pilot' or digest(pilot['examples']) != EXAMPLE_HASHES['pilot']:
            raise ValueError('Wrong pilot sample')
        stats = summarize(load_saved(root/'pilot', pilot), [q['id'] for q in pilot['examples']])
        if not stats['complete'] or not stats['all_controls_pass']:
            raise ValueError('A complete passing technical pilot is required')
    directory = root/phase
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/'manifest.json'
    if path.exists():
        if read(path) != manifest:
            raise ValueError('Incompatible resume; use a new run ID')
    elif list(directory.glob('question_*.json')):
        raise ValueError('Records without manifest')
    else:
        save_json(path, manifest)
        checkpoint()
    rows = load_saved(directory, manifest)
    done = {r['question_id'] for r in rows}
    if len(done) < len(items):
        tokenizer = AutoTokenizer.from_pretrained(settings['model'], revision=settings['revision'], trust_remote_code=True)
        model = AutoModel.from_pretrained(settings['model'], revision=settings['revision'],
                    code_revision=settings['revision'], trust_remote_code=True, torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True, **settings['model_kwargs']).to('cuda').eval()
        context_limit = getattr(model.config, 'max_position_embeddings', None) or model.config.max_sequence_length
        if model.config.mask_token_id != cfg.mask_id:
            raise ValueError('Unexpected model mask token')
        for q in items:
            if q['id'] in done:
                continue
            prompt = tokenizer.apply_chat_template([dict(role='user', content=q['question']+PROMPT_SUFFIX)],
                                                   add_generation_prompt=True, return_tensors='pt')[0].to('cuda')
            if (prompt == cfg.mask_id).any() or len(prompt)+cfg.gen_len > context_limit:
                raise ValueError('Invalid prompt/context length')
            torch.cuda.synchronize()
            start = time.monotonic()
            row = generate_pair(lambda x: forward_adapter(model, x), prompt, cfg, controls=True)
            torch.cuda.synchronize()
            row.update(question_id=q['id'], question=q['question'], gold=q['answer'],
                       prompt_ids=prompt.tolist(), seconds=time.monotonic()-start, manifest_sha256=digest(manifest))
            for arm in ('baseline', 'treated'):
                row[arm] = output(row[arm+'_ids'], tokenizer, q['question'], q['answer'], settings)
            save_json(directory/f"question_{q['id']:05d}.json", dict(row, sha256=digest(row)))
            rows.append(row)
            save_json(directory/'summary.json', summarize(rows, [q['id'] for q in items]))
            checkpoint()
            print(f"{phase} {len(rows)}/{len(items)} id={q['id']} seconds={row['seconds']:.1f} controls=pass", flush=True)
    stats = summarize(load_saved(directory, manifest), [q['id'] for q in items])
    save_json(directory/'summary.json', stats)
    checkpoint()
    return stats


# Modal entry points are shared; running either file selects its model explicitly.
def remote_experiment(phase: str, run_id: str, model_name: str):
    import importlib
    import sys
    import modal
    sys.path.insert(0, '/opt/experiment')
    experiment = importlib.import_module(model_name)
    results = modal.Volume.from_name('dlm-calendar-results')
    cache = modal.Volume.from_name('dlm-calendar-hf-cache')
    try:
        return collect(Path('/results')/run_id, phase, experiment.SETTINGS,
                       experiment.aligned_logits, checkpoint=results.commit, code_dir=Path('/opt/experiment'))
    finally:
        results.commit()
        cache.commit()


def make_app(model_name, remote_function):
    import modal
    app = modal.App('dlm-compact-'+model_name)
    image = (modal.Image.debian_slim(python_version='3.11')
        .pip_install('torch==2.9.0', 'transformers==4.51.3', 'datasets==3.6.0',
                     'huggingface-hub==0.34.4', 'accelerate==1.6.0', 'numpy==2.2.6',
                     'safetensors==0.6.2', 'sentencepiece==0.2.0')
        .env({'HF_HOME': '/cache/huggingface', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
              'CALENDAR_CODE_DIR': '/opt/experiment'}))
    for name in ('llada.py', 'dream.py', 'print_results.py'):
        image = image.add_local_file(HERE/name, '/opt/experiment/'+name)
    volumes = {path: modal.Volume.from_name(name, create_if_missing=True) for path, name in
               (('/cache', 'dlm-calendar-hf-cache'), ('/results', 'dlm-calendar-results'))}
    run = app.function(image=image, gpu='L40S', cpu=4, memory=32768, timeout=24*60*60,
                       max_containers=1, volumes=volumes)(remote_function)

    return app, run


def launch(run, model_name, phase, run_id):
    run_id = run_id or f'compact-{model_name}-v1'
    if phase not in ('pilot', 'main') or not re.fullmatch(f'compact-{model_name}-[a-zA-Z0-9_-]+', run_id):
        raise ValueError(f'Use pilot/main and a compact-{model_name}-... run ID')
    print(json.dumps(run.remote(phase, run_id), indent=2))


def run_experiment(phase: str, run_id: str):
    return remote_experiment(phase, run_id, 'llada')


def main(phase: str = 'pilot', run_id: str = ''):
    launch(run, 'llada', phase, run_id)

if __name__ == '__main__':
    print(__doc__)
else:
    try:
        import modal
    except ImportError:
        pass
    else:
        app, run = make_app('llada', run_experiment)
        main = app.local_entrypoint()(main)
