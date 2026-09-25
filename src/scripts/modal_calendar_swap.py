"""Launch explicitly: modal run src/scripts/modal_calendar_swap.py --phase pilot"""
from pathlib import Path
import hashlib
import json
import re

import modal

HERE = Path(__file__).resolve().parent
APP_NAME = 'dlm-calendar-swap'
app = modal.App(APP_NAME)
cache = modal.Volume.from_name('dlm-calendar-hf-cache', create_if_missing=True)
results = modal.Volume.from_name('dlm-calendar-results', create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version='3.11')
    .pip_install('torch==2.9.0', 'transformers==4.51.3', 'datasets==3.6.0',
                 'huggingface-hub==0.34.4', 'accelerate==1.6.0', 'numpy==2.2.6',
                 'safetensors==0.6.2', 'sentencepiece==0.2.0')
    .env({'HF_HOME': '/cache/huggingface', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'})
    .add_local_file(HERE / 'calendar_swap.py', '/opt/experiment/calendar_swap.py')
)


@app.function(image=image, gpu='L40S', cpu=4, memory=32768,
              timeout=24 * 60 * 60, max_containers=1,
              volumes={'/cache': cache, '/results': results})
def run_experiment(phase: str, run_id: str, launcher_sha256: str):
    import sys
    sys.path.insert(0, '/opt/experiment')
    from calendar_swap import collect
    try:
        return collect(Path('/results') / run_id, phase, checkpoint=results.commit,
                       launcher_sha256=launcher_sha256)
    finally:
        results.commit()
        cache.commit()


@app.local_entrypoint()
def main(phase: str = 'pilot', run_id: str = 'calendar-swap-v1'):
    if phase not in ('pilot', 'main') or not re.fullmatch(r'[a-zA-Z0-9_-]+', run_id):
        raise ValueError('phase must be pilot/main; run-id accepts letters, digits, _ and -')
    summary = run_experiment.remote(phase, run_id, hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    print(json.dumps(summary, indent=2))
