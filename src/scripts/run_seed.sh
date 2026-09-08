#!/usr/bin/env bash
# One model, end to end.
#
#   ./run_seed.sh s1 runs/baseline_s1_30k/best.pt
#   ./run_seed.sh s2 runs/baseline_s2_30k/best.pt
#   ./run_seed.sh s0 runs/baseline_s0/best.pt        # re-check seed 0 through the
#                                                    # same path, into results_s0/
#
# DLM_CKPT picks the weights, DLM_TAG picks where everything written lands.
# Checkpoints stay in runs/ for all three models; labels, caches, CSVs and
# figures go to runs_<tag>/ and results_<tag>/, so nothing collides.
#
# Split it if you are collecting on a GPU box and analysing on a laptop:
#   COLLECT_ONLY=1 ./run_seed.sh s1 runs/baseline_s1_30k/best.pt   (on the GPU)
#   rsync the runs_s1/ directory back
#   ANALYSE_ONLY=1 ./run_seed.sh s1                                (on the laptop)

set -euo pipefail

# Work from the repo root whatever directory this was launched from, and
# wherever the script itself was dropped (root, src/scripts/, anywhere).
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "${here}/src/dlm_interp/paths.py" ]]; do
  [[ "${here}" == "/" ]] && { echo "cannot find the repo root from $0"; exit 1; }
  here="$(dirname "${here}")"
done
cd "${here}"
echo "repo root  : ${here}"

SEED="${1:?usage: run_seed.sh <tag without underscore, e.g. s1> [checkpoint]}"
CKPT="${2:-baseline_${SEED}_30k/best.pt}"

export DLM_TAG="_${SEED}"
export DLM_CKPT="${CKPT}"

echo "model      : ${DLM_CKPT}"
echo "writes to  : runs${DLM_TAG}/ and results${DLM_TAG}/"
echo

COLLECTORS=(
  src/scripts/collect_labels.py
  src/scripts/collect_waitgain.py
  src/scripts/collect_penalty_horizons.py
)

ANALYSIS=(
  "src/scripts/part1_ladder.py predictability"
  "src/scripts/part1_ladder.py readiness"
  "src/scripts/part1_ladder.py"
  "src/scripts/depth_part1.py readiness"
  "src/scripts/bootstrap_horizon.py"
)

# Fail before doing any work rather than three hours in.
missing=0
for f in "${COLLECTORS[@]}"; do
  [[ -f "$f" ]] || { echo "missing: $f"; missing=1; }
done
# The analysis reads labels, never the weights, so only the collectors need it.
if [[ -z "${ANALYSE_ONLY:-}" && ! -f "${CKPT}" ]]; then
  echo "missing checkpoint: ${CKPT}"
  missing=1
fi
(( missing == 0 )) || { echo; echo "fix the paths above first"; exit 1; }

if [[ -z "${ANALYSE_ONLY:-}" ]]; then
  for f in "${COLLECTORS[@]}"; do
    echo "=== $f ==="
    time python "$f"
    echo
  done
fi

if [[ -z "${COLLECT_ONLY:-}" ]]; then
  for cmd in "${ANALYSIS[@]}"; do
    echo "=== $cmd ==="
    time python $cmd
    echo
  done
fi

echo "done. results${DLM_TAG}/ now holds:"
ls -1 "results${DLM_TAG}/"