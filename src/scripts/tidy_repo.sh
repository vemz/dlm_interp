#!/usr/bin/env bash
# Cut src/scripts/ down to the scripts that produce a published result.
#
#   bash tidy_repo.sh          # dry run: prints what it would do, changes nothing
#   bash tidy_repo.sh --go     # actually does it
#
# Nothing is deleted. Superseded scripts move to src/scripts/archive/, which
# keeps them reachable for anyone who reads results.md and asks "how was that
# negative measured" — git history does not answer that question for someone
# browsing a clone.
#
# Written for bash 3.2, the version macOS ships: no associative arrays.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "${here}/src/dlm_interp/paths.py" ]]; do
  [[ "${here}" == "/" ]] && { echo "run this from inside the repo"; exit 1; }
  here="$(dirname "${here}")"
done
cd "${here}"
echo "repo root  : ${here}"

GO=0
[[ "${1:-}" == "--go" ]] && GO=1

# ---------------------------------------------------------------- what stays
# One script per row of the results table. Nothing else.
KEEP="
collect_labels.py
collect_waitgain.py
collect_penalty_horizons.py
part1_ladder.py
depth_part1.py
bootstrap_horizon.py
checkpoint_curve_bootstrap.py
compare_seeds.py
run_seed.sh
tidy_repo.sh
"

# ------------------------------------------------------------- what moves out
# "file|reason". The successor is named so the decision is checkable rather
# than a matter of taste.
MOVE="
analysis.py|superseded by part1_ladder.py
probe.py|superseded by part1_ladder.py
probe_residual.py|superseded by part1_ladder.py
rq1_figure.py|superseded by bootstrap_horizon.py
probe_horizon.py|superseded by bootstrap_horizon.py
probe_penalty.py|superseded by bootstrap_horizon.py
checkpoint_curve.py|superseded by checkpoint_curve_bootstrap.py
depth_controls.py|superseded by depth_part1.py
collect_penalty.py|superseded by collect_penalty_horizons.py
inspect_labels.py|a debugging utility, not a result
commit_order.py|negative: the ordering rule, cited in results.md
compare.py|negative: the ordering rule, cited in results.md
collect_commitvalue.py|negative: the abandoned commit-value estimator
diagnose_value.py|negative: the abandoned commit-value estimator
test_ancestral_rate.py|the sampler rate check
"

names()  { printf '%s\n' "$MOVE" | while IFS='|' read -r f r; do
             [ -n "$f" ] && printf '%s\n' "$f"; done; }
reason() { printf '%s\n' "$MOVE" | while IFS='|' read -r f r; do
             [ "$f" = "$1" ] && printf '%s\n' "$r"; done; }
count()  { ls -1 src/scripts 2>/dev/null | grep -v '^archive$' | wc -l | tr -d ' '; }

echo "before     : $(count) files in src/scripts/"
echo

# ------------------------------------------------ safety: nothing imports them
echo "checking that no surviving script imports one that is about to move..."
fail=0
for f in $(names); do
  case "$f" in *.py) ;; *) continue ;; esac
  mod="${f%.py}"
  for k in $KEEP; do
    [ -f "src/scripts/$k" ] || continue
    # a real import statement only: the line must START with import/from, so
    # prose and comments that merely mention the old file do not trip this
    if grep -qE "^[[:space:]]*(import|from)[[:space:]]+[A-Za-z0-9_. ]*\b${mod}\b" "src/scripts/$k"; then
      echo "  BLOCKED: src/scripts/$k imports ${mod}"
      fail=1
    fi
  done
done
if [ $fail -eq 0 ]; then
  echo "  clean — the surviving scripts are self-contained"
else
  echo; echo "resolve the imports above before moving anything"; exit 1
fi
echo

# ----------------------------------------------------------------- do the move
[ $GO -eq 1 ] && mkdir -p src/scripts/archive
moved=0
for f in $(names); do
  [ -f "src/scripts/$f" ] || continue
  line="$(printf 'archive  %-28s -> %s' "$f" "$(reason "$f")")"
  if [ $GO -eq 1 ]; then
    echo "  $line"
    git mv "src/scripts/$f" "src/scripts/archive/$f" 2>/dev/null \
      || mv "src/scripts/$f" "src/scripts/archive/$f"
  else
    echo "  [dry] $line"
  fi
  moved=$((moved + 1))
done
[ $moved -eq 0 ] && echo "  nothing matched — already tidy, or the filenames differ"

# ------------------------------------------------------- the misplaced notebook
if [ -f src/dlm_interp/readiness_trace_tier1.ipynb ]; then
  msg="move     readiness_trace_tier1.ipynb -> notebooks/  (it is not library code)"
  if [ $GO -eq 1 ]; then
    echo "  $msg"
    mkdir -p notebooks
    git mv src/dlm_interp/readiness_trace_tier1.ipynb notebooks/ 2>/dev/null \
      || mv src/dlm_interp/readiness_trace_tier1.ipynb notebooks/
  else
    echo "  [dry] $msg"
  fi
fi

# --------------------------------------------------------------- archive README
if [ $GO -eq 1 ] && [ $moved -gt 0 ]; then
  {
    echo "# archive"
    echo
    echo "Nothing here is dead code — it is code that is no longer the way a"
    echo "result is produced."
    echo
    echo "Files marked *superseded* make the same measurement with the method the"
    echo "audit replaced: a standard error over split seeds instead of a bootstrap"
    echo "clustered on the grouping unit, or a fixed PCA budget instead of a"
    echo "dimension ladder. The successor is named for each."
    echo
    echo "Files marked *negative* are cited in results.md and are the only record"
    echo "of how the abandoned lines were measured."
    echo
    for f in $(names); do
      [ -f "src/scripts/archive/$f" ] && echo "- \`$f\` — $(reason "$f")"
    done
    true
  } > src/scripts/archive/README.md
  echo "  wrote    src/scripts/archive/README.md"
fi

echo
if [ $GO -eq 1 ]; then
  echo "after      : $(count) files in src/scripts/"
  echo "             $(ls -1 src/scripts/archive 2>/dev/null | wc -l | tr -d ' ') in src/scripts/archive/"
else
  echo "dry run. re-run with --go to apply."
fi
