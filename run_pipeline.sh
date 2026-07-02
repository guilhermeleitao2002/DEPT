#!/usr/bin/env bash
# run_pipeline.sh — run the full ADEPT pipeline end-to-end with per-phase timing.
# See: ./run_pipeline.sh --help

set -euo pipefail
export PYTHONUNBUFFERED=1

PYTHON="${PYTHON:-python3}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

START_PHASE=1
END_PHASE=6
SKIP_ADV=0

usage() {
    cat <<'EOF'
run_pipeline.sh — Run the full ADEPT pipeline end-to-end.

Phases (in order):
  1. Adversarial generation (TANTGen)   scripts/adversarial_generation/generate_adversarials.py
  2. Packet-to-Tensor                   scripts/packet_to_tensor/process_all_attacks.py
  3. Random Forest train + evaluate     scripts/rf/run_rf.py
  4. TCN train + evaluate               scripts/tcn/run_tcn.py
  5. Meta-Learner train + evaluate      scripts/metalearner/run_metalearner.py
  6. Reconstructor train + evaluate     scripts/reconstructor/run_reconstructor.py

Each phase is timed; per-phase and total wall-clock are printed in a summary
at the end. Live output is also mirrored to
  logs/pipeline_<timestamp>/phase_<N>_<name>.log
The script aborts on the first failing phase (set -euo pipefail).

Usage:
  ./run_pipeline.sh                 # all 6 phases
  ./run_pipeline.sh --skip-adv      # skip phase 1 (PCAPs already present)
  ./run_pipeline.sh --from N        # start at phase N  (1..6)
  ./run_pipeline.sh --to   N        # stop  after phase N (1..6)

Env overrides:
  PYTHON=python3   # interpreter to use (default: python3)
EOF
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-adv)  SKIP_ADV=1;       shift   ;;
        --from)      START_PHASE="$2"; shift 2 ;;
        --to)        END_PHASE="$2";   shift 2 ;;
        -h|--help)   usage 0 ;;
        *)           echo "Unknown argument: $1" >&2; usage 2 ;;
    esac
done

PHASE_NAMES=(
    [1]="adversarial_generation"
    [2]="packet_to_tensor"
    [3]="random_forest"
    [4]="tcn"
    [5]="metalearner"
    [6]="reconstructor"
)
PHASE_CMDS=(
    [1]="$PYTHON scripts/adversarial_generation/generate_adversarials.py"
    [2]="$PYTHON scripts/packet_to_tensor/process_all_attacks.py"
    [3]="$PYTHON scripts/rf/run_rf.py"
    [4]="$PYTHON scripts/tcn/run_tcn.py"
    [5]="$PYTHON scripts/metalearner/run_metalearner.py"
    [6]="$PYTHON scripts/reconstructor/run_reconstructor.py"
)

declare -A DURATIONS

SEP="==============================================================================="

section() {
    printf '\n%s\n PHASE %d/6 — %s\n%s\n' "$SEP" "$1" "$2" "$SEP"
}

format_duration() {
    local s=$1
    printf '%02d:%02d:%02d' $((s/3600)) $(((s%3600)/60)) $((s%60))
}

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/pipeline_${TS}"
mkdir -p "$LOG_DIR"

OVERALL_START=$(date +%s)

for i in 1 2 3 4 5 6; do
    if (( i < START_PHASE || i > END_PHASE )); then continue; fi
    if [[ $i -eq 1 && $SKIP_ADV -eq 1 ]]; then continue; fi

    name="${PHASE_NAMES[$i]}"
    cmd="${PHASE_CMDS[$i]}"
    log_file="$LOG_DIR/phase_${i}_${name}.log"

    section "$i" "$name"
    printf 'Command : %s\n'   "$cmd"
    printf 'Log     : %s\n'   "$log_file"
    printf 'Started : %s\n\n' "$(date '+%Y-%m-%d %H:%M:%S')"

    start=$(date +%s)
    set +e
    $cmd 2>&1 | tee "$log_file"
    rc=${PIPESTATUS[0]}
    set -e
    elapsed=$(( $(date +%s) - start ))
    DURATIONS[$i]=$elapsed

    printf '\nFinished: %s  (%s)\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$(format_duration "$elapsed")"

    if [[ $rc -ne 0 ]]; then
        printf '\nPHASE %d (%s) FAILED with exit code %d\n' "$i" "$name" "$rc" >&2
        exit "$rc"
    fi
done

OVERALL=$(( $(date +%s) - OVERALL_START ))

printf '\n%s\n PIPELINE SUMMARY\n%s\n' "$SEP" "$SEP"
for i in 1 2 3 4 5 6; do
    if [[ -n "${DURATIONS[$i]:-}" ]]; then
        printf '  phase %d  %-22s  %s\n' "$i" "${PHASE_NAMES[$i]}" "$(format_duration "${DURATIONS[$i]}")"
    fi
done
printf '  %-32s %s\n' "TOTAL" "$(format_duration "$OVERALL")"
printf '\nLogs: %s\n' "$LOG_DIR"
