#!/usr/bin/env bash
# run_pipeline.sh — run the full DEPT pipeline end-to-end with per-phase timing.
# See: ./run_pipeline.sh --help

set -euo pipefail
export PYTHONUNBUFFERED=1

PYTHON="${PYTHON:-python3}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

START_PHASE=1
END_PHASE=8
SKIP_ADV=0
PER_ATTACK=0

# The 10 attacks with TANTGen adversarial counterparts (mirrors the
# benign_sources list in config/rf.yaml / config/tcn.yaml).
ATTACKS=(brute_force ddos dns_ddos ldap_ddos mirai ntp_ddos portscan
         ssdp_ddos ssdp_flood syn_ddos)

usage() {
    cat <<'EOF'
run_pipeline.sh — Run the full DEPT pipeline end-to-end.

Phases (in order):
  1. Adversarial generation (TANTGen)   scripts/adversarial_generation/generate_adversarials.py
  2. Packet-to-Tensor                   scripts/packet_to_tensor/process_all_attacks.py
  3. Random Forest train + evaluate     scripts/rf/run_rf.py
  4. TCN train + evaluate               scripts/tcn/run_tcn.py
  5. Meta-Learner train + evaluate      scripts/metalearner/run_metalearner.py
  6. Reconstructor train + evaluate     scripts/reconstructor/run_reconstructor.py
  7. Gradient ablation (RQ1)            scripts/gradient_ablation/run_gradient_ablation.py
  8. Deployment baseline                scripts/baseline/run_baseline.py

Each phase is timed; per-phase and total wall-clock are printed in a summary
at the end. Live output is also mirrored to
  logs/pipeline_<timestamp>/phase_<N>_<name>.log
The script aborts on the first failing phase (set -euo pipefail).

Usage:
  ./run_pipeline.sh                 # all 8 phases (one global model per stage)
  ./run_pipeline.sh --skip-adv      # skip phase 1 (PCAPs already present)
  ./run_pipeline.sh --from N        # start at phase N  (1..8)
  ./run_pipeline.sh --to   N        # stop  after phase N (1..8)

Per-attack mode:
  ./run_pipeline.sh --per-attack            # phases 3-6 once per attack
  ./run_pipeline.sh --per-attack --from 5   # e.g. meta + reconstructor only
  ./run_pipeline.sh --attacks mirai,ddos    # subset (implies --per-attack)

  Runs the model phases (3-6) once per attack, passing --attack <name> to
  each stage script. Phases 1-2 are per-attack by construction (their
  outputs are keyed by attack), so they are skipped here — run them via a
  normal (global) invocation first. Phases 7-8 evaluate the GLOBAL models
  (their configs read data/models/<stage>/), so they are also skipped in
  per-attack mode — run them via a normal (global) invocation. Global
  artifacts stay at data/models/<stage>/; per-attack artifacts land under
  data/models/per_attack/<attack>/<stage>/. Logs go to
  logs/pipeline_<timestamp>/<attack>/phase_<N>_<name>.log.

Env overrides:
  PYTHON=python3   # interpreter to use (default: python3)
EOF
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-adv)   SKIP_ADV=1;       shift   ;;
        --from)       START_PHASE="$2"; shift 2 ;;
        --to)         END_PHASE="$2";   shift 2 ;;
        --per-attack) PER_ATTACK=1;     shift   ;;
        --attacks)    PER_ATTACK=1
                      IFS=',' read -r -a ATTACKS <<< "$2"; shift 2 ;;
        -h|--help)    usage 0 ;;
        *)            echo "Unknown argument: $1" >&2; usage 2 ;;
    esac
done

PHASE_NAMES=(
    [1]="adversarial_generation"
    [2]="packet_to_tensor"
    [3]="random_forest"
    [4]="tcn"
    [5]="metalearner"
    [6]="reconstructor"
    [7]="gradient_ablation"
    [8]="baseline"
)
PHASE_CMDS=(
    [1]="$PYTHON scripts/adversarial_generation/generate_adversarials.py"
    [2]="$PYTHON scripts/packet_to_tensor/process_all_attacks.py"
    [3]="$PYTHON scripts/rf/run_rf.py"
    [4]="$PYTHON scripts/tcn/run_tcn.py"
    [5]="$PYTHON scripts/metalearner/run_metalearner.py"
    [6]="$PYTHON scripts/reconstructor/run_reconstructor.py"
    [7]="$PYTHON scripts/gradient_ablation/run_gradient_ablation.py"
    [8]="$PYTHON scripts/baseline/run_baseline.py"
)

declare -A DURATIONS

SEP="==============================================================================="

section() {
    printf '\n%s\n PHASE %d/8 — %s\n%s\n' "$SEP" "$1" "$2" "$SEP"
}

format_duration() {
    local s=$1
    printf '%02d:%02d:%02d' $((s/3600)) $(((s%3600)/60)) $((s%60))
}

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/pipeline_${TS}"
mkdir -p "$LOG_DIR"

# run_phase <phase> [<attack>] — run one phase (optionally scoped to one
# attack), tee its output to the log, and record its duration.
run_phase() {
    local i="$1" attack="${2:-}"
    local name="${PHASE_NAMES[$i]}"
    local cmd="${PHASE_CMDS[$i]}"
    local label="$name" log_file key

    if [[ -n "$attack" ]]; then
        cmd+=" --attack $attack"
        label="$name  [attack: $attack]"
        log_file="$LOG_DIR/$attack/phase_${i}_${name}.log"
        key="$attack:$i"
        mkdir -p "$LOG_DIR/$attack"
    else
        log_file="$LOG_DIR/phase_${i}_${name}.log"
        key="$i"
    fi

    section "$i" "$label"
    printf 'Command : %s\n'   "$cmd"
    printf 'Log     : %s\n'   "$log_file"
    printf 'Started : %s\n\n' "$(date '+%Y-%m-%d %H:%M:%S')"

    local start rc elapsed
    start=$(date +%s)
    set +e
    $cmd 2>&1 | tee "$log_file"
    rc=${PIPESTATUS[0]}
    set -e
    elapsed=$(( $(date +%s) - start ))
    DURATIONS[$key]=$elapsed

    printf '\nFinished: %s  (%s)\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$(format_duration "$elapsed")"

    if [[ $rc -ne 0 ]]; then
        printf '\nPHASE %d (%s) FAILED with exit code %d\n' "$i" "$label" "$rc" >&2
        exit "$rc"
    fi
}

OVERALL_START=$(date +%s)

if (( PER_ATTACK )); then
    if (( START_PHASE < 3 )); then
        printf 'Note: --per-attack only applies to phases 3-6; starting at phase 3.\n'
        printf '      (Phases 1-2 are per-attack by construction — run them globally.)\n'
        START_PHASE=3
    fi
    if (( END_PHASE > 6 )); then
        printf 'Note: --per-attack only applies to phases 3-6; stopping after phase 6.\n'
        printf '      (Phases 7-8 evaluate the global models — run them globally.)\n'
        END_PHASE=6
    fi
    printf 'Per-attack run over %d attack(s): %s\n' \
        "${#ATTACKS[@]}" "${ATTACKS[*]}"
    for attack in "${ATTACKS[@]}"; do
        for i in 3 4 5 6; do
            if (( i < START_PHASE || i > END_PHASE )); then continue; fi
            run_phase "$i" "$attack"
        done
    done
else
    for i in 1 2 3 4 5 6 7 8; do
        if (( i < START_PHASE || i > END_PHASE )); then continue; fi
        if [[ $i -eq 1 && $SKIP_ADV -eq 1 ]]; then continue; fi
        run_phase "$i"
    done
fi

OVERALL=$(( $(date +%s) - OVERALL_START ))

printf '\n%s\n PIPELINE SUMMARY\n%s\n' "$SEP" "$SEP"
if (( PER_ATTACK )); then
    for attack in "${ATTACKS[@]}"; do
        attack_total=0
        attack_ran=0
        for i in 3 4 5 6; do
            if [[ -n "${DURATIONS[$attack:$i]:-}" ]]; then
                attack_total=$(( attack_total + DURATIONS[$attack:$i] ))
                attack_ran=1
            fi
        done
        if (( attack_ran )); then
            printf '  %s\n' "$attack"
            for i in 3 4 5 6; do
                if [[ -n "${DURATIONS[$attack:$i]:-}" ]]; then
                    printf '    phase %d  %-22s  %s\n' "$i" "${PHASE_NAMES[$i]}" \
                        "$(format_duration "${DURATIONS[$attack:$i]}")"
                fi
            done
            printf '    %-32s %s\n' "subtotal" "$(format_duration "$attack_total")"
        fi
    done
else
    for i in 1 2 3 4 5 6 7 8; do
        if [[ -n "${DURATIONS[$i]:-}" ]]; then
            printf '  phase %d  %-22s  %s\n' "$i" "${PHASE_NAMES[$i]}" "$(format_duration "${DURATIONS[$i]}")"
        fi
    done
fi
printf '  %-32s %s\n' "TOTAL" "$(format_duration "$OVERALL")"
printf '\nLogs: %s\n' "$LOG_DIR"
