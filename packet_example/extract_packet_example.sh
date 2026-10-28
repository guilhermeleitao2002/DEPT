#!/usr/bin/env bash
# =============================================================================
# extract_packet_example.sh
# -----------------------------------------------------------------------------
# Extract the flow shown in the thesis packet figure (Fig. 5.4) from the clean,
# adversarial and purified captures, and print its IPv4 total length at each
# stage plus a tshark validity check.
#
# Flow (Mirai, size-only variant, from the reconstructor's test split):
#     192.168.2.108:35220  ->  52.25.66.250:8280   (TCP, 4 packets)
#
# Sources:
#   clean       original malicious capture (TANTGen tree)
#   adversarial TANTGen size-attack output      (data/adversarial/)
#   purified    DEPT reconstructor realization  (data/models/reconstructor/)
#
# Requires: tshark, editcap (wireshark-common).
# Usage:    ./extract_packet_example.sh
# =============================================================================
set -euo pipefail

# --- Resolve repo roots (this script lives in DEPT/packet_example) ------------
HERE="$(cd "$(dirname "$0")" && pwd)"
DEPT_ROOT="$(cd "$HERE/.." && pwd)"
TANTGEN_ROOT="$(cd "$DEPT_ROOT/.." && pwd)/Transformer-based_Adversarial_Network_Traffic_Generator"

CLEAN_PCAP="$TANTGEN_ROOT/data/original/Kitsune/Malicious/Mirai_malicious.pcap"
ADV_PCAP="$DEPT_ROOT/data/adversarial/mirai/mirai_adversarial_size.pcap"
PUR_PCAP="$DEPT_ROOT/data/models/reconstructor/realized/mirai_adv_size.pcap"

# --- The flow, as a tshark display filter -------------------------------------
# Exclude ICMP errors that embed a copy of these packets.
SRC="192.168.2.108"; SPORT="35220"
DST="52.25.66.250";  DPORT="8280"
FILTER="ip.src==${SRC} && ip.dst==${DST} && tcp.srcport==${SPORT} && tcp.dstport==${DPORT} && not icmp"

# --- Output directory (one small capture per stage) ----------------------------
OUT_DIR="$HERE"

for tool in tshark editcap; do
    command -v "$tool" >/dev/null 2>&1 || { echo "ERROR: '$tool' not found (install wireshark-common)."; exit 1; }
done

extract () {
    # $1 stage label, $2 source pcap
    local stage="$1" src="$2" out="$OUT_DIR/flow_$1.pcap"
    if [[ ! -f "$src" ]]; then
        echo "ERROR: missing source capture for '$stage': $src" >&2
        exit 1
    fi
    # Slice out just the chosen flow into a small capture.
    tshark -r "$src" -Y "$FILTER" -w "$out" >/dev/null 2>&1
    echo "$out"
}

printf '%s\n' "==============================================================================="
printf ' DEPT packet example  —  flow %s:%s -> %s:%s (TCP)\n' "$SRC" "$SPORT" "$DST" "$DPORT"
printf '%s\n' "==============================================================================="

printf '\n%-14s %-8s %-10s %-12s %s\n' "stage" "pkt#" "ip.len" "tcp.flags" "frame.len"
printf '%s\n' "-------------------------------------------------------------------------------"
for pair in "clean:$CLEAN_PCAP" "adversarial:$ADV_PCAP" "purified:$PUR_PCAP"; do
    stage="${pair%%:*}"; src="${pair#*:}"
    out="$(extract "$stage" "$src")"
    # Per-packet IPv4 total length (the manipulated quantity), flags, frame size.
    tshark -r "$out" -T fields -e frame.number -e ip.len -e tcp.flags.str -e frame.len 2>/dev/null \
      | while IFS=$'\t' read -r n len flags flen; do
            printf '%-14s %-8s %-10s %-12s %s\n' "$stage" "$n" "$len" "$flags" "$flen"
        done
done

printf '\n%s\n' "Protocol compliance (tshark, checksum validation on):"
printf '%s\n' "-------------------------------------------------------------------------------"
for pair in "clean:$CLEAN_PCAP" "adversarial:$ADV_PCAP" "purified:$PUR_PCAP"; do
    stage="${pair%%:*}"
    out="$OUT_DIR/flow_$stage.pcap"
    total="$(tshark -r "$out" -T fields -e frame.number 2>/dev/null | grep -c .)"
    bad="$(tshark -r "$out" \
             -o ip.check_checksum:TRUE -o tcp.check_checksum:TRUE -o udp.check_checksum:TRUE \
             -Y '_ws.malformed || ip.checksum.status=="Bad" || tcp.checksum.status=="Bad" || udp.checksum.status=="Bad"' \
             -T fields -e frame.number 2>/dev/null | grep -c . || true)"
    printf '  %-14s %s/%s packets malformed or bad-checksum\n' "$stage" "$bad" "$total"
done

printf '\n%s\n' "The appended adversarial padding (real bytes, packet 1):"
printf '%s\n' "-------------------------------------------------------------------------------"
# The TANTGen size attack appends payload; show the difference in TCP payload
# length and the raw appended bytes for the first packet.
clean_pl="$(tshark -r "$OUT_DIR/flow_clean.pcap"       -T fields -e tcp.len -c 1 2>/dev/null)"
adv_pl="$(tshark   -r "$OUT_DIR/flow_adversarial.pcap" -T fields -e tcp.len -c 1 2>/dev/null)"
pur_pl="$(tshark   -r "$OUT_DIR/flow_purified.pcap"    -T fields -e tcp.len -c 1 2>/dev/null)"
printf '  TCP payload length  clean=%s B  ->  adversarial=%s B  ->  purified=%s B\n' \
       "$clean_pl" "$adv_pl" "$pur_pl"
printf '  appended bytes (adversarial, packet 1): '
tshark -r "$OUT_DIR/flow_adversarial.pcap" -T fields -e tcp.payload -c 1 2>/dev/null \
  | tr -d ':' | sed 's/\(..\)/\1 /g'

printf '\nSmall per-stage captures written under: %s\n' "$OUT_DIR"
