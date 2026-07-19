"""Reproduce TANTGen's cicflowmeter flow extraction for a single PCAP.

TANTGen calibrated its ML-NIDS on flow features from the cicflowmeter build it
vendors under ``src/Feature_Extraction/cicflowmeter`` (v0.4.2), whose column set
is what ``cicflowmeter_mapping`` renames. This runner drives that vendored
package's own ``create_sniffer`` → ``flush_flows`` path — the exact code TANTGen
used to build its training CSVs — so features stay train/inference-consistent.

It is invoked as a subprocess by :mod:`.ml_nids` under an interpreter that can
import the vendored package (which pulls in ``scapy``, ``requests`` and
``numpy``); ml_nids prepends the package directory via ``PYTHONPATH``.

The pip-distributed ``cicflowmeter`` (0.1.6) is deliberately *not* used: under
scapy >= 2.6 its sniffer hooks are dead (it writes an empty CSV), and its column
set is missing ``cwr_flag_count``, so it fails the mapping check anyway.

Usage:
    python cicflowmeter_runner.py <pcap> <out_csv>
"""
from __future__ import annotations

import sys


def extract(pcap_path: str, out_csv: str) -> None:
    """Extract flows from ``pcap_path`` into ``out_csv`` (TANTGen's file path)."""
    from cicflowmeter.sniffer import create_sniffer

    sniffer, session = create_sniffer(
        input_file=pcap_path,
        input_interface=None,
        output_mode="csv",
        output=out_csv,
    )
    sniffer.start()
    try:
        sniffer.join()
    finally:
        # Mirror sniffer.py main(): stop the periodic GC thread, then flush the
        # flows still open at end-of-capture so none are dropped.
        if hasattr(session, "_gc_stop"):
            session._gc_stop.set()
            session._gc_thread.join(timeout=2.0)
        session.flush_flows()


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(f"usage: {argv[0]} <pcap> <out_csv>", file=sys.stderr)
        return 2
    extract(argv[1], argv[2])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
