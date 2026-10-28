"""Reproduce TANTGen's cicflowmeter flow extraction for a single PCAP.

Runs the cicflowmeter build vendored by TANTGen
(``src/Feature_Extraction/cicflowmeter``, v0.4.2), the one its ML-NIDS was
trained on. Called as a subprocess by :mod:`.ml_nids`, with the package on
``PYTHONPATH``. The pip ``cicflowmeter`` (0.1.6) is not used: with scapy >= 2.6
it writes empty CSVs, and it lacks ``cwr_flag_count``.

The vendored ``CSVWriter.write`` is called from two threads and could write
the header twice, so it is wrapped in a lock.

Usage:
    python cicflowmeter_runner.py <pcap> <out_csv>
"""
from __future__ import annotations

import sys
import threading


def _serialise_writer(session) -> None:
    """Guard the session's output writer so concurrent writes can't race."""
    writer = session.output_writer
    lock = threading.Lock()
    original = writer.write

    def write(data, _original=original, _lock=lock):
        with _lock:
            _original(data)

    writer.write = write


def extract(pcap_path: str, out_csv: str) -> None:
    """Extract flows from ``pcap_path`` into ``out_csv`` (TANTGen's file path)."""
    from cicflowmeter.sniffer import create_sniffer

    sniffer, session = create_sniffer(
        input_file=pcap_path,
        input_interface=None,
        output_mode="csv",
        output=out_csv,
    )
    _serialise_writer(session)
    sniffer.start()
    try:
        sniffer.join()
    finally:
        # Mirror sniffer.py main(): stop the periodic GC thread (fully, so it is
        # not writing during flush), then flush the flows still open at
        # end-of-capture so none are dropped.
        if hasattr(session, "_gc_stop"):
            session._gc_stop.set()
            session._gc_thread.join(timeout=30.0)
        session.flush_flows()


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(f"usage: {argv[0]} <pcap> <out_csv>", file=sys.stderr)
        return 2
    extract(argv[1], argv[2])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
