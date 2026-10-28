#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Export every ./Images/*.svg to a plain ./Images/<name>.pdf next to it.
# -----------------------------------------------------------------------------
# These PDFs are only used when the build cannot run Inkscape (e.g. Overleaf);
# see Preamble_commands.tex. Run after editing or adding an SVG:
#     ./Images/export-svg-to-pdf.sh [--force]
set -euo pipefail

cd "$(dirname "$0")"

command -v inkscape >/dev/null 2>&1 || {
    echo "error: inkscape not found in PATH" >&2
    exit 1
}

force="${1:-}"

updated=0
for svg in *.svg; do
    [ -e "$svg" ] || continue
    pdf="${svg%.svg}.pdf"
    if [ "$force" != "--force" ] && [ -f "$pdf" ] && [ "$pdf" -nt "$svg" ]; then
        echo "up to date  $pdf"
        continue
    fi
    echo "exporting   $pdf"
    inkscape "$svg" --export-filename="$pdf" >/dev/null
    updated=$((updated + 1))
done

echo "done: $updated file(s) exported"
