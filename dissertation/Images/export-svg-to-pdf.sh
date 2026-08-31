#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Export every ./Images/*.svg to a plain ./Images/<name>.pdf next to it.
# -----------------------------------------------------------------------------
# The normal local build never needs this: the 'svg' package drives Inkscape at
# compile time (see Preamble_commands.tex) and caches into ./Images/.svgcache/.
# Overleaf, however, has no Inkscape and only allows restricted shell escape, so
# that conversion cannot happen there. The preamble detects this and falls back
# to \includegraphics on the .pdf files this script produces.
#
# Run it after editing/adding any SVG, then commit the regenerated PDFs:
#     ./Images/export-svg-to-pdf.sh
#
# The Inkscape invocation mirrors what svg.sty issues with inkscapelatex=false
# (default export area, text rendered into the PDF), so the fallback figures are
# byte-for-byte the same artwork the local build embeds.
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
