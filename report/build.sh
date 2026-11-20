#!/bin/sh
# Build the report: figures/*.py -> figures/*.pdf, then report.tex -> report.pdf.
# Needs python3 + matplotlib and a TeX distribution (pdflatex). Run from anywhere: sh report/build.sh
set -e
cd "$(dirname "$0")"
for f in figures/*.py; do [ "$(basename "$f")" = style.py ] || PYTHONDONTWRITEBYTECODE=1 python3 "$f"; done
for i in 1 2 3; do  # extra passes resolve the contents, references and page count
  pdflatex -interaction=batchmode -halt-on-error report.tex >/dev/null || { tail -20 report.log; exit 1; }
done
rm -f report.aux report.log report.out report.toc
echo "built report.pdf ($(pdfinfo report.pdf 2>/dev/null | awk '/Pages/{print $2}') pages)"
