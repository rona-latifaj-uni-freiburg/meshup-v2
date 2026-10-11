#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/../densematcher_visuals"
export TEXMFVAR=/tmp/meshup_texmf_var
export TEXMFCONFIG=/tmp/meshup_texmf_config
export TEXMFCACHE=/tmp/meshup_texmf_cache
pandoc presentation_guide.md --standalone --toc --toc-depth=2 --mathml --css=presentation_guide.css -o presentation_guide.html
pandoc presentation_guide.md --standalone --toc --toc-depth=2 --listings --lua-filter=presentation_guide_pdf.lua -V geometry:margin=20mm -V fontsize=10pt --include-in-header=presentation_guide_header.tex -o presentation_guide.tex
pdflatex -interaction=nonstopmode -halt-on-error presentation_guide.tex > presentation_guide_build.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error presentation_guide.tex >> presentation_guide_build.log 2>&1
