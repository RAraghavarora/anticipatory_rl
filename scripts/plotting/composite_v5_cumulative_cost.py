# Stack the 2-room and v5 cumulative-cost panels into one figure at the
# image level, not inside R -- building both panels' guide_legend() calls
# in one R session previously crashed ggplot2's guide-building (see
# plot_v5_cumulative_cost_ggdist.R's header for the full story). Run this
# after both source PDFs are up to date:
#   Rscript scripts/plotting/plot_canonical_cost_ggdist.R
#   Rscript scripts/plotting/plot_v5_cumulative_cost_ggdist.R
#   python3 scripts/plotting/composite_v5_cumulative_cost.py

import subprocess
import tempfile
from pathlib import Path

from PIL import Image

TOP = "results/canonical_planner/figures/cumulative_cost_ggdist.pdf"
BOTTOM = "results/v5/figures/cumulative_cost_ggdist_v5.pdf"
OUT_PDF = "results/v5/figures/thesis/cumulative_cost_ggdist_both.pdf"


def rasterize(pdf_path, out_prefix):
    subprocess.run(["pdftoppm", "-png", "-r", "150", pdf_path, out_prefix], check=True)
    return f"{out_prefix}-1.png"


with tempfile.TemporaryDirectory() as tmp:
    top_png = rasterize(TOP, f"{tmp}/top")
    bottom_png = rasterize(BOTTOM, f"{tmp}/bottom")

    a, b = Image.open(top_png), Image.open(bottom_png)
    w = max(a.width, b.width)

    def pad(im):
        if im.width == w:
            return im
        canvas = Image.new("RGB", (w, im.height), "white")
        canvas.paste(im, ((w - im.width) // 2, 0))
        return canvas

    a, b = pad(a), pad(b)
    combined = Image.new("RGB", (w, a.height + b.height), "white")
    combined.paste(a, (0, 0))
    combined.paste(b, (0, a.height))

    Path(OUT_PDF).parent.mkdir(parents=True, exist_ok=True)
    combined.save(OUT_PDF, "PDF", resolution=150.0)

print(f"wrote {OUT_PDF}")
