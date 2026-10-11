"""Assemble the self-contained showcase HTML for Artifact publishing.

Embeds cherry-picked deformation videos, poster stills, and the report's
comparison charts as base64 data URIs so the page has zero external
dependencies (required by the Artifact CSP).
"""

from __future__ import annotations

import base64
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VIDEOS = ROOT / "report_final" / "site_src" / "videos"
STILLS = ROOT / "report_final" / "site_src" / "posters"
CHARTS = ROOT / "report_final" / "figures"
REPORT_PDF = ROOT / "report_final" / "main_web.pdf"
OUT = ROOT / "report_final" / "site_src" / "index.html"


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def data_uri(path: Path, mime: str) -> str:
    return f"data:{mime};base64,{b64(path)}"


def pdf_page_count(path: Path) -> int:
    data = path.read_bytes()
    count = len(re.findall(rb"/Type\s*/Page(?!s)", data))
    if count:
        return count
    matches = re.findall(rb"/Count\s+(\d+)", data)
    return int(matches[-1]) if matches else 0


# (video basename without .mp4, still basename without .jpg, title, tag, blurb, metrics-or-None)
PAIRS = [
    (
        "01_horse2_giraffe4_deformation_slow_partfield",
        "01_horse2_giraffe4_pair",
        "Horse2 → Giraffe4",
        "best result",
        "Two compatible quadruped structures. PartField separates neck, body, and legs cleanly, "
        "and the deformation stretches proportions without losing part identity.",
        "Chamfer 0.0134 · F@0.02 0.992",
    ),
    (
        "02_panda3_bear_deformation_slow_partfield",
        "02_panda3_bear_pair",
        "Panda3 → Bear",
        "strong result",
        "Same-order body plan, different proportions. Local bucket Chamfer keeps limbs and head "
        "coherent through the shape change.",
        "Chamfer 0.0199 · F@0.02 0.973",
    ),
    (
        "06_fox_cat2_deformation_slow_partfield",
        "06_fox_cat2_pair",
        "Fox → Cat2",
        "strong result",
        "Small-carnivore body plans transfer well: ears, tail, and legs stay anchored to their "
        "own regions throughout.",
        "Chamfer 0.0224 · F@0.02 0.946",
    ),
    (
        "05_goldie_pug_deformation_slow_partfield",
        "05_goldie_pug_pair",
        "Goldie → Pug",
        "honest partial failure",
        "A real limitation, shown rather than hidden: PartField buckets match the tail region as "
        "a whole, but not tail-tip to tail-tip — exact endpoint identity can drift.",
        "Chamfer 0.0248 · F@0.02 0.906",
    ),
    (
        "10_elephant_giraffe_deformation_slow_partfield",
        "10_elephant_giraffe_pair",
        "Elephant → Giraffe",
        "DenseCorr3D benchmark",
        "Benchmark meshes, not SAM3D reconstructions. A structurally easier direction than the "
        "reverse — giraffe proportions are closer to what elephant topology can approximate.",
        "Chamfer 0.0279 · F@0.02 0.857",
    ),
    (
        "11_panther_horse2_deformation_slow_partfield",
        "11_panther_horse2_pair",
        "Panther → Horse2",
        "cross-dataset chain",
        "A DenseCorr3D source deformed toward a self-created SAM3D target — the pipeline does "
        "not care which dataset a mesh came from.",
        "Chamfer 0.0231 · F@0.02 0.892",
    ),
    (
        "12_horse2_dense_giraffe_deformation_slow_partfield",
        "12_horse2_dense_giraffe_pair",
        "Horse2 → Dense Giraffe",
        "cross-dataset chain",
        "The other half of the chain: a SAM3D source toward a DenseCorr3D-derived target.",
        "Chamfer 0.0189 · F@0.02 0.969",
    ),
    (
        "07_bulldog_cat_deformation_slow_partfield",
        "07_bulldog_cat_pair",
        "Bulldog → Cat",
        "domain diversity",
        "An early dog-morph asset from before the PartField pivot, kept here for range: stocky, "
        "short-limbed body plans stress the local-bucket assumption differently than lean animals.",
        None,
    ),
    (
        "03_cat_dachshund_deformation_slow_partfield",
        "03_cat_dachshund_pair",
        "Cat → Dachshund",
        "domain diversity",
        "Extreme proportion change — a short compact body toward a long low one — without "
        "breaking part identity.",
        None,
    ),
    (
        "04_elephant_moose_deformation_slow_partfield",
        "04_elephant_moose_pair",
        "Elephant → Moose",
        "hard stress case",
        "A genuinely hard DenseCorr3D pair: antlers and a moose muzzle are structure the elephant "
        "topology has no real counterpart for.",
        None,
    ),
    (
        "08_bulldog_dachshund_deformation_slow_partfield",
        "08_bulldog_dachshund_pair",
        "Bulldog → Dachshund",
        "domain diversity",
        "Another early dog-morph pair, included for a fuller picture of the dog-domain exploration "
        "that preceded the PartField pivot.",
        None,
    ),
    (
        "09_dachshund_cat_deformation_slow_partfield",
        "09_dachshund_cat_pair",
        "Dachshund → Cat",
        "domain diversity",
        "The reverse direction of pair 3, showing the deformation is not direction-locked.",
        None,
    ),
    (
        "13_panther_horse3_deformation_slow_partfield",
        "13_panther_horse3_pair",
        "Panther → Horse3",
        "cross-dataset chain",
        "A second chain variant using the horse3 asset instead of horse2.",
        None,
    ),
    (
        "14_horse3_dense_giraffe_deformation_slow_partfield",
        "14_horse3_dense_giraffe_pair",
        "Horse3 → Dense Giraffe",
        "cross-dataset chain",
        "Completes the horse3 chain variant.",
        None,
    ),
]

TAG_ORDER = {
    "best result": 0,
    "strong result": 1,
    "honest partial failure": 2,
    "hard stress case": 2,
    "DenseCorr3D benchmark": 3,
    "cross-dataset chain": 4,
    "domain diversity": 5,
}
PAIRS.sort(key=lambda p: TAG_ORDER.get(p[3], 9))


def build_cards() -> str:
    cards = []
    for video_base, still_base, title, tag, blurb, metrics in PAIRS:
        video_path = VIDEOS / f"{video_base}.mp4"
        still_path = STILLS / f"{video_base}.png"
        video_data = data_uri(video_path, "video/mp4")
        poster_data = data_uri(still_path, "image/png")
        metrics_html = f'<div class="card-metrics">{metrics}</div>' if metrics else ""
        cards.append(
            f"""
      <article class="card">
        <div class="card-media">
          <video class="card-video" muted loop playsinline preload="none"
                 poster="{poster_data}" src="{video_data}"></video>
          <span class="card-tag">{tag}</span>
        </div>
        <div class="card-body">
          <h3>{title}</h3>
          <p>{blurb}</p>
          {metrics_html}
        </div>
      </article>"""
        )
    return "\n".join(cards)


def main() -> None:
    chart_category = data_uri(CHARTS / "densematcher_category_comparison.png", "image/png")
    chart_hist = data_uri(CHARTS / "auc_spread_histogram.png", "image/png")
    chart_sam3d = data_uri(CHARTS / "sam3d_extended_fscore.png", "image/png")
    showcase_animals = data_uri(CHARTS / "domain_animals_showcase.png", "image/png")
    report_pdf = data_uri(REPORT_PDF, "application/pdf")
    report_size_mb = REPORT_PDF.stat().st_size / (1024 * 1024)
    report_pages = pdf_page_count(REPORT_PDF)

    cards_html = build_cards()

    template = TEMPLATE.format(
        chart_category=chart_category,
        chart_hist=chart_hist,
        chart_sam3d=chart_sam3d,
        showcase_animals=showcase_animals,
        report_pdf=report_pdf,
        report_size_mb=f"{report_size_mb:.1f}",
        report_pages=report_pages,
        cards=cards_html,
    )
    OUT.write_text(template, encoding="utf-8")
    size_mb = OUT.stat().st_size / (1024 * 1024)
    print(f"wrote {OUT} ({size_mb:.1f} MB)")


TEMPLATE = r"""<title>Semantic-Guided Mesh-to-Mesh Deformation</title>
<style>
:root {{
  --bg: #fbfaf7;
  --bg-panel: #f2f0ea;
  --ink: #20242c;
  --muted: #5b6270;
  --line: #dfe0d8;
  --teal: #146b63;
  --amber: #a1580f;
  --card-bg: #ffffff;
  --mono: ui-monospace, "SFMono-Regular", Menlo, Consolas, "Liberation Mono", monospace;
  --serif: "Iowan Old Style", "Palatino Linotype", Palatino, ui-serif, Georgia, "Times New Roman", serif;
}}
:root[data-theme="dark"] {{
  --bg: #14161a;
  --bg-panel: #1a1d22;
  --ink: #eae7de;
  --muted: #9aa0a8;
  --line: #2c3038;
  --teal: #59c9bb;
  --amber: #e3a45c;
  --card-bg: #1c1f25;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg: #14161a;
    --bg-panel: #1a1d22;
    --ink: #eae7de;
    --muted: #9aa0a8;
    --line: #2c3038;
    --teal: #59c9bb;
    --amber: #e3a45c;
    --card-bg: #1c1f25;
  }}
}}

* {{ box-sizing: border-box; }}
html {{ background: var(--bg); }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: var(--serif);
  line-height: 1.55;
  -webkit-font-smoothing: antialiased;
}}
a {{ color: var(--teal); }}
.wrap {{ max-width: 1180px; margin: 0 auto; padding: 0 28px; }}

.eyebrow {{
  font-family: var(--mono);
  font-size: 0.74rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--muted);
}}

header.hero {{
  border-bottom: 1px solid var(--line);
  background: var(--bg-panel);
  padding: 64px 0 48px;
}}
header.hero h1 {{
  font-family: var(--mono);
  color: var(--ink);
  font-size: clamp(2rem, 4.6vw, 3.2rem);
  line-height: 1.08;
  letter-spacing: -0.015em;
  margin: 14px 0 16px;
  text-wrap: balance;
  font-weight: 700;
  max-width: 18ch;
}}
header.hero p {{
  color: var(--muted);
  max-width: 58ch;
  font-size: 1.08rem;
  margin: 0;
  font-family: var(--serif);
}}

nav.stats {{
  border-bottom: 1px solid var(--line);
  background: var(--bg-panel);
}}
.stats-row {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(170px, 100%), 1fr));
  gap: 0;
}}
.stat {{
  padding: 26px 22px;
  border-left: 1px solid var(--line);
}}
.stat:first-child {{ border-left: none; }}
.stat .num {{
  font-family: var(--mono);
  font-size: 1.9rem;
  font-variant-numeric: tabular-nums;
  color: var(--teal);
  font-weight: 700;
  letter-spacing: -0.01em;
}}
.stat .vs {{
  font-family: var(--mono);
  font-size: 0.86rem;
  color: var(--amber);
  font-variant-numeric: tabular-nums;
  margin-top: 2px;
}}
.stat .label {{
  font-family: var(--mono);
  font-size: 0.7rem;
  letter-spacing: 0.05em;
  color: var(--muted);
  text-transform: uppercase;
  margin-top: 8px;
  line-height: 1.4;
}}

section {{ padding: 56px 0; }}
section.tight {{ padding: 40px 0; }}
h2 {{
  font-family: var(--mono);
  font-size: 1.05rem;
  letter-spacing: 0.02em;
  text-transform: uppercase;
  border-bottom: 1px solid var(--line);
  padding-bottom: 14px;
  margin: 0 0 28px;
}}
.lede {{ max-width: 68ch; color: var(--ink); font-size: 1.05rem; }}
.lede.muted {{ color: var(--muted); }}

.grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(min(300px, 100%), 1fr));
  gap: 22px;
}}
.card {{
  background: var(--card-bg);
  border: 1px solid var(--line);
  border-radius: 3px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}}
.card-media {{ position: relative; background: #0d0e10; }}
.card-video {{ width: 100%; aspect-ratio: 16/9; object-fit: cover; display: block; cursor: pointer; }}
.card-tag {{
  position: absolute;
  top: 10px;
  left: 10px;
  font-family: var(--mono);
  font-size: 0.64rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  background: rgba(20,22,25,0.72);
  color: #f2f1ec;
  padding: 4px 8px;
  border-radius: 2px;
}}
.card-body {{ padding: 16px 18px 18px; }}
.card-body h3 {{
  font-family: var(--mono);
  font-size: 0.95rem;
  margin: 0 0 8px;
  font-weight: 600;
}}
.card-body p {{ margin: 0; font-size: 0.9rem; color: var(--muted); }}
.card-metrics {{
  margin-top: 10px;
  font-family: var(--mono);
  font-size: 0.74rem;
  color: var(--teal);
  font-variant-numeric: tabular-nums;
}}

.chart-block {{ margin-bottom: 44px; }}
.chart-block img {{ width: 100%; height: auto; border: 1px solid var(--line); border-radius: 2px; }}
.chart-cap {{ font-size: 0.88rem; color: var(--muted); margin-top: 10px; max-width: 80ch; }}

.note {{
  background: var(--bg-panel);
  border-left: 3px solid var(--amber);
  padding: 18px 22px;
  font-size: 0.92rem;
  color: var(--ink);
  max-width: 80ch;
}}
.note strong {{ color: var(--amber); }}

.report-link {{
  display: inline-flex;
  align-items: center;
  gap: 14px;
  margin-top: 22px;
  padding: 16px 22px;
  border: 1px solid var(--ink);
  border-radius: 3px;
  text-decoration: none;
  color: var(--ink);
  background: var(--card-bg);
  transition: background 0.15s ease, color 0.15s ease;
}}
.report-link:hover {{ background: var(--ink); color: var(--bg); }}
.report-link:hover .report-link-meta {{ color: var(--bg); opacity: 0.7; }}
.report-link-title {{
  font-family: var(--mono);
  font-weight: 700;
  font-size: 0.95rem;
}}
.report-link-title::before {{ content: "\2193  "; }}
.report-link-meta {{
  font-family: var(--mono);
  font-size: 0.78rem;
  color: var(--muted);
}}

footer {{
  border-top: 1px solid var(--line);
  padding: 32px 0 48px;
  font-family: var(--mono);
  font-size: 0.78rem;
  color: var(--muted);
}}

@media (max-width: 680px) {{
  .stats-row {{ grid-template-columns: repeat(2, 1fr); }}
  .stat:nth-child(2n+1) {{ border-left: none; }}
}}
</style>

<header class="hero">
  <div class="wrap">
    <div class="eyebrow">Study Project &middot; University of Freiburg, LMB &middot; Summer Semester 2026</div>
    <h1>Semantic-Guided Mesh-to-Mesh Deformation</h1>
    <p>PartField-guided Chamfer optimization, scored against DenseMatcher&rsquo;s own predicted
    correspondences on identical pairs &mdash; with every confound the audit found disclosed, not hidden.</p>
  </div>
</header>

<nav class="stats">
  <div class="wrap stats-row">
    <div class="stat">
      <div class="num">21/23</div>
      <div class="label">categories beat DenseMatcher, no PF selection</div>
    </div>
    <div class="stat">
      <div class="num">10/12</div>
      <div class="label">clean-preparation categories &mdash; zero preprocessing advantage</div>
    </div>
    <div class="stat">
      <div class="num">0.806<span class="vs"> vs 0.713</span></div>
      <div class="label">mean AUC, ours vs DenseMatcher (clean-prep, zero PF)</div>
    </div>
    <div class="stat">
      <div class="num">5.52<span class="vs"> vs 9.65</span></div>
      <div class="label">mean Err, ours vs DenseMatcher (clean-prep, zero PF)</div>
    </div>
    <div class="stat">
      <div class="num">0.992</div>
      <div class="label">best F@0.02, Horse2&rarr;Giraffe4</div>
    </div>
  </div>
</nav>

<main class="wrap">
  <section class="tight">
    <p class="lede">This page collects the deformation videos behind the written report, plus the two
    figures that carry the report&rsquo;s central claim. Every clip below is a real optimization run,
    shown at full render resolution. Tags mark what each pair is doing in the argument &mdash; including
    the ones that are honestly labeled as partial failures.</p>
  </section>

  <section>
    <h2>Deformations</h2>
    <div class="grid">
{cards}
    </div>
  </section>

  <section>
    <h2>The comparison</h2>
    <p class="lede muted">Both methods scored by the identical graph-geodesic evaluator, on the
    identical directed pairs. Category labels below are colored by preprocessing provenance &mdash;
    teal categories used no ground-truth-informed alignment at all.</p>
    <div class="chart-block">
      <img src="{chart_category}" alt="This pipeline vs DenseMatcher AUC and Err per category">
    </div>
    <div class="chart-block">
      <img src="{chart_hist}" alt="AUC distribution histogram, this pipeline vs DenseMatcher">
      <p class="chart-cap">This pipeline&rsquo;s AUC is induced from the deformation by nearest-neighbor
      lookup after a Chamfer-optimized fit &mdash; close to tautological once surfaces overlap.
      DenseMatcher predicts correspondence directly, which is why its distribution is bimodal: a
      real peak of near-total failures near AUC 0.2, and a healthy peak near 0.8.</p>
    </div>
    <div class="note">
      <strong>Disclosed, not hidden:</strong> 11 of 23 categories needed assisted orientation before
      PartField ran &mdash; a canonicalization step DenseMatcher&rsquo;s released predictions never
      had access to. Only the 12 &ldquo;clean-preparation&rdquo; categories are treated as a fully
      apples-to-apples comparison in the report&rsquo;s headline claim. Broccoli and eggplant, both
      clean, genuinely lose.
    </div>
  </section>

  <section>
    <h2>Surface quality: animals vs. vehicles</h2>
    <p class="lede muted">No DenseCorr3D ground truth exists for these self-created SAM3D meshes, so
    only geometric surface metrics are reported here &mdash; never AUC/Err. The same recipe applied to
    self-created vehicle meshes is a genuine negative result.</p>
    <div class="chart-block">
      <img src="{chart_sam3d}" alt="F@0.02 across every evaluated SAM3D pair, animals vs cars">
    </div>
    <div class="chart-block">
      <img src="{showcase_animals}" alt="Animal deformation showcase, consistent PartField coloring">
      <p class="chart-cap">Four SAM3D animal pairs, source / final / target, all rendered with the
      same discrete PartField bucket coloring for a fair visual comparison.</p>
    </div>
  </section>

  <section class="tight">
    <h2>Read more</h2>
    <p class="lede">The full written report covers the pipeline architecture, the MeshUp-era
    predecessor this project moved on from, a root-cause trace for the hardest failure categories,
    and every caveat behind the numbers above in detail.</p>
    <a class="report-link" href="{report_pdf}" download="semantic_guided_mesh_deformation_report.pdf">
      <span class="report-link-title">Download the full report</span>
      <span class="report-link-meta">PDF &middot; {report_pages} pages &middot; {report_size_mb} MB</span>
    </a>
  </section>
</main>

<footer>
  <div class="wrap">Rona Latifaj &middot; University of Freiburg, LMB &middot; Study Project, Summer Semester 2026</div>
</footer>

<script>
document.querySelectorAll(".card-video").forEach(function (video) {{
  video.addEventListener("mouseenter", function () {{ video.play().catch(function () {{}}); }});
  video.addEventListener("mouseleave", function () {{ video.pause(); video.currentTime = 0; }});
  video.addEventListener("click", function () {{
    if (video.paused) {{ video.play().catch(function () {{}}); }} else {{ video.pause(); }}
  }});
}});
</script>
"""

if __name__ == "__main__":
    main()
