"""Assemble a minimal, self-contained showcase: main numbers + a video grid only.

A trimmed sister page to build_site.py's full showcase -- no hero image, no
comparison charts, no note boxes, no report download. Just the headline
stats and a grid of hover-to-play deformation clips with a plain caption
underneath each one (no numbering).
"""

from __future__ import annotations

import base64
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VIDEOS = ROOT / "report_final" / "site_src" / "videos"
STILLS = ROOT / "report_final" / "site_src" / "posters"
OUT = ROOT / "report_final" / "site_src" / "index_simple.html"


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def data_uri(path: Path, mime: str) -> str:
    return f"data:{mime};base64,{b64(path)}"


# (video basename without .mp4, still basename without .png, title, metrics-or-None)
PAIRS: list[tuple[str, str, str, str | None]] = [
    ("01_horse2_giraffe4_deformation_slow_partfield", "01_horse2_giraffe4_deformation_slow_partfield", "Horse → Giraffe", "Chamfer 0.0134 · F@0.02 0.992"),
    ("02_panda3_bear_deformation_slow_partfield", "02_panda3_bear_deformation_slow_partfield", "Panda → Bear", "Chamfer 0.0199 · F@0.02 0.973"),
    ("03_cat_dachshund_deformation_slow_partfield", "03_cat_dachshund_deformation_slow_partfield", "Cat → Dachshund", None),
    ("04_elephant_moose_deformation_slow_partfield", "04_elephant_moose_deformation_slow_partfield", "Elephant → Moose", None),
    ("05_goldie_pug_deformation_slow_partfield", "05_goldie_pug_deformation_slow_partfield", "Goldie → Pug", "Chamfer 0.0248 · F@0.02 0.906"),
    ("06_fox_cat2_deformation_slow_partfield", "06_fox_cat2_deformation_slow_partfield", "Fox → Cat", "Chamfer 0.0224 · F@0.02 0.946"),
    ("07_elephant_giraffe_deformation_slow_partfield", "07_elephant_giraffe_deformation_slow_partfield", "Elephant → Giraffe", "Chamfer 0.0279 · F@0.02 0.857"),
    ("08_mushs0t4_mushroom_pf01_4000_deformation_slow_partfield", "08_mushs0t4_mushroom_pf01_4000_deformation_slow_partfield", "Mushroom to Mushroom (DenseCorr3D)", None),
    ("09_chrs2t0_chairsfix_pf07_4000_deformation_slow_partfield", "09_chrs2t0_chairsfix_pf07_4000_deformation_slow_partfield", "Chair to Chair (DenseCorr3D)", None),
    ("10_bpks2t3_backpackfix_pf07_4000_deformation_slow_partfield", "10_bpks2t3_backpackfix_pf07_4000_deformation_slow_partfield", "Backpack to Backpack (DenseCorr3D)", None),
    ("11_knfs4t2_knifeonly_pf05_4000_deformation_slow_partfield", "11_knfs4t2_knifeonly_pf05_4000_deformation_slow_partfield", "Knife to Knife (DenseCorr3D)", None),
    ("12_shps1t0_shampoo_pf03_4000_deformation_slow_partfield", "12_shps1t0_shampoo_pf03_4000_deformation_slow_partfield", "Shampoo to Shampoo (DenseCorr3D)", None),
]


def build_cards() -> str:
    cards = []
    for video_base, still_base, title, metrics in PAIRS:
        video_path = VIDEOS / f"{video_base}.mp4"
        still_path = STILLS / f"{still_base if still_base.endswith('.png') else video_base}.png"
        if not still_path.exists():
            still_path = STILLS / f"{still_base}.png"
        video_data = data_uri(video_path, "video/mp4")
        poster_data = data_uri(still_path, "image/png")
        metrics_html = f'<div class="card-metrics">{metrics}</div>' if metrics else ""
        cards.append(
            f"""
      <article class="card">
        <div class="card-media">
          <video class="card-video" muted loop playsinline preload="none"
                 poster="{poster_data}" src="{video_data}"></video>
        </div>
        <div class="card-body">
          <h3>{title}</h3>
          {metrics_html}
        </div>
      </article>"""
        )
    return "\n".join(cards)


def main() -> None:
    cards_html = build_cards()
    template = TEMPLATE.format(cards=cards_html)
    OUT.write_text(template, encoding="utf-8")
    size_mb = OUT.stat().st_size / (1024 * 1024)
    print(f"wrote {OUT} ({size_mb:.1f} MB)")


TEMPLATE = r"""<title>Semantic-Guided Mesh-to-Mesh Deformation Showcase</title>
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
.wrap {{ max-width: 1180px; margin: 0 auto; padding: 0 28px; }}

header.top {{
  border-bottom: 1px solid var(--line);
  background: var(--bg-panel);
  padding: 40px 0 32px;
}}
header.top .eyebrow {{
  font-family: var(--mono);
  font-size: 0.74rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--muted);
}}
header.top h1 {{
  font-family: var(--mono);
  color: var(--ink);
  font-size: clamp(1.7rem, 3.6vw, 2.5rem);
  line-height: 1.1;
  letter-spacing: -0.015em;
  margin: 12px 0 0;
  text-wrap: balance;
  font-weight: 700;
}}

nav.stats {{
  border-bottom: 1px solid var(--line);
  background: var(--card-bg);
}}
.stats-row {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(170px, 100%), 1fr));
  gap: 4px;
}}
.stat {{ padding: 24px 28px; border-left: 1px solid var(--line); }}
.stat:first-child {{ border-left: none; }}
.stat .num {{
  font-family: var(--mono);
  font-size: 1.7rem;
  font-variant-numeric: tabular-nums;
  color: var(--teal);
  font-weight: 700;
  letter-spacing: -0.01em;
}}
.stat .vs {{
  font-family: var(--mono);
  font-size: 0.82rem;
  color: var(--amber);
  font-variant-numeric: tabular-nums;
  margin-top: 2px;
}}
.stat .label {{
  font-family: var(--mono);
  font-size: 0.68rem;
  letter-spacing: 0.05em;
  color: var(--muted);
  text-transform: uppercase;
  margin-top: 7px;
  line-height: 1.4;
}}

main.wrap {{ padding-top: 56px; padding-bottom: 56px; }}
.grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(min(280px, 100%), 1fr));
  gap: 28px;
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
.card-body {{ padding: 14px 16px 16px; }}
.card-body h3 {{
  font-family: var(--mono);
  font-size: 0.92rem;
  margin: 0;
  font-weight: 600;
}}
.card-metrics {{
  margin-top: 8px;
  font-family: var(--mono);
  font-size: 0.72rem;
  color: var(--teal);
  font-variant-numeric: tabular-nums;
}}

footer {{
  border-top: 1px solid var(--line);
  padding: 26px 0 40px;
  font-family: var(--mono);
  font-size: 0.76rem;
  color: var(--muted);
}}

@media (max-width: 680px) {{
  .stats-row {{ grid-template-columns: repeat(2, 1fr); }}
  .stat:nth-child(2n+1) {{ border-left: none; }}
}}
</style>

<header class="top">
  <div class="wrap">
    <div class="eyebrow">Study Project &middot; University of Freiburg, LMB</div>
    <h1>Semantic-Guided Mesh-to-Mesh Deformation Showcase</h1>
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
      <div class="label">clean-preparation categories</div>
    </div>
    <div class="stat">
      <div class="num">0.806<span class="vs"> vs 0.713</span></div>
      <div class="label">mean AUC, ours vs DenseMatcher (clean-prep)</div>
    </div>
    <div class="stat">
      <div class="num">5.52<span class="vs"> vs 9.65</span></div>
      <div class="label">mean Err, ours vs DenseMatcher (clean-prep)</div>
    </div>
    <div class="stat">
      <div class="num">0.992</div>
      <div class="label">best F@0.02, Horse2&rarr;Giraffe4</div>
    </div>
  </div>
</nav>

<main class="wrap">
  <div class="grid">
{cards}
  </div>
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
