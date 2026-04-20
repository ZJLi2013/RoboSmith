"""Generate a self-contained HTML asset gallery and open it in the browser.

Reads assets/catalog.json + reference.png thumbnails, produces a single
HTML file with embedded base64 images. Zero dependencies beyond stdlib.

For built-in URDF primitives (box/cylinder/sphere), generates inline SVG
previews by parsing the URDF geometry and color.

Usage:
    python scripts/part1/browse_assets.py            # generate & open
    python scripts/part1/browse_assets.py --no-open  # generate only
"""

from __future__ import annotations

import base64
import json
import math
import sys
import webbrowser
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_ROOT = REPO_ROOT / "assets"
CATALOG_PATH = ASSETS_ROOT / "catalog.json"
OUTPUT_HTML = REPO_ROOT / "assets" / "gallery.html"


def img_to_data_uri(path: Path) -> str | None:
    if not path.exists():
        return None
    data = base64.b64encode(path.read_bytes()).decode()
    suffix = path.suffix.lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(
        suffix.lstrip("."), "image/png"
    )
    return f"data:{mime};base64,{data}"


def _rgba_to_css(rgba: list[float]) -> str:
    r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
    return f"rgb({r},{g},{b})"


def _rgba_lighter(rgba: list[float], factor: float = 0.3) -> str:
    r = min(1.0, rgba[0] + factor)
    g = min(1.0, rgba[1] + factor)
    b = min(1.0, rgba[2] + factor)
    return _rgba_to_css([r, g, b, 1.0])


def _rgba_darker(rgba: list[float], factor: float = 0.15) -> str:
    r = max(0.0, rgba[0] - factor)
    g = max(0.0, rgba[1] - factor)
    b = max(0.0, rgba[2] - factor)
    return _rgba_to_css([r, g, b, 1.0])


def _parse_urdf_shape(urdf_path: Path) -> dict | None:
    """Extract first visual geometry shape + color from a URDF file."""
    if not urdf_path.exists():
        return None
    try:
        tree = ET.parse(urdf_path)
    except ET.ParseError:
        return None
    root = tree.getroot()

    materials = {}
    for mat in root.iter("material"):
        name = mat.get("name", "")
        color_el = mat.find("color")
        if color_el is not None and name:
            materials[name] = [float(v) for v in color_el.get("rgba", "0.5 0.5 0.5 1").split()]

    for link in root.iter("link"):
        for vis in link.iter("visual"):
            geom = vis.find("geometry")
            if geom is None:
                continue
            color = [0.5, 0.5, 0.5, 1.0]
            mat_el = vis.find("material")
            if mat_el is not None:
                color_el = mat_el.find("color")
                if color_el is not None:
                    color = [float(v) for v in color_el.get("rgba", "0.5 0.5 0.5 1").split()]
                elif mat_el.get("name", "") in materials:
                    color = materials[mat_el.get("name")]

            box = geom.find("box")
            if box is not None:
                sizes = [float(v) for v in box.get("size", "0.1 0.1 0.1").split()]
                return {"type": "box", "size": sizes, "color": color}
            cyl = geom.find("cylinder")
            if cyl is not None:
                return {"type": "cylinder",
                        "radius": float(cyl.get("radius", "0.05")),
                        "length": float(cyl.get("length", "0.1")),
                        "color": color}
            sph = geom.find("sphere")
            if sph is not None:
                return {"type": "sphere",
                        "radius": float(sph.get("radius", "0.05")),
                        "color": color}
    return None


def _generate_svg(shape: dict, name: str) -> str:
    """Generate an inline SVG preview for a URDF primitive."""
    c = shape["color"]
    fill = _rgba_to_css(c)
    hi = _rgba_lighter(c, 0.25)
    lo = _rgba_darker(c, 0.2)
    uid = name.replace(" ", "_")

    W, H = 200, 200
    cx, cy = W // 2, H // 2

    if shape["type"] == "box":
        sx, sy, sz = shape["size"]
        max_dim = max(sx, sy, sz)
        scale = 70 / max_dim if max_dim > 0 else 70
        w = sx * scale
        h = sz * scale
        d = sy * scale * 0.4
        # isometric box: top face, front face, side face
        top = f"{cx - w/2},{cy - h/2 - d} {cx + w/2},{cy - h/2 - d} {cx + w/2 + d},{cy - h/2 - d + d} {cx + d - w/2},{cy - h/2 - d + d}"
        front = f"{cx - w/2},{cy - h/2} {cx + w/2},{cy - h/2} {cx + w/2},{cy + h/2} {cx - w/2},{cy + h/2}"
        side = f"{cx + w/2},{cy - h/2} {cx + w/2 + d},{cy - h/2 - d + d} {cx + w/2 + d},{cy + h/2 - d + d} {cx + w/2},{cy + h/2}"
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">'
            f'<defs><linearGradient id="g_{uid}" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="0%" stop-color="{hi}"/><stop offset="100%" stop-color="{fill}"/>'
            f'</linearGradient></defs>'
            f'<polygon points="{front}" fill="url(#g_{uid})" stroke="{lo}" stroke-width="1.5"/>'
            f'<polygon points="{top}" fill="{hi}" stroke="{lo}" stroke-width="1"/>'
            f'<polygon points="{side}" fill="{lo}" stroke="{lo}" stroke-width="1"/>'
            f'</svg>'
        )

    elif shape["type"] == "cylinder":
        r = shape["radius"]
        length = shape["length"]
        ratio = length / (2 * r) if r > 0 else 2
        body_h = min(120, max(30, 60 * ratio))
        ell_rx = min(60, max(20, 60 / max(ratio, 0.5)))
        ell_ry = max(8, ell_rx * 0.25)
        top_cy = cy - body_h / 2
        bot_cy = cy + body_h / 2
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">'
            f'<defs><linearGradient id="g_{uid}" x1="0" y1="0" x2="1" y2="0">'
            f'<stop offset="0%" stop-color="{lo}"/>'
            f'<stop offset="45%" stop-color="{hi}"/>'
            f'<stop offset="100%" stop-color="{lo}"/>'
            f'</linearGradient></defs>'
            f'<rect x="{cx - ell_rx}" y="{top_cy}" width="{2*ell_rx}" height="{body_h}" '
            f'fill="url(#g_{uid})" stroke="{lo}" stroke-width="1"/>'
            f'<ellipse cx="{cx}" cy="{bot_cy}" rx="{ell_rx}" ry="{ell_ry}" '
            f'fill="{lo}" stroke="{lo}" stroke-width="1"/>'
            f'<ellipse cx="{cx}" cy="{top_cy}" rx="{ell_rx}" ry="{ell_ry}" '
            f'fill="{hi}" stroke="{lo}" stroke-width="1"/>'
            f'</svg>'
        )

    elif shape["type"] == "sphere":
        r = 55
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">'
            f'<defs><radialGradient id="g_{uid}" cx="40%" cy="35%" r="60%">'
            f'<stop offset="0%" stop-color="{hi}"/>'
            f'<stop offset="100%" stop-color="{lo}"/>'
            f'</radialGradient></defs>'
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="url(#g_{uid})" '
            f'stroke="{lo}" stroke-width="1.5"/>'
            f'</svg>'
        )

    return ""


def build_html(entries: list[dict]) -> str:
    cards_html = []
    for e in entries:
        asset_dir = ASSETS_ROOT / e["dir"]
        ref_img = asset_dir / "reference.png"
        data_uri = img_to_data_uri(ref_img)

        tags = ", ".join(e.get("tags", []))
        source = e.get("source", "unknown")
        badge_cls = "badge-gen" if source == "generated" else "badge-builtin"
        badge_text = "Generated" if source == "generated" else "Built-in"
        mass = e.get("mass_kg", "?")
        size = e.get("size_cm", [])
        size_str = " × ".join(str(s) for s in size) + " cm" if size else "—"
        desc = e.get("description", "")

        if data_uri:
            img_tag = f'<img src="{data_uri}" alt="{e["name"]}">'
        else:
            urdf_path = asset_dir / "model.urdf"
            shape = _parse_urdf_shape(urdf_path)
            if shape:
                img_tag = _generate_svg(shape, e["name"])
            else:
                img_tag = f'<div class="no-img">{e["name"][0].upper()}</div>'

        cards_html.append(f"""
      <div class="card">
        <div class="card-img">{img_tag}</div>
        <div class="card-body">
          <div class="card-title">{e["name"]}</div>
          <span class="badge {badge_cls}">{badge_text}</span>
          <div class="card-meta">
            <div><strong>Tags:</strong> {tags}</div>
            <div><strong>Mass:</strong> {mass} kg</div>
            <div><strong>Size:</strong> {size_str}</div>
          </div>
          {"<div class='card-desc'>" + desc + "</div>" if desc else ""}
        </div>
      </div>""")

    n_total = len(entries)
    n_gen = sum(1 for e in entries if e.get("source") == "generated")
    n_builtin = n_total - n_gen

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RoboSmith Asset Gallery</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0f1117;
    color: #e0e0e0;
    min-height: 100vh;
  }}
  .header {{
    background: linear-gradient(135deg, #1a1d2e 0%, #2d1f3d 100%);
    padding: 2rem 2rem 1.5rem;
    border-bottom: 1px solid #2a2d3e;
  }}
  .header h1 {{
    font-size: 1.8rem;
    font-weight: 700;
    color: #fff;
    margin-bottom: 0.5rem;
  }}
  .header .stats {{
    display: flex;
    gap: 1.5rem;
    font-size: 0.9rem;
    color: #9ca3af;
  }}
  .header .stats span {{ color: #60a5fa; font-weight: 600; }}
  .filters {{
    padding: 1rem 2rem;
    display: flex;
    gap: 0.5rem;
    border-bottom: 1px solid #1e2030;
  }}
  .filters button {{
    padding: 0.4rem 1rem;
    border-radius: 20px;
    border: 1px solid #3a3d50;
    background: transparent;
    color: #9ca3af;
    cursor: pointer;
    font-size: 0.85rem;
    transition: all 0.2s;
  }}
  .filters button:hover {{ border-color: #60a5fa; color: #60a5fa; }}
  .filters button.active {{
    background: #60a5fa;
    color: #fff;
    border-color: #60a5fa;
  }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 1.2rem;
    padding: 1.5rem 2rem;
  }}
  .card {{
    background: #1a1d2e;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #2a2d3e;
    transition: transform 0.2s, box-shadow 0.2s;
  }}
  .card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.4);
    border-color: #3a3d50;
  }}
  .card-img {{
    height: 200px;
    background: #f5f5f5;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
  }}
  .card-img img {{
    width: 100%;
    height: 100%;
    object-fit: contain;
  }}
  .no-img {{
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 3rem;
    font-weight: 700;
    color: #6b7280;
    background: linear-gradient(135deg, #e5e7eb, #d1d5db);
  }}
  .card-body {{ padding: 1rem; }}
  .card-title {{
    font-size: 1rem;
    font-weight: 600;
    color: #fff;
    margin-bottom: 0.4rem;
    word-break: break-word;
  }}
  .badge {{
    display: inline-block;
    padding: 0.15rem 0.6rem;
    border-radius: 10px;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.6rem;
  }}
  .badge-gen {{ background: #1e3a5f; color: #60a5fa; }}
  .badge-builtin {{ background: #1a3329; color: #34d399; }}
  .card-meta {{
    font-size: 0.8rem;
    color: #9ca3af;
    line-height: 1.6;
  }}
  .card-meta strong {{ color: #d1d5db; }}
  .card-desc {{
    margin-top: 0.5rem;
    font-size: 0.78rem;
    color: #6b7280;
    font-style: italic;
  }}
  .card[data-source="generated"] {{ }}
  .card[data-source="builtin"] {{ }}
</style>
</head>
<body>
  <div class="header">
    <h1>RoboSmith Asset Gallery</h1>
    <div class="stats">
      <div>Total: <span>{n_total}</span></div>
      <div>Built-in: <span>{n_builtin}</span></div>
      <div>Generated: <span>{n_gen}</span></div>
    </div>
  </div>
  <div class="filters">
    <button class="active" onclick="filterAssets('all', this)">All</button>
    <button onclick="filterAssets('builtin', this)">Built-in</button>
    <button onclick="filterAssets('generated', this)">Generated</button>
  </div>
  <div class="grid" id="gallery">
    {"".join(cards_html)}
  </div>
  <script>
    document.querySelectorAll('.card').forEach(card => {{
      const badge = card.querySelector('.badge');
      const src = badge && badge.classList.contains('badge-gen') ? 'generated' : 'builtin';
      card.dataset.source = src;
    }});
    function filterAssets(type, btn) {{
      document.querySelectorAll('.filters button').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      document.querySelectorAll('.card').forEach(card => {{
        if (type === 'all' || card.dataset.source === type)
          card.style.display = '';
        else
          card.style.display = 'none';
      }});
    }}
  </script>
</body>
</html>"""


def main():
    if not CATALOG_PATH.exists():
        print(f"catalog.json not found at {CATALOG_PATH}")
        print("Run: python -c \"from robotsmith.assets.library import AssetLibrary; AssetLibrary('./assets').save_catalog()\"")
        sys.exit(1)

    entries = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    html = build_html(entries)
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"Gallery generated: {OUTPUT_HTML}")
    print(f"  {len(entries)} assets ({sum(1 for e in entries if e.get('source') == 'generated')} generated)")

    if "--no-open" not in sys.argv:
        webbrowser.open(OUTPUT_HTML.as_uri())
        print("  Opened in browser.")


if __name__ == "__main__":
    main()
