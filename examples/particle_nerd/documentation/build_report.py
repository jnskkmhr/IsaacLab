# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export the methodology RST using the cable scaling report's A4 technical-report style.

Run with uv run --no-project --with docutils --with beautifulsoup4 --with playwright
--with pymupdf python examples/particle_nerd/documentation/build_report.py.
Requires Chrome/Chromium and network access for MathJax during rendering.
"""

import argparse
import asyncio
import base64
import shutil
from datetime import date
from pathlib import Path

import pymupdf
from bs4 import BeautifulSoup
from docutils.core import publish_parts
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[3]
STYLE = """
@page { size: A4; margin: 25mm 22mm 21mm; }
* { box-sizing: border-box; }
body { margin: 0; color: #191919; font-family: 'Nimbus Sans', Arial, sans-serif;
       font-size: 11pt; line-height: 1.22; }
.report-title { padding-top: 14mm; margin-bottom: 6mm; border-bottom: .5pt solid #ccc;
                padding-bottom: 6mm; }
.metadata { color: #555; font-size: 9pt; margin: 0 0 2mm; }
h1 { font-size: 25pt; line-height: 1.08; margin: 0 0 1mm; font-weight: 700; }
.subtitle { font-size: 14pt; color: #555; margin: 0; }
h2 { color: #76b900; font-size: 12pt; margin: 6mm 0 4mm; padding-bottom: 2.5mm;
     border-bottom: .5pt solid #ccc; break-after: avoid; }
h3 { font-size: 11pt; margin: 5mm 0 2mm; break-after: avoid; }
p { margin: 0 0 2.2mm; orphans: 3; widows: 3; }
.summary-box { background: #f2f2f2; border-left: 3pt solid #76b900; padding: 3mm 4mm;
               margin: 0 0 5mm; break-inside: avoid; }
.summary-box p:last-child { margin-bottom: 0; }
.admonition { color: #555; font-size: 9pt; margin: 4mm 0; }
.admonition-title { display: none; }
figure { margin: 4mm 0; break-inside: avoid; }
figure img { display: block; width: 100%; height: auto; }
figcaption { font-size: 9pt; line-height: 1.2; color: #555; margin-top: 2mm; }
table { border-collapse: collapse; width: 100%; font-size: 9pt; margin: 4mm 0;
        table-layout: fixed; break-inside: avoid; }
caption { text-align: center; color: #555; font-weight: normal; margin-bottom: 2mm; }
th { border-top: .7pt solid #222; border-bottom: .5pt solid #222; text-align: left; }
td, th { padding: 1.4mm 2mm; vertical-align: top; overflow-wrap: anywhere; }
tbody tr:last-child td { border-bottom: .7pt solid #222; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
td p, th p { margin: 0; }
code, pre { font-family: 'Nimbus Mono PS', monospace; font-size: 9pt; }
code { overflow-wrap: anywhere; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; margin: 3mm 0;
      padding: 2mm 3mm; background: #f5f5f5; break-inside: avoid; }
a { color: inherit; text-decoration: none; overflow-wrap: anywhere; }
ol, ul { padding-left: 6mm; margin: 2mm 0 4mm; }
li { margin-bottom: 1.5mm; }
p:has(+ div.math) { break-after: avoid; }
.math { break-inside: avoid; }
div.math { margin: 3mm 0; text-align: center; font-size: 10pt; }
mjx-container[display='true'] { margin: 0 !important; }
#references { break-inside: avoid; }
.citation-list { font-size: 9pt; }
.citation { margin-bottom: 3mm; }
.citation .label { font-weight: bold; }
@media screen { body { width: 166mm; margin: 15mm auto; } }
"""


def report_html(source: Path, report_date: date) -> str:
    """Convert the canonical RST and apply presentation-only report structure."""
    parts = publish_parts(
        source.read_text(),
        writer_name="html5",
        settings_overrides={
            "report_level": 2,
            "halt_level": 2,
            "math_output": "MathJax https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js",
            "stylesheet_path": [],
            "embed_stylesheet": False,
        },
    )
    soup = BeautifulSoup(parts["body"], "html.parser")
    abstract = soup.find("section", id="abstract")
    if abstract is None:
        raise ValueError("Expected an Abstract section in the canonical methodology source")
    heading = abstract.find(["h1", "h2"])
    heading.name = "h2"
    heading.string = "Executive summary"
    summary = soup.new_tag("div", attrs={"class": "summary-box"})
    for paragraph in list(abstract.find_all("p", recursive=False)):
        summary.append(paragraph.extract())
    heading.insert_after(summary)
    section_number = 0
    for section in soup.find_all("section", recursive=False):
        if section is abstract:
            continue
        section_number += 1
        heading = section.find(["h1", "h2"], recursive=False)
        heading.name = "h2"
        heading.insert(0, f"{section_number}  ")
        for sub_number, subsection in enumerate(section.find_all("section", recursive=False), start=1):
            subheading = subsection.find(["h2", "h3"], recursive=False)
            subheading.name = "h3"
            subheading.insert(0, f"{section_number}.{sub_number}  ")
    for picture in soup.find_all("img"):
        path = (source.parent / picture["src"]).resolve()
        picture["src"] = "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()
    title = f"""<div class="report-title">
<p class="metadata">{report_date.strftime("%B %d, %Y")} &nbsp; | &nbsp;
Source <code>d5cc675f5</code> &nbsp; | &nbsp; Methodology</p>
<h1>Particle NeRD Methodology</h1>
<p class="subtitle">Learned particle dynamics with analytical rigid-body coupling</p></div>"""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Particle NeRD Methodology — Isaac Lab</title><style>{STYLE}</style>
<script>window.MathJax = {{tex: {{inlineMath: [['\\\\(', '\\\\)']],
    displayMath: [['\\\\[', '\\\\]']] }}, svg: {{fontCache: 'local'}}}};</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js"></script>
</head><body>{title}{soup}</body></html>"""


def add_page_furniture(raw: Path, output: Path) -> int:
    """Add the reference report's first-page banner, running headers, and page numbers."""
    document = pymupdf.open(raw)
    vectors = {}
    for name in ("pipeline", "contact_geometry", "network_layers", "particle_selection", "dataset"):
        preview = pymupdf.Pixmap(str(ROOT / f"examples/particle_nerd/documentation/figures/{name}.jpg"))
        vector_path = ROOT / f"docs/_build/particle_nerd_figures/{name}.pdf"
        if not vector_path.exists():
            raise FileNotFoundError(f"Regenerate the methodology figures first: missing {vector_path}")
        vectors[(preview.width, preview.height)] = pymupdf.open(vector_path)
    replaced_figures = 0
    green = (118 / 255, 185 / 255, 0)
    grey = (0.33, 0.33, 0.33)
    for number, page in enumerate(document, start=1):
        # Isolate browser clipping/transforms before adding independent PDF overlays.
        page.wrap_contents()
        for image in page.get_images():
            vector = vectors.get((image[2], image[3]))
            if vector is not None:
                rectangles = page.get_image_rects(image[0])
                page.delete_image(image[0])
                for rectangle in rectangles:
                    page.show_pdf_page(rectangle, vector, 0)
                    replaced_figures += 1
        page.wrap_contents()
        width, height = page.rect.width, page.rect.height
        left, right = 22 * 72 / 25.4, width - 22 * 72 / 25.4
        if number == 1:
            page.draw_rect(pymupdf.Rect(0, 0, width, 40), color=None, fill=green)
            page.insert_text((left + 4, 25), "NVIDIA | ISAAC LAB", fontname="hebo", fontsize=14, color=(1, 1, 1))
            page.insert_textbox(
                pymupdf.Rect(right - 150, 13, right, 32),
                "Internal technical report",
                fontname="helv",
                fontsize=10,
                color=(1, 1, 1),
                align=2,
            )
        else:
            page.insert_text((left, 39), "NVIDIA | Isaac Lab Methodology Report", fontsize=9, color=grey)
            page.insert_textbox(
                pymupdf.Rect(right - 140, 29, right, 45), "Particle NeRD", fontsize=9, color=grey, align=2
            )
        page.insert_textbox(
            pymupdf.Rect(0, height - 28, width, height - 12), str(number), fontsize=9, color=grey, align=1
        )
    if replaced_figures != len(vectors):
        raise RuntimeError("Expected all methodology figures in the report")
    document.set_metadata({"title": "Particle NeRD Methodology", "subject": "Isaac Lab internal technical report"})
    document.save(output, garbage=4, deflate=True)
    count = len(document)
    document.close()
    for vector in vectors.values():
        vector.close()
    return count


async def build(output: Path, report_date: date, browser_path: str | None) -> None:
    """Render checked mathematical content, then assemble the final report PDF."""
    output.mkdir(parents=True, exist_ok=True)
    html_path = output / "particle_nerd_methodology.html"
    pdf_path = output / "particle_nerd_methodology.pdf"
    raw_path = output / "particle_nerd_methodology_body.pdf"
    html_path.write_text(report_html(ROOT / "examples/particle_nerd/documentation/methodology.rst", report_date))
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(executable_path=browser_path, args=["--no-sandbox"])
        page = await browser.new_page(viewport={"width": 1000, "height": 1200})
        await page.goto(html_path.as_uri(), wait_until="networkidle")
        await page.wait_for_function("window.MathJax && MathJax.startup && MathJax.startup.promise", timeout=60000)
        await page.evaluate("async () => { await MathJax.startup.promise; await document.fonts.ready; }")
        if await page.locator("mjx-container").count() == 0:
            raise RuntimeError("No rendered equations found")
        errors = await page.locator("[data-mml-node='merror']").all_text_contents()
        if errors:
            raise RuntimeError(f"MathJax errors: {errors}")
        await page.emulate_media(media="print")
        await page.pdf(path=str(raw_path), prefer_css_page_size=True, print_background=True)
        await browser.close()
    pages = add_page_furniture(raw_path, pdf_path)
    raw_path.unlink()
    print(f"Wrote {pdf_path} ({pages} A4 pages)")
    print(f"HTML source: {html_path}")


def main() -> None:
    """Parse report output options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/_build")
    parser.add_argument("--date", type=date.fromisoformat, default=date.today())
    parser.add_argument("--browser", default=shutil.which("google-chrome") or shutil.which("chromium"))
    args = parser.parse_args()
    asyncio.run(build(args.output.resolve(), args.date, args.browser))


if __name__ == "__main__":
    main()
