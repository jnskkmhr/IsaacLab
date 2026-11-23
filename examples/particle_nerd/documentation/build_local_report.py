# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render local_methodology.html to a checked, paginated methodology PDF.

Run with uv run --no-project --with playwright --with pymupdf python
examples/particle_nerd/documentation/build_local_report.py.
Pass --source, --name and --title to build another HTML document of this folder.
Requires Chrome/Chromium and network access for MathJax.
"""

import argparse
import asyncio
import shutil
from pathlib import Path

import pymupdf
from playwright.async_api import async_playwright


async def build(
    output: Path,
    browser_path: str | None,
    source_name: str = "local_methodology.html",
    name: str = "particle_nerd_local_methodology",
    title: str = "Particle NeRD Local | Methodology",
) -> None:
    """Typeset equations, reject overflowing content, and export A4 pages."""
    source = Path(__file__).with_name(source_name)
    output.mkdir(parents=True, exist_ok=True)
    html_path = output / f"{name}.html"
    pdf_path = output / f"{name}.pdf"
    raw_path = output / f"{name}_body.pdf"
    html_path.write_text(source.read_text())
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(executable_path=browser_path, args=["--no-sandbox"])
        page = await browser.new_page(viewport={"width": 900, "height": 1200})
        await page.goto(html_path.as_uri(), wait_until="networkidle")
        await page.wait_for_function("window.MathJax && MathJax.startup && MathJax.startup.promise")
        await page.evaluate("async () => { await MathJax.startup.promise; await document.fonts.ready; }")
        await page.emulate_media(media="print")
        # Fix the browser viewport to the printable width before checking bounds.
        await page.add_style_tag(content="body { width: 186mm; }")
        errors = await page.locator("[data-mml-node='merror']").all_text_contents()
        if errors or await page.locator("mjx-container").count() == 0:
            raise RuntimeError(f"Missing or invalid mathematics: {errors}")
        overflow = await page.evaluate("""() => {
            const bad = [];
            for (const el of document.querySelectorAll('td, th, .eq-body')) {
                const r = el.getBoundingClientRect();
                for (const math of el.querySelectorAll('mjx-container > svg')) {
                    const m = math.getBoundingClientRect();
                    if (m.right > r.right + 1 || m.left < r.left - 1)
                        bad.push({text: el.textContent.trim(), width: m.width, available: r.width});
                }
            }
            return bad;
        }""")
        if overflow:
            raise RuntimeError(f"Mathematics exceeds its table cell or equation column: {overflow}")
        await page.pdf(path=str(raw_path), prefer_css_page_size=True, print_background=True)
        await browser.close()
    with pymupdf.open(raw_path) as document:
        for number, page in enumerate(document, start=1):
            page.wrap_contents()
            page.insert_text((34, 24), title, fontsize=8, color=(0.35, 0.35, 0.35))
            page.insert_textbox(
                pymupdf.Rect(0, page.rect.height - 30, page.rect.width, page.rect.height - 12),
                str(number),
                fontsize=9,
                align=1,
            )
        document.set_metadata({"title": title.replace(" | ", " ")})
        document.save(pdf_path, garbage=4, deflate=True)
        print(f"Wrote {pdf_path} ({len(document)} pages)")
    raw_path.unlink()


def main() -> None:
    """Parse output directory, browser, and document options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[3] / "docs/_build")
    parser.add_argument("--browser", default=shutil.which("google-chrome") or shutil.which("chromium"))
    parser.add_argument("--source", default="local_methodology.html", help="HTML file name in this folder")
    parser.add_argument("--name", default="particle_nerd_local_methodology", help="Base name of the output files")
    parser.add_argument("--title", default="Particle NeRD Local | Methodology", help="Page header text")
    args = parser.parse_args()
    asyncio.run(build(args.output.resolve(), args.browser, args.source, args.name, args.title))


if __name__ == "__main__":
    main()
