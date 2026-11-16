# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Draw Module 3's particle–slice attention and motion-head data flow."""

from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring

BLUE = "#224c70"
GREEN = "#348242"
LIGHT = "#a8cda4"
svg = Element(
    "svg",
    {
        "xmlns": "http://www.w3.org/2000/svg",
        "viewBox": "0 0 1020 660",
        "class": "attention-diagram",
        "font-family": "Arial, sans-serif",
    },
)


def add(tag, **attrs):
    """Append an SVG element."""
    return SubElement(svg, tag, {k.replace("_", "-"): str(v) for k, v in attrs.items()})


def label(x, y, text, size=19, bold=False):
    """Draw a centered label."""
    element = add(
        "text", x=x, y=y, text_anchor="middle", font_size=size, font_weight="bold" if bold else "normal", fill=BLUE
    )
    element.text = text


def line(x, y, xx, yy, arrow=True, color=BLUE):
    """Draw a connection."""
    attrs = {"x1": x, "y1": y, "x2": xx, "y2": yy, "stroke": color, "stroke_width": 1.7}
    if arrow:
        attrs["marker_end"] = "url(#module3-arrow)"
    add("line", **attrs)


def node(x, y, particle=True):
    """Distinguish particle features from latent slice features."""
    if particle:
        add("circle", cx=x, cy=y, r=6, fill=GREEN)
    else:
        add("rect", x=x - 9, y=y - 9, width=18, height=18, fill="#e1eadd", stroke=BLUE, stroke_width=1.5)


def main():
    """Write a vector schematic with one representative attention head."""
    defs = SubElement(svg, "defs")
    marker = SubElement(
        defs,
        "marker",
        {
            "id": "module3-arrow",
            "viewBox": "0 0 10 10",
            "refX": "8",
            "refY": "5",
            "markerWidth": "5",
            "markerHeight": "5",
            "orient": "auto",
        },
    )
    SubElement(marker, "path", {"d": "M0 0L10 5L0 10Z", "fill": BLUE})
    add(
        "rect",
        x=346,
        y=43,
        width=656,
        height=529,
        rx=10,
        fill="none",
        stroke="#75976e",
        stroke_width=1.5,
        stroke_dasharray="7 4",
    )
    label(674, 25, "× L", 21, True)
    for x, title in [
        (170, "(a) Encoding"),
        (510, "(b) Slicing"),
        (844, "(c) Slice attention"),
    ]:
        label(x, 76, title, 19, True)
    label(170, 137, "Local features", 18)
    add("rect", x=79, y=165, width=182, height=47, rx=4, fill="#f3f7f0", stroke=GREEN)
    label(170, 195, "Encoder", 19)
    for k in range(6):
        node(95 + 30 * k, 246)
    label(170, 281, "N particle tokens · Eq. (12)", 18)
    line(285, 195, 366, 195)
    # N-to-S soft aggregation; spatial slice anchors are separate from features.
    for j in range(6):
        for k in range(3):
            line(407, 118 + j * 23, 600, 142 + k * 45, False, LIGHT)
    for j in range(6):
        node(407, 118 + j * 23)
    for k in range(3):
        node(600, 142 + k * 45, False)
    label(510, 281, "N → S", 18)
    label(510, 307, "A · Eqs. (13)–(14)", 17)
    line(627, 195, 713, 195)
    positions = [(790, 155), (902, 155), (846, 231)]
    for i, a in enumerate(positions):
        for b in positions[i + 1 :]:
            line(*a, *b, False, BLUE)
    for p in positions:
        node(*p, False)
    label(844, 110, "RoPE", 17)
    label(844, 281, "Eqs. (15)–(16)", 17)
    line(844, 324, 844, 354)
    # Lower row runs right to left back toward the motion head.
    for x, title in [
        (170, "(f) Motion prediction"),
        (510, "(e) Hidden-state update"),
        (844, "(d) Deslicing"),
    ]:
        label(x, 384, title, 18, True)
    for k in range(3):
        for j in range(6):
            line(755, 424 + k * 42, 948, 413 + j * 21, False, LIGHT)
    for k in range(3):
        node(755, 424 + k * 42, False)
    for j in range(6):
        node(948, 413 + j * 21)
    label(844, 547, "A + B · Eq. (17)", 17)
    line(716, 470, 649, 470)
    add("rect", x=371, y=412, width=278, height=119, rx=4, fill="#f3f7f0", stroke=GREEN)
    label(510, 440, "Projection", 18)
    label(510, 471, "+ Residual", 18)
    label(510, 502, "FFN + residual", 16)
    label(510, 550, "Eq. (17)", 17)
    # Repeat the block by returning the updated hidden state to the next slicing stage.
    add(
        "path",
        d="M510 412 V342 H380 V252 H397",
        fill="none",
        stroke=BLUE,
        stroke_width=1.6,
        stroke_dasharray="5 3",
        marker_end="url(#module3-arrow)",
    )
    line(371, 470, 288, 470)
    add("rect", x=54, y=414, width=234, height=112, rx=4, fill="#f3f7f0", stroke=GREEN)
    label(171, 454, "Motion head · Eq. (18)", 19)
    label(171, 504, "δx, δv, gates", 17)
    label(170, 550, "x, v · Eqs. (19)–(20)", 17)
    # Direct local skip bypasses the attention blocks and enters the head.
    add(
        "path",
        d="M54 137 H20 V605 H304 V498 H289",
        fill="none",
        stroke=GREEN,
        stroke_width=1.8,
        marker_end="url(#module3-arrow)",
    )
    label(165, 631, "Local skip", 16)
    target = Path(__file__).with_name("figures") / "local_attention.svg"
    target.write_text(tostring(svg, encoding="unicode"))
    print(target)


if __name__ == "__main__":
    main()
