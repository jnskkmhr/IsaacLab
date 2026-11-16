# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Draw an editable schematic of local continuous convolution and packing features."""

import math
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring

BLUE = "#224c70"
GREEN = "#348242"
LIGHT = "#c4ddbf"
svg = Element(
    "svg",
    {
        "xmlns": "http://www.w3.org/2000/svg",
        "viewBox": "0 0 1000 560",
        "class": "aggregation-diagram",
        "font-family": "Arial, sans-serif",
    },
)


def add(tag, **attrs):
    """Append an SVG element."""
    return SubElement(svg, tag, {k.replace("_", "-"): str(v) for k, v in attrs.items()})


def text(x, y, value, size=19, bold=False, color=BLUE):
    """Append a centered text label."""
    el = add(
        "text", x=x, y=y, text_anchor="middle", font_size=size, fill=color, font_weight="bold" if bold else "normal"
    )
    el.text = value


def line(x, y, xx, yy, color=BLUE, arrow=False, width=1.6):
    """Draw a connection, optionally with an arrowhead."""
    attrs = {"x1": x, "y1": y, "x2": xx, "y2": yy, "stroke": color, "stroke_width": width}
    if arrow:
        attrs["marker_end"] = "url(#aggregation-arrow)"
    add("line", **attrs)


def point(x, y, radius=6, color=GREEN):
    """Draw a particle or filter node."""
    add("circle", cx=x, cy=y, r=radius, fill=color)


def main():
    """Write a schematic; positions are illustrative rather than simulation samples."""
    defs = SubElement(svg, "defs")
    marker = SubElement(
        defs,
        "marker",
        {
            "id": "aggregation-arrow",
            "viewBox": "0 0 10 10",
            "refX": "8",
            "refY": "5",
            "markerWidth": "5",
            "markerHeight": "5",
            "orient": "auto",
        },
    )
    SubElement(marker, "path", {"d": "M0 0L10 5L0 10Z", "fill": BLUE})
    text(155, 26, "(a) Relative neighborhood", bold=True)
    text(480, 26, "(b) Trilinear interpolation", bold=True)
    text(825, 26, "(c) Shared 3D filters", bold=True)
    add("circle", cx=155, cy=140, r=76, fill="#f6faf4", stroke=GREEN, stroke_width=1.5, stroke_dasharray="5 4")
    for x, y in [(99, 105), (167, 80), (202, 118), (118, 173), (182, 193)]:
        line(155, 140, x, y, LIGHT)
        point(x, y)
        line(x, y, x + (155 - x) * 0.24, y + (140 - y) * 0.24, arrow=True)
    point(155, 140, 8, BLUE)
    text(155, 162, "i", 18)
    text(155, 242, "Offsets + relative velocities")
    text(155, 267, "Nearby particles count more")
    line(246, 139, 343, 139, arrow=True)
    text(295, 109, "Normalize", 16)
    text(295, 181, "Map to cube", 16)

    # One interpolation cell in the mapped 4×4×4 filter lattice.
    def vertex(a, b, c):
        return 405 + 120 * a + 44 * c, 205 - 104 * b - 40 * c

    fraction = (0.25, 0.4, 0.6)
    query = vertex(*fraction)
    for a in range(2):
        for b in range(2):
            for c in range(2):
                x, y = vertex(a, b, c)
                for axis, coordinate in enumerate([a, b, c]):
                    if coordinate == 0:
                        nxt = [a, b, c]
                        nxt[axis] = 1
                        xx, yy = vertex(*nxt)
                        line(x, y, xx, yy, "#9ca9b2")
                weight = math.prod(t if side else 1 - t for t, side in zip(fraction, [a, b, c]))
                line(*query, x, y, LIGHT, width=0.8 + weight * 7)
                point(x, y, 3 + 15 * weight)
    point(*query, 7, BLUE)
    text(480, 242, "One offset → eight grid nodes")
    text(480, 267, "Weights sum to one")
    line(582, 139, 655, 139, arrow=True)
    # Four planes, each with 4×4 filter nodes, representing 64 spatial nodes.
    for depth in range(3, -1, -1):
        x0, y0 = 681 + depth * 15, 116 - depth * 14
        add("rect", x=x0, y=y0, width=84, height=84, fill="#f4f8f2", stroke="#71886a", stroke_width=1)
        for k in range(4):
            line(x0 + k * 28, y0, x0 + k * 28, y0 + 84, LIGHT)
            line(x0, y0 + k * 28, x0 + 84, y0 + k * 28, LIGHT)
    line(819, 139, 868, 139, arrow=True)
    for k in range(6):
        add("rect", x=890, y=79 + k * 20, width=25, height=17, fill=GREEN if k % 2 else LIGHT, stroke=GREEN)
    text(904, 224, "aᵢ", 22, True)
    text(825, 242, "4 × 4 × 4 spatial filter bank")
    text(825, 267, "Weighted sum → embedding")
    line(25, 294, 975, 294, "#b9c7b5")
    text(500, 325, "(d) The same neighbors also produce six packing features · Eqs. (9)–(10)", 20, True)
    # Three intuitive summaries of the six output channels.
    for x, label in [(165, "Amount nearby"), (500, "Compression or expansion"), (835, "Average relative motion")]:
        text(x, 363, label, 19, True)
    for k, (dx, dy) in enumerate([(-40, 0), (-18, -24), (22, -20), (40, 15), (-18, 25)]):
        point(165 + dx, 408 + dy, 4 + k % 3)
    point(165, 408, 7, BLUE)
    for center, inward in [(449, True), (551, False)]:
        point(center, 408, 6, BLUE)
        for sign in [-1, 1]:
            point(center + sign * 33, 408, 5)
            start, end = (28, 13) if inward else (13, 28)
            line(center + sign * start, 408, center + sign * end, 408, arrow=True)
    text(449, 443, "Positive rate", 16)
    text(551, 443, "Negative rate", 16)
    for dy in [-20, 0, 20]:
        line(785, 408 + dy, 823, 408 + dy, arrow=True)
    line(841, 408, 886, 408, arrow=True, width=3)
    text(165, 485, "Neighbor count + weighted count", 18)
    text(500, 485, "Approach increases neighbor weight", 18)
    text(835, 485, "Weighted mean of velocity offsets", 18)
    text(165, 516, "2 channels", 18)
    text(500, 516, "1 channel", 18)
    text(835, 516, "3 channels (x, y, z)", 18)
    text(
        500,
        550,
        "Boundary features use a separate filter bank to produce bᵢ with the same interpolation procedure.",
        18,
    )
    target = Path(__file__).with_name("figures") / "local_aggregation.svg"
    target.write_text(tostring(svg, encoding="unicode"))
    print(target)


if __name__ == "__main__":
    main()
