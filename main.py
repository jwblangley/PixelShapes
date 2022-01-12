import xml.etree.ElementTree as ET

import argparse

import re

import torch
from PIL import Image

import matplotlib.pyplot as plt

import torchvision.transforms.functional as TF

from cairosvg import svg2png

from median_cut import median_cut, paint_lines, paint_centroids


XML_NAMESPACE = "http://www.w3.org/2000/svg"

PAINT_THICKNESS = 5


def edge_detect(img):
    filter = torch.tensor([[1, 2, 1]], dtype=torch.float32)
    img = torch.nn.functional.conv2d(
        img.unsqueeze(0).mean(dim=1, keepdim=True), filter.unsqueeze(0).unsqueeze(0)
    )

    filter = torch.tensor([[-1, 0, 1]], dtype=torch.float32)
    img = (
        torch.nn.functional.conv2d(img, filter.unsqueeze(0).unsqueeze(0))
        .squeeze()
        .abs()
    )
    return img


def manipulate_svg(svg, box_x, box_y, box_width, box_height, col):
    ET.register_namespace("", XML_NAMESPACE)
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg"), "Shape is not an svg"

    shape_width = root.get("width")
    shape_height = root.get("height")
    assert shape_width is not None, "Shape SVG does not have a 'width' attribute"
    assert shape_height is not None, "Shape SVG does not have a 'height' attribute"

    shape_width = float(re.sub(r"[^\d|\.]", "", shape_width))
    shape_height = float(re.sub(r"[^\d|\.]", "", shape_height))

    # Set location and size

    # Due to cairosvg limitations during rasterization, nested SVG centering must be done manually.
    # This can otherwise be achieved with just x, y and viewBox
    limiting_width = box_width / shape_width < box_height / shape_height

    new_shape_width = (
        box_width if limiting_width else shape_width * (box_height / shape_height)
    )
    new_shape_height = (
        shape_height * (box_width / shape_width) if limiting_width else box_height
    )

    new_shape_x = box_x + (box_width - new_shape_width) // 2
    new_shape_y = box_y + (box_height - new_shape_height) // 2

    root.set("x", f"{new_shape_x}")
    root.set("y", f"{new_shape_y}")
    root.set("width", f"{new_shape_width}")
    root.set("height", f"{new_shape_height}")
    root.set(
        "viewBox",
        f"0 0 {shape_width} {shape_height}",
    )
    root.set("preserveAspectRatio", "meet")

    # Re-colour
    col_str = f"rgb({col[0]},{col[1]},{col[2]})"

    for elem_type in (
        "rect",
        "circle",
        "ellipse",
        "line",
        "polyline",
        "polygon",
        "path",
    ):
        for elem in root.findall(f".//{{{XML_NAMESPACE}}}{elem_type}"):
            # Fill
            fill = elem.get("fill")
            if not (
                fill is None or fill.lower() == "none" or fill.lower() == "transparent"
            ):
                elem.set("fill", col_str)

            # Stroke
            stroke = elem.get("stroke")
            if not (
                stroke is None
                or stroke.lower() == "none"
                or stroke.lower() == "transparent"
            ):
                elem.set("stroke", col_str)

            # Style
            style = elem.get("style")
            if style is not None:
                # Fill
                re_fill = re.compile(r"fill:(.+?)(;|$)")
                fill = re_fill.search(style)
                if (
                    fill is not None
                    and fill.group(1).lower() != "none"
                    and fill.group(1).lower() != "transparent"
                ):
                    style = re_fill.sub(rf"fill:{col_str}\g<2>", style)

                # Stroke
                re_stroke = re.compile(r"stroke:(.+?)(;|$)")
                stroke = re_stroke.search(style)
                if (
                    stroke is not None
                    and stroke.group(1).lower() != "none"
                    and stroke.group(1).lower() != "transparent"
                ):
                    style = re_stroke.sub(rf"stroke:{col_str}\g<2>", style)

                elem.set("style", style)

    svg_str = ET.tostring(root, method="xml", encoding="unicode", xml_declaration=False)
    return svg_str


def shape_at_centroid(shape, img, centroid):
    _, _, bbox = centroid
    x1, y1, x2, y2 = bbox
    col = img[:, y1:y2, x1:x2].flatten(1).mean(dim=1) * 255
    col = col.int()

    svg_line = manipulate_svg(shape, x1, y1, x2 - x1, y2 - y1, col)
    return svg_line


def generate_svg_lines(shape, img, centroids, bg=None):
    yield f"""<svg version="1.1"
    width="{img.size(2)}" height="{img.size(1)}"
    xmlns="{XML_NAMESPACE}">\n\n"""

    if bg is not None:
        yield f'\t<rect x="0" y="0" width="{img.size(2)}" height="{img.size(1)}" fill="{bg}" />\n'

    for centroid in centroids:
        yield f"{shape_at_centroid(shape, img, centroid)}\n"

    yield "\n</svg>\n"


parser = argparse.ArgumentParser()

parser.add_argument("input", help="Path to input image", type=str)
parser.add_argument("shape", help="Path to svg shape", type=str)
parser.add_argument(
    "iterations", help="Number of iterations to run median cut for", type=int
)
parser.add_argument("output", help="Path to save result image to", type=str)
parser.add_argument("-d", "--debug", action="store_true", help="Show debug popup")
parser.add_argument(
    "-r",
    "--rasterize",
    action="store_true",
    help="Rasterize the vector graphics to png",
)
parser.add_argument(
    "-bg",
    "--background",
    type=str,
    help="SVG colour for the background. Transparent by default",
    default=None,
)

args = parser.parse_args()

if __name__ == "__main__":
    img = Image.open(args.input)
    img = TF.to_tensor(img)

    with open(args.shape, "r") as shape_svg:
        shape = shape_svg.read()

    edges = edge_detect(img)

    lines, centroids = median_cut(
        edges, (0, 0, edges.size(1), edges.size(0)), args.iterations
    )

    if args.debug:
        mc_img = paint_lines(edges, lines, PAINT_THICKNESS)
        mc_img = paint_centroids(mc_img, centroids, PAINT_THICKNESS)

        plt.imshow(mc_img)
        plt.show()

    out_svg = "\n".join(generate_svg_lines(shape, img, centroids, bg=args.background))

    if args.rasterize:
        svg2png(
            bytestring=out_svg,
            write_to=args.output,
        )
    else:
        with open(args.output, "w") as outfile:
            outfile.write(out_svg)
