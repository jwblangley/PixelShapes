import argparse

import torch
from PIL import Image

import matplotlib.pyplot as plt

import torchvision.transforms.functional as TF

from cairosvg import svg2png

from median_cut import median_cut, paint_lines, paint_centroids

PAINT_THICKNESS = 30


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


def centroid_to_shape(img, centroid):
    _, _, bbox = centroid
    x1, y1, x2, y2 = bbox
    col = img[:, y1:y2, x1:x2].flatten(1).mean(dim=1) * 255

    cx = x1 + (x2 - x1) // 2
    cy = y1 + (y2 - y1) // 2
    r = min(x2 - x1, y2 - y1) // 2
    col = col.int()

    svg_line = (
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="rgb({col[0]},{col[1]},{col[2]})" />'
    )
    return svg_line


def generate_svg_lines(img, centroids, bg=None):
    yield f"""<svg version="1.1"
    width="{img.size(2)}" height="{img.size(1)}"
    xmlns="http://www.w3.org/2000/svg">\n\n"""

    if bg is not None:
        yield f'\t<rect x="0" y="0" width="{img.size(2)}" height="{img.size(1)}" fill="{bg}" />\n'

    for centroid in centroids:
        yield f"\t{centroid_to_shape(img, centroid)}\n"

    yield "\n</svg>\n"


parser = argparse.ArgumentParser()

parser.add_argument("input", help="Path to input image", type=str)
parser.add_argument("output", help="Path to save result image to", type=str)
parser.add_argument(
    "iterations", help="Number of iterations to run median cut for", type=int
)
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

    edges = edge_detect(img)

    lines, centroids = median_cut(
        edges, (0, 0, edges.size(1), edges.size(0)), args.iterations
    )

    if args.debug:
        mc_img = paint_lines(edges, lines, PAINT_THICKNESS)
        mc_img = paint_centroids(mc_img, centroids, PAINT_THICKNESS)

        plt.imshow(mc_img)
        plt.show()

    if args.rasterize:
        svg2png(
            bytestring="".join(generate_svg_lines(img, centroids, bg=args.background)),
            write_to=args.output,
        )
    else:
        with open(args.output, "w") as outfile:
            for svg_line in generate_svg_lines(img, centroids):
                outfile.write(svg_line)
