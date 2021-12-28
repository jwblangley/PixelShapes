import torch
from PIL import Image

import matplotlib.pyplot as plt

import torchvision.transforms.functional as TF

from median_cut import median_cut, paint_lines, paint_centroids

PAINT_THICKNESS = 10


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


def generate_svg_lines(img, centroids):
    yield f"""<svg version="1.1"
    width="{img.size(2)}" height="{img.size(1)}"
    xmlns="http://www.w3.org/2000/svg">\n\n"""

    for centroid in centroids:
        _, _, bbox = centroid
        x1, y1, x2, y2 = bbox
        col = img[:, y1:y2, x1:x2].flatten(1).mean(dim=1) * 255

        cx = x1 + (x2 - x1) // 2
        cy = y1 + (y2 - y1) // 2
        r = min(x2 - x1, y2 - y1) // 2
        col = col.int()

        yield f'\t<circle cx="{cx}" cy="{cy}" r="{r}" fill="rgb({col[0]},{col[1]},{col[2]})" />\n'

    yield "\n</svg>\n"


if __name__ == "__main__":

    img = Image.open("picture.jpg")
    img = TF.to_tensor(img)

    edges = edge_detect(img)

    lines, centroids = median_cut(edges, (0, 0, edges.size(1), edges.size(0)), 13)

    mc_img = paint_lines(edges, lines, PAINT_THICKNESS)
    mc_img = paint_centroids(mc_img, centroids, PAINT_THICKNESS)

    plt.imshow(mc_img)
    plt.show()

    with open("out.svg", "w") as outfile:
        for svg_line in generate_svg_lines(img, centroids):
            outfile.write(svg_line)
