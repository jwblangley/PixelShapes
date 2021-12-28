import numpy as np


def calculate_centroid(img, bbox):
    assert len(img.shape) == 2, "Invalid image shape"

    x1, y1, x2, y2 = bbox
    assert x1 < x2 and y1 < y2, "Invalid coordinates: recursion too deep"

    subimg = img[y1:y2, x1:x2]
    pdf = subimg.flatten() / subimg.sum()
    cdf = pdf.cumsum(0)

    split_index = np.where(cdf >= 0.5)[0][0]
    split_y = split_index // subimg.size(0)
    split_x = split_index % subimg.size(0)

    return (x1 + split_x, y1 + split_y, bbox)


def calculate_split(img, bbox, wide):
    assert len(img.shape) == 2, "Invalid image shape"

    x1, y1, x2, y2 = bbox
    assert x1 < x2 and y1 < y2, "Invalid coordinates: recursion too deep"

    subimg = img[y1:y2, x1:x2]
    dim = 0 if wide else 1

    pdf = subimg.sum(dim=dim) / subimg.sum()
    cdf = pdf.cumsum(0)

    split_index = np.where(cdf >= 0.5)[0][0]

    base = x1 if wide else y1

    return base + split_index


def median_cut(img, bbox, its):
    x1, y1, x2, y2 = bbox
    assert x1 < x2 and y1 < y2, "Invalid coordinates: recursion too deep"

    if its <= 0:
        return [], [calculate_centroid(img, bbox)]

    width = x2 - x1
    height = y2 - y1

    if width >= height:
        split = calculate_split(img, bbox, True)

        child1_lines, child1_centroids = median_cut(img, (x1, y1, split, y2), its - 1)
        child2_lines, child2_centroids = median_cut(img, (split, y1, x2, y2), its - 1)

        this_lines = [(split, y1, split, y2)]
        return (
            this_lines + child1_lines + child2_lines,
            child1_centroids + child2_centroids,
        )
    else:
        split = calculate_split(img, bbox, False)

        child1_lines, child1_centroids = median_cut(img, (x1, y1, x2, split), its - 1)
        child2_lines, child2_centroids = median_cut(img, (x1, split, x2, y2), its - 1)

        this_lines = [(x1, split, x2, split)]
        return (
            this_lines + child1_lines + child2_lines,
            child1_centroids + child2_centroids,
        )


def paint_lines(img, lines, thickness):
    img = img.clone()

    LINE_COLOUR = 1.0

    for line in lines:
        x1, y1, x2, y2 = line

        if x1 == x2:
            img[y1:y2, x1 - thickness // 2 : x1 + 1 + thickness // 2] = LINE_COLOUR
        elif y1 == y2:
            img[y1 - thickness // 2 : y1 + 1 + thickness // 2, x1:x2] = LINE_COLOUR
        else:
            raise NotImplementedError("Only perpendicular lines supported")

    return img


def paint_centroids(img, centroids, thickness):
    img = img.clone()

    CENTROID_COLOUR = 1.0

    for centroid in centroids:
        x, y, bbox = centroid

        img[
            y - thickness // 2 : y + 1 + thickness // 2,
            x - thickness // 2 : x + 1 + thickness // 2,
        ] = CENTROID_COLOUR

    return img
