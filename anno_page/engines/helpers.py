import torch
import numpy as np

from shapely.geometry import Polygon


DTYPE_MAPPING = {
    "float32": torch.float32,
    "fp32":    torch.float32,
    "float16": torch.float16,
    "fp16":    torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16":     torch.bfloat16,
}


def config_get_dtype(config, key, fallback=torch.float32):
    dtype_str = config.get(key, None)
    if dtype_str is not None:
        dtype = DTYPE_MAPPING.get(dtype_str.lower(), None)
        if dtype is not None:
            return dtype

    return fallback


def find_lines_in_bbox(bbox, page_layout, threshold=0.5):
    lines = []

    x1, y1, x2, y2 = bbox
    bbox_polygon = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

    for line in page_layout.lines_iterator():
        line_polygon = Polygon(line.polygon)
        intersection = bbox_polygon.intersection(line_polygon)
        if intersection.area / line_polygon.area >= threshold:
            lines.append(line)

    return lines


def find_nearest_region(bbox, page_layout, categories):
    nearest_region = None
    min_distance = float('inf')

    bbox_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])

    for region in page_layout.regions:
        if region.category in categories:
            region_bbox = region.get_polygon_bounding_box()
            region_center = np.array([(region_bbox[0] + region_bbox[2]) / 2, (region_bbox[1] + region_bbox[3]) / 2])
            distance = np.linalg.norm(bbox_center - region_center)
            if distance < min_distance:
                min_distance = distance
                nearest_region = region

    return nearest_region

