import cv2
import numpy as np
from lxml import etree as ET
from pero_ocr.core.layout import RegionLayout

from anno_page.enums.category import Category


def region_to_altoxml(region: RegionLayout, page_content_element):
    category = Category.from_string(region.category)

    # TODO: find a better way to get the composed block ID
    composed_block_id = len(page_content_element.findall(".//ComposedBlock")) + 1

    page_element = page_content_element
    while page_element.tag != "Page":
        page_element = page_element.getparent()

    page_id = page_element.attrib["ID"]

    composed_block_element = ET.SubElement(page_content_element, "ComposedBlock")
    composed_block_element.set("ID", f"{page_id}_CB{composed_block_id:04d}")
    composed_block_element.set("TYPE", str(category))

    graphical_element = ET.SubElement(composed_block_element, "GraphicalElement")
    graphical_element.set("ID", region.id)

    bounding_box = region.get_polygon_bounding_box()
    set_position_and_size(composed_block_element, bounding_box)
    set_position_and_size(graphical_element, bounding_box)

    if region.metadata is not None:
        composed_block_element.set("TAGREFS", region.metadata.tag_id)

    return composed_block_element


def set_position_and_size(block, bounding_box):
    x_min, y_min, x_max, y_max = bounding_box
    height = y_max - y_min
    width = x_max - x_min

    block.set("HEIGHT", str(round(height)))
    block.set("WIDTH", str(round(width)))
    block.set("VPOS", str(round(y_min)))
    block.set("HPOS", str(round(x_min)))


def render_to_image(image, page_layout):
    render = np.copy(image)

    for region in page_layout.regions:
        if region.category in (None, "text"):
            continue

        x_min, y_min, x_max, y_max = region.get_polygon_bounding_box()
        cv2.rectangle(render, (round(x_min), round(y_min)), (round(x_max), round(y_max)), (0, 255, 0), 2)

    return render
