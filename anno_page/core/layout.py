import cv2
import numpy as np

from lxml import etree as ET
from lxml.etree import Element
from pero_ocr.core.layout import RegionLayout, PageLayout, create_ocr_processing_element, ALTOVersion

from anno_page import globals
from anno_page.enums import Category


def region_to_altoxml(region: RegionLayout, page_content_element):
    category = Category.from_string(region.category)

    page_element = page_content_element
    while not (page_element.tag == "Page" or page_element.tag.endswith("}Page")):
        page_element = page_element.getparent()

    page_id = page_element.attrib["ID"]

    composed_block_id = get_next_id(page_content_element, "ComposedBlock", prefix=f"{page_id}_CB", padding=4)
    composed_block_element = ET.SubElement(page_content_element, "ComposedBlock")
    composed_block_element.set("ID", composed_block_id)
    composed_block_element.set("TYPE", str(category))

    graphical_element_id = get_next_id(page_content_element, "GraphicalElement", prefix=f"{page_id}_GE", padding=4)
    graphical_element = ET.SubElement(composed_block_element, "GraphicalElement")
    graphical_element.set("ID", graphical_element_id)

    bounding_box = region.get_polygon_bounding_box()
    set_position_and_size(composed_block_element, bounding_box)
    set_position_and_size(graphical_element, bounding_box)

    if region.graphical_metadata is not None:
        composed_block_element.set("TAGREFS", region.graphical_metadata.tag_id)

    return composed_block_element


def get_next_id(parent_element, element_tag, prefix="", padding=4):
    existing_elements = parent_element.findall(f".//{element_tag}")
    existing_ids = set([element.attrib["ID"] for element in existing_elements if "ID" in element.attrib])
    index = 1

    new_id = f"{prefix}{str(index).zfill(padding)}"

    while new_id in existing_ids:
        index += 1
        new_id = f"{prefix}{str(index).zfill(padding)}"

    return new_id



def set_position_and_size(block, bounding_box):
    x_min, y_min, x_max, y_max = bounding_box
    height = y_max - y_min
    width = x_max - x_min

    block.set("HEIGHT", str(round(height)))
    block.set("WIDTH", str(round(width)))
    block.set("VPOS", str(round(y_min)))
    block.set("HPOS", str(round(x_min)))


def add_page_layout_to_alto(page_layout: PageLayout, alto_root: Element, alto_version=ALTOVersion.ALTO_v4_4):
    namespaces = alto_root.nsmap
    mods_namespace = namespaces.get("mods", None)
    if mods_namespace is None:
        mods_namespace = "http://www.loc.gov/mods/v3"
        namespaces["mods"] = mods_namespace

    description_element = alto_root.find("Description", namespaces)
    if description_element is None:
        description_element = ET.SubElement(alto_root, "Description")

    processing_element = create_ocr_processing_element(id=globals.software_name,
                                                       software_creator_str=globals.software_creator,
                                                       software_name_str=globals.software_name,
                                                       software_version_str=globals.software_version,
                                                       alto_version=alto_version)

    description_element.append(processing_element)

    tags_element = alto_root.find("Tags", namespaces)
    if tags_element is None:
        tags_element = ET.SubElement(alto_root, "Tags")

    layout_element = alto_root.find("Layout", namespaces)
    if layout_element is None:
        layout_element = ET.SubElement(alto_root, "Layout")

    page_element = layout_element.find("Page", namespaces)
    if page_element is None:
        page_element = ET.SubElement(layout_element, "Page")

    print_space_element = page_element.find("PrintSpace", namespaces)
    if print_space_element is None:
        print_space_element = ET.SubElement(page_element, "PrintSpace")

    for region in page_layout.regions:
        if region.category in (None, "text"):
            continue

        region_to_altoxml(region, print_space_element)
        if region.graphical_metadata is not None:
            region.graphical_metadata.to_altoxml(tags_element,
                                       category=region.category,
                                       bounding_box=region.get_polygon_bounding_box(),
                                       confidence=region.detection_confidence,
                                       mods_namespace=mods_namespace)

    return alto_root


def render_to_image(image, page_layout):
    render = np.copy(image)

    for region in page_layout.regions:
        if region.category in (None, "text"):
            continue

        x_min, y_min, x_max, y_max = region.get_polygon_bounding_box()
        cv2.rectangle(render, (round(x_min), round(y_min)), (round(x_max), round(y_max)), (0, 255, 0), 2)

    return render
