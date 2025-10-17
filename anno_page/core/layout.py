import cv2
import numpy as np

from lxml import etree as ET
from lxml.etree import Element
from pero_ocr.core.layout import RegionLayout, PageLayout, create_ocr_processing_element, ALTOVersion

from anno_page.enums.category import Category


def region_to_altoxml(region: RegionLayout, page_content_element):
    # import IPython; IPython.embed(); exit(1)

    category = Category.from_string(region.category)

    # TODO: find a better way to get the composed block ID or at least check for duplicates
    composed_block_id = len(page_content_element.findall(".//ComposedBlock")) + 1

    page_element = page_content_element
    while not (page_element.tag == "Page" or page_element.tag.endswith("}Page")):
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


def add_page_layout_to_alto(page_layout: PageLayout, alto_root: Element, alto_version=ALTOVersion.ALTO_v4_4):
    # import IPython; IPython.embed(); exit(1)

    namespaces = alto_root.nsmap
    mods_namespace = namespaces.get("mods", None)
    if mods_namespace is None:
        mods_namespace = "http://www.loc.gov/mods/v3"
        namespaces["mods"] = mods_namespace

    description_element = alto_root.find("Description", namespaces)
    if description_element is None:
        description_element = ET.SubElement(alto_root, "Description")

    processing_element = create_ocr_processing_element(id="AnnoPageProcessing",
                                                       software_creator_str="AnnoPage",
                                                       software_name_str="AnnoPage",
                                                       software_version_str="0.1",
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
        if region.metadata is not None:
            region.metadata.to_altoxml(tags_element,
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
