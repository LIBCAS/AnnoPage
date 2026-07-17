import cv2
import json
import numpy as np

from io import BytesIO
from typing import Tuple, Optional
from lxml import etree as ET
from lxml.etree import Element

from pero_ocr.core.layout import RegionLayout, PageLayout, create_ocr_processing_element, ALTOVersion

from anno_page import globals
from anno_page.core.services import DateTimeService
from anno_page.enums import Category
from anno_page.core.metadata import GraphicalObjectMetadata


class AnnoPageRegionLayout(RegionLayout):
    def __init__(self,
                 id: str,
                 polygon: np.ndarray,
                 region_type: Optional[str] = None,
                 category: Optional[str] = None,
                 detection_confidence: Optional[float] = None,
                 graphical_metadata: Optional[GraphicalObjectMetadata] = None):
        super().__init__(id=id,
                         polygon=polygon,
                         region_type=region_type,
                         category=category,
                         detection_confidence=detection_confidence)

        self.graphical_metadata: Optional[GraphicalObjectMetadata] = graphical_metadata

    def to_altoxml(self, print_space_element, tags, mods_namespace, arabic_helper, min_line_confidence,
                   print_space_coords: Tuple[int, int, int, int], version: ALTOVersion, word_splitters=["-"]) -> Tuple[int, int, int, int]:
        category = Category.from_string(self.category)
        page_element = get_page_element(print_space_element)
        page_id = page_element.attrib["ID"]

        composed_block_id = get_next_id(print_space_element, "ComposedBlock", prefix=f"{page_id}_CB", padding=4)
        composed_block_element = ET.SubElement(print_space_element, "ComposedBlock")
        composed_block_element.set("ID", composed_block_id)
        composed_block_element.set("TYPE", str(category))

        graphical_element_id = get_next_id(print_space_element, "GraphicalElement", prefix=f"{page_id}_GE", padding=4)
        graphical_element = ET.SubElement(composed_block_element, "GraphicalElement")
        graphical_element.set("ID", graphical_element_id)

        bounding_box = self.get_polygon_bounding_box()
        set_position_and_size(composed_block_element, bounding_box)
        set_position_and_size(graphical_element, bounding_box)

        x_min, y_min, x_max, y_max = bounding_box
        block_height, block_width, block_vpos, block_hpos = y_max - y_min, x_max - x_min, y_min, x_min
        print_space_height, print_space_width, print_space_vpos, print_space_hpos = print_space_coords

        print_space_height = max([print_space_vpos + print_space_height, block_vpos + block_height])
        print_space_width = max([print_space_hpos + print_space_width, block_hpos + block_width])
        print_space_vpos = min([print_space_vpos, block_vpos])
        print_space_hpos = min([print_space_hpos, block_hpos])
        print_space_height = print_space_height - print_space_vpos
        print_space_width = print_space_width - print_space_hpos

        if self.graphical_metadata is not None:
            if self.graphical_metadata.confidence is None:
                self.graphical_metadata.confidence = self.detection_confidence

            self.graphical_metadata.to_altoxml(tags,
                                               category=self.category,
                                               bounding_box=bounding_box,
                                               mods_namespace=mods_namespace)

            composed_block_element.set("TAGREFS", self.graphical_metadata.tag_id)

        return print_space_height, print_space_width, print_space_vpos, print_space_hpos

    @classmethod
    def from_altoxml(cls, composed_block_element):
        category = composed_block_element.attrib.get("TYPE", None)
        if category is None:
            return None

        height = composed_block_element.attrib.get("HEIGHT", None)
        width = composed_block_element.attrib.get("WIDTH", None)
        vpos = composed_block_element.attrib.get("VPOS", None)
        hpos = composed_block_element.attrib.get("HPOS", None)

        if None in (height, width, vpos, hpos):
            return None

        height = int(height)
        width = int(width)
        vpos = int(vpos)
        hpos = int(hpos)

        if not cls.check_graphical_element_exists(composed_block_element, height, width, vpos, hpos):
            return None

        region_coords = [[hpos, vpos], [hpos + width, vpos], [hpos + width, vpos + height], [hpos, vpos + height]]
        region_coords = np.array(region_coords)

        region_id = composed_block_element.attrib["ID"]

        region = cls(region_id, region_coords, category=category)

        return region

    @staticmethod
    def check_graphical_element_exists(composed_block_element, composed_block_height, composed_block_width, composed_block_vpos, composed_block_hpos):
        composed_block_children = composed_block_element.getchildren()
        if len(composed_block_children) == 1 and composed_block_children[0].tag.endswith("GraphicalElement"):
            graphical_element = composed_block_children[0]
            graphical_element_height = int(graphical_element.attrib.get("HEIGHT", 0))
            graphical_element_width = int(graphical_element.attrib.get("WIDTH", 0))
            graphical_element_vpos = int(graphical_element.attrib.get("VPOS", 0))
            graphical_element_hpos = int(graphical_element.attrib.get("HPOS", 0))

            if (graphical_element_height == composed_block_height and
                graphical_element_width == composed_block_width and
                graphical_element_vpos == composed_block_vpos and
                graphical_element_hpos == composed_block_hpos):
                return True

        return False

    def to_pagexml(self, page_element: ET.SubElement, validate_id: bool = False):
        custom = {
            "category": self.category,
            "detection_confidence": round(self.detection_confidence, 3),
            "metadata": self.graphical_metadata.to_dict() if self.graphical_metadata is not None else None
        }

        region_element = ET.SubElement(page_element, "ImageRegion")
        region_element.attrib["id"] = self.id
        region_element.attrib["custom"] = json.dumps(custom)

        if self.region_type is not None:
            region_element.attrib["type"] = self.region_type

        coords = ET.SubElement(region_element, "Coords")
        coords.attrib["points"] = " ".join([f"{int(x)},{int(y)}" for x, y in self.polygon])

        return region_element

    @classmethod
    def from_pagexml(cls, region_element: ET.SubElement, page_layout=None):
        region_id = region_element.attrib["id"]
        region_type = region_element.attrib.get("type", None)

        polygon = []
        coords_element = region_element.find("Coords", region_element.nsmap)
        if coords_element is not None:
            points_str = coords_element.attrib.get("points", "")
            for point_str in points_str.split():
                x_str, y_str = point_str.split(",")
                polygon.append((float(x_str), float(y_str)))

        polygon = np.array(polygon)

        category = None
        detection_confidence = None
        graphical_metadata = None

        if "custom" in region_element.attrib:
            custom = json.loads(region_element.attrib["custom"])
            category = custom.get("category", None)
            detection_confidence = custom.get("detection_confidence", None)
            metadata_dict = custom.get("metadata", None)
            if metadata_dict is not None:
                graphical_metadata = GraphicalObjectMetadata.from_dict(metadata_dict, page_layout)

        region = cls(region_id,
                     polygon=polygon,
                     region_type=region_type,
                     category=category,
                     detection_confidence=detection_confidence,
                     graphical_metadata=graphical_metadata)

        return region


class AnnoPagePageLayout(PageLayout):
    def __init__(self, id, page_size):
        super().__init__(id, page_size)

        self.from_altoxml_ended += altoxml_load_regions
        self.from_pagexml_ended += pagexml_load_regions

        self.to_altoxml_processing_added += altoxml_add_processing_step
        self.to_altoxml_regions_ended += altoxml_postprocess_lines
        self.to_pagexml_processing_added += pagexml_add_processing_step


def pagexml_load_regions(page_layout, page_tree):
    root = page_tree.getroot()
    for region_element in root.findall(".//ImageRegion", root.nsmap):
        region = AnnoPageRegionLayout.from_pagexml(region_element, page_layout)
        if region is not None:
            page_layout.regions.append(region)


def altoxml_load_regions(page_layout, root):
    print_space_element = root.find(".//PrintSpace", root.nsmap)
    if print_space_element is None:
        return

    tags_element = root.find(".//Tags", root.nsmap)

    for composed_block_element in print_space_element.findall(".//ComposedBlock", print_space_element.nsmap):
        region = AnnoPageRegionLayout.from_altoxml(composed_block_element)
        if region is not None:
            page_layout.regions.append(region)

            tagrefs = composed_block_element.attrib.get("TAGREFS", None)
            if tagrefs is not None:
                tagrefs = tagrefs.split()
                for tagref in tagrefs:
                    tag_element = tags_element.find(f".//{{*}}LayoutTag[@ID='{tagref}']", tags_element.nsmap)
                    if tag_element is not None:
                        metadata = GraphicalObjectMetadata.from_altoxml(page_layout, tag_element, tags_element, print_space_element)
                        if metadata is not None:
                            region.graphical_metadata = metadata
                            region.detection_confidence = metadata.confidence


def get_page_element(print_space_element):
    page_element = print_space_element
    while not (page_element.tag == "Page" or page_element.tag.endswith("}Page")):
        page_element = page_element.getparent()

    return page_element


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


def add_page_layout_to_alto(page_layout: AnnoPagePageLayout, alto_root: Element, alto_version=ALTOVersion.ALTO_v4_4):
    namespaces = alto_root.nsmap
    mods_namespace = namespaces.get("mods", None)
    if mods_namespace is None:
        mods_namespace = "http://www.loc.gov/mods/v3"
        namespaces["mods"] = mods_namespace

    description_element = find_or_create_element(alto_root, "Description", namespaces)
    tags_element = find_or_create_element(alto_root, "Tags", namespaces)
    layout_element = find_or_create_element(alto_root, "Layout", namespaces)
    page_element = find_or_create_element(layout_element, "Page", namespaces)
    print_space_element = find_or_create_element(page_element, "PrintSpace", namespaces)

    altoxml_add_processing_step(page_layout, description_element, alto_version)

    print_space_coords = get_print_space_coords(page_layout, print_space_element)

    for region in page_layout.regions:
        if isinstance(region, AnnoPageRegionLayout):
            print_space_coords = region.to_altoxml(print_space_element, tags_element, mods_namespace, None, 0.0, print_space_coords, alto_version)

    update_print_space_and_margins(page_layout, page_element, print_space_coords)

    altoxml_postprocess_lines(page_layout, print_space_element, alto_version)

    return alto_root


def remove_element(alto, element_specification):
    element = alto.find(element_specification, namespaces=alto.nsmap)
    if element is not None:
        parent = element.getparent()
        parent.remove(element)

    return alto


def remove_tagrefs(alto, tag_id):
    elements = alto.xpath(
        '//*[@TAGREFS and contains(concat(" ", normalize-space(@TAGREFS), " "), '
        'concat(" ", $tag, " "))]',
        tag=tag_id,
    )

    for element in elements:
        tagrefs = element.attrib.get("TAGREFS", "")
        tagrefs_list = tagrefs.split()
        tagrefs_list = [tag for tag in tagrefs_list if tag != tag_id]
        if tagrefs_list:
            element.attrib["TAGREFS"] = " ".join(tagrefs_list)
        else:
            del element.attrib["TAGREFS"]

    return alto

def check_tag_source(alto, tag_id, tag_name):
    record_content_source_element = alto.find(f".//{tag_name}[@ID='{tag_id}']/XmlData/mods:mods/mods:recordInfo/mods:recordContentSource", namespaces=alto.nsmap)
    if record_content_source_element is None:
        return False

    try:
        record_content_source = record_content_source_element.text.split()[0]
    except:
        return False

    if record_content_source != globals.software_name:
        return False

    return True


def remove_annopage_elements(alto):
    alto_bytes_io = BytesIO(ET.tostring(alto, encoding="utf-8", xml_declaration=True))

    page_layout = AnnoPagePageLayout(id="temp", page_size=(0, 0))
    page_layout.from_altoxml(alto_bytes_io)

    region_ids = []
    tag_ids = []
    caption_tag_ids = []
    reference_tag_ids = []

    for region in page_layout.regions:
        if isinstance(region, AnnoPageRegionLayout):
            region_ids.append(region.id)
            if region.graphical_metadata is not None:
                metadata = region.graphical_metadata

                if check_tag_source(alto, metadata.tag_id, "LayoutTag"):
                    tag_ids.append(metadata.tag_id)

                if metadata.caption_lines_metadata is not None and check_tag_source(alto, metadata.caption_lines_metadata.tag_id, "StructureTag"):
                    caption_tag_ids.append(metadata.caption_lines_metadata.tag_id)

                if metadata.reference_lines_metadata is not None and check_tag_source(alto, metadata.reference_lines_metadata.tag_id, "OtherTag"):
                    reference_tag_ids.append(metadata.reference_lines_metadata.tag_id)

    for region_id in region_ids:
        alto = remove_element(alto, f".//ComposedBlock[@ID='{region_id}']")

    for tag_id in tag_ids:
        alto = remove_element(alto, f".//LayoutTag[@ID='{tag_id}']")
        alto = remove_tagrefs(alto, tag_id)

    for caption_tag_id in caption_tag_ids:
        alto = remove_element(alto, f".//StructureTag[@ID='{caption_tag_id}']")
        alto = remove_tagrefs(alto, caption_tag_id)

    for reference_tag_id in reference_tag_ids:
        alto = remove_element(alto, f".//OtherTag[@ID='{reference_tag_id}']")
        alto = remove_tagrefs(alto, reference_tag_id)

    return alto


def get_print_space_coords(page_layout, print_space_element):
    print_space_height = print_space_element.attrib.get("HEIGHT", 0)
    print_space_width = print_space_element.attrib.get("WIDTH", 0)
    print_space_vpos = print_space_element.attrib.get("VPOS", page_layout.page_size[0])
    print_space_hpos = print_space_element.attrib.get("HPOS", page_layout.page_size[1])

    print_space_height = int(print_space_height)
    print_space_width = int(print_space_width)
    print_space_vpos = int(print_space_vpos)
    print_space_hpos = int(print_space_hpos)

    return print_space_height, print_space_width, print_space_vpos, print_space_hpos


def update_print_space_and_margins(page_layout, page_element, print_space_coords):
    print_space_height, print_space_width, print_space_vpos, print_space_hpos = print_space_coords

    top_margin = find_or_create_element(page_element, "TopMargin")
    left_margin = find_or_create_element(page_element, "LeftMargin")
    right_margin = find_or_create_element(page_element, "RightMargin")
    bottom_margin = find_or_create_element(page_element, "BottomMargin")
    print_space = find_or_create_element(page_element, "PrintSpace")

    top_margin.set("HEIGHT", "{}".format(int(print_space_vpos)))
    top_margin.set("WIDTH", "{}".format(int(page_layout.page_size[1])))
    top_margin.set("VPOS", "0")
    top_margin.set("HPOS", "0")

    left_margin.set("HEIGHT", "{}".format(int(page_layout.page_size[0])))
    left_margin.set("WIDTH", "{}".format(int(print_space_hpos)))
    left_margin.set("VPOS", "0")
    left_margin.set("HPOS", "0")

    right_margin.set("HEIGHT", "{}".format(int(page_layout.page_size[0])))
    right_margin.set("WIDTH", "{}".format(int(page_layout.page_size[1] - (print_space_hpos + print_space_width))))
    right_margin.set("VPOS", "0")
    right_margin.set("HPOS", "{}".format(int(print_space_hpos + print_space_width)))

    bottom_margin.set("HEIGHT", "{}".format(int(page_layout.page_size[0] - (print_space_vpos + print_space_height))))
    bottom_margin.set("WIDTH", "{}".format(int(page_layout.page_size[1])))
    bottom_margin.set("VPOS", "{}".format(int(print_space_vpos + print_space_height)))
    bottom_margin.set("HPOS", "0")

    print_space.set("HEIGHT", str(int(print_space_height)))
    print_space.set("WIDTH", str(int(print_space_width)))
    print_space.set("VPOS", str(int(print_space_vpos)))
    print_space.set("HPOS", str(int(print_space_hpos)))


def find_or_create_element(parent_element, tag, namespaces=None):
    if namespaces is None:
        namespaces = parent_element.nsmap

    element = parent_element.find(tag, namespaces)
    if element is None:
        element = ET.SubElement(parent_element, tag)

    return element


def altoxml_postprocess_lines(page_layout, print_space_element, alto_version=ALTOVersion.ALTO_v4_4):
    for region in page_layout.regions:
        if not isinstance(region, AnnoPageRegionLayout):
            continue

        if region.graphical_metadata is not None:
            metadata: GraphicalObjectMetadata = region.graphical_metadata
            if metadata.caption_lines_metadata is not None:
                for line in metadata.caption_lines_metadata.lines:
                    line_id = line.id
                    if not line_id.startswith("line_"):
                        line_id = f"line_{line_id}"

                    line_element = print_space_element.find(f".//TextLine[@ID='{line_id}']")

                    if line_element is None:
                        continue

                    current_tag_refs = line_element.attrib["TAGREFS"] if "TAGREFS" in line_element.attrib else None
                    if current_tag_refs is not None:
                        current_tag_refs = set(current_tag_refs.split())
                    else:
                        current_tag_refs = set()

                    current_tag_refs.add(metadata.caption_lines_metadata.tag_id)
                    line_element.set("TAGREFS", " ".join(current_tag_refs))

            if metadata.continuing_line is not None:
                line_id = metadata.continuing_line.id
                if not line_id.startswith("line_"):
                    line_id = f"line_{line_id}"

                line_element = print_space_element.find(f".//{{*}}TextLine[@ID='{line_id}']")
                if line_element is not None:
                    string_element = line_element.find(".//{*}String")
                    if string_element is not None:
                        string_element.attrib["SUBS_CONTENT"] = region.transcription + string_element.attrib["CONTENT"]
                        string_element.attrib["SUBS_TYPE"] = "Abbreviation"


def altoxml_add_processing_step(page_layout, description_element: ET.Element, alto_version=ALTOVersion.ALTO_v4_4):
    processing_element = create_ocr_processing_element(id=globals.software_name,
                                                       software_creator_str=globals.software_creator,
                                                       software_name_str=globals.software_name,
                                                       software_version_str=globals.software_version,
                                                       alto_version=alto_version)

    description_element.append(processing_element)


def pagexml_add_processing_step(page_layout, metadata: ET.Element):
    metadata_item = ET.SubElement(metadata, "MetadataItem")
    metadata_item.set("type", "processingStep")
    metadata_item.set("name", "Non-textual elements detection and analysis")
    metadata_item.set("value", globals.software_fullname)
    metadata_item.set("date", DateTimeService.get_datetime_now().isoformat())


def render_to_image(image, page_layout):
    render = np.copy(image)

    for region in page_layout.regions:
        if region.category in (None, "text"):
            continue

        x_min, y_min, x_max, y_max = region.get_polygon_bounding_box()
        cv2.rectangle(render, (round(x_min), round(y_min)), (round(x_max), round(y_max)), (0, 255, 0), 2)

    return render
