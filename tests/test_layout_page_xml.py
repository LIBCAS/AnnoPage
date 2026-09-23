import os
import numpy as np
from lxml import etree as ET

from pero_ocr.core.layout import RegionLayout, TextLine, PAGEVersion
from pero_ocr.core.services import UuidService as PeroOcrUuidService, DateTimeService as PeroOcrDateTimeService

from anno_page.enums import Language, LineRelation
from anno_page.core.layout import AnnoPagePageLayout, AnnoPageRegionLayout, remove_annopage_elements
from anno_page.core.metadata import GraphicalObjectMetadata, RelatedLinesMetadata, ColorInfo, DominantColorInfo
from anno_page.core.services import UuidService as AnnoPageUuidService, DateTimeService as AnnoPageDateTimeService

from utils import generate_uuid, get_datetime_now, load_xml, assert_xml_equal


def setup():
    PeroOcrUuidService.generate_uuid = generate_uuid
    PeroOcrDateTimeService.get_datetime_now = get_datetime_now
    AnnoPageUuidService.generate_uuid = generate_uuid
    AnnoPageDateTimeService.get_datetime_now = get_datetime_now


def test_page_xml_image_no_metadata():
    setup()

    top = 20
    left = 10
    bottom = 100
    right = 80
    polygon = np.array([[left, top], [right, top], [right, bottom], [left, bottom]])
    region = AnnoPageRegionLayout(id="image_region", polygon=polygon, category="Image", detection_confidence=0.9)

    page_layout = AnnoPagePageLayout(id="test_page", page_size=(297, 210))
    page_layout.regions.append(region)

    generated_page_xml = page_layout.to_pagexml_string(version=PAGEVersion.PAGE_2019_07_15)

    resources_dir = os.path.join(os.path.dirname(__file__), "resources")
    expected_page_xml = load_xml(os.path.join(resources_dir, "page-xml_image_no_metadata.xml"))

    assert_xml_equal(generated_page_xml, expected_page_xml, load_custom_as_json=True)


def test_page_xml_image_with_basic_metadata():
    setup()

    top = 20
    left = 10
    bottom = 100
    right = 80
    polygon = np.array([[left, top], [right, top], [right, bottom], [left, bottom]])
    region = AnnoPageRegionLayout(id="image_region", polygon=polygon, category="Image", detection_confidence=0.9)
    region.graphical_metadata = GraphicalObjectMetadata(
        tag_id="image_001",
        mods_id="MODS_PICT_0001",
        description={
            Language.ENGLISH: "This is a description of an image",
            Language.CZECH: "Toto je popis obrázku"
        },
        caption={
            Language.ENGLISH: "This is a caption of an image",
            Language.CZECH: "Toto je titulek obrázku"
        },
        topics={
            Language.ENGLISH: ["topic1", "topic2"],
            Language.CZECH: ["topic1", "topic2"]
        },
        color={
            Language.ENGLISH: ColorInfo(
                color_mode="grayscale",
                dominant_colors=[DominantColorInfo(name="gray", coverage=0.7),
                                 DominantColorInfo(name="white", coverage=0.2),
                                 DominantColorInfo(name="black", coverage=0.1)]
            ),
            Language.CZECH: ColorInfo(
                color_mode="šedotónový",
                dominant_colors=[DominantColorInfo(name="šedá", coverage=0.7),
                                 DominantColorInfo(name="bílá", coverage=0.2),
                                 DominantColorInfo(name="černá", coverage=0.1)]
            ),
        },
        title="Fig. 1: Overview"
    )

    page_layout = AnnoPagePageLayout(id="test_page", page_size=(297, 210))
    page_layout.regions.append(region)

    generated_page_xml = page_layout.to_pagexml_string(version=PAGEVersion.PAGE_2019_07_15)

    resources_dir = os.path.join(os.path.dirname(__file__), "resources")
    expected_page_xml = load_xml(os.path.join(resources_dir, "page-xml_image_with_basic_metadata.xml"))

    assert_xml_equal(generated_page_xml, expected_page_xml, load_custom_as_json=True)


def test_page_xml_image_with_basic_metadata_and_text_lines_tagrefs_ai_models():
    setup()

    image_top = 20
    image_left = 10
    image_bottom = 100
    image_right = 80
    image_polygon = np.array([[image_left, image_top],
                              [image_right, image_top],
                              [image_right, image_bottom],
                              [image_left, image_bottom]])
    image_region = AnnoPageRegionLayout(id="image_region", polygon=image_polygon, category="Image", detection_confidence=0.9)
    image_region.graphical_metadata = GraphicalObjectMetadata(
        tag_id="image_001",
        mods_id="MODS_PICT_0001",
        description={
            Language.ENGLISH: "This is a description of an image",
            Language.CZECH: "Toto je popis obrázku"
        },
        caption={
            Language.ENGLISH: "This is a caption of an image",
            Language.CZECH: "Toto je titulek obrázku"
        },
        topics={
            Language.ENGLISH: ["topic1", "topic2"],
            Language.CZECH: ["topic1", "topic2"]
        },
        color={
            Language.ENGLISH: ColorInfo(
                color_mode="grayscale",
                dominant_colors=[DominantColorInfo(name="gray", coverage=0.7),
                                 DominantColorInfo(name="white", coverage=0.2),
                                 DominantColorInfo(name="black", coverage=0.1)]
            ),
            Language.CZECH: ColorInfo(
                color_mode="šedotónový",
                dominant_colors=[DominantColorInfo(name="šedá", coverage=0.7),
                                 DominantColorInfo(name="bílá", coverage=0.2),
                                 DominantColorInfo(name="černá", coverage=0.1)]
            ),
        },
        title="This is line no. 2",
        used_ai_models={
            "element-detection": "annopage-yolo-1.0",
        }
    )

    text_top = 120
    text_left = 10
    text_bottom = 200
    text_right = 200

    text_polygon = np.array([[text_left, text_top],
                             [text_right, text_top],
                             [text_right, text_bottom],
                             [text_left, text_bottom]])

    text_region = RegionLayout(id="text_region", polygon=text_polygon, category="text")

    line_left = 10
    line_right = 200

    line_tops = [120, 140, 160, 180]
    line_bottoms = [140, 160, 180, 200]
    baseline_tops = [bottom - 5 for bottom in line_bottoms]

    line_polygons = [np.array([[line_left, top],
                               [line_right, top],
                               [line_right, bottom],
                               [line_left, bottom]]) for top, bottom in zip(line_tops, line_bottoms)]

    baselines = [np.array([[line_left, baseline_top],
                           [line_right, baseline_top]]) for baseline_top in baseline_tops]

    line1 = TextLine(id="line1", polygon=line_polygons[0], baseline=baselines[0], transcription="This is line no. 1.")
    line2 = TextLine(id="line2", polygon=line_polygons[1], baseline=baselines[1], transcription="This is line no. 2.")
    line3 = TextLine(id="line3", polygon=line_polygons[2], baseline=baselines[2], transcription="This is line no. 3.")
    line4 = TextLine(id="line4", polygon=line_polygons[3], baseline=baselines[3], transcription="This is line no. 4.")

    text_region.lines.append(line1)
    text_region.lines.append(line2)
    text_region.lines.append(line3)
    text_region.lines.append(line4)

    image_region.graphical_metadata.caption_lines_metadata = RelatedLinesMetadata(
        tag_id="fc.image_001",
        mods_id="MODS_PICT_0001_CAPTION_0001",
        lines=[line2],
        relation=LineRelation.CAPTION,
        description=line2.transcription,
        title={
            Language.ENGLISH: line2.transcription
        },
        confidence=0.8
    )

    page_layout = AnnoPagePageLayout(id="test_page", page_size=(297, 210))
    page_layout.regions.append(image_region)
    page_layout.regions.append(text_region)

    generated_page_xml = page_layout.to_pagexml_string(version=PAGEVersion.PAGE_2019_07_15)

    resources_dir = os.path.join(os.path.dirname(__file__), "resources")
    expected_page_xml = load_xml(os.path.join(resources_dir, "page-xml_image_with_basic_metadata_and_text_lines_tagrefs_ai_models.xml"))

    assert_xml_equal(generated_page_xml, expected_page_xml, load_custom_as_json=True)
