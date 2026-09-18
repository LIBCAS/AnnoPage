import os
import numpy as np
from lxml import etree as ET

from pero_ocr.core.layout import RegionLayout, TextLine, ALTOVersion
from pero_ocr.core.services import UuidService as PeroOcrUuidService, DateTimeService as PeroOcrDateTimeService

from anno_page.enums import Language, LineRelation
from anno_page.core.layout import AnnoPagePageLayout, AnnoPageRegionLayout, remove_annopage_elements
from anno_page.core.metadata import GraphicalObjectMetadata, RelatedLinesMetadata
from anno_page.core.services import UuidService as AnnoPageUuidService, DateTimeService as AnnoPageDateTimeService

from utils import generate_uuid, get_datetime_now, load_xml, assert_xml_equal


def setup():
    PeroOcrUuidService.generate_uuid = generate_uuid
    PeroOcrDateTimeService.get_datetime_now = get_datetime_now
    AnnoPageUuidService.generate_uuid = generate_uuid
    AnnoPageDateTimeService.get_datetime_now = get_datetime_now


def test_alto_image_no_metadata():
    setup()

    top = 20
    left = 10
    bottom = 100
    right = 80
    polygon = np.array([[left, top], [right, top], [right, bottom], [left, bottom]])
    region = AnnoPageRegionLayout(id="image_region", polygon=polygon, category="Image", detection_confidence=0.9)

    page_layout = AnnoPagePageLayout(id="test_page", page_size=(297, 210))
    page_layout.regions.append(region)

    generated_alto = page_layout.to_altoxml_string(version=ALTOVersion.ALTO_v4_4)

    resources_dir = os.path.join(os.path.dirname(__file__), "resources")
    expected_alto = load_xml(os.path.join(resources_dir, "alto_image_no_metadata.xml"))

    assert_xml_equal(generated_alto, expected_alto)


def test_alto_image_with_basic_metadata():
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
            Language.ENGLISH: "grayscale",
            Language.CZECH: "černobílý"
        },
        title="Fig. 1: Overview"
    )

    page_layout = AnnoPagePageLayout(id="test_page", page_size=(297, 210))
    page_layout.regions.append(region)

    generated_alto = page_layout.to_altoxml_string(version=ALTOVersion.ALTO_v4_4)

    resources_dir = os.path.join(os.path.dirname(__file__), "resources")
    expected_alto = load_xml(os.path.join(resources_dir, "alto_image_with_basic_metadata.xml"))

    assert_xml_equal(generated_alto, expected_alto)


def test_alto_image_with_basic_metadata_and_text_lines():
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
            Language.ENGLISH: "grayscale",
            Language.CZECH: "černobílý"
        },
        title="This is line no. 2"
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

    line1_top = 120
    line1_bottom = 140
    baseline1_top = 135

    line2_top = 140
    line2_bottom = 160
    baseline2_top = 155

    line3_top = 160
    line3_bottom = 180
    baseline3_top = 175

    line4_top = 180
    line4_bottom = 200
    baseline4_top = 195

    line1_polygon = np.array([[line_left, line1_top],
                              [line_right, line1_top],
                              [line_right, line1_bottom],
                              [line_left, line1_bottom]])

    line2_polygon = np.array([[line_left, line2_top],
                              [line_right, line2_top],
                              [line_right, line2_bottom],
                              [line_left, line2_bottom]])

    line3_polygon = np.array([[line_left, line3_top],
                              [line_right, line3_top],
                              [line_right, line3_bottom],
                              [line_left, line3_bottom]])

    line4_polygon = np.array([[line_left, line4_top],
                              [line_right, line4_top],
                              [line_right, line4_bottom],
                              [line_left, line4_bottom]])

    baseline1 = np.array([[line_left, baseline1_top],
                          [line_right, baseline1_top]])

    baseline2 = np.array([[line_left, baseline2_top],
                          [line_right, baseline2_top]])

    baseline3 = np.array([[line_left, baseline3_top],
                          [line_right, baseline3_top]])

    baseline4 = np.array([[line_left, baseline4_top],
                          [line_right, baseline4_top]])

    line1 = TextLine(id="line1", polygon=line1_polygon, baseline=baseline1, transcription="This is line no. 1.")
    line2 = TextLine(id="line2", polygon=line2_polygon, baseline=baseline2, transcription="This is line no. 2.")
    line3 = TextLine(id="line3", polygon=line3_polygon, baseline=baseline3, transcription="This is line no. 3.")
    line4 = TextLine(id="line4", polygon=line4_polygon, baseline=baseline4, transcription="This is line no. 4.")

    text_region.lines.append(line1)
    text_region.lines.append(line2)
    text_region.lines.append(line3)
    text_region.lines.append(line4)

    page_layout = AnnoPagePageLayout(id="test_page", page_size=(297, 210))
    page_layout.regions.append(image_region)
    page_layout.regions.append(text_region)

    generated_alto = page_layout.to_altoxml_string(version=ALTOVersion.ALTO_v4_4)

    resources_dir = os.path.join(os.path.dirname(__file__), "resources")
    expected_alto = load_xml(os.path.join(resources_dir, "alto_image_with_basic_metadata_and_text_lines.xml"))

    assert_xml_equal(generated_alto, expected_alto)


def test_alto_image_with_basic_metadata_and_text_lines_tagrefs():
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
            Language.ENGLISH: "grayscale",
            Language.CZECH: "černobílý"
        },
        title="This is line no. 2"
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

    generated_alto = page_layout.to_altoxml_string(version=ALTOVersion.ALTO_v4_4)

    resources_dir = os.path.join(os.path.dirname(__file__), "resources")
    expected_alto = load_xml(os.path.join(resources_dir, "alto_image_with_basic_metadata_and_text_lines_tagrefs.xml"))

    assert_xml_equal(generated_alto, expected_alto)


def test_alto_image_with_basic_metadata_and_text_lines_tagrefs_ai_models():
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
            Language.ENGLISH: "grayscale",
            Language.CZECH: "černobílý"
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

    generated_alto = page_layout.to_altoxml_string(version=ALTOVersion.ALTO_v4_4)

    resources_dir = os.path.join(os.path.dirname(__file__), "resources")
    expected_alto = load_xml(os.path.join(resources_dir, "alto_image_with_basic_metadata_and_text_lines_tagrefs_ai_models.xml"))

    assert_xml_equal(generated_alto, expected_alto)


def test_alto_remove_annopage_elements():
    setup()

    resources_dir = os.path.join(os.path.dirname(__file__), "resources")
    input_alto = load_xml(os.path.join(resources_dir, "alto_remove_annopage_elements_before.xml"))
    output_alto = load_xml(os.path.join(resources_dir, "alto_remove_annopage_elements_after.xml"))

    alto = ET.fromstring(input_alto.encode())
    result_alto = remove_annopage_elements(alto)
    result = ET.tostring(result_alto, encoding='unicode')

    assert_xml_equal(result, output_alto)
