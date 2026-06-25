import os
import uuid
import datetime
import numpy as np

import xmltodict

from pero_ocr.core.layout import PageLayout, RegionLayout, TextLine, ALTOVersion
from pero_ocr.core.services import UuidService as PeroOcrUuidService, DateTimeService as PeroOcrDateTimeService

from anno_page.enums import Language, LineRelation
from anno_page.core.layout import set_handlers, AnnoPageRegionLayout
from anno_page.core.metadata import GraphicalObjectMetadata, RelatedLinesMetadata
from anno_page.core.services import UuidService as AnnoPageUuidService, DateTimeService as AnnoPageDateTimeService


def generate_uuid(*args, **kwargs):
    return uuid.UUID("01234567-0123-0123-0123-0123456789ab")


def get_datetime_now(*args, **kwargs):
    return datetime.datetime(1970, 1, 1, 12, 0, 0)


def assert_lists_equal(list1, list2):
    if len(list1) != len(list2):
        raise AssertionError(f"Lists differ in length: {len(list1)} != {len(list2)}")

    unmatched_items = list(list2)
    for item1 in list1:
        for index, item2 in enumerate(unmatched_items):
            try:
                if isinstance(item1, dict) and isinstance(item2, dict):
                    assert_dicts_equal(item1, item2)
                elif isinstance(item1, list) and isinstance(item2, list):
                    assert_lists_equal(item1, item2)
                elif item1 != item2:
                    raise AssertionError
            except AssertionError:
                continue

            del unmatched_items[index]
            break
        else:
            raise AssertionError(f"List item {item1} is missing in the second list.")


def assert_dicts_equal(dict1, dict2):
    all_keys = set(dict1.keys()) | set(dict2.keys())
    for key in all_keys:
        if key not in dict1:
            raise AssertionError(f"Key '{key}' is missing in the first dictionary.")
        if key not in dict2:
            raise AssertionError(f"Key '{key}' is missing in the second dictionary.")

        value1 = dict1[key]
        value2 = dict2[key]

        if isinstance(value1, dict) and isinstance(value2, dict):
            assert_dicts_equal(value1, value2)

        elif isinstance(value1, list) and isinstance(value2, list):
            assert_lists_equal(value1, value2)

        elif value1 != value2:
            raise AssertionError(f"Values for key '{key}' differ: {value1} != {value2}")


def assert_xml_equal(actual_xml: str, expected_xml: str) -> None:
    actual_dict = xmltodict.parse(actual_xml)
    expected_dict = xmltodict.parse(expected_xml)
    assert_dicts_equal(actual_dict, expected_dict)


def load_xml(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


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

    page_layout = PageLayout(id="test_page", page_size=(297, 210))
    page_layout.regions.append(region)

    set_handlers(page_layout)
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

    page_layout = PageLayout(id="test_page", page_size=(297, 210))
    page_layout.regions.append(region)

    set_handlers(page_layout)
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

    page_layout = PageLayout(id="test_page", page_size=(297, 210))
    page_layout.regions.append(image_region)
    page_layout.regions.append(text_region)

    set_handlers(page_layout)
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

    image_region.graphical_metadata.caption_lines_metadata = RelatedLinesMetadata(
        tag_id="fc.image_001",
        mods_id="MODS_PICT_0001_CAPTION_0001",
        lines=[line2],
        relation=LineRelation.CAPTION,
        description=line2.transcription,
        title={
            Language.ENGLISH: line2.transcription
        }
    )

    page_layout = PageLayout(id="test_page", page_size=(297, 210))
    page_layout.regions.append(image_region)
    page_layout.regions.append(text_region)

    set_handlers(page_layout)
    generated_alto = page_layout.to_altoxml_string(version=ALTOVersion.ALTO_v4_4)

    resources_dir = os.path.join(os.path.dirname(__file__), "resources")
    expected_alto = load_xml(os.path.join(resources_dir, "alto_image_with_basic_metadata_and_text_lines_tagrefs.xml"))

    assert_xml_equal(generated_alto, expected_alto)
