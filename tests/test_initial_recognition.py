import os
import io
import json
import numpy as np

from configparser import ConfigParser

from pero_ocr.core.layout import RegionLayout, TextLine, ALTOVersion
from pero_ocr.core.services import UuidService as PeroOcrUuidService, DateTimeService as PeroOcrDateTimeService

from anno_page.enums import Language, LineRelation
from anno_page.core.layout import AnnoPagePageLayout, AnnoPageRegionLayout
from anno_page.core.metadata import GraphicalObjectMetadata
from anno_page.core.services import UuidService as AnnoPageUuidService, DateTimeService as AnnoPageDateTimeService
from anno_page.engines.initial import InitialRecognitionEngine, InitialRecognitionResult, LLMResult

from utils import generate_uuid, get_datetime_now, load_xml, assert_xml_equal


def setup():
    PeroOcrUuidService.generate_uuid = generate_uuid
    PeroOcrDateTimeService.get_datetime_now = get_datetime_now
    AnnoPageUuidService.generate_uuid = generate_uuid
    AnnoPageDateTimeService.get_datetime_now = get_datetime_now


def test_alto_initial_no_space():
    setup()

    resources_dir = os.path.join(os.path.dirname(__file__), "resources")
    expected_alto = load_xml(os.path.join(resources_dir, "alto_initial_no_space.xml"))

    def prepare_prompt_data(image, page_layout, region):
        continuing_line = [line for line in page_layout.lines_iterator() if line.id == "line2"][0]
        return None, None, continuing_line

    def process_initial(region, initial_crop, context_crop, continuing_line) -> LLMResult:
        return LLMResult(data=InitialRecognitionResult(initial="XXX", include_space=False))

    config_dict = {
        "categories": json.dumps(["initial"]),
        "max_attempts": "3",
        "api": "test",
        "api_key": "test_key",
        "prompt_settings": os.path.join(resources_dir, "initial_recognition_prompt.json")
    }

    config = ConfigParser()
    config.add_section("INITIAL_RECOGNITION")
    for key, value in config_dict.items():
        config.set("INITIAL_RECOGNITION", key, value)

    engine = InitialRecognitionEngine(config["INITIAL_RECOGNITION"], device="cpu", config_path="")
    engine._process_initial = process_initial
    engine._prepare_prompt_data = prepare_prompt_data

    initial_top = 50
    initial_left = 10
    initial_right = 65
    initial_bottom = 105

    initial_polygon = np.array([[initial_left, initial_top],
                                [initial_right, initial_top],
                                [initial_right, initial_bottom],
                                [initial_left, initial_bottom]])
    initial_region = AnnoPageRegionLayout(id="initial_region", polygon=initial_polygon, category="Initial", detection_confidence=0.9)
    initial_region.graphical_metadata = GraphicalObjectMetadata(
        tag_id="initial_001",
        mods_id="MODS_PICT_0001",
    )

    line_tops = [10, 30, 50, 70, 90, 110, 130]
    line_lefts = [10, 10, 70, 70, 70, 10, 10]
    line_rights = [200, 200, 200, 200, 200, 200, 200]
    line_bottoms = [25, 45, 65, 85, 105, 125, 145]

    line_baseline_tops = [bottom - 5 for bottom in line_bottoms]

    line_transcriptions = ["This is line no. 1.",
                           "This is line no. 2.",
                           "A line no. 3.",
                           "A line no. 4.",
                           "A line no. 5.",
                           "This is line no. 6.",
                           "This is line no. 7."]

    lines = []
    for i, (line_top, line_left, line_right, line_bottom, line_baseline_top, line_transcription) in (
            enumerate(zip(line_tops, line_lefts, line_rights, line_bottoms, line_baseline_tops, line_transcriptions))):
        line_polygon = np.array([[line_left, line_top],
                                 [line_right, line_top],
                                 [line_right, line_bottom],
                                 [line_left, line_bottom]])
        line_baseline = np.array([[line_left, line_baseline_top],
                                  [line_right, line_baseline_top]])
        lines.append(TextLine(id=f"line{i}", polygon=line_polygon, baseline=line_baseline, transcription=line_transcription))

    text_region_1_top = min(line_tops[:2])
    text_region_1_left = min(line_lefts[:2])
    text_region_1_right = max(line_rights[:2])
    text_region_1_bottom = max(line_bottoms[:2])

    text_region_2_top = min(line_tops[2:])
    text_region_2_left = min(line_lefts[2:])
    text_region_2_right = max(line_rights[2:])
    text_region_2_bottom = max(line_bottoms[2:])

    text_region_1_polygon = np.array([[text_region_1_left, text_region_1_top],
                                      [text_region_1_right, text_region_1_top],
                                      [text_region_1_right, text_region_1_bottom],
                                      [text_region_1_left, text_region_1_bottom]])

    text_region_2_polygon = np.array([[text_region_2_left, text_region_2_top],
                                      [text_region_2_right, text_region_2_top],
                                      [text_region_2_right, text_region_2_bottom],
                                      [text_region_2_left, text_region_2_bottom]])

    text_region_1 = RegionLayout(id="text_region_1", polygon=text_region_1_polygon, category="text")
    text_region_2 = RegionLayout(id="text_region_2", polygon=text_region_2_polygon, category="text")

    text_region_1.lines = lines[:2]
    text_region_2.lines = lines[2:]

    page_layout = AnnoPagePageLayout(id="test_page", page_size=(297, 210))
    page_layout.regions.append(text_region_1)
    page_layout.regions.append(text_region_2)
    page_layout.regions.append(initial_region)

    page_layout = engine.process_page(None, page_layout)

    generated_alto = page_layout.to_altoxml_string(version=ALTOVersion.ALTO_v4_4)

    assert_xml_equal(generated_alto, expected_alto)


def test_alto_initial_with_space():
    setup()

    resources_dir = os.path.join(os.path.dirname(__file__), "resources")
    expected_alto = load_xml(os.path.join(resources_dir, "alto_initial_with_space.xml"))

    def prepare_prompt_data(image, page_layout, region):
        continuing_line = [line for line in page_layout.lines_iterator() if line.id == "line2"][0]
        return None, None, continuing_line

    def process_initial(region, initial_crop, context_crop, continuing_line) -> LLMResult:
        return LLMResult(data=InitialRecognitionResult(initial="XXX", include_space=True))

    config_dict = {
        "categories": json.dumps(["initial"]),
        "max_attempts": "3",
        "api": "test",
        "api_key": "test_key",
        "prompt_settings": os.path.join(resources_dir, "initial_recognition_prompt.json")
    }

    config = ConfigParser()
    config.add_section("INITIAL_RECOGNITION")
    for key, value in config_dict.items():
        config.set("INITIAL_RECOGNITION", key, value)

    engine = InitialRecognitionEngine(config["INITIAL_RECOGNITION"], device="cpu", config_path="")
    engine._process_initial = process_initial
    engine._prepare_prompt_data = prepare_prompt_data

    initial_top = 50
    initial_left = 10
    initial_right = 65
    initial_bottom = 105

    initial_polygon = np.array([[initial_left, initial_top],
                                [initial_right, initial_top],
                                [initial_right, initial_bottom],
                                [initial_left, initial_bottom]])
    initial_region = AnnoPageRegionLayout(id="initial_region", polygon=initial_polygon, category="Initial", detection_confidence=0.9)
    initial_region.graphical_metadata = GraphicalObjectMetadata(
        tag_id="initial_001",
        mods_id="MODS_PICT_0001",
    )

    line_tops = [10, 30, 50, 70, 90, 110, 130]
    line_lefts = [10, 10, 70, 70, 70, 10, 10]
    line_rights = [200, 200, 200, 200, 200, 200, 200]
    line_bottoms = [25, 45, 65, 85, 105, 125, 145]

    line_baseline_tops = [bottom - 5 for bottom in line_bottoms]

    line_transcriptions = ["This is line no. 1.",
                           "This is line no. 2.",
                           "A line no. 3.",
                           "A line no. 4.",
                           "A line no. 5.",
                           "This is line no. 6.",
                           "This is line no. 7."]

    lines = []
    for i, (line_top, line_left, line_right, line_bottom, line_baseline_top, line_transcription) in (
            enumerate(zip(line_tops, line_lefts, line_rights, line_bottoms, line_baseline_tops, line_transcriptions))):
        line_polygon = np.array([[line_left, line_top],
                                 [line_right, line_top],
                                 [line_right, line_bottom],
                                 [line_left, line_bottom]])
        line_baseline = np.array([[line_left, line_baseline_top],
                                  [line_right, line_baseline_top]])
        lines.append(TextLine(id=f"line{i}", polygon=line_polygon, baseline=line_baseline, transcription=line_transcription))

    text_region_1_top = min(line_tops[:2])
    text_region_1_left = min(line_lefts[:2])
    text_region_1_right = max(line_rights[:2])
    text_region_1_bottom = max(line_bottoms[:2])

    text_region_2_top = min(line_tops[2:])
    text_region_2_left = min(line_lefts[2:])
    text_region_2_right = max(line_rights[2:])
    text_region_2_bottom = max(line_bottoms[2:])

    text_region_1_polygon = np.array([[text_region_1_left, text_region_1_top],
                                      [text_region_1_right, text_region_1_top],
                                      [text_region_1_right, text_region_1_bottom],
                                      [text_region_1_left, text_region_1_bottom]])

    text_region_2_polygon = np.array([[text_region_2_left, text_region_2_top],
                                      [text_region_2_right, text_region_2_top],
                                      [text_region_2_right, text_region_2_bottom],
                                      [text_region_2_left, text_region_2_bottom]])

    text_region_1 = RegionLayout(id="text_region_1", polygon=text_region_1_polygon, category="text")
    text_region_2 = RegionLayout(id="text_region_2", polygon=text_region_2_polygon, category="text")

    text_region_1.lines = lines[:2]
    text_region_2.lines = lines[2:]

    page_layout = AnnoPagePageLayout(id="test_page", page_size=(297, 210))
    page_layout.regions.append(text_region_1)
    page_layout.regions.append(text_region_2)
    page_layout.regions.append(initial_region)

    page_layout = engine.process_page(None, page_layout)

    generated_alto = page_layout.to_altoxml_string(version=ALTOVersion.ALTO_v4_4)

    assert_xml_equal(generated_alto, expected_alto)
