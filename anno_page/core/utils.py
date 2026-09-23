import os
import json
import logging
import numpy as np

logger = logging.getLogger(__name__)


def compose_path(file_path, reference_path):
    if reference_path and not os.path.isabs(file_path):
        file_path = os.path.join(reference_path, file_path)
    return file_path


def config_get_list(config, key, fallback=None, make_lowercase=False):
    if key not in config:
        return fallback

    try:
        value = json.loads(config[key])
    except json.decoder.JSONDecodeError as e:
        logger.info(f'Failed to parse list from config key "{key}", returning fallback {fallback}:\n{e}')
        return fallback

    if not isinstance(value, list):
        logger.info(f'Config key "{key}" is not a list (got {type(value).__name__}), returning fallback {fallback}.')
        return fallback

    if make_lowercase:
        value = [str(item).lower() if isinstance(item, str) else item for item in value]

    return value


def find_textline(print_space_element, line, namespaces):
    text_line_element = print_space_element.find(f".//TextLine[@ID='{line.id}']", namespaces)
    if text_line_element is None:
        text_line_element = find_textline_by_geometry_and_content(print_space_element, line, namespaces)

    return text_line_element


def find_textline_by_geometry_and_content(print_space_element, line, namespaces):
    result = None

    textline_elements = print_space_element.findall(".//TextLine", namespaces)
    for textline_element in textline_elements:
        vpos = int(textline_element.attrib.get("VPOS", 0))
        hpos = int(textline_element.attrib.get("HPOS", 0))
        width = int(textline_element.attrib.get("WIDTH", 0))
        height = int(textline_element.attrib.get("HEIGHT", 0))

        polygon = np.asarray([[hpos, vpos],
                              [hpos + width, vpos],
                              [hpos + width, vpos + height],
                              [hpos, vpos + height]])

        transcription = ""
        for child in textline_element.getchildren():
            if child.tag.endswith("String"):
                transcription += child.attrib.get("CONTENT", "")
            elif child.tag.endswith("SP"):
                transcription += " "

        transcription = transcription.strip()

        if np.all(polygon == line.polygon) and transcription == line.transcription:
            result = textline_element
            break

    return result
