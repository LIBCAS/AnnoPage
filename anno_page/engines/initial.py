import os
import cv2
import json
import numpy as np
import base64
import requests

from json import JSONDecodeError
from jinja2 import Template
from pydantic import BaseModel, ValidationError
from shapely.geometry import Polygon

from anno_page.core.utils import compose_path, config_get_list
from anno_page.engines.base import LayoutProcessingEngine
from anno_page.core.metadata import GraphicalObjectMetadata
from anno_page.core.llm_api_aliases import get_llm_api_aliases


class InitialRecognitionResult(BaseModel):
    initial: str
    include_space: bool


class InitialRecognitionEngine(LayoutProcessingEngine):
    def __init__(self, config, device, config_path):
        super().__init__(config, device, config_path, requires_lines=True)

        self.categories = config_get_list(self.config, key="categories", fallback=["initial"], make_lowercase=True)
        self.max_attempts = config.getint("max_attempts", fallback=3)

        self.top_down_target_coefficient = 0.0
        self.left_right_target_coefficient = 2
        self.top_down_context_coefficient = 1.0
        self.left_right_context_coefficient = 1.0

        llm_api_aliases = get_llm_api_aliases()
        api = self.config["api"].lower()
        if api in llm_api_aliases and "completions" in llm_api_aliases[api]:
            self.api_url = llm_api_aliases[api]["completions"]
        else:
            self.api_url = api

        self.api_key = self.config.get("api_key", None)
        api_key_path = compose_path(self.api_key, self.config_path)
        if os.path.exists(api_key_path):
            with open(api_key_path, 'r') as f:
                self.api_key = f.read().strip()

        prompt_settings_path = compose_path(config.get("prompt_settings"), self.config_path)
        with open(prompt_settings_path, 'r') as f:
            self.prompt_settings = json.load(f)

        self.prompt_model = self.prompt_settings["model"]
        self.prompt_text = self._normalize_category_names(self.prompt_settings["text"])

    @staticmethod
    def _normalize_category_names(prompt_text):
        if type(prompt_text) == dict:
            normalized_prompt = {}
            for category, text in prompt_text.items():
                normalized_category = category.lower()
                normalized_prompt[normalized_category] = text
            return normalized_prompt
        else:
            return prompt_text

    def process_page(self, image, page_layout):
        for region in page_layout.regions:
            if region.category is None or region.category.lower() == "text":
                continue

            if self.categories is None or region.category.lower() in self.categories:
                initial_crop, context_crop, continuing_line = self._prepare_prompt_data(image, page_layout, region)

                result = self._process_initial(region, initial_crop, context_crop, continuing_line)

                if result is not None:
                    region.transcription = result.initial
                    if result.include_space:
                        region.transcription += " "

                    metadata: GraphicalObjectMetadata = region.graphical_metadata
                    if metadata is not None:
                        metadata.tag_description = result.initial
                        metadata.continuing_line = continuing_line

        return page_layout

    def _process_initial(self, region, initial_crop, context_crop, continuing_line) -> InitialRecognitionResult|None:
        example_output = InitialRecognitionResult(initial="X", include_space=True)

        prompt_template = self.prompt_text
        if isinstance(prompt_template, dict):
            prompt_template = prompt_template[region.category.lower()]
        prompt_template = Template(prompt_template)

        prompt_text = prompt_template.render(example_output=example_output.model_dump_json(indent=4),
                                             continuing_line=continuing_line.transcription)

        request_args = {
            "model": self.prompt_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_text
                        },
                        {
                            "type": "image_url",
                            "image_url": "data:image/jpeg;base64," + base64.b64encode(cv2.imencode('.jpg', initial_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])[1]).decode('utf-8')
                        },
                        {
                            "type": "image_url",
                            "image_url": "data:image/jpeg;base64," + base64.b64encode(cv2.imencode('.jpg', context_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])[1]).decode('utf-8')
                        }
                    ]
                }
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "response_schema",
                    "strict": True,
                    "schema": InitialRecognitionResult.model_json_schema()
                },
            },
            "reasoning_effort": "low"
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        for _ in range(self.max_attempts):
            response = requests.post(self.api_url, headers=headers, json=request_args)
            result = response.json()["choices"][0]["message"]["content"]

            try:
                result_json = json.loads(result)
                initial_result = InitialRecognitionResult.model_validate(result_json)
                self.logger.info(f"Successfully parsed initial result for region {region.id}")
                return initial_result
            except JSONDecodeError:
                self.logger.info(f"Failed to parse JSON for region {region.id}: {response.text}")
            except ValidationError:
                self.logger.info(f"Initial result for region {region.id} does not conform to expected format: {result_json}")
            except Exception as e:
                self.logger.info(f"Exception for region {region.id}: {e}")

        self.logger.warning(f"Could not get valid result for region {region.id} after {self.max_attempts} attempts")
        return None

    def _prepare_prompt_data(self, image, page_layout, region):
        region_bbox = region.get_polygon_bounding_box()
        x_min, y_min, x_max, y_max = region_bbox

        median_line_height = np.median([sum(line.heights) for line in page_layout.lines_iterator()])

        initial_crop = image[y_min:y_max, x_min:x_max]
        nearby_lines, continuing_line = self._get_initial_lines(page_layout, region_bbox, median_line_height)

        context_crop = self._get_context_crop(image, region_bbox, nearby_lines, median_line_height)

        return initial_crop, context_crop, continuing_line

    def _get_initial_lines(self, page_layout, region_bbox, median_line_height):
        x_min, y_min, x_max, y_max = region_bbox
        width = x_max - x_min



        target_top = y_min - median_line_height * self.top_down_target_coefficient
        target_bottom = y_max + median_line_height * self.top_down_target_coefficient
        target_left = x_min + width / 2
        target_right = x_max + median_line_height * self.left_right_target_coefficient

        target_polygon = Polygon([[target_left, target_top],
                                  [target_right, target_top],
                                  [target_right, target_bottom],
                                  [target_left, target_bottom]])

        nearby_lines = []
        for line in page_layout.lines_iterator():
            line_polygon = Polygon(line.polygon)
            if line_polygon.intersects(target_polygon):
                nearby_lines.append(line)

        continuing_line = None
        continuing_line_baseline_point = None
        for line in nearby_lines:
            left_baseline_point = sorted(line.baseline, key=lambda pair: pair[0])[0]

            if target_left < left_baseline_point[0] < target_right and target_top < left_baseline_point[1] < target_bottom:
                if continuing_line_baseline_point is None or continuing_line_baseline_point[1] > left_baseline_point[1]:
                    continuing_line = line
                    continuing_line_baseline_point = left_baseline_point

        return nearby_lines, continuing_line

    def _get_context_crop(self, image, region_bbox, nearby_lines, median_line_height):
        x_min, y_min, x_max, y_max = region_bbox

        context_x_min = round(min(x_min, min([point[0] for line in nearby_lines for point in line.polygon])) - median_line_height * self.left_right_context_coefficient)
        context_x_max = round(max(x_max, max([point[0] for line in nearby_lines for point in line.polygon])) + median_line_height * self.left_right_context_coefficient)
        context_y_min = round(min(y_min, min([point[1] for line in nearby_lines for point in line.polygon])) - median_line_height * self.top_down_context_coefficient)
        context_y_max = round(max(y_min, max([point[1] for line in nearby_lines for point in line.polygon])) + median_line_height * self.top_down_context_coefficient)

        context_crop = image[context_y_min:context_y_max, context_x_min:context_x_max]

        return context_crop
