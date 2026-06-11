import os
import cv2
import json
import base64
import requests

from json import JSONDecodeError
from jinja2 import Template
from pydantic import BaseModel, ValidationError
from shapely.geometry import Polygon

from anno_page.core.utils import compose_path, config_get_list
from anno_page.engines.base import LayoutProcessingEngine
from anno_page.core.metadata import GraphicalObjectMetadata


class InitialRecognitionResult(BaseModel):
    initial: str
    line_id: int|None
    include_space: bool
    text: str|None


class InitialRecognitionEngine(LayoutProcessingEngine):
    def __init__(self, config, device, config_path):
        super().__init__(config, device, config_path, requires_lines=True)

        self.categories = config_get_list(self.config, key="categories", fallback=["initial"], make_lowercase=True)
        self.scaling_factor = config.getfloat("scaling_factor", fallback=0.5)
        self.max_attempts = config.getint("max_attempts", fallback=3)

        self.api_key = self.config.get("api_key", None)
        api_key_path = compose_path(self.api_key, self.config_path)
        if os.path.exists(api_key_path):
            with open(api_key_path, 'r') as f:
                self.api_key = f.read().strip()

        self.llm_service_url_aliases = {}
        llm_service_url_aliases_path = self.config.get('llm_service_url_aliases', fallback=None)
        if llm_service_url_aliases_path is not None:
            self._load_llm_service_url_aliases(compose_path(llm_service_url_aliases_path, self.config_path))

        api = self.config["api"].lower()
        if api in self.llm_service_url_aliases and "completions" in self.llm_service_url_aliases[api]:
            self.api_url = self.llm_service_url_aliases[api]["completions"]
        else:
            self.api_url = api

        prompt_settings_path = compose_path(config.get("prompt_settings"), self.config_path)
        with open(prompt_settings_path, 'r') as f:
            self.prompt_settings = json.load(f)

        self.prompt_model = self.prompt_settings["model"]
        self.prompt_text = self._normalize_category_names(self.prompt_settings["text"])

    def _load_llm_service_url_aliases(self, path):
        with open(path, 'r') as f:
            url_aliases = json.load(f)

        for url_alias in url_aliases:
            for alias in url_alias["aliases"]:
                self.llm_service_url_aliases[alias.lower()] = url_alias["urls"]

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
                x_min, y_min, x_max, y_max = region.get_polygon_bounding_box()

                initial_crop = image[y_min:y_max, x_min:x_max]

                width = x_max - x_min
                height = y_max - y_min

                extended_x_min = round(x_min - width * self.scaling_factor)
                extended_x_max = round(x_max + width * self.scaling_factor)
                extended_y_min = round(y_min - height * self.scaling_factor)
                extended_y_max = round(y_max + height * self.scaling_factor)

                bounding_box = [[extended_x_min, extended_y_min],
                                [extended_x_max, extended_y_min],
                                [extended_x_max, extended_y_max],
                                [extended_x_min, extended_y_max]]

                initial_polygon = Polygon(bounding_box)

                nearby_lines = []
                for line in page_layout.lines_iterator():
                    line_polygon = Polygon(line.polygon)
                    if line_polygon.intersects(initial_polygon):
                        nearby_lines.append(line)

                context_x_min = round(min([point[0] for line in nearby_lines for point in line.polygon]) - width * self.scaling_factor)
                context_x_max = round(max([point[0] for line in nearby_lines for point in line.polygon]) + width * self.scaling_factor)
                context_y_min = round(min([point[1] for line in nearby_lines for point in line.polygon]) - height * self.scaling_factor)
                context_y_max = round(max([point[1] for line in nearby_lines for point in line.polygon]) + height * self.scaling_factor)

                context_crop = image[context_y_min:context_y_max, context_x_min:context_x_max]
                context_lines = [{"line_id": i, "transcription": line.transcription} for i, line in enumerate(nearby_lines)]

                initial_result = self.process_initial(region, initial_crop, context_crop, context_lines)

                if initial_result is not None:
                    region.transcription = initial_result.initial
                    if initial_result.include_space:
                        region.transcription += " "

                    metadata: GraphicalObjectMetadata = region.graphical_metadata
                    if metadata is not None:
                        metadata.tag_description = initial_result.text
                        if initial_result.line_id is not None and initial_result.line_id < len(nearby_lines):
                            metadata.continuation_line = nearby_lines[initial_result.line_id]
                            self.logger.info(f"Setting continuation line to {nearby_lines[initial_result.line_id].id}")

        return page_layout

    def process_initial(self, region, initial_crop, context_crop, context_lines) -> InitialRecognitionResult|None:
        example_output = InitialRecognitionResult(initial="X", line_id=42, include_space=True,
                                                  text="X is the twenty-fourth letter of the Latin alphabet.")

        prompt_template = self.prompt_text
        if isinstance(prompt_template, dict):
            prompt_template = prompt_template[region.category.lower()]
        prompt_template = Template(prompt_template)

        prompt_text = prompt_template.render(example_output=example_output.model_dump_json(indent=4),
                                             context_lines=json.dumps(context_lines, indent=4))

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
