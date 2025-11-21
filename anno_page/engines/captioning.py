import os
import cv2
import json
import torch
import base64
import requests
import numpy as np

from jinja2 import Template
from multiprocessing import Pool

from pero_ocr.utils import compose_path, config_get_list

from anno_page.core.metadata import RelatedLinesMetadata
from anno_page.engines import LayoutProcessingEngine
from anno_page.engines.detection import YoloDetector
from anno_page.enums import Language, LineRelation
from anno_page.engines.helpers import find_nearest_region, find_lines_in_bbox


class CaptionYoloNearestEngine(LayoutProcessingEngine):
    def __init__(self, config, device, config_path):
        super().__init__(config, device, config_path, requires_lines=True)

        self.detector = YoloDetector(model_path=compose_path(self.config["YOLO_PATH"], self.config_path),
                                     device=self.device,
                                     detection_threshold=self.config.getfloat("YOLO_DETECTION_THRESHOLD", 0.2),
                                     image_size=self.config.getint("YOLO_IMAGE_SIZE", 640))

    def process_page(self, page_image, page_layout):
        yolo_result = self.detector(page_image)
        captions = yolo_result.boxes.xyxy.cpu().numpy().astype(np.int32).tolist()

        if len(captions) == 0:
            self.logger.info("No captions detected by YOLO engine.")
            return page_layout

        for caption in captions:
            caption_lines = find_lines_in_bbox(caption, page_layout, threshold=0.5)
            caption_lines_text = " ".join([line.transcription for line in caption_lines if line.transcription])

            linked_region = find_nearest_region(caption, page_layout, categories=["Image", "Photograph"])

            caption_lines_metadata = RelatedLinesMetadata(tag_id=f"fc.{linked_region.id}",
                                                          mods_id=f"{linked_region.graphical_metadata.mods_id}_CAPTION_0001",
                                                          lines=caption_lines,
                                                          relation=LineRelation.CAPTION,
                                                          description=caption_lines_text,
                                                          title=caption_lines_text)

            linked_region.graphical_metadata.caption_lines_metadata = caption_lines_metadata

        return page_layout


class CaptionYoloKeypointsEngine(LayoutProcessingEngine):
    def __init__(self, config, device, config_path):
        super().__init__(config, device, config_path, requires_lines=True)

        self.detector = YoloDetector(model_path=compose_path(self.config["YOLO_PATH"], self.config_path),
                                     device=self.device,
                                     detection_threshold=self.config.getfloat("YOLO_DETECTION_THRESHOLD", 0.2),
                                     image_size=self.config.getint("YOLO_IMAGE_SIZE", 640))

        self.yolo_keypoint_threshold = self.config.getfloat("yolo_keypoint_threshold", fallback=0.5)

    def process_page(self, page_image, page_layout):
        yolo_result = self.detector(page_image)

        captions = yolo_result.boxes.xyxy.cpu().numpy().astype(np.int32).tolist()
        captions_keypoints = yolo_result.keypoints.xy.cpu().numpy().astype(np.int32).tolist()

        captions_keypoints_confs = yolo_result.keypoints.conf
        if captions_keypoints_confs is not None:
            captions_keypoints_confs = captions_keypoints_confs.cpu().numpy().astype(np.float32).tolist()
        else:
            captions_keypoints_confs = []

        if len(captions) == 0:
            self.logger.info("No captions detected by YOLO engine.")
            return page_layout

        for caption, caption_keypoints, caption_keypoints_confs in zip(captions, captions_keypoints, captions_keypoints_confs):
            caption_lines = find_lines_in_bbox(caption, page_layout, threshold=0.5)
            caption_lines_text = " ".join([line.transcription for line in caption_lines if line.transcription])

            for caption_keypoint, caption_keypoint_conf in zip(caption_keypoints, caption_keypoints_confs):
                if caption_keypoint_conf >= self.yolo_keypoint_threshold:
                    x, y = caption_keypoint
                    linked_region = find_nearest_region((x, y, x, y), page_layout, categories=["Image", "Photograph"])
                    if linked_region is not None:
                        caption_lines_metadata = RelatedLinesMetadata(tag_id=f"fc.{linked_region.id}",
                                                                    mods_id=f"{linked_region.graphical_metadata.mods_id}_CAPTION_0001",
                                                                    lines=caption_lines,
                                                                    relation=LineRelation.CAPTION,
                                                                    description=caption_lines_text,
                                                                    title=caption_lines_text)

                        linked_region.graphical_metadata.caption_lines_metadata = caption_lines_metadata

        return page_layout


class CaptionYoloOrganizerEngine(LayoutProcessingEngine):
    def __init__(self, config, device, config_path):
        super().__init__(config, device, config_path, requires_lines=True)

        self.detector = YoloDetector(model_path=compose_path(self.config["YOLO_PATH"], self.config_path),
                                     device=self.device,
                                     detection_threshold=self.config.getfloat("YOLO_DETECTION_THRESHOLD", 0.2),
                                     image_size=self.config.getint("YOLO_IMAGE_SIZE", 640))

        self.caption_organizer = CaptionOrganizer(model_path=compose_path(self.config['organizer_path'], self.config_path),
                                                  device=self.device,
                                                  categories=config_get_list(self.config, key="organizer_categories", fallback=[]))

    def process_page(self, page_image, page_layout):
        yolo_result = self.detector(page_image)
        captions = yolo_result.boxes.xyxy.cpu().numpy().astype(np.int32).tolist()

        if len(captions) == 0:
            self.logger.info("No captions detected by YOLO engine.")
            return page_layout

        regions = [region for region in page_layout.regions if region.category not in ("text", None)]
        assignment = self.caption_organizer.assign_captions_to_regions(regions, captions, page_image)

        for linked_region, caption in assignment:
            caption_lines = find_lines_in_bbox(caption, page_layout, threshold=0.5)
            caption_lines_text = " ".join([line.transcription for line in caption_lines if line.transcription])

            caption_lines_metadata = RelatedLinesMetadata(tag_id=f"fc.{linked_region.id}",
                                                          mods_id=f"{linked_region.graphical_metadata.mods_id}_CAPTION_0001",
                                                          lines=caption_lines,
                                                          relation=LineRelation.CAPTION,
                                                          description=caption_lines_text,
                                                          title=caption_lines_text)

            linked_region.graphical_metadata.caption_lines_metadata = caption_lines_metadata

        return page_layout


class CaptionOrganizer:
    def __init__(self, model_path, device, categories):
        self.model_path = model_path
        self.device = device
        self.categories = categories

        self.model = torch.jit.load(self.model_path).to(self.device)

    def assign_captions_to_regions(self, regions, captions, page_image):
        bboxes, query_types = self.prepare_input_data(regions, captions, page_image)
        bboxes = bboxes.to(self.device)
        query_types = query_types.to(self.device)

        with torch.no_grad():
            relation_matrix = self.model(bboxes, query_types).cpu().numpy()[0]

        assignment = []
        for region_index, region in enumerate(regions):
            target_index = np.argmax(relation_matrix[:, region_index])
            if target_index >= len(regions):
                caption_index = target_index - len(regions)
                caption = captions[caption_index]
                assignment.append((region, caption))

        return assignment

    def prepare_input_data(self, regions, captions, page_image, target_length=64):
        h, w = page_image.shape[:2]

        bboxes = torch.zeros((1, target_length, 4), dtype=torch.float32)
        query_types = torch.full((1, target_length), self.categories.index("Padding"), dtype=torch.int64)

        for region_index, region in enumerate(regions):
            if region.category in self.categories:
                x1, y1, x2, y2 = region.get_polygon_bounding_box()
                bboxes[0, region_index] = torch.tensor([x1/w, y1/h, x2/w, y2/h], dtype=torch.float32)
                query_types[0, region_index] = self.categories.index(region.category)

        for caption_index, caption in enumerate(captions):
            matrix_caption_index = caption_index + len(regions)
            x1, y1, x2, y2 = caption
            bboxes[0, matrix_caption_index] = torch.tensor([x1/w, y1/h, x2/w, y2/h], dtype=torch.float32)
            query_types[0, matrix_caption_index] = self.categories.index("Image caption")

        return bboxes, query_types


class PromptData:
    def __init__(self, image=None, region=None, metadata=None, prompt=None, result=None):
        self.image = image
        self.region = region
        self.metadata = metadata
        self.prompt = prompt
        self.result = result


class ChatGPTImageCaptioningEngine(LayoutProcessingEngine):
    def __init__(self, config, device, config_path):
        super().__init__(config, device, config_path)

        self.api_key = self.config["api_key"]
        self.max_image_size = self.config.getint('max_image_size', fallback=None)
        self.categories = config_get_list(self.config, key="categories", fallback=None) if "categories" in self.config else None
        self.num_processes = self.config.getint('num_processes', fallback=1)
        self.prompt_settings = compose_path(self.config["prompt_settings"], self.config_path)
        self.max_attempts = self.config.getint('max_attempts', fallback=3)

        api_key_path = compose_path(self.api_key, self.config_path)
        if os.path.exists(api_key_path):
            with open(api_key_path, 'r') as f:
                self.api_key = f.read().strip()

        with open(self.prompt_settings, 'r') as f:
            prompt_settings = json.load(f)

        self.prompt_model = prompt_settings.get("model", "gpt-4o-mini")
        self.prompt_text = prompt_settings["text"]
        self.prompt_max_tokens = prompt_settings.get("max_tokens", 500)

    def process_page(self, page_image, page_layout):
        data = []

        for region in page_layout.regions:
            if (self.categories is None and region.category not in (None, 'text')) or (self.categories is not None and region.category in self.categories):
                image = self.crop_region_image(page_image, region)

                if image.size == 0:
                    self.logger.warning(f"Empty region detected {region.id} ({region.category}), skipping captioning.")

                else:
                    data.append(self.prepare_prompt_data(image, region, page_layout))

        current_attempt = 0

        unfinished_data = data
        while len(unfinished_data) > 0 and current_attempt < self.max_attempts:
            self.process_elements(unfinished_data)
            self.process_image_captions(unfinished_data)

            current_attempt += 1

            unfinished_data = [item for item in data if item.result is None]
            self.logger.info(f"Captioning attempt {current_attempt} completed, {len(unfinished_data)} item{'s' if len(unfinished_data) > 1 else ''} remaining.")

        return page_layout

    def crop_region_image(self, page_image, region):
        x1, y1, x2, y2 = region.get_polygon_bounding_box()

        original_width = x2 - x1
        original_height = y2 - y1

        image = page_image[y1:y2, x1:x2]

        if self.max_image_size is not None and image.size > 0:
            if original_width > self.max_image_size or original_height > self.max_image_size:
                if original_width > original_height:
                    image = cv2.resize(image, (self.max_image_size,
                                               round(self.max_image_size * original_height / original_width)))
                else:
                    image = cv2.resize(image, (round(self.max_image_size * original_width / original_height),
                                               self.max_image_size))

        return image

    def prepare_prompt_data(self, image, region, page_layout):
        if type(self.prompt_text) == dict:
            if region.category in self.prompt_text:
                prompt_template = Template(self.prompt_text[region.category])
            elif "default" in self.prompt_text:
                prompt_template = Template(self.prompt_text["default"])
            else:
                raise ValueError(f"No prompt template found for category '{region.category}' and no default template provided.")
        else:
            prompt_template = Template(self.prompt_text)

        page_metadata = page_layout.metadata.get("anno_page_metadata", None)

        prompt = prompt_template.render(page_metadata if page_metadata else [], category=region.category)

        return PromptData(
            image=image,
            region=region,
            metadata=page_metadata,
            prompt=prompt
        )

    def process_elements(self, data: list[PromptData]):
        if self.num_processes > 1:
            self.logger.debug(f"Processing image captions in parallel using {self.num_processes} processes.")
            with Pool(self.num_processes) as p:
                image_captions = p.map(self.generate_image_caption, data)

            for item, image_caption in zip(data, image_captions):
                item.result = image_caption
        else:
            self.logger.debug("Processing image captions sequentially.")
            for item in data:
                image_caption = self.generate_image_caption(item)
                item.result = image_caption

    def generate_image_caption(self, item: PromptData):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        self.logger.debug(f"Generating caption for region {item.region.id} with prompt: {item.prompt}")

        payload = {
            "model": self.prompt_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": item.prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{self.encode_image(item.image)}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": self.prompt_max_tokens
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        try:
            image_caption = response.json()["choices"][0]["message"]["content"]
            self.logger.info(f"Successfully generated caption for region {item.region.id}: {image_caption}")
        except:
            image_caption = None
            self.logger.warning(f"Failed to get caption from API for region {item.region.id}: {response.text}")

        return image_caption

    def process_image_captions(self, data: list[PromptData]):
        for item in data:
            if item.result is None:
                self.logger.debug(f"No caption result for region {item.region.id}, skipping processing.")
                continue

            try:
                image_caption_json = json.loads(item.result)
            except:
                self.logger.error(f"Failed to parse image caption JSON for region {item.region.id}: {item.result}")
                item.result = None
                continue

            if len(image_caption_json) == 0:
                self.logger.warning(f"Empty image caption JSON for region {item.region.id}: {item.result}")
                item.result = None
                continue

            item.region.graphical_metadata.caption = {
                Language.ENGLISH: image_caption_json.get("caption_en", None),
                Language.CZECH: image_caption_json.get("caption_cz", None)
            }

            item.region.graphical_metadata.topics = {
                Language.ENGLISH: image_caption_json.get("topics_en", None),
                Language.CZECH: image_caption_json.get("topics_cz", None)
            }

            item.region.graphical_metadata.color = {
                Language.ENGLISH: image_caption_json.get("color_en", None),
                Language.CZECH: image_caption_json.get("color_cz", None)
            }

            self.logger.info(f"Successfully processed caption for region {item.region.id}")

    @staticmethod
    def encode_image(image):
        image_jpg = cv2.imencode('.jpg', image)[1]
        image_base64 = base64.b64encode(image_jpg).decode('utf-8')
        return image_base64

