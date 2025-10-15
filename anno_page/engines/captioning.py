import os
import cv2
import json
import torch
import base64
import logging
import requests
import numpy as np

from multiprocessing import Pool

from pero_ocr.utils import compose_path, config_get_list
from pero_ocr.layout_engines.cnn_layout_engine import LayoutEngineYolo

from anno_page.core.metadata import GraphicalObjectMetadata, RelatedLinesMetadata
from anno_page.enums.language import Language
from anno_page.enums.line_relation import LineRelation
from anno_page.engines.helpers import find_nearest_region, find_lines_in_bbox


class DummyImageCaptioning:
    def __init__(self, config, device, config_path):
        self.device = device

        self.categories = config["categories"] if "categories" in config else None
        np.random.seed(42)

        self.logger = logging.getLogger(__name__)

    def process_page(self, page_image, page_layout):
        text_lines = list(page_layout.lines_iterator(["text", None]))

        if self.categories is not None:
            for region in page_layout.regions:
                if region.category in self.categories:
                    caption_lines_metadata = None
                    reference_lines_metadata = None

                    if len(text_lines) > 0:
                        np.random.shuffle(text_lines)
                        caption_lines = text_lines[:np.random.randint(2, 5)]
                        self.logger.info(f"Caption lines for {region.id}: {len(caption_lines)}")

                        np.random.shuffle(text_lines)
                        reference_lines = text_lines[:np.random.randint(3, 8)]
                        print(f"Reference lines for {region.id}: {len(reference_lines)}")

                        reference_lines_text = " ".join([line.transcription for line in reference_lines if line.transcription])
                        caption_lines_text = " ".join([line.transcription for line in caption_lines if line.transcription])

                        reference_lines_metadata = RelatedLinesMetadata(tag_id=f"rtf.{region.id}",
                                                                        mods_id=f"MODS_{region.id}_RELATED_0001",
                                                                        lines=reference_lines,
                                                                        relation=LineRelation.REFERENCE,
                                                                        description=reference_lines_text,
                                                                        title=reference_lines_text)

                        caption_lines_metadata = RelatedLinesMetadata(tag_id=f"fc.{region.id}",
                                                                      mods_id=f"MODS_{region.id}_CAPTION_0001",
                                                                      lines=caption_lines,
                                                                      relation=LineRelation.CAPTION,
                                                                      description=caption_lines_text,
                                                                      title=caption_lines_text)

                        for reference_line in reference_lines:
                            reference_line.metadata = [reference_lines_metadata]

                        for caption_line in caption_lines:
                            caption_line.metadata = [caption_lines_metadata]

                    metadata = GraphicalObjectMetadata(tag_id=region.id,
                                                       mods_id=f"MODS_{region.id}",
                                                       caption={
                                                           Language.ENGLISH: "This is a caption",
                                                           Language.CZECH: "Toto je popis"
                                                       },
                                                       topics={
                                                           Language.ENGLISH: "Here, are, the, topics",
                                                           Language.CZECH: "Tady, jsou, témata"
                                                       },
                                                       color={
                                                           Language.ENGLISH: "color",
                                                           Language.CZECH: "barva"
                                                       },
                                                       description="Object description",
                                                       title="Object title",
                                                       reference_lines_metadata=reference_lines_metadata,
                                                       caption_lines_metadata=caption_lines_metadata)

                    if region.metadata is None:
                        region.metadata = metadata
                    else:
                        region.metadata.update(metadata)

        return page_layout


class CaptionYoloNearestEngine:
    def __init__(self, config, device, config_path):
        self.device = device
        self.yolo_engine = LayoutEngineYolo(model_path=compose_path(config['yolo_path'], config_path),
                                            device=device,
                                            image_size=config.get("yolo_image_size", fallback=None),
                                            detection_threshold=config.getfloat("yolo_detection_threshold", fallback=0.2))

        self.logger = logging.getLogger(__name__)

    def process_page(self, page_image, page_layout):
        yolo_result = self.yolo_engine.detect(page_image)
        captions = yolo_result.boxes.xyxy.cpu().numpy().astype(np.int32).tolist()

        if len(captions) == 0:
            self.logger.info("No captions detected by YOLO engine.")
            return page_layout

        for caption in captions:
            caption_lines = find_lines_in_bbox(caption, page_layout, threshold=0.5)
            caption_lines_text = " ".join([line.transcription for line in caption_lines if line.transcription])

            linked_region = find_nearest_region(caption, page_layout, categories=["Image", "Photograph"])

            caption_lines_metadata = RelatedLinesMetadata(tag_id=f"fc.{linked_region.id}",
                                                          mods_id=f"MODS_{linked_region.id}_CAPTION_0001",
                                                          lines=caption_lines,
                                                          relation=LineRelation.CAPTION,
                                                          description=caption_lines_text,
                                                          title=caption_lines_text)

            metadata = GraphicalObjectMetadata(tag_id=linked_region.id,
                                               mods_id=f"MODS_{linked_region.id}",
                                               caption_lines_metadata=caption_lines_metadata)

            if linked_region.metadata is None:
                linked_region.metadata = metadata
            else:
                linked_region.metadata.update(metadata)

        return page_layout


class CaptionYoloKeypointsEngine:
    def __init__(self, config, device, config_path):
        self.device = device
        self.yolo_engine = LayoutEngineYolo(model_path=compose_path(config['yolo_path'], config_path),
                                            device=device,
                                            image_size=config.get("yolo_image_size", fallback=None),
                                            detection_threshold=config.getfloat("yolo_detection_threshold", fallback=0.2))

        self.yolo_keypoint_threshold = config.getfloat("yolo_keypoint_threshold", fallback=0.5)

        self.logger = logging.getLogger(__name__)

    def process_page(self, page_image, page_layout):
        yolo_result = self.yolo_engine.detect(page_image)

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
                    region = find_nearest_region((x, y, x, y), page_layout, categories=["Image", "Photograph"])
                    if region is not None:
                        caption_lines_metadata = RelatedLinesMetadata(tag_id=f"fc.{region.id}",
                                                                      mods_id=f"MODS_{region.id}_CAPTION_0001",
                                                                      lines=caption_lines,
                                                                      relation=LineRelation.CAPTION,
                                                                      description=caption_lines_text,
                                                                      title=caption_lines_text)

                        metadata = GraphicalObjectMetadata(tag_id=region.id,
                                                           mods_id=f"MODS_{region.id}",
                                                           caption_lines_metadata=caption_lines_metadata)

                        if region.metadata is None:
                            region.metadata = metadata
                        else:
                            region.metadata.update(metadata)

        return page_layout


class CaptionYoloOrganizerEngine:
    def __init__(self, config, device, config_path):
        self.device = device
        self.yolo_engine = LayoutEngineYolo(model_path=compose_path(config['yolo_path'], config_path),
                                            device=device,
                                            image_size=config.get("yolo_image_size", fallback=None),
                                            detection_threshold=config.getfloat("yolo_detection_threshold", fallback=0.2))

        self.caption_organizer = CaptionOrganizer(model_path=compose_path(config['organizer_path'], config_path),
                                                  device=self.device,
                                                  categories=config_get_list(config, key="organizer_categories", fallback=[]))

        self.logger = logging.getLogger(__name__)

    def process_page(self, page_image, page_layout):
        yolo_result = self.yolo_engine.detect(page_image)
        captions = yolo_result.boxes.xyxy.cpu().numpy().astype(np.int32).tolist()

        if len(captions) == 0:
            self.logger.info("No captions detected by YOLO engine.")
            return page_layout

        regions = [region for region in page_layout.regions if region.category not in ("text", None)]
        assignment = self.caption_organizer.assign_captions_to_regions(regions, captions, page_image)

        for region, caption in assignment:
            caption_lines = find_lines_in_bbox(caption, page_layout, threshold=0.5)
            caption_lines_text = " ".join([line.transcription for line in caption_lines if line.transcription])
            caption_lines_metadata = RelatedLinesMetadata(tag_id=f"fc.{region.id}",
                                                          mods_id=f"MODS_{region.id}_CAPTION_0001",
                                                          lines=caption_lines,
                                                          relation=LineRelation.CAPTION,
                                                          description=caption_lines_text,
                                                          title=caption_lines_text)

            metadata = GraphicalObjectMetadata(tag_id=region.id,
                                               mods_id=f"MODS_{region.id}",
                                               caption_lines_metadata=caption_lines_metadata)

            if region.metadata is None:
                region.metadata = metadata
            else:
                region.metadata.update(metadata)

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


class ChatGPTImageCaptioning:
    def __init__(self, config, device, config_path):
        self.config_path = config_path

        self.api_key = config["api_key"]
        self.max_image_size = config.getint('max_image_size', fallback=None)
        self.categories = config_get_list(config, key="categories", fallback=None) if "categories" in config else None
        self.num_processes = config.getint('num_processes', fallback=1)
        self.prompt_settings = compose_path(config["prompt_settings"], self.config_path)
        self.max_attempts = config.getint('max_attempts', fallback=3)

        api_key_path = os.path.join(self.config_path, self.api_key)
        if os.path.exists(api_key_path):
            with open(api_key_path, 'r') as f:
                self.api_key = f.read().strip()

        with open(self.prompt_settings, 'r') as f:
            prompt_settings = json.load(f)
        self.prompt_model = prompt_settings.get("model", "gpt-4o-mini")
        self.prompt_text = prompt_settings["text"]
        self.prompt_max_tokens = prompt_settings.get("max_tokens", 500)

        self.logger = logging.getLogger(__name__)

    def process_page(self, page_image, page_layout):
        regions = []
        images = []

        for region in page_layout.regions:
            if self.categories is None and region.category not in (None, 'text') or region.category in self.categories:
                image = self.crop_region_image(page_image, region)

                if image.size == 0:
                    self.logger.warning(f"Empty region detected {region.id} ({region.category}): {x1},{y1} {x2},{y2}")

                else:
                    images.append(image)
                    regions.append(region)

        current_attempt = 0
        while len(images) > 0 and current_attempt < self.max_attempts:
            image_captions = self.process_images_and_regions(images, regions)

            current_attempt += 1
            next_attempt_images = []
            next_attempt_regions = []

            for image, region, image_caption in zip(images, regions, image_captions):
                result = self.process_image_caption(region, image_caption)
                if not result:
                    next_attempt_images.append(image)
                    next_attempt_regions.append(region)

            images = next_attempt_images
            regions = next_attempt_regions

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

    def process_images_and_regions(self, images, regions):
        if self.num_processes > 1:
            with Pool(self.num_processes) as p:
                image_captions = p.map(self.generate_image_caption, images)
        else:
            image_captions = [self.generate_image_caption(image) for image in images]

        return image_captions

    def generate_image_caption(self, image):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.prompt_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.prompt_text
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{self.encode_image(image)}"
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
        except:
            image_caption = ""

        return image_caption

    def process_image_caption(self, region, image_caption):
        try:
            image_caption_json = json.loads(image_caption)
        except json.JSONDecodeError:
            self.logger.error(f"Failed to parse image caption JSON for region {region.id}: {image_caption}")
            return False

        if len(image_caption_json) == 0:
            self.logger.warning(f"Empty image caption JSON for region {region.id}: {image_caption}")
            return False

        caption_en = image_caption_json.get("caption_en", None)
        caption_cz = image_caption_json.get("caption_cz", None)
        topics_en = image_caption_json.get("topics_en", None)
        topics_cz = image_caption_json.get("topics_cz", None)
        color_en = image_caption_json.get("color_en", None)
        color_cz = image_caption_json.get("color_cz", None)

        # TODO: jak se ma system chovat, kdyz metadata pro region uz existuji? maji se jen doplnit, nebo prepsat?
        metadata = GraphicalObjectMetadata(tag_id=region.id,
                                           mods_id=f"MODS_{region.id}",
                                           caption={
                                               Language.ENGLISH: caption_en,
                                               Language.CZECH: caption_cz
                                           },
                                           topics={
                                               Language.ENGLISH: topics_en,
                                               Language.CZECH: topics_cz
                                           },
                                           color={
                                               Language.ENGLISH: color_en,
                                               Language.CZECH: color_cz
                                           },
                                           caption_lines_metadata=None,
                                           reference_lines_metadata=None)

        if region.metadata is None:
            region.metadata = metadata
        else:
            region.metadata.update(metadata)

        self.logger.info(f"Successfully processed caption for region {region.id}")

        return True

    @staticmethod
    def encode_image(image):
        image_jpg = cv2.imencode('.jpg', image)[1]
        image_base64 = base64.b64encode(image_jpg).decode('utf-8')
        return image_base64

