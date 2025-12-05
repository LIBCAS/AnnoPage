import numpy as np

import uuid
from ultralytics import YOLO
from pero_ocr.core.layout import RegionLayout

from anno_page.core.utils import compose_path, config_get_list
from anno_page.core.metadata import GraphicalObjectMetadata
from anno_page.engines import LayoutProcessingEngine
from anno_page.enums import Category, Language


class YoloDetectionEngine(LayoutProcessingEngine):
    def __init__(self, config, device, config_path):
        super().__init__(config, device, config_path)

        self.detector = YoloDetector(model_path=compose_path(self.config["MODEL_PATH"], self.config_path),
                                     device=self.device,
                                     detection_threshold=self.config.getfloat("DETECTION_THRESHOLD", 0.2),
                                     image_size=self.config.getint("IMAGE_SIZE", 640))

        self.categories = config_get_list(self.config, key="categories", fallback=None)

    def process_page(self, page_image, page_layout):
        results = self.detector(page_image)

        boxes = results[0].boxes.data.cpu()
        for box in boxes:
            x_min, y_min, x_max, y_max, conf, class_id = box.tolist()
            category = self.detector.names[int(class_id)]

            if self.categories is None or self.categories is not None and category in self.categories:
                category_name = Category.from_string(category).to_string(Language.MODS_GENRE_EN)
                region_id = self.get_next_region_id(page_layout, category, prefix=category_name)
                polygon = np.array([[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min], [x_min, y_min]])

                region = RegionLayout(region_id, polygon, category=category, detection_confidence=conf)

                mods_id = self.get_next_mods_id(page_layout)
                region.graphical_metadata = GraphicalObjectMetadata(tag_id=region_id,
                                                                    mods_id=mods_id,
                                                                    mods_uuid=uuid.uuid4())

                page_layout.regions.append(region)

        return page_layout

    @staticmethod
    def get_next_region_id(page_layout, category, prefix, padding=3):
        existing_region_ids = set([region.id for region in page_layout.regions if region.category == category])

        index = 1
        new_id = f"{prefix}_{str(index).zfill(padding)}"

        while new_id in existing_region_ids:
            index += 1
            new_id = f"{prefix}_{str(index).zfill(padding)}"

        return new_id
    
    @staticmethod
    def get_next_mods_id(page_layout, prefix="MODS_PICT", padding=4):
        existing_mods_ids = set([region.graphical_metadata.mods_id for region in page_layout.regions if region.graphical_metadata and region.graphical_metadata.mods_id])

        index = 1
        new_id = f"{prefix}_{str(index).zfill(padding)}"

        while new_id in existing_mods_ids:
            index += 1
            new_id = f"{prefix}_{str(index).zfill(padding)}"

        return new_id


class YoloDetector:
    def __init__(self, model_path, device, detection_threshold=0.2, image_size=640):
        self.model = YOLO(model_path).to(device)
        self.detection_threshold = detection_threshold
        self.image_size = image_size

    def __call__(self, *args, **kwargs):
        return self.detect(*args, **kwargs)

    def detect(self, image):
        results = self.model(image, conf=self.detection_threshold, imgsz=self.image_size, verbose=False)
        return results[0]

    @property
    def names(self):
        return self.model.names
