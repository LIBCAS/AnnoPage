import re
import torch
import logging

from pero_ocr.document_ocr.page_parser import LayoutExtractorYolo

from anno_page.engines.embedding import ClipEmbeddingEngine
from anno_page.engines.captioning import (ChatGPTImageCaptioning, CaptionYoloNearestEngine,
                                          CaptionYoloKeypointsEngine, CaptionYoloOrganizerEngine)


def operation_factory(config, device, config_path):
    engine = None
    logger = logging.getLogger(__name__)

    if "METHOD" not in config:
        logger.warning("Config does not contain 'METHOD' key.")
        return None

    if config['METHOD'] == 'LAYOUT_YOLO':
        logger.info("Creating LayoutExtractorYolo engine")
        engine = LayoutExtractorYolo(config, device, config_path=config_path)
    elif config['METHOD'] == 'CLIP_EMBEDDING':
        logger.info("Creating ClipEmbedding engine")
        engine = ClipEmbeddingEngine(config, device, config_path=config_path)
    elif config['METHOD'] == 'GPT_IMAGE_CAPTIONING':
        logger.info("Creating GPTImageCaptioning engine")
        engine = ChatGPTImageCaptioning(config, device, config_path=config_path)
    elif config['METHOD'] == 'CAPTION_YOLO_NEAREST':
        logger.info("Creating CaptionYoloNearestEngine engine")
        engine = CaptionYoloNearestEngine(config, device, config_path=config_path)
    elif config['METHOD'] == 'CAPTION_YOLO_ORGANIZER':
        logger.info("Creating CaptionYoloOrganizerEngine engine")
        engine = CaptionYoloOrganizerEngine(config, device, config_path=config_path)
    elif config['METHOD'] == 'CAPTION_YOLO_KEYPOINTS':
        logger.info("Creating CaptionYoloKeypointsEngine engine")
        engine = CaptionYoloKeypointsEngine(config, device, config_path=config_path)
    else:
        logger.warning(f"Unknown operation method: {config['METHOD']}")

    return engine


def get_default_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class PageParser:
    def __init__(self, config, device=None, config_path=''):
        if not config.sections():
            raise ValueError('Config file is empty or does not exist.')

        self.logger = logging.getLogger(__name__)

        self.device = device if device is not None else get_default_device()

        self.engines = self.init_engines(config, config_path, operation_factory)

    def process_page(self, image, page_layout):
        for engine in self.engines:
            self.logger.debug(f"Running {engine.__class__.__name__} engine")
            page_layout = engine.process_page(image, page_layout)

        return page_layout

    def init_engines(self, config, config_path, engine_factory) -> list:
        engines = []

        for section_name in config.sections():
            self.logger.debug(f"Processing section {section_name}")

            if section_name in ("PAGE_PARSER", "PARSE_FOLDER"):
                self.logger.debug(f"Skipping section {section_name}")
                continue

            engine = engine_factory(config[section_name], config_path=config_path, device=self.device)
            if engine is not None:
                engines.append(engine)
            else:
                self.logger.info(f"Engine for section {section_name} could not be created.")

        return engines
