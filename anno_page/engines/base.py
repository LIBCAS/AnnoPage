import logging

from abc import ABC, abstractmethod


class BaseEngine(ABC):
    def __init__(self, config, device, config_path, requires_lines=False):
        self.config = config
        self.device = device
        self.config_path = config_path
        self.requires_lines = requires_lines

        self.logger = logging.getLogger(self.__class__.__name__)


class LayoutProcessingEngine(BaseEngine):
    @abstractmethod
    def process_page(self, image, page_layout):
        pass
