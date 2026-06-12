import json

import cv2
import numpy as np

from skimage.color import deltaE_ciede2000, rgb2lab
from pydantic import BaseModel

from anno_page.engines import LayoutProcessingEngine
from anno_page.core.utils import compose_path, config_get_list
from anno_page.core.metadata import GraphicalObjectMetadata, DominantColorInfo, ColorInfo
from anno_page.enums import Language


class ColorDefinition(BaseModel):
    names: dict[str, str]
    variants: dict[str, str]


class ColorCoverage:
    def __init__(self, names: dict[str, str], coverage: float):
        self.names = names
        self.coverage = coverage


class NamedColors:
    def __init__(self, colors, color_names, colors_mapping, colors_lab):
        self.colors: np.ndarray = colors
        self.color_names: list[dict[str, str]] = color_names
        self.colors_mapping: np.ndarray = colors_mapping
        self.colors_lab: np.ndarray = colors_lab


class DominantColorsEngine(LayoutProcessingEngine):
    def __init__(self, config, device, config_path):
        super().__init__(config, device, config_path)

        self.named_colors = self.load_colors(compose_path(self.config["colors"], self.config_path))
        self.categories = config_get_list(self.config, key="categories", fallback=None, make_lowercase=True)
        self.coverage_threshold = self.config.getfloat("coverage_threshold", 0.1)
        self.max_size = self.config.getint("max_size", 256)
        self.gaussian_blur_kernel_size = self.config.getint("gaussian_blur_kernel_size", fallback=0)

    @staticmethod
    def load_colors(path) -> NamedColors:
        with open(path, "r", encoding="utf8") as fh:
            data = json.load(fh)

        color_definitions = [ColorDefinition(**item) for item in data]

        colors = []
        color_names = []
        colors_mapping = []

        for i, color_definition in enumerate(color_definitions):
            color_names.append(color_definition.names)
            for color_variant_name, color_variant in color_definition.variants.items():
                colors.append(DominantColorsEngine.hex_to_rgb(color_variant))
                colors_mapping.append(i)

        colors = np.array(colors, dtype=np.uint8)
        colors_mapping = np.array(colors_mapping)
        colors_lab = rgb2lab(colors).reshape(-1, 3)

        for i, color_name in enumerate(color_names):
            color_variants = colors[colors_mapping == i]
            color_name["color"] = np.clip(np.mean(color_variants, axis=0), 0, 255).astype(np.uint8)

        named_colors = NamedColors(colors, color_names, colors_mapping, colors_lab)

        return named_colors

    def process_page(self, page_image, page_layout):
        for i, region in enumerate(page_layout.regions):
            if region.category is None or region.category.lower() == "text":
                continue

            if self.categories is None or region.category.lower() in self.categories:
                x_min, y_min, x_max, y_max = region.get_polygon_bounding_box()
                region_image = page_image[y_min:y_max, x_min:x_max]

                if region_image.size == 0:
                    continue

                if self.gaussian_blur_kernel_size > 0:
                    region_image = cv2.GaussianBlur(region_image, (self.gaussian_blur_kernel_size, self.gaussian_blur_kernel_size), 0)

                region_image = cv2.cvtColor(region_image, cv2.COLOR_BGR2RGB)
                region_image = self.resize_to_max_size(region_image)

                region_colors = self.process_crop(region_image)
                metadata: GraphicalObjectMetadata = region.graphical_metadata

                for region_color in region_colors:
                    if region_color.coverage >= self.coverage_threshold:
                        if metadata.color is None:
                            metadata.color = ColorInfo()

                        if isinstance(metadata.color, dict):
                            for language in metadata.color:
                                dominant_color = DominantColorInfo(name=region_color.names[language.to_string()], coverage=region_color.coverage)

                                if metadata.color[language].dominant_colors is None:
                                    metadata.color[language].dominant_colors = [dominant_color]
                                else:
                                    metadata.color[language].dominant_colors.append(dominant_color)

                        elif isinstance(metadata.color, ColorInfo):
                            if Language.ENGLISH.to_string() in region_color.names:
                                color_name = region_color.names[Language.ENGLISH.to_string()]
                            elif Language.CZECH.to_string() in region_color.names:
                                color_name = region_color.names[Language.CZECH.to_string()]
                            else:
                                color_name = list(region_color.names.values())[0]

                            dominant_color = DominantColorInfo(name=color_name, coverage=region_color.coverage)

                            if metadata.color.dominant_colors is None:
                                metadata.color.dominant_colors = [dominant_color]
                            else:
                                metadata.color.dominant_colors.append(dominant_color)

        return page_layout

    def process_crop(self, image) -> list[ColorCoverage]:
        pixels_lab = rgb2lab(image).reshape(-1, 3)
        distances = deltaE_ciede2000(pixels_lab[:, None, :], self.named_colors.colors_lab[None, :, :])
        assignments = np.argmin(distances, axis=1)
        color_assignments = self.named_colors.colors_mapping[assignments]
        counts = np.bincount(color_assignments, minlength=len(self.named_colors.color_names))
        total_pixels = len(assignments)

        coverages = [ColorCoverage(names=name, coverage=counts[index] / total_pixels) for index, name in enumerate(self.named_colors.color_names)]
        return coverages

    def resize_to_max_size(self, image):
        if self.max_size <= 0:
            return image

        height, width = image.shape[:2]
        current_size = max(width, height)
        if current_size <= self.max_size:
            return image

        scale = self.max_size / current_size
        new_size = (round(width * scale), round(height * scale))
        resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        return resized

    @staticmethod
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i: i + 2], 16) for i in (0, 2, 4))
