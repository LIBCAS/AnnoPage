import uuid

from PIL import Image
from datetime import datetime
from sentence_transformers import SentenceTransformer

from anno_page import globals
from anno_page.core.utils import config_get_list
from anno_page.engines import BaseEngine, LayoutProcessingEngine
from anno_page.engines.helpers import set_model_dtype
from anno_page.enums import Category, Language
from anno_page.core.embedding import ObjectEmbedding, ProcessingInfo


class ClipImageEmbeddingEngine(LayoutProcessingEngine):
    def __init__(self, config, device, config_path):
        super().__init__(config, device, config_path)

        self.model_name = self.config["MODEL"]
        self.decimal_places = self.config.getint("DECIMAL_PLACES", None)
        self.precision = self.config.get("PRECISION", "float16")
        self.categories = config_get_list(self.config, key="categories", fallback=None)

        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.model = set_model_dtype(self.model, self.precision)

    def process_page(self, page_image, page_layout):
        for region in page_layout.regions:
            if (self.categories is None and region.category not in (None, 'text')) or (self.categories is not None and region.category in self.categories):
                x_min, y_min, x_max, y_max = region.get_polygon_bounding_box()
                region_image = page_image[y_min:y_max, x_min:x_max]

                if region_image.size == 0:
                    continue

                object_uuid = region.graphical_metadata.mods_uuid if region.graphical_metadata is not None else uuid.uuid4()

                region_embedding = self.model.encode(Image.fromarray(region_image), show_progress_bar=False).tolist()

                if self.decimal_places is not None:
                    region_embedding = [round(value, self.decimal_places) for value in region_embedding]

                category_name = Category.from_string(region.category).to_string(Language.MODS_GENRE_EN)

                region_object_embedding = ObjectEmbedding(
                    id=f"uuid:{object_uuid}",
                    tag_id=region.id,
                    page_uuid=page_layout.id,
                    category=category_name,
                    embedding=region_embedding,
                    processing_info=ProcessingInfo(
                        system=globals.software_name,
                        version=globals.software_version,
                        datetime=datetime.now().isoformat(),
                        model=self.model_name,
                        decimal_places=self.decimal_places,
                        precision=self.precision
                    )
                )

                region.embeddings.append(region_object_embedding)

        return page_layout


class ClipTextEmbeddingEngine(BaseEngine):
    def __init__(self, config, device, config_path):
        super().__init__(config, device, config_path)

        self.model_name = self.config["MODEL"]
        self.decimal_places = self.config.getint("DECIMAL_PLACES", None)
        self.precision = self.config.get("PRECISION", "float16")

        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.model = set_model_dtype(self.model, self.precision)

    def process(self, data: str | list[str]) -> list[ObjectEmbedding]:
        if isinstance(data, str):
            data = [data]

        embeddings = self.model.encode(data, show_progress_bar=False).tolist()

        if self.decimal_places is not None:
            embeddings = [[round(value, self.decimal_places) for value in embedding] for embedding in embeddings]

        result = []
        for i, embedding in enumerate(embeddings):
            result.append(ObjectEmbedding(
                id="",
                tag_id="",
                page_uuid="",
                category="text",
                source=data[i],
                embedding=embedding,
                processing_info=ProcessingInfo(
                    system=globals.software_name,
                    version=globals.software_version,
                    datetime=datetime.now().isoformat(),
                    model=self.model_name,
                    decimal_places=self.decimal_places,
                    precision=self.precision
                )
            ))

        return result
