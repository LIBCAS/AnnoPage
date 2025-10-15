import uuid

from PIL import Image
from datetime import datetime
from sentence_transformers import SentenceTransformer

from pero_ocr.utils import compose_path, config_get_list

from anno_page.core.embedding import ElementEmbeddings, Element, ProcessingInfo


class ClipEmbeddingEngine:
    def __init__(self, config, device, config_path):
        self.device = device
        self.model_name = config["MODEL"]
        self.precision = int(config["PRECISION"]) if "PRECISION" in config and config["PRECISION"] is not None else None
        self.categories = config_get_list(config, key="categories", fallback=None) if "categories" in config else None

        self.model = SentenceTransformer(self.model_name, device=device)

    def process_page(self, page_image, page_layout):
        if page_layout.embedding_data is None:
            page_layout.embedding_data = ElementEmbeddings()

        page_embeddings = page_layout.embedding_data

        for region in page_layout.regions:
            if self.categories is None and region.category not in (None, 'text') or region.category in self.categories:
                x_min, y_min, x_max, y_max = region.get_polygon_bounding_box()
                region_image = page_image[y_min:y_max, x_min:x_max]

                if region_image.size == 0:
                    continue

                object_uuid = region.metadata.mods_uuid if region.metadata is not None else uuid.uuid4()

                region_embedding = self.model.encode(Image.fromarray(region_image), show_progress_bar=False).tolist()

                if self.precision is not None:
                    region_embedding = [round(value, self.precision) for value in region_embedding]

                region_embedding_data = Element(
                    id=f"uuid:{object_uuid}",
                    alto_id=region.id,
                    page_id=page_layout.id,
                    category=str(region.category),
                    embedding=region_embedding,
                    processing_info=ProcessingInfo(
                        system="AnnoPage",
                        version="0.1",
                        datetime=datetime.now().isoformat(),
                        model=self.model_name,
                        precision=self.precision
                    )
                )

                page_embeddings.append(region_embedding_data)

        return page_layout