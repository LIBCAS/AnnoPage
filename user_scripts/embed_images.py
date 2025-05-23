import os
import cv2
import uuid
import argparse

from PIL import Image
from typing import Optional
from datetime import datetime
from pydantic import BaseModel
from pero_ocr.core.layout import PageLayout
from sentence_transformers import SentenceTransformer

from anno_page.enums.language import Language
from anno_page.enums.category import Category


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Path to the image.")
    parser.add_argument("--page-xml", type=str, help="Path to the PAGE XML file.")
    parser.add_argument("--model", type=str, help="Name of the SentenceTransformer model.", default="clip-ViT-L-14")
    parser.add_argument("--precision", type=int, help="Number of decimal places of the embeddings. If None, the output of the model is not changed.", default=None)
    parser.add_argument("--output", type=str, help="Path to the output JSON file.")
    args = parser.parse_args()
    return args


class ObjectEmbedding(BaseModel):
    id: str
    alto: str
    category: str
    embedding: list[float]


class ProcessingInfo(BaseModel):
    system: str
    version: str
    datetime: str
    model: str
    precision: Optional[int]


class EmbeddingsData(BaseModel):
    page_id: str
    info: ProcessingInfo
    categories: dict[str, str]
    data: list[ObjectEmbedding]


def main():
    args = parse_args()

    model = SentenceTransformer(args.model)
    image = cv2.imread(args.image)
    layout = PageLayout(file=args.page_xml)

    output_dir = os.path.dirname(args.output)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    embeddings_data = EmbeddingsData(
        page_id=layout.id,
        info=ProcessingInfo(
            system="AnnoPage",
            version="0.1",
            datetime=datetime.now().isoformat(),
            model=args.model,
            precision=args.precision
        ),
        categories={str(category.value): category.to_string(Language.MODS_GENRE_EN) for category in Category},
        data = []
    )

    for region in layout.regions:
        if region.category is None or region.category == "text":
            continue

        x_min, y_min, x_max, y_max = region.get_polygon_bounding_box()
        region_image = image[y_min:y_max, x_min:x_max]

        if "metadata" in dir(region) and region.metadata is not None:
            object_uuid = region.metadata.object_uuid
        else:
            object_uuid = uuid.uuid4()
            print(f"Warning: No metadata found for region {region.id}. Generating random UUID: {object_uuid}")

        embeddings_data.data.append(
            ObjectEmbedding(
                id=f"uuid:{object_uuid}",
                alto=region.id,
                category=str(Category.from_string(region.category).value),
                embedding=model.encode(Image.fromarray(region_image)).tolist(),
            )
        )

    if args.precision is not None:
        for obj in embeddings_data.data:
            obj.embedding = [round(val, args.precision) for val in obj.embedding]

    with open(args.output, "w") as file:
        file.write(embeddings_data.model_dump_json(indent=2))

    return 0


if __name__ == "__main__":
    exit(main())
