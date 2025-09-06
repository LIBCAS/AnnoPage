import os
import cv2
import argparse

from pero_ocr.core.layout import PageLayout

from anno_page.engines.embedding import ClipEmbeddingEngine


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Path to the image.")
    parser.add_argument("--page-xml", type=str, help="Path to the PAGE XML file.")
    parser.add_argument("--model", type=str, help="Name of the SentenceTransformer model.", default="clip-ViT-L-14")
    parser.add_argument("--precision", type=int, help="Number of decimal places of the embeddings. If None, the output of the model is not changed.", default=None)
    parser.add_argument("--device", type=str, help="Processing device ('cpu' or 'cuda')", choices=['cpu', 'cuda'])
    parser.add_argument("--output", type=str, help="Path to the output JSON (JSON lines) file.")
    parser.add_argument("--jsonlines", action='store_true', help="If set, the output is saved in JSON Lines format instead of a single JSON array.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    image = cv2.imread(args.image)
    layout = PageLayout(file=args.page_xml)

    output_dir = os.path.dirname(args.output)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    embedding_engine = ClipEmbeddingEngine(
        config={
            "MODEL": args.model,
            "PRECISION": args.precision
        },
        device=args.device,
        config_path=None
    )

    layout = embedding_engine.process_page(image, layout)

    if layout.embedding_data is not None:
        with open(args.output, 'w') as file:
            if args.jsonlines:
                file.write(layout.embedding_data.model_dump_jsonlines() + "\n")
            else:
                file.write(layout.embedding_data.model_dump_json(indent=2) + "\n")

        print(f"Embeddings saved to {args.output}")
    else:
        print("No embeddings were generated for the page layout.")

    return 0


if __name__ == "__main__":
    exit(main())
