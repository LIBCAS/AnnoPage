import os
import json
import logging
import argparse

from doc_api.adapter import Adapter
from doc_api.connector import Connector
from doc_api.api.schemas.base_objects import Engine
from doc_client.doc_client_wrapper import DocClientWrapper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, help="Path to directory with images.")
    parser.add_argument("--output", type=str, help="Path to the output dir.")
    parser.add_argument("--alto-xmls", type=str, help="Path to directory with ALTO XMLs.", required=False, default=None)
    parser.add_argument("--page-xmls", type=str, help="Path to directory with PAGE XMLs.", required=False, default=None)
    parser.add_argument("--metadata", type=str, help="Path to the metadata JSON.")
    parser.add_argument("--engine-name", type=str, help="Name of the processing engine to use.")

    parser.add_argument("--output-alto", action="store_true", help="Whether to output ALTO XMLs.")
    parser.add_argument("--output-embeddings", action="store_true", help="Whether to output embeddings.")
    parser.add_argument("--output-embeddings-jsonlines", action="store_true", help="Whether to output embeddings in JSONL format.")
    parser.add_argument("--output-renders", action="store_true", help="Whether to output rendered images.")
    parser.add_argument("--output-crops", action="store_true", help="Whether to output image crops.")
    parser.add_argument("--output-image-captioning-prompts", action="store_true", help="Whether to output image captioning prompts.")
    parser.add_argument("--image-captioning-settings", type=str, help="Path to image captioning settings JSON or JSON string.", required=False, default=None)

    parser.add_argument("--api-url", type=str, help="URL of the API endpoint.")
    parser.add_argument("--api-key", type=str, help="API key for authentication.")

    parser.add_argument("--polling-interval", default=1.0, type=float, help="Time in seconds to wait between result checks.")
    parser.add_argument("--logging-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO", help="Logging level.")

    parser.add_argument("--list-engine-names", action="store_true", help="List available engine names and exit.")

    args = parser.parse_args()
    return args


class AnnoPageClient(DocClientWrapper):
    pass


def setup_logging(logging_level):
    level = logging.getLevelName(logging_level)

    console_log_formatter = logging.Formatter('%(message)s')

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if not root_logger.handlers:
        console_handler = logging.StreamHandler()
        root_logger.addHandler(console_handler)

    root_handler = root_logger.handlers[0]
    root_handler.setFormatter(console_log_formatter)


def format_engines(engines: list[Engine]) -> str:
    if not engines:
        return "No available engines."

    lines = ["Available engines:", ""]
    for engine in engines:
        lines.append(f"Name: '{engine.name}'")
        lines.append(f"Description: {engine.description}")
        lines.append("")

    output = "\n".join(lines).strip()
    return output


def build_engine_settings(output_alto: bool = False,
                          output_embeddings: bool = False,
                          output_embeddings_jsonlines: bool = False,
                          output_renders: bool = False,
                          output_crops: bool = False,
                          output_image_captioning_prompts: bool = False,
                          image_captioning_settings: str | None = None) -> dict:
    outputs = {}
    if output_alto:
        outputs["alto"] = True
    if output_embeddings:
        outputs["embeddings"] = True
    if output_embeddings_jsonlines:
        outputs["embeddings_jsonlines"] = True
    if output_renders:
        outputs["renders"] = True
    if output_crops:
        outputs["crops"] = True
    if output_image_captioning_prompts:
        outputs["image_captioning_prompts"] = True

    engine_settings = {
        "outputs": outputs
    }

    if image_captioning_settings is not None:
        if os.path.exists(image_captioning_settings):
            with open(image_captioning_settings, "r", encoding="utf-8") as file:
                settings = json.load(file)
        else:
            settings = json.loads(image_captioning_settings)

        engine_settings["image_captioning"] = settings

    return engine_settings


def main():
    args = parse_args()

    setup_logging(args.logging_level)
    logger = logging.getLogger(__name__)

    connector = Connector(args.api_key, user_agent="AnnoPageClient/1.0")

    if args.list_engine_names:
        adapter = Adapter(args.api_url, connector)
        engines = adapter.get_engines()
        output = format_engines(engines.data)
        logger.info(output)

    else:
        engine_settings = build_engine_settings(output_alto=args.output_alto,
                                                output_embeddings=args.output_embeddings,
                                                output_embeddings_jsonlines=args.output_embeddings_jsonlines,
                                                output_renders=args.output_renders,
                                                output_crops=args.output_crops,
                                                output_image_captioning_prompts=args.output_image_captioning_prompts,
                                                image_captioning_settings=args.image_captioning_settings)

        client = AnnoPageClient(api_url=args.api_url,
                                connector=connector,
                                polling_interval=args.polling_interval)

        client.run_job_pipeline(
            images_dir=args.images,
            result_dir=args.output,
            alto_dir=args.alto_xmls,
            page_xml_dir=args.page_xmls,
            meta_file=args.metadata,
            engine_name=args.engine_name,
            engine_settings=engine_settings
        )

    return 0


if __name__ == "__main__":
    exit(main())
