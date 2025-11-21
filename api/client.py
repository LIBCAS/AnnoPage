import logging
import argparse

from doc_api.adapter import Adapter
from doc_api.connector import Connector
from doc_api.api.schemas.base_objects import Engine
from doc_client.doc_client_wrapper import DocClientWrapper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, help="Path to directory with images.")
    parser.add_argument("--output", type=str, help="Path to the output file.")
    parser.add_argument("--alto-xmls", type=str, help="Path to directory with ALTO XMLs.", required=False, default=None)
    parser.add_argument("--page-xmls", type=str, help="Path to directory with PAGE XMLs.", required=False, default=None)
    parser.add_argument("--metadata", type=str, help="Path to the metadata JSON.")
    parser.add_argument("--engine-name", type=str, help="Name of the processing engine to use.")

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
        client = AnnoPageClient(args.api_url, connector, polling_interval=args.polling_interval)
        client.run_job_pipeline(
            images_dir=args.images,
            result_dir=args.output,
            alto_dir=args.alto_xmls,
            page_xml_dir=args.page_xmls,
            meta_file=args.metadata,
            engine_name=args.engine_name
        )

    return 0


if __name__ == "__main__":
    exit(main())
