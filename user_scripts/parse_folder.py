import os
import re
import cv2
import json
import time
import torch
import logging
import argparse
import traceback
import configparser

from lxml import etree as ET
from safe_gpu import safe_gpu
from multiprocessing import Pool
from pero_ocr.core.layout import PageLayout, ALTOVersion

from anno_page.core.layout import render_to_image, add_page_layout_to_alto
from anno_page.core.page_parser import PageParser


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to config file.")
    parser.add_argument("--input-image-path", help="Path to directory with images to process.")

    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument("--input-xml-path", help="Path to directory with PAGE XML files", required=False, default=None)
    group1.add_argument("--input-alto-path", help="Path to directory with ALTO XML files", required=False, default=None)

    parser.add_argument("--input-metadata-path", help="Path to JSON file with metadata for input images.", required=False, default=None)

    parser.add_argument("--output-xml-path", help="Path to directory where PAGE XML files will be saved.")
    parser.add_argument("--output-alto-path", help="Path to directory where ALTO files will be saved.")
    parser.add_argument("--output-render-path", help="Path to directory where rendered images will be saved.")
    parser.add_argument("--output-crops-path", help="Path to directory where region crops will be saved.")
    parser.add_argument("--output-image-captioning-prompts-path", help="Path to directory where image captioning prompts will be saved.")
    parser.add_argument("--output-embeddings-path", help="Path to directory where embeddings will be saved.")
    parser.add_argument("--embeddings-jsonlines", action='store_true', help="If set, the embedding output is saved in JSON Lines format instead of a single JSON array.")
    parser.add_argument('-s', '--skip-processed', action='store_true', required=False, help='If set, already processed files are skipped.')

    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--gpu-id", type=int, default=None, help="If set, the computation runs of the specified GPU, otherwise safe-gpu is used to allocate first unused GPU.")

    parser.add_argument("--process-count", type=int, default=1, help="Number of parallel processes.")

    parser.add_argument("--logging-level", default="WARNING", help="Logging level. Possible values: DEBUG, INFO, WARNING, ERROR, CRITICAL")

    args = parser.parse_args()
    return args


def setup_logging(logging_level):
    level = logging.getLevelName(logging_level)

    log_formatter = logging.Formatter('[%(levelname)s|%(asctime)s|%(filename)s:%(name)s]: %(message)s', datefmt="%Y-%m-%d_%H-%M-%S")

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    root_handler = root_logger.handlers[0]
    root_handler.setFormatter(log_formatter)

    root_logger.info("Logging initialized")

def get_device(device, gpu_index=None, logger=None):
    if gpu_index is None:
        if device == "gpu":
            safe_gpu.claim_gpus(logger=logger)
            torch_device = torch.device("cuda")
        else:
            torch_device = torch.device("cpu")
    else:
        torch_device = torch.device(f"cuda:{gpu_index}")

    return torch_device


def get_value_or_none(config, section, key, getboolean: bool = False):
    if config.has_option(section, key):
        if getboolean:
            value = config.getboolean(section, key)
        else:
            value = config[section][key]
    else:
        value = None
    return value


def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_already_processed_files_in_directory(directory):
    already_processed = set()

    if directory is not None:
        file_pattern = r"(.+?)(\.xml|\.jpg)"
        regex = re.compile(file_pattern)

        for file in os.listdir(directory):
            matched = regex.match(file)
            if matched:
                already_processed.add(matched.groups()[0])

    return already_processed


def load_already_processed_files(directories):
    already_processed = set()
    first = True

    for directory in directories:
        if directory is not None:
            files = load_already_processed_files_in_directory(directory)

            if first:
                already_processed = files
                first = False
            else:
                already_processed = already_processed.intersection(files)

    return already_processed


class Computator:
    def __init__(self,
                 page_parser,
                 input_image_path,
                 input_xml_path,
                 input_alto_path,
                 output_xml_path,
                 output_alto_path,
                 output_embeddings_path,
                 output_render_path,
                 output_crops_path,
                 output_image_captioning_prompts_path,
                 embeddings_jsonlines=False):
        self.page_parser = page_parser
        self.input_image_path = input_image_path
        self.input_xml_path = input_xml_path
        self.input_alto_path = input_alto_path
        self.output_xml_path = output_xml_path
        self.output_alto_path = output_alto_path
        self.output_embeddings_path = output_embeddings_path
        self.output_render_path = output_render_path
        self.output_crops_path = output_crops_path
        self.output_image_captioning_prompts_path = output_image_captioning_prompts_path
        self.embeddings_jsonlines = embeddings_jsonlines

        self.logger = logging.getLogger(self.__class__.__name__)

    def __call__(self, image_file_name, file_id, index, ids_count, file_metadata=None):
        self.logger.info(f"Processing {file_id}")
        start_time = time.time()

        try:
            if self.input_image_path is not None:
                image = cv2.imread(os.path.join(self.input_image_path, image_file_name), 1)
                if image is None:
                    raise Exception(f'Unable to read image "{os.path.join(self.input_image_path, image_file_name)}"')
            else:
                image = None

            alto_file_path = None
            page_layout = PageLayout(id=file_id, page_size=(image.shape[0], image.shape[1]))

            self.logger.info(f"Created empty page layout for id: '{file_id}'.")

            if self.input_alto_path is not None:
                if not self.page_parser.requires_lines:
                    self.logger.info("Page parser does not require lines, skipping ALTO file loading.")
                else:
                    alto_file_path = os.path.join(self.input_alto_path, file_id + '.xml')
                    if os.path.isfile(alto_file_path):
                        page_layout.from_altoxml(alto_file_path)
                        self.logger.info(f"Loaded ALTO file: '{alto_file_path}'.")
                    else:
                        self.logger.warning(f"ALTO file does not exist: '{alto_file_path}'.")
            elif self.input_xml_path is not None:
                xml_file_path = os.path.join(self.input_xml_path, file_id + '.xml')
                if os.path.isfile(xml_file_path):
                    page_layout = PageLayout(file=xml_file_path)
                    self.logger.info(f"Loaded PAGE XML file: '{xml_file_path}'.")
                else:
                    self.logger.warning(f"PAGE XML file does not exist: '{xml_file_path}'.")

            page_layout.metadata["anno_page_metadata"] = file_metadata
            page_layout = self.page_parser.process_page(image, page_layout)

            if self.output_xml_path is not None:
                page_layout.to_pagexml(os.path.join(self.output_xml_path, file_id + '.xml'))

            if self.output_alto_path is not None:
                output_alto_path = os.path.join(self.output_alto_path, file_id + '.xml')

                if alto_file_path is not None:
                    parser = ET.XMLParser(remove_blank_text=True)
                    alto = ET.parse(alto_file_path, parser)
                    add_page_layout_to_alto(page_layout, alto.getroot())

                    with open(output_alto_path, 'w', encoding="utf-8") as file:
                        file.write(ET.tostring(alto, pretty_print=True, encoding="utf-8", xml_declaration=True).decode("utf-8"))

                else:
                    page_layout.to_altoxml(output_alto_path, version=ALTOVersion.ALTO_v4_4)

            if self.output_embeddings_path is not None:
                embeddings = page_layout.get_all_embeddings()

                extension = 'jsonl' if self.embeddings_jsonlines else 'json'
                embeddings_file = os.path.join(self.output_embeddings_path, f"{file_id}.{extension}")
                with open(embeddings_file, 'w') as file:
                    if self.embeddings_jsonlines:
                        for embedding in embeddings:
                            file.write(embedding.model_dump_json() + "\n")
                    else:
                        json.dump([embedding.model_dump() for embedding in embeddings], file, ensure_ascii=False, indent=4)

            if self.output_render_path is not None:
                render = render_to_image(image, page_layout)
                render_file = str(os.path.join(self.output_render_path, file_id + '.jpg'))
                cv2.imwrite(render_file, render, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

            if self.output_crops_path is not None:
                for region in page_layout.regions:
                    if region.category in (None, "text"):
                        continue

                    x1, y1, x2, y2 = region.get_polygon_bounding_box()
                    crop = image[y1:y2, x1:x2]

                    suffix = f"{region.id}"
                    if region.graphical_metadata is not None:
                        suffix = f"{region.graphical_metadata.tag_id}"

                    crop_path = os.path.join(self.output_crops_path, f"{file_id}_{suffix}.jpg")
                    cv2.imwrite(crop_path, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            if self.output_image_captioning_prompts_path is not None:
                for region in page_layout.regions:
                    if region.category in (None, "text"):
                        continue

                    if region.graphical_metadata is not None:
                        region_prompts = region.graphical_metadata.prompts
                        if region_prompts is not None:
                            suffix = f"{region.graphical_metadata.tag_id}"
                            prompts_path = os.path.join(self.output_image_captioning_prompts_path, f"{file_id}_{suffix}.txt")
                            with open(prompts_path, 'w', encoding='utf-8') as file:
                                json.dump(region_prompts, file, ensure_ascii=False, indent=4)


        except KeyboardInterrupt:
            traceback.print_exc()
            self.logger.warning("Terminated by user.")
            exit()
        except Exception as e:
            self.logger.error(f"Failed to process file '{file_id}'.")
            self.logger.error(e)
            traceback.print_exc()

        end_time = time.time()
        self.logger.info(f"DONE {index + 1}/{ids_count} ({100 * (index + 1) / ids_count:.2f} %) [id: {file_id}] Time:{end_time - start_time:.2f}")


def main():
    args = parse_arguments()
    config_path = args.config
    skip_already_processed_files = args.skip_processed

    setup_logging(args.logging_level)
    logger = logging.getLogger(__name__)

    if not os.path.isfile(config_path):
        logger.error(f"Config file does not exist: '{config_path}'.")
        exit(-1)

    config = configparser.ConfigParser()
    config.read(config_path)

    if 'PARSE_FOLDER' not in config:
        config.add_section('PARSE_FOLDER')

    if args.input_image_path is not None:
        config['PARSE_FOLDER']['INPUT_IMAGE_PATH'] = args.input_image_path

    if args.input_xml_path is not None:
        config['PARSE_FOLDER']['INPUT_XML_PATH'] = args.input_xml_path

    if args.input_alto_path is not None:
        config['PARSE_FOLDER']['INPUT_ALTO_PATH'] = args.input_alto_path

    if args.input_metadata_path is not None:
        config['PARSE_FOLDER']['INPUT_METADATA_PATH'] = args.input_metadata_path

    if args.output_xml_path is not None:
        config['PARSE_FOLDER']['OUTPUT_XML_PATH'] = args.output_xml_path

    if args.output_alto_path is not None:
        config['PARSE_FOLDER']['OUTPUT_ALTO_PATH'] = args.output_alto_path

    if args.output_embeddings_path is not None:
        config['PARSE_FOLDER']['OUTPUT_EMBEDDINGS_PATH'] = args.output_embeddings_path

    if args.output_render_path is not None:
        config['PARSE_FOLDER']['OUTPUT_RENDER_PATH'] = args.output_render_path

    if args.output_crops_path is not None:
        config['PARSE_FOLDER']['OUTPUT_CROPS_PATH'] = args.output_crops_path

    if args.output_image_captioning_prompts_path is not None:
        config['PARSE_FOLDER']['OUTPUT_IMAGE_CAPTIONING_PROMPTS_PATH'] = args.output_image_captioning_prompts_path

    device = get_device(args.device, args.gpu_id, logger)

    page_parser = PageParser(config, config_path=os.path.dirname(config_path), device=device)

    input_image_path = get_value_or_none(config, 'PARSE_FOLDER', 'INPUT_IMAGE_PATH')
    input_xml_path = get_value_or_none(config, 'PARSE_FOLDER', 'INPUT_XML_PATH')
    input_alto_path = get_value_or_none(config, 'PARSE_FOLDER', 'INPUT_ALTO_PATH')
    input_metadata_path = get_value_or_none(config, 'PARSE_FOLDER', 'INPUT_METADATA_PATH')
    output_xml_path = get_value_or_none(config, 'PARSE_FOLDER', 'OUTPUT_XML_PATH')
    output_alto_path = get_value_or_none(config, 'PARSE_FOLDER', 'OUTPUT_ALTO_PATH')
    output_embeddings_path = get_value_or_none(config, 'PARSE_FOLDER', 'OUTPUT_EMBEDDINGS_PATH')
    output_render_path = get_value_or_none(config, 'PARSE_FOLDER', 'OUTPUT_RENDER_PATH')
    output_crops_path = get_value_or_none(config, 'PARSE_FOLDER', 'OUTPUT_CROPS_PATH')
    output_image_captioning_prompts_path = get_value_or_none(config, 'PARSE_FOLDER', 'OUTPUT_IMAGE_CAPTIONING_PROMPTS_PATH')

    embeddings_jsonlines = args.embeddings_jsonlines

    if output_xml_path is not None:
        create_dir_if_not_exists(output_xml_path)

    if output_alto_path is not None:
        create_dir_if_not_exists(output_alto_path)

    if output_embeddings_path is not None:
        create_dir_if_not_exists(output_embeddings_path)

    if output_render_path is not None:
        create_dir_if_not_exists(output_render_path)

    if output_crops_path is not None:
        create_dir_if_not_exists(output_crops_path)

    if output_image_captioning_prompts_path is not None:
        create_dir_if_not_exists(output_image_captioning_prompts_path)

    files_metadata = {}
    if input_metadata_path is not None:
        with open(input_metadata_path, 'r') as file:
            files_metadata = json.load(file)

    images_to_process = []
    ids_to_process = []

    if input_image_path is not None:
        logger.info(f'Reading images from {input_image_path}.')
        ignored_extensions = ['', '.xml', '.logits']
        images_to_process = [f for f in os.listdir(input_image_path) if os.path.splitext(f)[1].lower() not in ignored_extensions]
        images_to_process = sorted(images_to_process)
        ids_to_process = [os.path.splitext(os.path.basename(file))[0] for file in images_to_process]

    if skip_already_processed_files:
        already_processed_files = load_already_processed_files([output_xml_path, output_alto_path, output_render_path])

        if len(already_processed_files) > 0:
            logger.info(f"Already processed {len(already_processed_files)} file(s).")

            images_to_process = [image for id, image in zip(ids_to_process, images_to_process) if id not in already_processed_files]
            ids_to_process = [id for id in ids_to_process if id not in already_processed_files]

    computator = Computator(page_parser=page_parser,
                            input_image_path=input_image_path,
                            input_xml_path=input_xml_path,
                            input_alto_path=input_alto_path,
                            output_xml_path=output_xml_path,
                            output_alto_path=output_alto_path,
                            output_embeddings_path=output_embeddings_path,
                            output_render_path=output_render_path,
                            output_crops_path=output_crops_path,
                            output_image_captioning_prompts_path=output_image_captioning_prompts_path,
                            embeddings_jsonlines=embeddings_jsonlines)

    results = []
    if args.process_count > 1:
        with Pool(processes=args.process_count) as pool:
            tasks = []
            for index, (file_id, image_file_name) in enumerate(zip(ids_to_process, images_to_process)):
                file_metadata = files_metadata.get(image_file_name, None)
                tasks.append((image_file_name, file_id, index, len(ids_to_process), file_metadata))
            results = pool.starmap(computator, tasks)
    else:
        for index, (file_id, image_file_name) in enumerate(zip(ids_to_process, images_to_process)):
            file_metadata = files_metadata.get(image_file_name, None)
            results.append(computator(image_file_name, file_id, index, len(ids_to_process), file_metadata))

    return 0


if __name__ == "__main__":
    exit(main())
