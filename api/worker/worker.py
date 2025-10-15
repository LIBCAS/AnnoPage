import os
import cv2
import sys
import time
import json
import argparse
import subprocess
import configparser
import logging.config

from collections import defaultdict

from api.config import config
from api.schemas.base_objects import Job, ProcessingState
from api.worker.adapter import Adapter
from api.worker.connector import Connector
from api.worker.processing_setup import ProcessingSetup
from api.worker.utils import create_zip_archive

logger = logging.getLogger(__name__)


def parse_arguments():
    logger.info(' '.join(sys.argv))

    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", help="If set, the worker will wait the specified number of seconds before getting new job if there was no job previously.", required=False, default=None, type=int)
    parser.add_argument("--default-configs", help="Path to directory with default configs.", required=True, type=str)

    return parser.parse_args()


def process_job(adapter: Adapter, job: Job, default_configs: str):
    setup = ProcessingSetup.from_job(job)

    download_data(adapter, setup)
    download_json(adapter, setup)

    prepare_processing_config(setup, default_configs)

    run_processing(setup)

    prepare_processing_result(setup)

    upload_results(adapter, setup)
    set_job_finished(adapter, setup)


def download_data(adapter: Adapter, setup: ProcessingSetup):
    images = adapter.get_images()

    # TODO: log if images is None

    for image_info in images:
        download_image(adapter, setup, image_info)
        download_alto(adapter, setup, image_info)


def download_image(adapter: Adapter, setup: ProcessingSetup, image_info):
    image = adapter.get_image(image_info.id)

    if image is not None:
        image_path = os.path.join(setup.images_directory, f"{image_info.name}.jpg")
        os.makedirs(setup.images_directory, exist_ok=True)
        cv2.imwrite(image_path, image)
    else:
        # TODO: log error
        pass


def download_alto(adapter: Adapter, setup: ProcessingSetup, image_info):
    alto = adapter.get_alto(image_info.id)
    if alto is not None:
        alto_path = os.path.join(setup.alto_directory, f"{image_info.name}.xml")
        os.makedirs(setup.alto_directory, exist_ok=True)
        with open(alto_path, "w", encoding="utf-8") as file:
            file.write(alto)
    else:
        # TODO: log error
        pass


def download_json(adapter: Adapter, setup: ProcessingSetup):
    meta_json = adapter.get_meta_json()
    if meta_json is not None:
        with open(setup.meta_json, "w", encoding="utf-8") as file:
            file.write(meta_json)
    else:
        # TODO: log error
        pass


def prepare_processing_config(setup: ProcessingSetup, default_configs: str):
    with open(setup.meta_json, "r") as file:
        meta_json = json.load(file)

    processing_config = configparser.ConfigParser()
    processing_config.add_section("PAGE_PARSER")
    processing_config["PAGE_PARSER"]["RUN_LAYOUT_PARSER"] = "no"
    processing_config["PAGE_PARSER"]["RUN_LINE_CROPPER"] = "no"
    processing_config["PAGE_PARSER"]["RUN_OCR"] = "no"
    processing_config["PAGE_PARSER"]["RUN_DECODER"] = "no"
    processing_config["PAGE_PARSER"]["RUN_OPERATIONS"] = "no"

    sections_counter = defaultdict(int)

    if "object_detection" in meta_json and meta_json["object_detection"]:
        processing_config["PAGE_PARSER"]["RUN_LAYOUT_PARSER"] = "yes"
        add_object_detection_section(processing_config, meta_json["object_detection"], sections_counter, default_configs)

    if "image_captioning" in meta_json and meta_json["image_captioning"]:
        processing_config["PAGE_PARSER"]["RUN_OPERATIONS"] = "yes"

        captioning_config = meta_json["image_captioning"]
        if "engine" not in captioning_config or captioning_config["engine"] == "chatgpt":
            add_gpt_image_captioning_section(processing_config, captioning_config, sections_counter, default_configs)
            create_prompt_settings_json(setup, captioning_config, default_configs)

    if "image_embedding" in meta_json and meta_json["image_embedding"]:
        processing_config["PAGE_PARSER"]["RUN_OPERATIONS"] = "yes"
        add_image_embedding_section(processing_config, meta_json["image_embedding"], sections_counter, default_configs)

    save_config(setup, processing_config)


def add_object_detection_section(processing_config: configparser.ConfigParser, object_detection_config_json: dict, sections_counter: dict, default_configs: str, config_name="object_detection.ini"):
    object_detection_config_path = os.path.join(default_configs, config_name)

    object_detection_config = configparser.ConfigParser()
    object_detection_config.read(object_detection_config_path)

    for config_section in object_detection_config.sections():
        section_id = sections_counter[config_section]
        new_section_name = f"{config_section}_{section_id}"
        processing_config.add_section(new_section_name)
        sections_counter[config_section] += 1

        for key, value in object_detection_config[config_section].items():
            processing_config[new_section_name][key] = value


def add_image_embedding_section(processing_config: configparser.ConfigParser, image_embedding_config_json: dict, sections_counter: dict, default_configs: str, config_name="image_embedding.ini"):
    image_embedding_config_path = os.path.join(default_configs, config_name)

    image_embedding_config = configparser.ConfigParser()
    image_embedding_config.read(image_embedding_config_path)

    for config_section in image_embedding_config.sections():
        section_id = sections_counter[config_section]
        new_section_name = f"{config_section}_{section_id}"
        processing_config.add_section(new_section_name)
        sections_counter[config_section] += 1

        for key, value in image_embedding_config[config_section].items():
            processing_config[new_section_name][key] = value


def add_gpt_image_captioning_section(processing_config: configparser.ConfigParser, image_captioning_config_json: dict, sections_counter: dict, default_configs: str, config_name="gpt_image_captioning.ini"):
    image_captioning_config_path = os.path.join(default_configs, config_name)

    image_captioning_config = configparser.ConfigParser()
    image_captioning_config.read(image_captioning_config_path)

    for config_section in image_captioning_config.sections():
        section_id = sections_counter[config_section]
        new_section_name = f"{config_section}_{section_id}"
        processing_config.add_section(new_section_name)
        sections_counter[config_section] += 1

        for key, value in image_captioning_config[config_section].items():
            processing_config[new_section_name][key] = value

        if "api_key" in processing_config[new_section_name]:
            processing_config[new_section_name]["api_key"] = image_captioning_config_json["api_key"]


def create_prompt_settings_json(setup: ProcessingSetup, image_captioning_config_json: dict, default_configs: str, prompt_settings_name="image_captioning_prompt.json"):
    default_prompt_settings_path = os.path.join(default_configs, prompt_settings_name)

    with open(default_prompt_settings_path, "r") as file:
        prompt_settings = json.load(file)

    if "model" in image_captioning_config_json:
        prompt_settings["model"] = image_captioning_config_json["model"]

    if "text" in image_captioning_config_json:
        prompt_settings["text"] = image_captioning_config_json["text"]

    if "max_tokens" in image_captioning_config_json:
        prompt_settings["max_tokens"] = image_captioning_config_json["max_tokens"]

    with open(setup.prompt_settings, "w") as file:
        json.dump(prompt_settings, file)


def save_config(setup: ProcessingSetup, config: configparser.ConfigParser):
    with open(setup.config, 'w') as file:
        config.write(file)


def run_processing(setup: ProcessingSetup):
    parse_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../user_scripts/parse_folder.py")

    process_env = os.environ.copy()
    process_params = [
        "python", parse_folder_path,
        "--config", setup.config,
        "--input-image-path", setup.images_directory,
        # "--input-alto-path", setup.alto_directory,
        "--output-alto-path", setup.output_directory,
        "--output-embeddings-path", setup.output_directory,
        "--jsonlines"
    ]

    subprocess.run(process_params, env=process_env)


def prepare_processing_result(setup: ProcessingSetup):
    files = [os.path.join(setup.output_directory, file) for file in os.listdir(setup.output_directory)]
    create_zip_archive(setup.result, files)


def upload_results(adapter: Adapter, setup: ProcessingSetup):
    adapter.post_result(setup.result)


def set_job_finished(adapter: Adapter, setup: ProcessingSetup):
    data = {
        "id": str(setup.job.id),
        "state": ProcessingState.DONE,
        "log": "",
        "log_user": ""
    }

    adapter.post_finish_job(data)


def main():
    args = parse_arguments()

    connector = Connector(worker_key=config.ADMIN_KEY)
    adapter = Adapter(connector)
    job = None

    while True:
        try:
            job = adapter.get_job()
            adapter.job = job

            if job is not None:
                logger.info(f"Processing job {job.id}.")
                process_job(adapter, job, args.default_configs)

            else:
                if args.wait is not None:
                    logger.info(f"No job found, waiting {args.wait} second{'s' if args.wait > 1 else ''}.")
                    time.sleep(args.wait)
                else:
                    logger.info('No job found, terminating worker!')
                    break

        except KeyboardInterrupt as e:
            logger.warning('Keyboard interrupt, terminating worker!')

            if job is not None:
                pass

            break

        except Exception as e:
            logger.critical('Unknown error occurred, worker will be terminated!')

            if job is not None:
                pass

            raise

    return 0


if __name__ == "__main__":
    exit(main())
