import argparse
import logging
import os
import sys
import subprocess

from typing import Optional

from doc_api.api.schemas.base_objects import Job
from doc_api.connector import Connector
from doc_worker.doc_worker_wrapper import DocWorkerWrapper, WorkerResponse

logger = logging.getLogger(__name__)


def parse_arguments():
    logger.info(' '.join(sys.argv))

    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", type=str, help="URL of the API endpoint.")
    parser.add_argument("--api-key", type=str, help="API key for authentication.")

    parser.add_argument("--base-dir", help="Base directory for jobs and engines (creates subdirectories 'jobs' and 'engines')")
    parser.add_argument("--jobs-dir", help="Directory for job data (overrides base-dir/jobs)")
    parser.add_argument("--engines-dir", help="Directory for engine files (overrides base-dir/engines)")

    parser.add_argument("--polling-interval", default=1.0, type=float, help="Time in seconds to wait between job requests.")
    parser.add_argument("--cleanup-job-dir", action="store_true", help="Remove job directory after successful processing")
    parser.add_argument("--cleanup-old-engines", action="store_true", help="Remove old engine versions when downloading new ones")

    parser.add_argument("--logging-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Logging level")

    return parser.parse_args()


def setup_logging(logging_level):
    level = logging.getLevelName(logging_level)

    console_log_formatter = logging.Formatter('[%(levelname)s|%(asctime)s|%(filename)s:%(name)s]: %(message)s', datefmt="%Y-%m-%d_%H-%M-%S")

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if not root_logger.handlers:
        console_handler = logging.StreamHandler()
        root_logger.addHandler(console_handler)

    root_handler = root_logger.handlers[0]
    root_handler.setFormatter(console_log_formatter)


class AnnoPageWorker(DocWorkerWrapper):
    processing_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../user_scripts/parse_folder.py")

    def process_job(self,
                    job: Job,
                    images_dir: str,
                    results_dir: str,
                    alto_dir: Optional[str] = None,
                    page_xml_dir: Optional[str] = None,
                    meta_file: Optional[str] = None,
                    engine_dir: Optional[str] = None) -> WorkerResponse:

        process_env = os.environ.copy()
        process_params = [
            "python", self.processing_script,
            "--config", os.path.join(engine_dir, "config.ini"),
            "--input-image-path", images_dir,
        ]

        if job.alto_required:
            process_params += ["--input-alto-path", alto_dir]

        if job.meta_json_required:
            process_params += ["--input-metadata-path", meta_file]

        process_params += ["--output-alto-path", os.path.join(results_dir, "alto")]
        process_params += ["--output-embeddings-path", os.path.join(results_dir, "embeddings")]
        process_params.append("--embeddings-jsonlines")

        # if job.setup.output_alto:
        #     process_params += ["--output-alto-path", os.path.join(results_dir, "alto")]
        #
        # if job.setup.output_embeddings:
        #     process_params += ["--output-embeddings-path", os.path.join(results_dir, "embeddings")]
        #
        # if job.setup.embeddings_jsonlines:
        #     process_params.append("--embeddings-jsonlines")

        process = subprocess.Popen(
            process_params,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=process_env,
            text=True
        )

        stdout, stderr = process.communicate()

        if process.returncode != 0:
            logger.error(f"Job {job.id} processing failed with return code {process.returncode}")
            logger.error(f"Stdout: {stdout}")
            logger.error(f"Stderr: {stderr}")
            result = WorkerResponse.fail(f"AnnoPage processing failed with return code {process.returncode}")
        else:
            logger.info(f"Job {job.id} processed successfully.")
            logger.debug(f"Stdout: {stdout}")
            logger.debug(f"Stderr: {stderr}")
            result = WorkerResponse.ok()

        return result


def main():
    args = parse_arguments()
    
    setup_logging(args.logging_level)

    connector = Connector(args.api_key, user_agent="AnnoPageWorker/1.0")
    logger.debug("Connector initialized.")

    worker = AnnoPageWorker(
        api_url=args.api_url,
        connector=connector,
        base_dir=args.base_dir,
        jobs_dir=args.jobs_dir,
        engines_dir=args.engines_dir,
        polling_interval=args.polling_interval,
        cleanup_job_dir=args.cleanup_job_dir,
        cleanup_old_engines=args.cleanup_old_engines
    )
    logger.debug("AnnoPageWorker initialized.")

    logger.debug("Starting worker ...")
    worker.start()
    logger.debug("Worker finished.")

    return 0


if __name__ == "__main__":
    exit(main())
