import datetime
import os

from doc_api.api.schemas.base_objects import JobWithImages

from api.helpers.utils import get_temp_directory


class ProcessingSetup:
    def __init__(self, root_directory: str, job: JobWithImages):
        self.root_directory = root_directory
        self.job = job

        self.output_alto = False
        self.output_embeddings = False
        self.embeddings_jsonlines = False

    @staticmethod
    def from_job(job: JobWithImages, prefix="anno-page_processing"):
        tmp_directory = get_temp_directory()
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        directory = os.path.join(tmp_directory, f"{prefix}_{job.id}_{current_time}")
        return ProcessingSetup(root_directory=directory, job=job)

    def _get_sub_path(self, *args):
        return os.path.abspath(os.path.join(self.root_directory, *args))

    @property
    def images_directory(self):
        return self._get_sub_path("images")

    @property
    def alto_directory(self):
        return self._get_sub_path("alto")

    @property
    def meta_json(self):
        return self._get_sub_path("meta.json")

    @property
    def output_directory(self):
        return self._get_sub_path("output")

    @property
    def config(self):
        return self._get_sub_path("config.ini")

    @property
    def prompt_settings(self):
        return self._get_sub_path("image_captioning_prompt.json")

    @property
    def result(self):
        return self._get_sub_path(f"{self.job.id}.zip")

