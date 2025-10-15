import os
import cv2
import numpy as np
import logging

from http import HTTPStatus
from urllib.parse import urljoin

from api.config import config
from api.schemas.base_objects import Job, Image
from api.worker.connector import Connector

logger = logging.getLogger(__name__)


def compose_url(*args):
    args = [str(arg).strip("/") for arg in args]
    route = os.path.join(*args)
    return urljoin(config.APP_URL_ROOT, route)


class Adapter:
    def __init__(self, connector: Connector, job: Job | None = None):
        self.connector = connector
        self.job = job

    def get_job_id(self, job_id=None):
        if job_id is not None:
            return job_id
        elif self.job is not None:
            return self.job.id
        else:
            raise ValueError("Job ID must be provided either directly or via the Adapter's job attribute.")

    def get_me(self, route="/api/user/me"):
        url = compose_url(route)
        response = self.connector.get(url)

        result = None
        if response.status_code == HTTPStatus.OK:
            result = response.json()
            logger.debug("User info successfully obtained.")
        else:
            logger.warning(f"Response: {response.status_code} {response.text}")

        return result

    def get_job(self, route="/api/worker/job") -> Job | None:
        url = compose_url(route)
        response = self.connector.get(url)

        result = None
        if response.status_code == HTTPStatus.OK:
            result = Job.model_validate(response.json())
            logger.debug("Job successfully obtained.")
        elif response.status_code == HTTPStatus.NOT_FOUND:
            logger.debug(f"No job found.")
        else:
            logger.warning(f"Response: {response.status_code} {response.text}")

        return result

    def get_images(self, job_id=None, route="/api/worker/images") -> list[Image] | None:
        job_id = self.get_job_id(job_id)

        url = compose_url(route, job_id)
        response = self.connector.get(url)

        result = None
        if response.status_code == HTTPStatus.OK:
            result_json = response.json()
            result = [Image.model_validate(item) for item in result_json]
            logger.debug(f"Images for job '{job_id}' successfully obtained.")
        else:
            logger.warning(f"Response: {response.status_code} {response.text}")

        return result

    def get_image(self, image_id, job_id=None, route="/api/worker/image") -> bytes | None:
        job_id = self.get_job_id(job_id)

        url = compose_url(route, job_id, image_id)
        response = self.connector.get(url)

        result = None
        if response.status_code == HTTPStatus.OK:
            result = cv2.imdecode(np.asarray(bytearray(response.content), dtype="uint8"), cv2.IMREAD_COLOR)
            logger.info(f"Image '{image_id}' for job '{job_id}' successfully downloaded.")
        else:
            logger.error(f"Downloading image '{image_id}' failed. Response: {response.status_code} {response.text}")

        return result

    def get_alto(self, image_id, job_id=None, route="/api/worker/alto") -> str | None:
        job_id = self.get_job_id(job_id)

        url = compose_url(route, job_id, image_id)
        response = self.connector.get(url)

        result = None
        if response.status_code == HTTPStatus.OK:
            result = response.content.decode()
            logger.info(f"ALTO '{image_id}' for job '{job_id}' successfully downloaded.")
        else:
            logger.error(f"Downloading ALTO '{image_id}' failed. Response: {response.status_code} {response.text}")

        return result

    def get_meta_json(self, job_id=None, route="/api/worker/meta_json") -> str | None:
        job_id = self.get_job_id(job_id)

        url = compose_url(route, job_id)
        response = self.connector.get(url)

        result = None
        if response.status_code == HTTPStatus.OK:
            result = response.content.decode()
            logger.info(f"Meta JSON for job '{job_id}' successfully downloaded.")
        else:
            logger.error(f"Downloading Meta JSON for job '{job_id}' failed. Response: {response.status_code} {response.text}")

        return result

    def post_result(self, result_path, job_id=None, route="/api/worker/result") -> bool:
        job_id = self.get_job_id(job_id)

        url = compose_url(route, job_id)
        response = self.connector.post(url, data=None, files=[("result", open(result_path, 'rb'))])

        if response.status_code == HTTPStatus.OK:
            logger.info(f"Result for job '{job_id}' successfully uploaded.")
            return True
        else:
            logger.error(f"Uploading result for job '{job_id}' failed. Response: {response.status_code} {response.text}")
            return False

    def post_finish_job(self, job_data: dict, job_id=None, route="/api/worker/finish_job") -> bool:
        job_id = self.get_job_id(job_id)

        url = compose_url(route, job_id)
        response = self.connector.post(url, json=job_data)

        if response.status_code == HTTPStatus.OK:
            logger.info(f"Job '{job_id}' successfully marked as finished.")
            return True
        else:
            logger.error(f"Marking job '{job_id}' as finished failed. Response: {response.status_code} {response.text}")
            return False

    def put_job_update(self, job_update: dict, job_id=None, route="/api/worker/update_job") -> bool:
        job_id = self.get_job_id(job_id)

        url = compose_url(route, job_id)
        response = self.connector.put(url, data=job_update)

        if response.status_code == HTTPStatus.OK:
            logger.info(f"Job '{job_id}' successfully updated.")
            return True
        else:
            logger.error(f"Updating job '{job_id}' failed. Response: {response.status_code} {response.text}")
            return False
