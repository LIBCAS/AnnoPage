import requests
import os
import time
import json
import argparse

from http import HTTPStatus

from api.worker.connector import Connector
from api.schemas.base_objects import ProcessingState


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, help="Path to directory with images.")
    parser.add_argument("--json-config", type=str, help="Path to the JSON file with configuration.")
    parser.add_argument("--api-url", type=str, help="URL of the API endpoint.")
    parser.add_argument("--api-key", type=str, help="API key for authentication.")
    parser.add_argument("--output", type=str, help="Path to the output file.")
    parser.add_argument("--wait", default=1, type=int, help="Wait time between status checks in seconds.")
        
    args = parser.parse_args()
    return args


def create_job(connector, api_url, images):
    data = {
        "images": [{"name": os.path.splitext(image)[0], "order": i} for i, image in enumerate(images)],
        "alto_required": False,
        "meta_json_required": True,
    }

    response = connector.post(f"{api_url}/api/user/job", json=data)

    result = None
    if response.status_code == HTTPStatus.OK:
        result = response.json()
        print(f"Job '{result['id']}' successfully created.")
    else:
        print(f"Creating job failed. Response: {response.status_code} {response.text}")
        exit(1)

    return result


def upload_images(connector, api_url, job_info, image_paths):
    job_id = job_info['id']
    for image_path in image_paths:
        with open(image_path, 'rb') as file:
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            response = connector.post(f"{api_url}/api/user/image/{job_id}/{image_name}", files={'file': file})
            if response.status_code == HTTPStatus.OK:
                print(f"Image '{image_name}' successfully uploaded.")
            else:
                print(f"Uploading image '{image_name}' failed. Response: {response.status_code} {response.text}")
                exit(1)


def upload_json(connector, api_url, job_info, json_path):
    job_id = job_info['id']

    with open(json_path, 'r') as file:
        data = json.load(file)

    response = connector.post(f"{api_url}/api/user/meta_json/{job_id}", params={"meta_json": json.dumps(data)})
    if response.status_code == HTTPStatus.OK:
        print(f"JSON configuration successfully uploaded.")
    else:
        print(f"Uploading JSON configuration failed. Response: {response.status_code} {response.text}")
        exit(1)


def wait_for_completion(connector, api_url, job_info, wait=1):
    job_id = job_info['id']

    status = get_job_status(connector, api_url, job_id)
    if status == ProcessingState.NEW:
        print(f"Job '{job_id}' is in '{ProcessingState.NEW}' state - something was not uploaded.")
        exit(1)

    while status not in (ProcessingState.DONE, ProcessingState.ERROR, ProcessingState.FAILED, ProcessingState.CANCELLED):
        if status == ProcessingState.QUEUED:
            print(f"Job '{job_id}' is in '{ProcessingState.QUEUED}' state - waiting for processing to start.")
        elif status == ProcessingState.PROCESSING:
            print(f"Job '{job_id}' is in '{ProcessingState.PROCESSING}' state - processing in progress.")

        time.sleep(wait)
        status = get_job_status(connector, api_url, job_id)

    if status == ProcessingState.DONE:
        print(f"Job '{job_id}' is in '{ProcessingState.DONE}' state - processing finished successfully.")
    elif status == ProcessingState.ERROR:
        print(f"Job '{job_id}' is in '{status}' state - processing finished with errors.")
        exit(1)
    elif status == ProcessingState.FAILED:
        print(f"Job '{job_id}' is in '{status}' state - processing failed.")
        exit(1)
    elif status == ProcessingState.CANCELLED:
        print(f"Job '{job_id}' is in '{status}' state - processing was cancelled.")
        exit(1)


def get_job_status(connector, api_url, job_id):
    response = connector.get(f"{api_url}/api/user/job/{job_id}")
    result = None
    if response.status_code == HTTPStatus.OK:
        job_info = response.json()
        result = ProcessingState(job_info['state'])
    else:
        print(f"Getting job status failed. Response: {response.status_code} {response.text}")
        exit(1)

    return result


def download_result(connector, api_url, job_info, output_path):
    job_id = job_info['id']
    response = connector.get(f"{api_url}/api/user/result/{job_id}")
    if response.status_code == HTTPStatus.OK:
        with open(output_path, 'wb') as file:
            file.write(response.content)
        print(f"Results for job '{job_id}' successfully downloaded to '{output_path}'.")
    else:
        print(f"Downloading results failed. Response: {response.status_code} {response.text}")
        exit(1)


def cancel_job(connector, api_url, job_info):
    job_id = job_info['id']
    response = connector.put(f"{api_url}/api/user/cancel_job/{job_id}")
    if response.status_code == HTTPStatus.OK:
        print(f"Job '{job_id}' successfully cancelled.")
    else:
        print(f"Cancelling job failed. Response: {response.status_code} {response.text}")


def main():
    args = parse_args()

    connector = Connector(args.api_key, user_agent="AnnoPageClient/1.0")

    image_names = [file for file in os.listdir(args.images) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_paths = [os.path.join(args.images, name) for name in image_names]

    job_info = None

    try:
        job_info = create_job(connector, args.api_url, image_names)
        upload_images(connector, args.api_url, job_info, image_paths)
        upload_json(connector, args.api_url, job_info, args.json_config)
        wait_for_completion(connector, args.api_url, job_info, wait=args.wait)
        download_result(connector, args.api_url, job_info, args.output)
    except KeyboardInterrupt:
        if job_info is not None:
            cancel_job(connector, args.api_url, job_info)

        print("Process interrupted by user.")
        exit(1)
    except:
        if job_info is not None:
            cancel_job(connector, args.api_url, job_info)

        print("An error occurred during processing.")
        raise
    
    return 0


if __name__ == "__main__":
    exit(main())
