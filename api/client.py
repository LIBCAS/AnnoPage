import argparse
import os
import time

from doc_api.api.schemas.base_objects import ProcessingState

from api.helpers.adapter import Adapter
from api.helpers.connector import Connector


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, help="Path to directory with images.")
    parser.add_argument("--alto", type=str, help="Path to directory with ALTO files.", required=False, default=None)
    parser.add_argument("--json-config", type=str, help="Path to the JSON file with configuration.")
    parser.add_argument("--api-url", type=str, help="URL of the API endpoint.")
    parser.add_argument("--api-key", type=str, help="API key for authentication.")
    parser.add_argument("--output", type=str, help="Path to the output file.")
    parser.add_argument("--wait", default=1, type=int, help="Wait time between status checks in seconds.")
        
    args = parser.parse_args()
    return args


def create_job(adapter: Adapter, images, alto_required=False):
    data = {
        "images": [{"name": os.path.splitext(image)[0], "order": i} for i, image in enumerate(images)],
        "alto_required": alto_required,
        "meta_json_required": True,
    }

    result = adapter.post_job(data, set_if_successful=True)

    return result


def upload_files(adapter: Adapter, file_paths, file_type):
    for file_path in file_paths:
        adapter.put_file(file_path, file_type)


def upload_json(adapter: Adapter, json_path):
    adapter.put_meta_json(json_path)


def wait_for_completion(adapter: Adapter, wait=1):
    state = get_job_state(adapter)

    if state is None:
        print(f"Job state could not be retrieved.")
        exit(1)

    if state == ProcessingState.NEW:
        print(f"Job is in '{ProcessingState.NEW}' state - something was not uploaded.")
        exit(1)

    while state not in (ProcessingState.DONE, ProcessingState.ERROR, ProcessingState.FAILED, ProcessingState.CANCELLED):
        if state == ProcessingState.QUEUED:
            print(f"Job is in '{ProcessingState.QUEUED}' state - waiting for processing to start.")
        elif state == ProcessingState.PROCESSING:
            print(f"Job is in '{ProcessingState.PROCESSING}' state - processing in progress.")

        time.sleep(wait)
        state = get_job_state(adapter)
        if state is None:
            print(f"Job state could not be retrieved.")
            exit(1)

    if state == ProcessingState.DONE:
        print(f"Job is in '{ProcessingState.DONE}' state - processing finished successfully.")
    elif state == ProcessingState.ERROR:
        print(f"Job is in '{state}' state - processing finished with errors.")
        exit(1)
    elif state == ProcessingState.FAILED:
        print(f"Job is in '{state}' state - processing failed.")
        exit(1)
    elif state == ProcessingState.CANCELLED:
        print(f"Job is in '{state}' state - processing was cancelled.")
        exit(1)


def get_job_state(adapter: Adapter):
    result = adapter.get_job()
    state = result.state if result is not None else None
    return state


def download_result(adapter: Adapter, output_path):
    result = adapter.get_result()

    if result is not None:
        with open(output_path, 'wb') as file:
            file.write(result)

        return True

    return False


def cancel_job(adapter: Adapter):
    result = adapter.patch_job_cancel()
    return result


def main():
    args = parse_args()

    connector = Connector(args.api_key, user_agent="AnnoPageClient/1.0")
    adapter = Adapter(args.api_url, connector)

    image_names = [file for file in os.listdir(args.images) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_paths = [os.path.join(args.images, name) for name in image_names]

    alto_required = args.alto is not None
    alto_paths = None

    if alto_required:
        alto_names = [os.path.splitext(name)[0] + '.xml' for name in image_names]
        alto_paths = [os.path.join(args.alto, name) for name in alto_names]

    job = None

    try:
        job = create_job(adapter, image_names, alto_required=alto_required)
        print(f"Created job with ID: {job.id}")

        upload_files(adapter, image_paths, file_type="image")
        print("Uploaded image files.")

        if alto_required:
            upload_files(adapter, alto_paths, file_type="alto")
            print("Uploaded ALTO files.")

        upload_json(adapter, args.json_config)
        print("Uploaded JSON configuration.")

        wait_for_completion(adapter, wait=args.wait)

        result = download_result(adapter, args.output)
        if result:
            print(f"Result successfully downloaded to '{args.output}'.")
        else:
            print(f"Downloading result failed.")

    except KeyboardInterrupt:
        if job is not None:
            result = cancel_job(adapter)
            if result:
                print("Job cancelled successfully.")
            else:
                print("Failed to cancel the job.")

        print("Process interrupted by user.")
        exit(1)
    except:
        if job is not None:
            result = cancel_job(adapter)
            if result:
                print("Job cancelled successfully.")
            else:
                print("Failed to cancel the job after error.")

        print("An error occurred during processing.")
        raise
    
    return 0


if __name__ == "__main__":
    exit(main())
