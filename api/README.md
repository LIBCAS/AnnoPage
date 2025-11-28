# Worker, client, and routes for AnnoPageAPI

This directory contains the implementation of the worker, client, and routes for AnnoPageAPI, which provides endpoints for managing processing jobs for AnnoPage. The API is based on [DocAPI](https://github.com/DCGM/DocAPI) and in this repository, we have custom implementations for the worker, client, and AnnoPage-specific routes.

## Worker
The worker is responsible for processing jobs using AnnoPage. It periodically checks for new jobs, processes them, sends the results back to API, and updates their status in the API. Example usage of the worker:
```bash
python worker.py \
    --api-url=ANNOPAGE_API_URL \
    --api-key=ANNOPAGE_API_KEY \
    --base-dir=/path/to/base/processing/dir \
    --polling-interval=5 \
    --logging-level=DEBUG
```

## Client

The client provides a way to interact with the AnnoPageAPI programmatically. It allows you to create a job and, when it is finished, to download the results. Example usage of the client to create a processing job with various output options:
```bash
python client.py \
    --images=/path/to/directory/with/images \
    --metadata=/path/to/metadata.json \
    --image-captioning-settings=/path/to/image_captioning_settings.json
    --api-key=ANNOPAGE_API_KEY \
    --api-url=ANNOPAGE_API_URL \
    --output=/path/to/output/directory \
    --output-alto \
    --output-embeddings \
    --output-embeddings-jsonlines \
    --output-renders \
    --output-crops \
    --output-image-captioning-prompts
```

You can also use the client to list all the available processing engines:
```bash
python client.py \
    --list-engines \
    --api-key=ANNOPAGE_API_KEY \
    --api-url=ANNOPAGE_API_URL \
```
