import json
import logging
import os

import aiofiles
from fastapi import Depends, HTTPException, status, UploadFile, File
from fastapi.responses import FileResponse

from aiofiles import os as aiofiles_os

from sqlalchemy.ext.asyncio import AsyncSession

from api.authentication import require_api_key
from api.cruds import cruds
from api.database import get_async_session
from api.routes.route_guards import challenge_worker_access_to_job
from api.schemas import base_objects
from api.db import model
from api.routes import worker_router
from api.config import config

from typing import List
from uuid import UUID


logger = logging.getLogger(__name__)


require_worker_key = require_api_key(key_role=base_objects.KeyRole.WORKER)


@worker_router.get("/job", response_model=base_objects.Job, tags=["Worker"])
async def get_job(
        key: model.Key = Depends(require_worker_key),
        db: AsyncSession = Depends(get_async_session)):
    db_job = await cruds.get_job_for_worker(db, key.id)
    return db_job


@worker_router.get("/images/{job_id}", response_model=List[base_objects.Image], tags=["Worker"])
async def get_images_for_job(job_id: UUID,
        key: model.Key = Depends(require_worker_key),
        db: AsyncSession = Depends(get_async_session)):
    await challenge_worker_access_to_job(db, key, job_id)

    db_images = await cruds.get_images(db, job_id)
    return [base_objects.Image.model_validate(db_image) for db_image in db_images]


@worker_router.get("/image/{job_id}/{image_id}", response_class=FileResponse, tags=["Worker"])
async def get_image(job_id: UUID, image_id: UUID,
        key: model.Key = Depends(require_worker_key),
        db: AsyncSession = Depends(get_async_session)):
    await challenge_worker_access_to_job(db, key, job_id)

    db_image = await cruds.get_image(db, image_id)
    if not db_image.image_uploaded:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "IMAGE_NOT_UPLOADED", "message": f"Image '{db_image.name}' (ID: {image_id}) is not uploaded"},
        )
    image_path = os.path.join(config.BATCH_UPLOADED_DIR, str(db_image.job_id), f"{db_image.id}.jpg")
    return FileResponse(image_path, media_type="image/jpeg", filename=db_image.name)


@worker_router.get("/alto/{job_id}/{image_id}", response_class=FileResponse, tags=["Worker"])
async def get_alto(job_id: UUID, image_id: UUID,
        key: model.Key = Depends(require_worker_key),
        db: AsyncSession = Depends(get_async_session)):
    await challenge_worker_access_to_job(db, key, job_id)

    db_image = await cruds.get_image(db, image_id)
    if not db_image.alto_uploaded:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "ALTO_NOT_UPLOADED", "message": f"ALTO for image '{db_image.name}' (ID: {image_id}) is not uploaded"},
        )
    alto_path = os.path.join(config.BATCH_UPLOADED_DIR, str(db_image.job_id), f"{db_image.id}.xml")
    return FileResponse(alto_path, media_type="application/xml", filename=f"{os.path.splitext(db_image.name)[0]}.xml")


@worker_router.get("/meta_json/{job_id}", response_class=FileResponse, tags=["Worker"])
async def get_meta_json(job_id: UUID,
        key: model.Key = Depends(require_worker_key),
        db: AsyncSession = Depends(get_async_session)):
    await challenge_worker_access_to_job(db, key, job_id)

    db_job = await cruds.get_job(db, job_id)
    if not db_job.meta_json_uploaded:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "META_JSON_NOT_UPLOADED", "message": f"Meta JSON for job '{job_id}' is not uploaded"},
        )
    meta_json_path = os.path.join(config.BATCH_UPLOADED_DIR, str(job_id), "meta.json")
    return FileResponse(meta_json_path, media_type="application/json", filename="meta.json")


@worker_router.put("/update_job/{job_id}", tags=["Worker"])
async def update_job(job_id: UUID,
        job_update: base_objects.JobUpdate,
        key: model.Key = Depends(require_worker_key),
        db: AsyncSession = Depends(get_async_session)):
    await challenge_worker_access_to_job(db, key, job_id)

    db_job = await cruds.get_job(db, job_id)
    if db_job.state not in {base_objects.ProcessingState.QUEUED, base_objects.ProcessingState.PROCESSING}:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "JOB_NOT_UPDATABLE", "message": f"Job '{job_id}' must be in '{base_objects.ProcessingState.QUEUED.value}' or '{base_objects.ProcessingState.PROCESSING.value}' state to be updated, current state: '{db_job.state.value}'"},
        )
    await cruds.update_job(db, job_update, key.id)
    return {"code": "JOB_UPDATED", "message": f"Job '{job_id}' updated successfully"}


@worker_router.post("/result/{job_id}", tags=["Worker"])
async def upload_result(
    job_id: UUID,
    result: UploadFile = File(...),
    key: model.Key = Depends(require_worker_key),
    db: AsyncSession = Depends(get_async_session)):
    await challenge_worker_access_to_job(db, key, job_id)

    db_job = await cruds.get_job(db, job_id)
    if db_job.state != base_objects.ProcessingState.PROCESSING:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "JOB_NOT_PROCESSING",
                "message": (
                    f"Job '{job_id}' must be in "
                    f"'{base_objects.ProcessingState.PROCESSING.value}' state to upload result, "
                    f"current state: '{db_job.state.value}'"
                ),
            },
        )

    await aiofiles_os.makedirs(config.RESULTS_DIR, exist_ok=True)
    await aiofiles_os.makedirs(os.path.join(config.RESULTS_DIR, str(job_id)), exist_ok=True)
    result_file_path = os.path.join(config.RESULTS_DIR, str(job_id), f"{job_id}.zip")

    async with aiofiles.open(result_file_path, "wb") as f:
        while chunk := await result.read(1024 * 1024):  # 1MB chunks
            await f.write(chunk)

    return {"code": "RESULT_UPLOADED", "message": f"Result for job '{job_id}' uploaded successfully"}


@worker_router.post("/finish_job/{job_id}", tags=["Worker"])
async def finish_job(job_id: UUID,
        job_finish: base_objects.JobFinish,
        key: model.Key = Depends(require_worker_key),
        db: AsyncSession = Depends(get_async_session)):
    await challenge_worker_access_to_job(db, key, job_id)

    db_job = await cruds.get_job(db, job_id)
    if db_job.state != base_objects.ProcessingState.PROCESSING:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "JOB_NOT_FINISHABLE", "message": f"Job '{job_id}' must be in '{base_objects.ProcessingState.PROCESSING.value}' state, current state: '{db_job.state.value}'"},
        )
    if job_finish.state == base_objects.ProcessingState.DONE:
        result_path = os.path.join(config.RESULTS_DIR, str(job_id), f"{job_id}.zip")
        if not await aiofiles_os.path.exists(result_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "RESULT_NOT_FOUND", "message": f"Result file for job '{job_id}' not found at expected location: '{result_path}'"},
            )
    await cruds.finish_job(db, job_finish)
    return {"code": "JOB_FINISHED", "message": f"Job '{job_id}' finished successfully"}