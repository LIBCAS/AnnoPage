import logging
import secrets
from datetime import datetime, timezone
from typing import List
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import select, exc, exists, literal, or_, and_, not_, update
from sqlalchemy.ext.asyncio import AsyncSession

from api.authentication import hmac_sha256_hex
from api.database import DBError
from api.db import model
from api.schemas import base_objects


logger = logging.getLogger(__name__)

class ImageForJobDefinition(BaseModel):
    name: str
    order: int


class JobDefinition(BaseModel):
    images: List[ImageForJobDefinition]
    alto_required: bool = False
    meta_json_required: bool = False


async def create_job(db: AsyncSession, key_id: UUID, job_definition: JobDefinition) -> model.Job:
    try:
        result = await db.execute(
            select(model.Key).where(model.Key.id == key_id)
        )
        db_key = result.scalar_one_or_none()
        if db_key is None:
            raise DBError(f"Key '{key_id}' does not exist", code="KEY_NOT_FOUND", status_code=404)

        db_job = model.Job(
            owner_key_id=key_id,
            definition=job_definition.model_dump(mode="json"),
            alto_required=job_definition.alto_required,
            meta_json_required=job_definition.meta_json_required)

        db.add(db_job)

        for img in job_definition.images:
            db_image = model.Image(
                job=db_job,
                name=img.name,
                order=img.order
            )
            db.add(db_image)

        await db.commit()
        return db_job
    except exc.SQLAlchemyError as e:
        raise DBError("Failed creating new job in database", status_code=500) from e


async def get_image_by_job_and_name(db: AsyncSession, job_id: UUID, image_name: str) -> model.Image:
    try:
        result = await db.execute(
            select(model.Image).where(
                model.Image.job_id == job_id,
                model.Image.name == image_name
            )
        )
        db_image = result.scalar_one_or_none()
        if db_image is None:
            raise DBError(f"Image '{image_name}' for Job '{job_id}' does not exist", code="IMAGE_NOT_FOUND", status_code=404)
        return db_image
    except exc.SQLAlchemyError as e:
        raise DBError(f"Failed reading image from database", status_code=500) from e


async def get_job(db: AsyncSession, job_id: UUID) -> model.Job:
    try:
        result = await db.execute(
            select(model.Job).where(model.Job.id == job_id)
        )
        db_job = result.scalar_one_or_none()
        if db_job is None:
            raise DBError(f"Job '{job_id}' does not exist", code="JOB_NOT_FOUND", status_code=404)
        return db_job
    except exc.SQLAlchemyError as e:
        raise DBError(f"Failed reading job from database", status_code=500) from e


async def update_job(db: AsyncSession, job_update: base_objects.JobUpdate, worker_key_id) -> None:
    try:
        result = await db.execute(
            select(model.Job).where(model.Job.id == job_update.id)
        )
        db_job = result.scalar_one_or_none()
        if db_job is None:
            raise DBError(f"Job '{job_update.id}' does not exist", code="JOB_NOT_FOUND", status_code=404)

        if job_update.progress is not None:
            db_job.progress = job_update.progress
        if job_update.log is not None:
            if db_job.log is None:
                db_job.log = job_update.log
            else:
                db_job.log += "\n" + job_update.log
        if job_update.log_user is not None:
            if db_job.log_user is None:
                db_job.log_user = job_update.log_user
            else:
                db_job.log_user += "\n" + job_update.log_user

        last_change = datetime.now(timezone.utc)
        if job_update.state == base_objects.ProcessingState.PROCESSING and db_job.state == base_objects.ProcessingState.QUEUED:
            db_job.started_date = last_change
            db_job.state = job_update.state

        db_job.last_change = last_change

        await db.commit()

    except exc.SQLAlchemyError as e:
        raise DBError(f"Failed updating job '{job_update.id}' in database", status_code=500) from e


async def finish_job(db: AsyncSession, job_finish: base_objects.JobFinish) -> None:
    try:
        result = await db.execute(
            select(model.Job).where(model.Job.id == job_finish.id)
        )
        db_job = result.scalar_one_or_none()
        if db_job is None:
            raise DBError(f"Job '{job_finish.id}' does not exist", code="JOB_NOT_FOUND", status_code=404)

        db_job.state = job_finish.state
        if job_finish.log is not None:
            if db_job.log is None:
                db_job.log = job_finish.log
            else:
                db_job.log += "\n" + job_finish.log
        if job_finish.log_user is not None:
            if db_job.log_user is None:
                db_job.log_user = job_finish.log_user
            else:
                db_job.log_user += "\n" + job_finish.log_user

        if db_job.previous_attempts is None:
            db_job.previous_attempts = 0
        else:
            db_job.previous_attempts += 1

        db_job.progress = 1.0

        finished_date = datetime.now(timezone.utc)
        db_job.finished_date = finished_date
        db_job.last_change = finished_date

        await db.commit()

    except exc.SQLAlchemyError as e:
        raise DBError("Failed finishing job in database", status_code=500) from e


async def get_jobs(db: AsyncSession, key_id: UUID) -> List[model.Job]:
    try:
        result = await db.scalars(
            select(model.Job)
              .where(model.Job.owner_key_id == key_id)
              .order_by(model.Job.created_date.desc())
        )
        return list(result.all())
    except exc.SQLAlchemyError as e:
        raise DBError('Failed reading jobs from database', status_code=500) from e


async def get_job_for_worker(db: AsyncSession, worker_key_id: UUID) -> model.Job:
    try:
        db_job = await db.execute(
            select(model.Job)
              .where(model.Job.state == base_objects.ProcessingState.QUEUED)
              .order_by(model.Job.created_date.asc())
              .with_for_update(skip_locked=True)
              .limit(1)
        )
        job = db_job.scalar_one_or_none()

        if job is None:
            raise DBError("No job available", code="NO_JOB_AVAILABLE", status_code=404)

        job.state = base_objects.ProcessingState.PROCESSING
        job.started_date = datetime.now(timezone.utc)
        job.last_change = job.started_date
        job.worker_key_id = worker_key_id

        await db.commit()

        return job

    except exc.SQLAlchemyError as e:
        raise DBError('Failed reading jobs from database', status_code=500) from e


async def start_job(db: AsyncSession, job_id: UUID) -> bool:
    try:
        # EXISTS: is there any image not uploaded?
        img_missing = exists(
            select(literal(1))
              .select_from(model.Image)
              .where(
                  model.Image.job_id == job_id,
                  model.Image.image_uploaded.is_(False),
              )
        )

        # EXISTS: is there any ALTO not uploaded?
        alto_missing = exists(
            select(literal(1))
              .select_from(model.Image)
              .where(
                  model.Image.job_id == job_id,
                  model.Image.alto_uploaded.is_(False),
              )
        )

        # meta condition: either not required OR (required AND already uploaded)
        meta_ok = or_(
            model.Job.meta_json_required.is_(False),
            and_(
                model.Job.meta_json_required.is_(True),
                model.Job.meta_json_uploaded.is_(True),
            ),
        )

        # readiness condition:
        # - if alto not required: all images uploaded  -> NOT img_missing
        # - if alto required: all images & all alto -> NOT img_missing AND NOT alto_missing
        ready = or_(
            and_(model.Job.alto_required.is_(False), not_(img_missing)),
            and_(model.Job.alto_required.is_(True),  not_(img_missing), not_(alto_missing)),
        )

        stmt = (
            update(model.Job)
            .where(
                model.Job.id == job_id,
                model.Job.state == base_objects.ProcessingState.NEW,
                meta_ok,
                ready,
            )
            .values(
                state=base_objects.ProcessingState.QUEUED,
                last_change=datetime.now(timezone.utc),
            )
            .returning(model.Job.id)   # tells us if an update happened
        )

        res = await db.execute(stmt)
        updated = res.scalar_one_or_none() is not None
        if updated:
            await db.commit()
            return True
        return False

    except exc.SQLAlchemyError as e:
        raise DBError("Failed updating job state in database", status_code=500) from e


async def cancel_job(db: AsyncSession, job_id: UUID) -> None:
    try:
        result = await db.execute(
            select(model.Job).where(model.Job.id == job_id).with_for_update()
        )
        db_job = result.scalar_one_or_none()
        if db_job is None:
            raise DBError(f"Job '{job_id}' does not exist", code="JOB_NOT_FOUND", status_code=404)

        db_job.state = base_objects.ProcessingState.CANCELLED
        db_job.finished_date = datetime.now(timezone.utc)
        await db.commit()

    except exc.SQLAlchemyError as e:
        raise DBError("Failed updating job state in database", status_code=500) from e


async def get_image(db: AsyncSession, image_id: UUID) -> model.Image:
    try:
        result = await db.execute(
            select(model.Image).where(model.Image.id == image_id)
        )
        db_image = result.scalar_one_or_none()
        if db_image is None:
            raise DBError(f"Image '{image_id}' does not exist", code="IMAGE_NOT_FOUND", status_code=404)
        return db_image
    except exc.SQLAlchemyError as e:
        raise DBError(f"Failed reading image from database", status_code=500) from e


async def get_images(db: AsyncSession, job_id: UUID) -> List[model.Image]:
    try:
        result = await db.execute(
            select(model.Job).where(model.Job.id == job_id)
        )
        db_job = result.scalar_one_or_none()
        if db_job is None:
            raise DBError(f"Job '{job_id}' does not exist", code="JOB_NOT_FOUND", status_code=404)

        result = await db.scalars(
            select(model.Image)
              .where(model.Image.job_id == job_id)
              .order_by(model.Image.order.asc())
        )
        return list(result.all())
    except exc.SQLAlchemyError as e:
        raise DBError('Failed reading images from database', status_code=500) from e


async def get_keys(db: AsyncSession) -> List[model.Key]:
    try:
        result = await db.scalars(select(model.Key).order_by(model.Key.label))
        return list(result.all())
    except exc.SQLAlchemyError as e:
        raise DBError('Failed reading keys from database', status_code=500) from e


KEY_PREFIX = "mk_"
KEY_BYTES = 32  # 32 bytes ≈ 256-bit entropy (recommended)

def generate_raw_key() -> str:
    # URL-safe Base64 without padding-ish chars; good for headers, query, and cookies
    return KEY_PREFIX + secrets.token_urlsafe(KEY_BYTES)

async def new_key(db: AsyncSession, label: str) -> str:
    """
    Create a new API key, store HMAC(key), return the RAW key string.
    Callers must display/return this once to the user and never log it.
    """

    try:
        #result = await db.execute(
        #    select(model.Key).where(model.Key.label == label)
        #)
        #key = result.scalar_one_or_none()
        #if key is not None:
        #    raise DBError(f"Key with label '{label}' already exists", status_code=409)

        # Retry loop in the vanishingly unlikely case of a hash collision
        for _ in range(3):
            raw_key = generate_raw_key()
            key_hash = hmac_sha256_hex(raw_key)

            # ensure uniqueness before insert (cheap existence check)
            existing = await db.execute(
                select(model.Key.key_hash).where(model.Key.key_hash == key_hash)
            )
            if existing.scalar_one_or_none() is not None:
                continue  # collision; regenerate

            try:
                db.add(model.Key(
                    label=label,
                    key_hash=key_hash
                ))
                await db.commit()
                return raw_key
            except exc.SQLAlchemyError:
                await db.rollback()
                continue

    except exc.SQLAlchemyError as e:
        raise DBError("Failed adding new key to database", status_code=500) from e
    raise DBError("Failed adding new key to database", status_code=409)


async def update_key(db: AsyncSession, key_update: base_objects.KeyUpdate) -> None:
    try:
        result = await db.execute(
            select(model.Key).where(model.Key.id == key_update.id)
        )
        db_key = result.scalar_one_or_none()
        if db_key is None:
            raise DBError(f"Key '{key_update.id}' does not exist", code="KEY_NOT_FOUND", status_code=404)

        result = await db.execute(
            select(model.Key).where(model.Key.label == key_update.label)
        )
        key = result.scalar_one_or_none()
        if key is not None:
            raise DBError(f"Key label '{key_update.label}' already exists", code="KEY_LABEL_ALREADY_EXISTS", status_code=409)

        if key_update.label is not None:
            db_key.label = key_update.label
        if key_update.active is not None:
            db_key.active = key_update.active
        if key_update.role is not None:
            db_key.role = key_update.role

        await db.commit()

    except exc.SQLAlchemyError as e:
        raise DBError("Failed updating key in database", status_code=500) from e






