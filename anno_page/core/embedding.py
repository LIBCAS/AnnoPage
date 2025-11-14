from typing import Optional, List, Iterator, Any
from pydantic import BaseModel, RootModel

from anno_page import globals


class ProcessingInfo(BaseModel):
    system: str = globals.software_name
    version: str = globals.software_version
    datetime: str
    model: str
    decimal_places: Optional[int]
    precision: str


class ObjectEmbedding(BaseModel):
    id: str
    tag_id: str
    page_uuid: str
    category: str
    processing_info: ProcessingInfo
    embedding: list[float]
