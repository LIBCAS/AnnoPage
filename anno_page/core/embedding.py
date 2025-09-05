from typing import Optional
from pydantic import BaseModel, RootModel


class ProcessingInfo(BaseModel):
    system: str = "AnnoPage"
    version: str = "0.1"
    datetime: str
    model: str
    precision: Optional[int]


class Element(BaseModel):
    id: str
    alto_id: str
    page_id: str
    category: str
    processing_info: ProcessingInfo
    embedding: list[float]


class ElementEmbeddings(RootModel[list[Element]]):
    pass
