from typing import Optional, List, Iterator, Any
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


class ElementEmbeddings(RootModel[List[Element]]):
    def __init__(self, items: Optional[List[Element]] = None):
        super().__init__(items or [])

    def __iter__(self) -> Iterator[Element]:
        return iter(self.root)
    def __len__(self) -> int:
        return len(self.root)
    def __getitem__(self, i: int) -> Element:
        return self.root[i]
    def __setitem__(self, i: int, e: Element) -> None:
        self.root[i] = e

    def append(self, e: Element) -> None:
        self.root.append(e)
    def extend(self, l) -> None:
        self.root.extend(l)
    def insert(self, i: int, e: Element) -> None:
        self.root.insert(i, e)
    def pop(self, i: int = -1) -> Element:
        return self.root.pop(i)
    def clear(self) -> None:
        self.root.clear()

    def model_dump_jsonlines(self, **kwargs: Any) -> str:
        return "\n".join(e.model_dump_json(**kwargs) for e in self.root)
