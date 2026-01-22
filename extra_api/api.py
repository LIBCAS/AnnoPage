import uvicorn
import fastapi
import configparser

from typing import Optional, Union, Dict, Any
from fastapi import FastAPI, APIRouter, Request, Depends, Body
from pydantic import BaseModel, ConfigDict
from contextlib import asynccontextmanager

from anno_page.engines.embedding import HuggingfaceTextEmbeddingEngine
from anno_page.engines.captioning import PromptBuilderEngine
from anno_page.engines.translation import TranslationEngine


api_router = APIRouter()


class LoadedEngines:
    translation_engine: TranslationEngine | None = None
    clip_text_embedding_engine: HuggingfaceTextEmbeddingEngine | None = None
    siglip_text_embedding_engine: HuggingfaceTextEmbeddingEngine | None = None
    prompt_builder_engine: PromptBuilderEngine | None = None


def load_engines():
    loaded_engines = LoadedEngines()
    loaded_engines.translation_engine = load_translation_engine()
    loaded_engines.clip_text_embedding_engine = load_clip_text_embedding_engine()
    loaded_engines.siglip_text_embedding_engine = load_siglip_text_embedding_engine()
    loaded_engines.prompt_builder_engine = load_prompt_builder_engine()
    return loaded_engines


def dict_to_config_section(data: dict):
    config = configparser.ConfigParser()
    section_name = "DEFAULT"
    config[section_name] = data
    return config[section_name]


def load_translation_engine() -> TranslationEngine:
    translation_engine_model_name = "Helsinki-NLP/opus-mt-cs-en"
    translation_engine_config = dict_to_config_section(data={
                                                           "TOKENIZER": translation_engine_model_name,
                                                           "MODEL": translation_engine_model_name
                                                       })
    translation_engine = TranslationEngine(config=translation_engine_config,
                                           device="cpu",
                                           config_path="")

    return translation_engine


def load_clip_text_embedding_engine() -> HuggingfaceTextEmbeddingEngine:
    clip_text_embedding_engine_config = dict_to_config_section(data={
                                                                   "MODEL": "openai/clip-vit-large-patch14",
                                                                   "PRECISION": "float16",
                                                                   "DECIMAL_PLACES": "6"
                                                               })
    clip_text_embedding_engine = HuggingfaceTextEmbeddingEngine(config=clip_text_embedding_engine_config,
                                                                device="cpu",
                                                                config_path="")

    return clip_text_embedding_engine


def load_siglip_text_embedding_engine() -> HuggingfaceTextEmbeddingEngine:
    siglip_text_embedding_engine_config = dict_to_config_section(data={
                                                                     "MODEL": "google/siglip2-large-patch16-512",
                                                                     "PRECISION": "float16",
                                                                     "DECIMAL_PLACES": "6"
                                                                 })
    siglip_text_embedding_engine = HuggingfaceTextEmbeddingEngine(config=siglip_text_embedding_engine_config,
                                                                  device="cpu",
                                                                  config_path="")

    return siglip_text_embedding_engine


def load_prompt_builder_engine() -> PromptBuilderEngine:
    prompt_builder_engine = PromptBuilderEngine()
    return prompt_builder_engine


def get_loaded_engines(request: Request) -> LoadedEngines:
    return request.app.state.loaded_engines


class PromptEvaluationBody(BaseModel):
    prompt: Union[str, Dict[str, str]]
    category: Optional[str] = None
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prompt": "For the given object of type {{ category }}, provide details about its content in JSON format with keys: "
                          "\"description_en\",  \"description_cz\",  \"caption_en\",  \"caption_cz\",  \"topics_en\",  \"topics_cz\",  \"color_en\",  \"color_cz\". "
                          "Do not include markdown or any other sequence, return just the JSON. "
                          "Values of \"description_en\" and \"description_cz\" should be a description of the image in full sentences in english and czech, respectively. "
                          "Values of \"caption_en\" and \"caption_cz\" should be one short sentence in english and czech, respectively, which could potentially appear in the document page near the image as its caption. "
                          "Values of \"topics_en\" and \"topics_cz\" should be a JSON list with each item being a topic/keyword for the image in english and czech, respectively. "
                          "Select the \"color_en\" and \"color_cz\" from the following list according to the appearance of the image in english and czech: [black-and-white, grayscale, duotone, color] and [černobílý, šedotónový, dvojbarevný, barevný]. "
                          "{% if title and title is not none %} The picture has the following title on the page: '{{ title }}'.{% endif %}"
                          "{% if document_author and document_author is not none %} The author of the document is '{{ document_author }}'.{% endif %}"
                          "{% if document_name and document_name is not none %} The name of the document is '{{ document_name }}'.{% endif %}"
                          "{% if document_year and document_year is not none %} The year of publishing is '{{ document_year }}'.{% endif %}"
                          "{% if document_type and document_type is not none %} The type of the document is '{{ document_type }}'.{% endif %}",
                "category": "photograph",
                "title": "Starý muž s holí a chlapec na cestě krajinou.",
                "metadata": {
                    "document_name": "Orbis Pictus",
                    "document_author": "Jan Ámos Komenský",
                    "document_year": 1658,
                }
            }
        }
    )


@api_router.get(
    "/text/translation",
    summary="Text Translation",
    openapi_extra={"x-order": 1},
    description="Translate text from Czech to English.",
    status_code=fastapi.status.HTTP_200_OK
)
async def text_translation(text: str, loaded_engines: LoadedEngines = Depends(get_loaded_engines)):
    if loaded_engines.translation_engine is not None:
        result = loaded_engines.translation_engine.process(text)[0]
    else:
        result = fastapi.HTTPException(status_code=fastapi.status.HTTP_503_SERVICE_UNAVAILABLE,
                                       detail="Translation engine is not available.")
    return result


@api_router.get(
    "/text/embedding/clip",
    summary="Text Embedding",
    openapi_extra={"x-order": 2},
    description="Converts text to embedding.",
    status_code=fastapi.status.HTTP_200_OK)
async def text_embedding_clip(text: str, loaded_engines: LoadedEngines = Depends(get_loaded_engines)):
    if loaded_engines.clip_text_embedding_engine is not None:
        result = loaded_engines.clip_text_embedding_engine.process(text)[0]
    else:
        result = fastapi.HTTPException(status_code=fastapi.status.HTTP_503_SERVICE_UNAVAILABLE,
                                       detail="Engine for CLIP is not available.")
    return result

@api_router.get(
    "/text/embedding/siglip",
    summary="Text Embedding",
    openapi_extra={"x-order": 3},
    description="Converts text to embedding.",
    status_code=fastapi.status.HTTP_200_OK)
async def text_embedding_siglip(text: str, loaded_engines: LoadedEngines = Depends(get_loaded_engines)):
    if loaded_engines.siglip_text_embedding_engine is not None:
        result = loaded_engines.siglip_text_embedding_engine.process(text)[0]
    else:
        result = fastapi.HTTPException(status_code=fastapi.status.HTTP_503_SERVICE_UNAVAILABLE,
                                       detail="Engine for SigLIP is not available.")
    return result


@api_router.post(
    "/prompt/evaluation",
    summary="Prompt Evaluation",
    openapi_extra={"x-order": 4},
    description="Evaluate prompt for image captioning.",
    status_code=fastapi.status.HTTP_200_OK)
async def prompt_evaluation(
        data: PromptEvaluationBody,
        loaded_engines: LoadedEngines = Depends(get_loaded_engines)):
    if loaded_engines.prompt_builder_engine is not None:
        result = loaded_engines.prompt_builder_engine.process(prompt=data.prompt,
                                                              category=data.category,
                                                              title=data.title,
                                                              metadata=data.metadata)
    else:
        result = fastapi.HTTPException(status_code=fastapi.status.HTTP_503_SERVICE_UNAVAILABLE,
                                       detail="PromptBuilderEngine is not available.")
    return result


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    fastapi_app.state.loaded_engines = load_engines()
    yield


app = FastAPI(title="AnnoPageExtraAPI", lifespan=lifespan)
app.include_router(api_router, prefix="/v1")


def main():
    uvicorn.run("anno_page.extra_api.api:app",
                host="127.0.0.1",
                port=8666,
                reload=False)


if __name__ == "__main__":
    exit(main())
