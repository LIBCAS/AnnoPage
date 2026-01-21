import configparser
import fastapi
from fastapi import Depends
from typing import Optional

from doc_api.api.routes import root_router
from doc_api.api.schemas import base_objects, responses
from doc_api.api.authentication import require_api_key
from doc_api.db import model

from anno_page.engines.translation import TranslationEngine
from anno_page.engines.embedding import HuggingfaceTextEmbeddingEngine
from anno_page.engines.captioning import PromptBuilderEngine


class LoadedEngines:
    translation_engine: TranslationEngine | None = None
    clip_text_embedding_engine: HuggingfaceTextEmbeddingEngine | None = None
    siglip_text_embedding_engine: HuggingfaceTextEmbeddingEngine | None = None
    prompt_builder_engine: PromptBuilderEngine | None = None


class RouteDefinition:
    def __init__(self,
                 path: str,
                 endpoint,
                 methods: list[str],
                 summary: Optional[str] = None,
                 tags: Optional[list[str]] = None,
                 description: Optional[str] = None,
                 status_code: Optional[int] = None,
                 openapi_extra: Optional[dict] = None):
        self.path = path
        self.endpoint = endpoint
        self.methods = methods
        self.summary = summary
        self.tags = tags
        self.description = description
        self.status_code = status_code
        self.openapi_extra = openapi_extra


loaded_engines = None


def setup():
    global loaded_engines
    if loaded_engines is None:
        loaded_engines = LoadedEngines()

    load_engines(loaded_engines)
    load_routes()


def load_engines(loaded_engines: LoadedEngines):
    loaded_engines.translation_engine = load_translation_engine()
    loaded_engines.clip_text_embedding_engine = load_clip_text_embedding_engine()
    loaded_engines.siglip_text_embedding_engine = load_siglip_text_embedding_engine()
    loaded_engines.prompt_builder_engine = load_prompt_builder_engine()


def load_routes():
    for route in get_routes():
        root_router.add_api_route(path=route.path,
                                  endpoint=route.endpoint,
                                  methods=route.methods,
                                  summary=route.summary,
                                  tags=route.tags,
                                  description=route.description,
                                  status_code=route.status_code,
                                  openapi_extra=route.openapi_extra)


def get_routes() -> list[RouteDefinition]:
    routes = [
        RouteDefinition(
            path="/v1/text/translation",
            endpoint=text_translation,
            methods=["GET"],
            summary="Text Translation",
            tags=["User"],
            description="Translate text from Czech to English.",
            status_code=fastapi.status.HTTP_200_OK,
            openapi_extra={"x-order": 666}
        ),
        RouteDefinition(
            path="/v1/text/embedding/clip",
            endpoint=text_embedding_clip,
            methods=["GET"],
            summary="Text Embedding using CLIP",
            tags=["User"],
            description="Converts text to embedding using CLIP model.",
            status_code=fastapi.status.HTTP_200_OK,
            openapi_extra={"x-order": 667}
        ),
        RouteDefinition(
            path="/v1/text/embedding/siglip",
            endpoint=text_embedding_siglip,
            methods=["GET"],
            summary="Text Embedding using SigLIP",
            tags=["User"],
            description="Converts text to embedding using SigLIP model.",
            status_code=fastapi.status.HTTP_200_OK,
            openapi_extra={"x-order": 668}
        ),
        RouteDefinition(
            path="/v1/prompt/evaluation",
            endpoint=prompt_evaluation,
            methods=["POST"],
            summary="Prompt Evaluation",
            tags=["User"],
            description="Evaluate prompt for image captioning.",
            status_code=fastapi.status.HTTP_200_OK,
            openapi_extra={"x-order": 669}
        )
    ]
    return routes


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


async def text_translation(
    text: str,
    key: model.Key = Depends(require_api_key(base_objects.KeyRole.READONLY, base_objects.KeyRole.USER))):
    if loaded_engines.translation_engine is not None:
        result = loaded_engines.translation_engine.process(text)[0]
    else:
        result = responses.DocAPIResponseServerError(status=fastapi.status.HTTP_503_SERVICE_UNAVAILABLE, code='INTERNAL_ERROR', detail="Translation engine is currently not available.")
    return result


async def text_embedding_clip(
        text: str,
        key: model.Key = Depends(require_api_key(base_objects.KeyRole.READONLY, base_objects.KeyRole.USER))):
    if loaded_engines.clip_text_embedding_engine is not None:
        result = loaded_engines.clip_text_embedding_engine.process(text)[0]
    else:
        result = responses.DocAPIResponseServerError(status=fastapi.status.HTTP_503_SERVICE_UNAVAILABLE, code='INTERNAL_ERROR', detail="Clip text embedding engine is currently not available.")
    return result


async def text_embedding_siglip(
        text: str,
        key: model.Key = Depends(require_api_key(base_objects.KeyRole.READONLY, base_objects.KeyRole.USER))):
    if loaded_engines.siglip_text_embedding_engine is not None:
        result = loaded_engines.siglip_text_embedding_engine.process(text)[0]
    else:
        result = responses.DocAPIResponseServerError(status=fastapi.status.HTTP_503_SERVICE_UNAVAILABLE, code='INTERNAL_ERROR', detail="Siglip text embedding engine is currently not available.")
    return result


async def prompt_evaluation(
        prompt: str|dict[str, str],
        category: Optional[str] = None,
        title: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
        key: model.Key = Depends(require_api_key(base_objects.KeyRole.READONLY, base_objects.KeyRole.USER))):
    if loaded_engines.prompt_builder_engine is not None:
        result = loaded_engines.prompt_builder_engine.process(prompt=prompt,
                                                              category=category,
                                                              title=title,
                                                              metadata=metadata)
    else:
        result = responses.DocAPIResponseServerError(status=fastapi.status.HTTP_503_SERVICE_UNAVAILABLE, code='INTERNAL_ERROR', detail="Prompt builder engine is currently not available.")
    return result
