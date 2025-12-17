import configparser
import fastapi
from fastapi import Depends
from typing import Optional

from doc_api.api.routes import root_router
from doc_api.api.schemas import base_objects
from doc_api.api.authentication import require_api_key
from doc_api.db import model

from anno_page.engines.translation import TranslationEngine
from anno_page.engines.embedding import ClipTextEmbeddingEngine
from anno_page.engines.captioning import PromptBuilderEngine


def dict_to_config_section(data: dict):
    config = configparser.ConfigParser()
    section_name = "DEFAULT"
    config[section_name] = data
    return config[section_name]

translation_engine_model_name = "Helsinki-NLP/opus-mt-cs-en"
translation_engine_config = dict_to_config_section(data={
                                                       "TOKENIZER": translation_engine_model_name,
                                                       "MODEL": translation_engine_model_name
                                                   })
translation_engine = TranslationEngine(config=translation_engine_config,
                                       device="cpu",
                                       config_path="")

text_embedding_engine_config = dict_to_config_section(data={
                                                         "MODEL": "clip-ViT-L-14",
                                                         "PRECISION": "float32",
                                                         "DECIMAL_PLACES": "6"
                                                      })
text_embedding_engine = ClipTextEmbeddingEngine(config=text_embedding_engine_config,
                                                 device="cpu",
                                                 config_path="")

prompt_builder_engine = PromptBuilderEngine()


@root_router.get(
    "/v1/text/translation",
    summary="Text Translation",
    tags=["User"],
    openapi_extra={"x-order": 666},
    description="Translate text from Czech to English.",
    status_code=fastapi.status.HTTP_200_OK
)
async def text_translation(
    text: str,
    key: model.Key = Depends(require_api_key(base_objects.KeyRole.READONLY, base_objects.KeyRole.USER))):
    output = translation_engine.process(text)[0]
    return output


@root_router.get(
    "/v1/text/embedding",
    summary="Text Embedding",
    tags=["User"],
    openapi_extra={"x-order": 667},
    description="Converts text to embedding.",
    status_code=fastapi.status.HTTP_200_OK)
async def text_embedding(
        text: str,
        key: model.Key = Depends(require_api_key(base_objects.KeyRole.READONLY, base_objects.KeyRole.USER))):
    embeddings = text_embedding_engine.process(text)[0]
    return embeddings


@root_router.post(
    "/v1/prompt/evaluation",
    summary="Prompt Evaluation",
    tags=["User"],
    openapi_extra={"x-order": 668},
    description="Evaluate prompt for image captioning.",
    status_code=fastapi.status.HTTP_200_OK)
async def prompt_evaluation(
        prompt: str|dict[str, str],
        category: Optional[str] = None,
        title: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
        key: model.Key = Depends(require_api_key(base_objects.KeyRole.READONLY, base_objects.KeyRole.USER))):
    output = prompt_builder_engine.process(prompt=prompt,
                                           category=category,
                                           title=title,
                                           metadata=metadata)
    return output
