# AnnoPageExtraAPI

AnnoPageExtraAPI provides additional API endpoints for AnnoPage. These endpoints offer functionality that is not used during document processing but can be useful for research and development purposes. Also, the implementation shows how to use corresponding AnnoPage engines.

The following endpoints are available:
- `/text/translation`: Translates the input text from Czech to English using the MarianMT model. ([engines/translation/TranslationEngine](https://github.com/LIBCAS/AnnoPage/blob/main/anno_page/engines/translation.py))
- `/text/embedding/clip`: Generates CLIP text embeddings for the input text using the CLIP model. ([engines/embedding/HuggingfaceImageEmbeddingEngine](https://github.com/LIBCAS/AnnoPage/blob/main/anno_page/engines/embedding.py))
- `/text/embedding/siglip`: Generates SigLIP text embeddings for the input text using the SigLIP 2 model. ([engines/embedding/HuggingfaceImageEmbeddingEngine](https://github.com/LIBCAS/AnnoPage/blob/main/anno_page/engines/embedding.py))
- `/text/prompt/evaluation`: Evaluates a prompt using jinja templating. ([engines/prompt/PromptBuilderEngine](https://github.com/LIBCAS/AnnoPage/blob/main/anno_page/engines/captioning.py))

To run the AnnoPageExtraAPI, run:
```bash
python api.py
```
