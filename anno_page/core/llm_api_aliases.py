import json
import logging


logger = logging.getLogger(__name__)

_llm_api_aliases = None


def get_llm_api_aliases():
    if _llm_api_aliases is None:
        logger.warning("LLM API aliases have not been loaded yet. Please call load_llm_api_aliases(path) first.")
        return {}

    return _llm_api_aliases


def load_llm_api_aliases(path, reload=False):
    global _llm_api_aliases
    if _llm_api_aliases is None or reload is True:
        with open(path, 'r') as f:
            api_aliases = json.load(f)

        _llm_api_aliases = {}
        for api_alias in api_aliases:
            for alias in api_alias["aliases"]:
                _llm_api_aliases[alias.lower()] = api_alias["urls"]

    return _llm_api_aliases
