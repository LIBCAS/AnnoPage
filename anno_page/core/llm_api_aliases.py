import json

_llm_api_aliases = None


def get_llm_api_aliases():
    if _llm_api_aliases is None:
        raise ValueError("LLM API aliases has not been initialized yet")

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
