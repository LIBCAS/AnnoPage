import os
import json
import logging

logger = logging.getLogger(__name__)


def compose_path(file_path, reference_path):
    if reference_path and not os.path.isabs(file_path):
        file_path = os.path.join(reference_path, file_path)
    return file_path


def config_get_list(config, key, fallback=None, make_lowercase=False):
    if key not in config:
        return fallback

    try:
        value = json.loads(config[key])
    except json.decoder.JSONDecodeError as e:
        logger.info(f'Failed to parse list from config key "{key}", returning fallback {fallback}:\n{e}')
        return fallback

    if make_lowercase:
        value = [str(item).lower() if isinstance(item, str) else item for item in value]

    return value
