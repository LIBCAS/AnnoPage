import os
from doc_api.run import main as doc_api_main


def main():
    config = {
        "SERVER_NAME": "AnnoPageAPI",
        "APP_VERSION": "1.0.0",
        "APP_HOST": "0.0.0.0",
        "APP_PORT": "8000",
        "PRODUCTION": "False",
        "KEY_PREFIX": "annopage",
        "BASE_DIR": "./annopage_api_data",
    }

    for key, value in config.items():
        if not os.environ.get(key, None):
            os.environ[key] = value

    doc_api_main()


if __name__ == "__main__":
    exit(main())
