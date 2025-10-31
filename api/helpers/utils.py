import os
import tempfile
import zipfile


def get_temp_directory():
    return tempfile.gettempdir()


def create_zip_archive(path, files):
    with open(path, "wb") as zip_file:
        with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zip_archive:
            for file in files:
                zip_archive.write(file, arcname=os.path.basename(file))
