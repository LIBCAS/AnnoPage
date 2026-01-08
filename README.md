# AnnoPage

AnnoPage is software being developed as part of the [Orbis Pictus project](https://orbis.lib.cas.cz/).

The goal is to create machine-learning–based software that automatically analyzes the content of a digitized page. The system identifies areas of the page that contain non-textual elements, marks these areas, and determines their type (map, illustration, photograph, drawing, table, graph, etc.). The system will then attempt to produce a brief textual description of these areas along with their boundaries (identification of objects using keywords, identification of people, finding and using textual annotations, etc.). This information will be used to supplement standard metadata, which can then be further indexed for search purposes and displayed to users in digital libraries using the Kramerius system, for example through the [Czech Digital Library](https://ceskadigitalniknihovna.cz/).

The development is carried out by the [Orbis Pictus project team](https://orbis.lib.cas.cz/resitelsky-tym/).

## Documentation

For documentation see the [Wiki](https://github.com/LIBCAS/AnnoPage/wiki).

## References

If you use AnnoPage in your research, please cite the following publications:

* Kišš, M., Hradiš, M., Dvořáková, M., Jiroušek, V., Kersch, F. (2026). AnnoPage Dataset: Dataset of Non-textual Elements in Documents with Fine-Grained Categorization. In: Jin, L., Zanibbi, R., Eglin, V. (eds) Document Analysis and Recognition – ICDAR 2025 Workshops. ICDAR 2025. Lecture Notes in Computer Science, vol 16226. Springer, Cham. https://doi.org/10.1007/978-3-032-09371-4_4

## Installation

Currently, AnnoPage can only be installed directly from the GitHub repository.
To install AnnoPage using `pip`, run the following command:

```bash
pip install "anno-page[full] @ git+https://github.com/LIBCAS/AnnoPage.git"
```

The `full` option installs all optional dependencies required for complete functionality.
If you want to run only the API or client, you can install AnnoPage using the `api` or `client` options, respectively.
In case you want to run the processing tool, you need to install the `tool` or the `full` option.

Since the installation is done directly from the GitHub repository, which also contains the publicly available models, it is recommended to set environment variable 
```bash 
export GIT_LFS_SKIP_SMUDGE=1     # Linux/MacOS
$env:GIT_LFS_SKIP_SMUDGE = "1"   # Windows (PowerShell)
``` 
to not download the models in case you have `git lfs` installed.
