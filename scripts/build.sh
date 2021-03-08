#!/bin/bash

set -eu
poetry install
flake8 .
mypy .
pytest tests
poetry config http-basic.pypi "__token__" "${PYPI_API_TOKEN}"
poetry build
poetry publish

sphinx-apidoc -f -o ./docs .
cd docs
make html
