#!/bin/bash

set -eu
poetry install
run flake8 .
run mypy .
run pytest tests
poetry config http-basic.pypi "__token__" "${PYPI_API_TOKEN}"
poetry build
poetry publish
