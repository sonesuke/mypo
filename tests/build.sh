#!/bin/bash

set -eu
poetry update
poetry run pytest tests
poetry config http-basic.pypi "__token__" "${PYPI_API_TOKEN}"
poetry build
poetry publish
