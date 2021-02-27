#!/bin/bash

set -eu
poetry update
poetry run flake8 .
poetry run mypy .
poetry run pytest tests