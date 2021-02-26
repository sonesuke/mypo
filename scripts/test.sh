#!/bin/bash

set -eu
poetry run flake8 .
poetry run mypy .
poetry run pytest tests