#!/bin/bash

set -eu
isort .
black .
flake8 .
mypy .
pytest