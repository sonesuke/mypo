#!/bin/bash

set -eu
poetry install
sphinx-apidoc -f -o ./docs .
cd docs
make html
