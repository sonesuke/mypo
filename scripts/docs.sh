#!/bin/bash

set -eu
poetry install
sphinx-build -b singlehtml ./docs ./docs/_build
