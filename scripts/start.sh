#!/bin/bash

set -eu
poetry install
poetry run jupyter notebook --ip=* --allow-root
