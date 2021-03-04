#!/bin/bash

set -eu
poetry install
jupyter notebook --ip=* --allow-root --NotebookApp.token=$JUPYTER_TOKEN
