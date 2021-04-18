#!/bin/bash

set -eu

# install libraries
poetry install

# build documentation
jupyter nbconvert --execute  /app/docs/tutorial/rel*.ipynb --to notebook --inplace
cd docs
make clean
make html
