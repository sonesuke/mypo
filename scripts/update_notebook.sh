#!/bin/bash

set -eu

# build documentation
jupyter nbconvert --execute  /app/docs/tutorial/*.ipynb --to notebook --inplace
