#!/bin/bash

set -eu
poetry update
poetry run pytest tests
