#!/bin/bash

set -eu
poetry update
poetry export --without-hashes --output requirements.txt
