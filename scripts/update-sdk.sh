#!/bin/bash

set -eu
poetry update
poetry export --without-hashes --dev --output requirements.txt
