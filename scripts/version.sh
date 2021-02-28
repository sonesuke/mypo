#!/bin/bash

set -eu
if [ $# != 1 ]
then
  echo "bash version.sh version-number"
  exit 1
fi
sed -e "s/__version__ = \".*\"/__version__ = \"${1}\"/" -i /app/mypo/__init__.py
sed -e "s/version = \".*\"/version = \"${1}\"/" -i /app/pyproject.toml