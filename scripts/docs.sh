set -eu

sphinx-apidoc -f -o ./docs .
sphinx-build -b singlehtml ./docs ./docs/_build
