poetry update && poetry run pytest tests && poetry config http-basic.pypi "__token__" "${PYPI_API_TOKEN}" &&	poetry publish --build
