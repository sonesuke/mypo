poetry update && poetry run pytest tests && oetry config http-basic.pypi "__token__" "${PYPI_API_TOKEN}" &&	poetry publish --build
