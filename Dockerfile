FROM amd64/python:3.9.2-slim-buster

RUN apt update && \
    apt install -y make && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

ADD poetry.toml /
ADD pyproject.toml /

RUN pip install poetry && \
    poetry install --no-root
