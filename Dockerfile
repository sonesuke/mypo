FROM amd64/python:3.9.2-slim-buster

RUN apt update && \
    apt install -y make && \
    rm -rf /var/lib/apt/lists/*
ADD https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py /get-poetry.py
RUN python get-poetry.py
ENV PATH $PATH:/root/.poetry/bin
ENV SHELL /bin/bash

# for intelliJ
ADD requirements.txt /requirements.txt
RUN pip install -r /requirements.txt
