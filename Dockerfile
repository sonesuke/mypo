FROM amd64/python:3.9.1-buster

RUN curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python
ENV PATH $PATH:/root/.poetry/bin
ENV SHELL /bin/bash

# for intelliJ
ADD requirements.txt /requirements.txt
RUN pip install -r /requirements.txt
