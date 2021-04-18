FROM python:3.7.10-slim-buster

RUN { \
        echo 'deb http://deb.debian.org/debian buster main contrib non-free'; \
        echo 'deb-src http://deb.debian.org/debian buster main contrib non-free'; \
        echo 'deb http://deb.debian.org/debian-security/ buster/updates main contrib non-free'; \
        echo 'deb-src http://deb.debian.org/debian-security/ buster/updates main contrib non-free'; \
        echo 'deb http://deb.debian.org/debian buster-updates main contrib non-free'; \
        echo 'deb-src http://deb.debian.org/debian buster-updates main contrib non-free'; \
    } > /etc/apt/sources.list

RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y make pandoc g++ intel-mkl && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/lib/x86_64-linux-gnu/libblas.so     libblas.so-x86_64-linux-gnu      /usr/lib/x86_64-linux-gnu/libmkl_rt.so 150
RUN update-alternatives --install /usr/lib/x86_64-linux-gnu/libblas.so.3   libblas.so.3-x86_64-linux-gnu    /usr/lib/x86_64-linux-gnu/libmkl_rt.so 150
RUN update-alternatives --install /usr/lib/x86_64-linux-gnu/liblapack.so   liblapack.so-x86_64-linux-gnu    /usr/lib/x86_64-linux-gnu/libmkl_rt.so 150
RUN update-alternatives --install /usr/lib/x86_64-linux-gnu/liblapack.so.3 liblapack.so.3-x86_64-linux-gnu  /usr/lib/x86_64-linux-gnu/libmkl_rt.so 150

RUN echo "/usr/lib/x86_64-linux-gnu"     >  /etc/ld.so.conf.d/mkl.conf
RUN ldconfig

ADD poetry.toml /
ADD poetry.lock /
ADD pyproject.toml /

RUN pip install poetry && \
    poetry install --no-root
