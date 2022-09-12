FROM ubuntu:bionic

ENV TZ=Europe/Rome
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y --no-install-recommends \
    make \
    automake \
    gcc \
    swig \
    python3.8 \
    python-tk \
    build-essential \
    gfortran \
    g++ \
    cpp \
    libc6-dev \
    man-db \
    autoconf \
    pkg-config \
    python3-pip \
    python3-dev


RUN pip3 install --upgrade pip

RUN pip3 install cython wheel setuptools
RUN pip3 install flake8 pytest pytest-cov
RUN pip3 install tk cytoolz
RUN pip3 install numpy

ADD requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

COPY ./ $HOME/pyitab/
WORKDIR ./pyitab/
RUN cd pyitab & ls
RUN pip3 install .
RUN python3 -m pytest --pyargs pyitab

ENTRYPOINT ["/usr/local/bin/bash"]
