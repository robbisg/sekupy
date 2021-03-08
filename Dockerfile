FROM ubuntu:bionic

ENV TZ=Europe/Rome
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y --no-install-recommends \
    make \
    automake \
    gcc \
    swig \
    python3.6 \
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


RUN alias python=python3
RUN alias pip=pip3
RUN which pip3

RUN apt install -y python-blosc
RUN pip3 install setuptools

RUN pip3 install cython wheel
RUN pip3 install tk cytoolz

RUN pip3 install --upgrade pip
RUN pip3 install flake8 pytest
RUN pip3 install numpy

ADD requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

COPY ./* $HOME/pyitab/
WORKDIR ./pyitab/
RUN cd pyitab & ls
RUN python3 setup.py develop

ENTRYPOINT ["/usr/local/bin/bash"]
