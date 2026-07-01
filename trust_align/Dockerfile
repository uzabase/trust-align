FROM --platform=linux/amd64 nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    git \
    build-essential \
    cmake \
    ninja-build \
    curl \
    wget \
    openjdk-11-jdk-headless \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

RUN python -m pip install --upgrade pip setuptools wheel Cython setuptools_scm pybind11 numpy

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt --no-build-isolation

COPY . .
