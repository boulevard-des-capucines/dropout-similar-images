FROM ubuntu:devel

ARG DEBIAN_FRONTEND=noninteractive

ENV APP_VENV_PATH=/opt/app-venv
ENV PATH="$APP_VENV_PATH/bin:$PATH"
ENV PROJECT_PATH=/project

RUN apt update \
    && apt install -y \
       python3 \
       python3-pip \
       python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv $APP_VENV_PATH  \
  && python3 -m pip install  \
     opencv-contrib-python-headless==4.7.0.72 \
     pandas==2.0.2 \
     matplotlib==3.7 \
     imutils==0.5.4 \
     build

COPY . $PROJECT_PATH

RUN cd $PROJECT_PATH  \
    && python3 -m build  \
    && cd dist  \
    && python3 -m pip install dataset_processor*.whl
