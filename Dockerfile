FROM ubuntu:devel

ARG DEBIAN_FRONTEND=noninteractive

ENV APP_VENV_PATH=/opt/app-venv
ENV PATH="$APP_VENV_PATH/bin:$PATH"

RUN apt update \
    && apt install -y \
       python3 \
       python3-pip \
       python3-venv  \
    && python3 -m venv $APP_VENV_PATH

RUN pip3 install \
    opencv-contrib-python-headless==4.7.0.72 \
    imutils==0.5.4 \
    matplotlib==3.7 \
    seaborn==0.12 \
    scikit-learn==1.2
