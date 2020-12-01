ARG PYTORCH="1.7.0-cuda11.0-cudnn8-runtime"

FROM pytorch/pytorch:${PYTORCH}

RUN apt-get update && apt-get install -y git

RUN git clone https://github.com/dacon-ai/torso-ct.git /torso-ct

WORKDIR /torso-ct

RUN python -m pip install --upgrade pip

RUN python -m pip install -r requirements.txt
