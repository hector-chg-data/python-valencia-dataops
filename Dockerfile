FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git \
      build-essential \
      libpq-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/requirements.txt

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# The application code will be provided via bind mount in docker-compose,
# but copying it here keeps the image usable on its own.
COPY . /workspace

