FROM python:3.10-slim

ENV PATH="/root/.local/bin:/cargo/bin:$PATH"
ENV DATA_PATH="/data"
ENV VIRTUAL_ENV_PATH="/code/.venv"
ENV PROFILE_PATH="/data/run-environment/profile"

RUN mkdir -p /code /cargo

RUN apt-get update && \
  apt-get install --no-install-recommends -y \
  curl \
  git \
  ffmpeg && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | CARGO_HOME=/cargo sh

COPY uv.lock pyproject.toml /code/
COPY src /code/src
COPY scripts /code/scripts
COPY pipeline-steps.sh /code/pipeline-steps.sh
COPY README.md /code/README.md

WORKDIR /code

RUN uv venv $VIRTUAL_ENV_PATH
RUN . $VIRTUAL_ENV_PATH/bin/activate
RUN uv sync --no-cache --link-mode=copy

ENV PYTHONPATH="/code/src"

ENTRYPOINT ["/bin/bash", "-c", "cd /code && . /code/.venv/bin/activate && ./pipeline-steps.sh"]
