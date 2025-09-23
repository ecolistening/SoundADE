FROM python:3.10-slim-bookworm

# install global dependencies
RUN apt-get update
RUN apt-get install -y --no-install-recommends curl ca-certificates git ffmpeg
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# install UV
ADD https://astral.sh/uv/0.7.13/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

# move code into container
ADD . /app
WORKDIR /app

# install local dependencies
RUN uv sync --locked
