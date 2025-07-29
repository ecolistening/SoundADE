FROM python:3.10-slim-bookworm

RUN apt-get update
RUN apt-get install --no-install-recommends -y curl git ffmpeg
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/0.7.8/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin:$PATH"

ADD . /app
WORKDIR /app
RUN uv sync --locked

ENTRYPOINT []
CMD ["uv", "run", "main.py", "pipeline"]
