# syntax=docker/dockerfile:1
#
# Layering for fast CI:
#   1) runtime-deps  — torch/transformers/etc. from docker/requirements.runtime.txt
#   2) app           — only `src/` + `pip install --no-deps -e .`
#
# Publish a shared deps image (optional, rebuild when requirements change):
#   bash scripts/docker-export-requirements.sh
#   docker build -f docker/Dockerfile.runtime-deps \
#     -t docker.cat/gitlab/express/runtime-deps:py3.12 .
#
# Point CI build at prebuilt deps (if your docker template supports build-arg):
#   docker build --build-arg RUNTIME_IMAGE=docker.cat/gitlab/express/runtime-deps:py3.12 .

ARG PYTHON_IMAGE=docker.cat/docker/python:3.12
ARG RUNTIME_IMAGE=

FROM ${PYTHON_IMAGE} AS runtime-deps-build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_INDEX_URL=http://pypi.dog.cat/root/pypi/+simple/ \
    PIP_TRUSTED_HOST=pypi.dog.cat \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /deps
COPY docker/requirements.runtime.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

FROM runtime-deps-build AS app-with-local-deps

FROM ${RUNTIME_IMAGE:-app-with-local-deps} AS app
WORKDIR /app
COPY pyproject.toml README.md LICENSE.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --no-deps -e .

CMD ["express"]
