# ================================================
# BUILD STAGE
# ================================================
# RUN pip install -U pip virtualenv

FROM python:3.10-slim-buster AS builder

ENV POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VER=1.2.0

# add poetry to the path
ENV PATH="$POETRY_HOME/bin:$PATH"

# install poetry and update packages
RUN apt-get update \
    && apt-get -y install --no-install-recommends \ 
        curl \
        build-essential \
        libffi-dev \
    #&& apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && pip install "poetry==${POETRY_VER}" --ignore-installed

WORKDIR /app
# create virtualenv with package deps
COPY poetry.lock pyproject.toml ./
RUN python -m venv --copies /app/venv
RUN . /app/venv/bin/activate \
    && poetry install --only main --no-root --no-interaction --no-ansi

# ================================================
# PRODUCTION STAGE
# ================================================
FROM python:3.10-slim-buster AS prod

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/venv/bin:$PATH"

RUN apt-get update && apt-get install -y curl

# copy the venv folder from builder image 
COPY --from=builder /app/venv /app/venv/
WORKDIR /app
COPY ./ ./
#HEALTHCHECK --interval=5s --timeout=3s \
#  CMD curl -f http://localhost/health || exit 1
# deploy api
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "80"]
# try 127.0.0.1
