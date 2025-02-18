FROM python:3.10-slim

RUN apt-get update -qq -y \
    && apt-get install curl libgomp1 make tree vim -qq -y \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean -qq -y

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/energy-demand-forecasting/.venv

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY . /energy-demand-forecasting

WORKDIR /energy-demand-forecasting

RUN uv sync --frozen --no-cache

ENV PATH="/energy-demand-forecasting/.venv/bin:$PATH" \
    UV_TOOL_BIN_DIR=/opt/uv-bin/

CMD ["make"]