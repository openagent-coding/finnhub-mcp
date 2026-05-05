FROM python:3.13-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY server.py .
RUN uv sync --frozen --no-dev

ENV MCP_PORT=8000
EXPOSE ${MCP_PORT}

CMD ["uv", "run", "python", "server.py"]
