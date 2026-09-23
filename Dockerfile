# Pin the base image so identical source commits produce reproducible layers.
FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea

# Apply Debian security updates before installing the application.
RUN apt-get update && \
    apt-get upgrade -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir . && \
    adduser --disabled-password --gecos "" --uid 1000 appuser
USER appuser
EXPOSE 11435
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:11435/health')" || exit 1
CMD ["ollama-queue-proxy"]
