
FROM python:3.12-slim

# Python runtime settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install only the serving dependencies
COPY docker/requirements-serving.txt /app/requirements-serving.txt

RUN pip install --no-cache-dir \
    -r /app/requirements-serving.txt

# Copy application code and configuration
COPY src/ /app/src/
COPY configs/ /app/configs/

# Create a non-root user
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Verify the API health endpoint
HEALTHCHECK --interval=30s \
    --timeout=5s \
    --start-period=10s \
    --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
