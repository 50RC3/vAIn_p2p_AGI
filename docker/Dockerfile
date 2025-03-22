FROM python:3.9-slim

# Add system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create non-root user
RUN groupadd -r vain && useradd -r -g vain vain \
    && chown -R vain:vain /app

# Install dependencies first for better caching
COPY --chown=vain:vain requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=vain:vain . .

# Switch to non-root user
USER vain

# Set production environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NODE_ENV=production

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Use tini as init
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "main.py"]
