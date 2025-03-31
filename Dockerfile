FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app

# Make config directory and ensure it's writable
RUN mkdir -p /app/config/backups && \
    chmod -R 777 /app/config

# Create default configs at build time
RUN python -c "from tools.config_manager import ConfigManager; ConfigManager().create_default_configs()"

# Default command runs with interactive configuration
ENTRYPOINT ["python"]
CMD ["main.py", "--config", "/app/config/network.json"]

# Environment variables that can be overridden at runtime
ENV NODE_ENV=development \
    PORT=8000 \
    LOG_LEVEL=INFO \
    INTERACTION_LEVEL=NORMAL
