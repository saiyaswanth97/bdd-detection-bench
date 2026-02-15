FROM python:3.11-slim

# Avoid interactions during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm -rf /tmp/requirements.txt

# Set working directory
WORKDIR /workspace

# Default command
CMD ["bash"]
