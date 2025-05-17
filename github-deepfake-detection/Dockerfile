# Use Python 3.10 slim base image
FROM python:3.10-slim

# Set up system user (required by Hugging Face)
RUN useradd -m -u 1000 user

# Install system dependencies as root
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libpng16-16 \
    libfreetype6 \
    && rm -rf /var/lib/apt/lists/*

# Switch to the non-root user AFTER installing packages
USER user
ENV PATH="/home/user/.local/bin:${PATH}"

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire application (including models)
COPY --chown=user . .

# Hugging Face requires port 7860
EXPOSE 7860

# Start Gunicorn server
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "2", "--timeout", "120", "app:app"]