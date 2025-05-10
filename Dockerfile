FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Create models directory
RUN mkdir -p models

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files (excluding models in .dockerignore)
COPY . .

# Copy models directly (no need for curl since we'll include them in build)
COPY models/best_deepfake_detector_resnet18.pth ./models/
COPY models/audio_model.h5 ./models/
COPY models/deepfake_video_new.h5 ./models/

# Environment variables
ENV PORT=10000
ENV MODELS_DIR=/app/models

EXPOSE 10000

# Start the application with optimized Gunicorn settings
CMD ["gunicorn", "--workers=1", "--threads=2", "--timeout=120", "--bind=0.0.0.0:10000", "app:app"]