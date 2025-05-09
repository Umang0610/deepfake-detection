FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y ffmpeg curl && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /opt/render/project/src/models
RUN curl -L -o /opt/render/project/src/models/audio_model.tflite https://github.com/Umang0610/deepfake-detection/raw/main/models/audio_model.tflite
RUN curl -L -o /opt/render/project/src/models/deepfake_video_new.tflite https://github.com/Umang0610/deepfake-detection/raw/main/models/deepfake_video_new.tflite
RUN curl -L -o /opt/render/project/src/models/best_deepfake_detector_resnet18_quantized.pth https://github.com/Umang0610/deepfake-detection/raw/main/models/best_deepfake_detector_resnet18_quantized.pth
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 10000
ENV PORT=10000
CMD ["gunicorn", "--workers=1", "--threads=1", "--timeout=180", "--bind=0.0.0.0:10000", "app:app"]