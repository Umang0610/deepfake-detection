FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y ffmpeg git-lfs && rm -rf /var/lib/apt/lists/*
COPY . .
RUN git lfs install && git lfs pull
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 10000
ENV PORT=10000
CMD ["gunicorn", "--workers=1", "--threads=1", "--timeout=180", "--bind=0.0.0.0:10000", "app:app"]