FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 10000
ENV PORT=10000
CMD ["gunicorn", "--workers=1", "--threads=1", "--timeout=180", "--bind=0.0.0.0:10000", "app:app"]