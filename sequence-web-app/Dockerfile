FROM python:3.11-slim

# Install ffmpeg and verify it works
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && ffmpeg -version \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Verify ffmpeg is on PATH at runtime
RUN which ffmpeg && which ffprobe

# Run
CMD gunicorn app:app --bind 0.0.0.0:$PORT --timeout 600 --workers 2 --threads 4