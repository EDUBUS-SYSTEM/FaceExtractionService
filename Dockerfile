FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV and InsightFace
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY face_extraction_app.py .

# Pre-download InsightFace models to avoid first-request delay
RUN python -c "from insightface.app import FaceAnalysis; app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider']); app.prepare(ctx_id=-1, det_size=(640, 640))"

# Expose port
EXPOSE 5001

# Run the application
CMD ["python", "face_extraction_app.py"]
