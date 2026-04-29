FROM python:3.12-slim

# Install Node.js for frontend build
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl libgl1 libglib2.0-0 && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (CPU-only PyTorch to keep image small)
COPY requirements.deploy.txt .
RUN pip install --no-cache-dir -r requirements.deploy.txt

# Build frontend
COPY frontend/ ./frontend/
RUN cd frontend && npm install && npm run build

# Copy backend and model weights
COPY deployment/ ./deployment/
COPY runs/ ./runs/

ENV PORT=7860
EXPOSE 7860

CMD ["python", "deployment/app.py"]
