
FROM python:3.11-slim

LABEL maintainer="ClinicalTriageEnv"
LABEL description="OpenEnv Healthcare AI Training Environment"
LABEL version="1.0.0"

# System dependencies
  RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for Docker layer caching
    COPY requirements.txt /app/requirements.txt

# Install Python dependencies
      RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all source files
                     COPY . /app/

# Make sure server module is importable
ENV PYTHONPATH="/app:$PYTHONPATH"

         # Hugging Face Spaces uses port 7860
ENV PORT=7860

# Health check
        HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
               CMD curl -f http://localhost:7860/health || exit 1

# Expose port
EXPOSE 7860

CMD python app.py           ← shell form, works everywhere

