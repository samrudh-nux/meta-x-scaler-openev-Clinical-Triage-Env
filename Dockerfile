FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create server dir if not present (for openenv multi-mode compliance)
RUN mkdir -p server && \
    if [ ! -f server/__init__.py ]; then touch server/__init__.py; fi

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
