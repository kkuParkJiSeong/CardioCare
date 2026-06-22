FROM python:3.12-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy codebase and data (needed for inference if testing locally in container)
COPY src/ /app/src/

# Set the default entrypoint to the inference script
ENTRYPOINT ["python", "src/inference.py"]