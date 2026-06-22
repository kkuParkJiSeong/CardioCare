FROM python:3.12-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy codebase, data, and sample input
COPY src/ /app/src/
COPY data/ /app/data/

# Set the default entrypoint to the inference script
ENTRYPOINT ["python", "src/inference.py"]