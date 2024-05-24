FROM python:3.8-slim

# Install necessary packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy the script and preprocessing module
COPY src /src
COPY data /data
WORKDIR /src

# Runs the script
CMD ["python", "main.py"]