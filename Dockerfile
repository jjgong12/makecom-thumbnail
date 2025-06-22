FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt /requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /requirements.txt

# Copy handler
COPY handler.py /handler.py

# Set the handler
CMD ["python", "-u", "/handler.py"]
