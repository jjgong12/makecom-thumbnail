FROM runpod/base:0.6.2-cuda12.2.0

WORKDIR /

# Install system dependencies including Python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt /requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /requirements.txt

# Copy handler
COPY handler.py /handler.py

# Set the handler
CMD ["python", "-u", "/handler.py"]
