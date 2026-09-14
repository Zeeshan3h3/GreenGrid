FROM python:3.10-slim

# 1. Install system dependencies (Rasterio / GeoPandas require GDAL and build tools)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgdal-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Set up a non-root user (Required by Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user

# 3. Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

# 4. Set the working directory
WORKDIR $HOME/app

# 5. Copy the current directory contents into the container
COPY --chown=user . $HOME/app

# 6. Install Python dependencies
# We install the CPU version of PyTorch first to save massive amounts of disk space (CUDA is not needed on HF free tier)
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# 7. Expose the default port for Hugging Face
EXPOSE 7860

# 8. Start the FastAPI server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
