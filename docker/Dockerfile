# Use an official NVIDIA PyTorch image with CUDA support
FROM nvidia/cuda:12.4.0-devel-ubuntu20.04

# Set the environment variable to avoid interactive prompts (like timezone)
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies and Miniconda
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    bzip2 \
    vim \
    x11-apps \
    libgtk-3-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -f -p /opt/conda \
    && rm miniconda.sh

# Set environment variables for Conda
ENV PATH="/opt/conda/bin:$PATH"
ENV DISPLAY=:0

# Initialize Conda
RUN conda init bash

# Create a new conda environment with Python 3.10
RUN conda create -n sam2_realtime python=3.10

# Activate the environment and install dependencies
RUN echo "conda activate sam2_realtime" >> ~/.bashrc
SHELL ["conda", "run", "-n", "sam2_realtime", "/bin/bash", "-c"]

# Install PyTorch, TorchVision, and other dependencies
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
RUN pip install matplotlib opencv-python imageio flask requests

# Install SAMURAI dependencies
WORKDIR /workspace
# COPY ./ /workspace/segment-anything-2-real-time
# RUN cd /workspace/segment-anything-2-real-time/ && pip install -e .
# RUN cd checkpoints && ./download_ckpts.sh && cd ..

# RealSense SDK installation
RUN apt-get update && apt-get install -y \
    lsb-release \
    && mkdir -p /etc/apt/keyrings \
    && curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | tee /etc/apt/keyrings/librealsense.pgp > /dev/null \
    && apt-get install -y apt-transport-https \
    && echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/librealsense.list \
    && apt-get update \
    && apt-get install -y librealsense2-dkms librealsense2-utils \
    && pip install pyrealsense2

# Set up your work environment
WORKDIR /workspace/segment-anything-2-real-time

# Expose ports if you want to run a web app or any service
EXPOSE 5000

# Set the default command to bash (so you can run any command interactively)
CMD ["bash"]
