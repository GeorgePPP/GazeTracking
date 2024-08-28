FROM ubuntu:22.04

# Install dependencies and show their versions
RUN apt-get update && apt-get install -y \
    python3 && \
    python3 --version && \
    apt-get install -y python3-pip && \
    pip3 --version && \
    apt-get install -y cmake && \
    cmake --version && \
    apt-get install -y libsm6 && \
    apt-get install -y libxext6 && \
    apt-get install -y libxrender1 && \
    apt-get install -y libfontconfig1 && \
    apt-get install -y libgtk2.0-dev && \
    apt-get install -y pkg-config && \
    apt-get install -y ffmpeg && \
    apt-get install -y libgl1-mesa-glx && \
    pkg-config --version

# Upgrade pip and show its version
RUN pip3 install --upgrade pip && \
    pip3 --version

# Copy the project files
COPY . /home/GazeTracking
WORKDIR /home/GazeTracking

# Install Python dependencies and show installed packages
RUN pip3 install -r requirements.txt && \
    pip3 list
