#!/bin/bash

echo "Building Docker image..."
docker build -t mpi-nvidia-ubuntu:latest .

echo "Starting container..."
docker run -it --rm \
    -v "$(pwd)":/workspace \
    -w /workspace \
    --name mpi-nvidia-dev \
    mpi-nvidia-ubuntu:latest \
    /bin/bash