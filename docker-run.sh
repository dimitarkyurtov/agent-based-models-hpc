#!/bin/bash

# Build the Docker image
echo "Building Docker image..."
docker build -t mpi-nvidia-ubuntu:latest .

# Run the container
echo "Starting container..."
docker run -it --rm \
    -v "$(pwd)":/workspace \
    -w /workspace \
    --name mpi-nvidia-dev \
    mpi-nvidia-ubuntu:latest \
    /bin/bash