# Use NVIDIA CUDA base image with Ubuntu
FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Install system dependencies and build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    vim \
    gdb \
    valgrind \
    clang \
    clang-format \
    clang-tidy \
    libboost-all-dev \
    libssl-dev \
    openssl \
    python3 \
    python3-pip \
    ssh \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libglfw3-dev \
    libglew-dev \
    xorg-dev \
    && rm -rf /var/lib/apt/lists/*

# Install OpenMPI
ARG OPENMPI_VERSION=4.1.6
RUN cd /tmp && \
    wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-${OPENMPI_VERSION}.tar.gz && \
    tar -xzf openmpi-${OPENMPI_VERSION}.tar.gz && \
    cd openmpi-${OPENMPI_VERSION} && \
    ./configure --prefix=/usr/local \
                --with-cuda=/usr/local/cuda \
                --enable-mpi-cxx \
                --enable-mpi-fortran=no && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd / && \
    rm -rf /tmp/openmpi-${OPENMPI_VERSION}*

# Set up MPI environment variables
ENV PATH=/usr/local/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ENV MANPATH=/usr/local/share/man:$MANPATH

# Set up CUDA environment variables
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda

# Create a working directory
WORKDIR /workspace

# Verify installations
RUN mpirun --version && \
    nvcc --version && \
    echo "OpenMPI and NVIDIA CUDA setup complete!"

# Default command
CMD ["/bin/bash"]