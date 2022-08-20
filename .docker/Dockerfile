FROM nvidia/cuda:10.2-base-ubuntu18.04
LABEL maintainer="whata <waku.hatakeyama@gmail.com>"

RUN apt-get update -y \
  && apt-get install -y --no-install-recommends \
    sudo \
    bc \
    bzip2 \
    ca-certificates \
    curl \
    git \
    libssl-dev \
    libffi-dev \
	  libncurses-dev \
    libgl1 \
    jq \
    nfs-common \
    parallel \
    unzip \
	  vim \
    wget \
    build-essential \
    sshfs \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*
SHELL ["/bin/bash", "-c"]

WORKDIR /workspace

# Add user
ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
RUN usermod -aG sudo user
RUN passwd -d user
USER user

RUN ["/bin/bash"]