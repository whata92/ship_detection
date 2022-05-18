FROM python
LABEL maintainer="whata <waku.hatakeyama@gmail.com>"

# install requirements
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
    python3.9 \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools \
    unzip \
	  vim \
    wget \
    build-essential \
    sshfs \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Create pipenv environment
COPY requirements.txt /tmp
RUN python3.9 -m pip install -r /tmp/requirements.txt

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