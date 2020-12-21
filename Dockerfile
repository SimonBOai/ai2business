# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.8-slim-buster

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

ARG gh_username=AI2Business
ARG ai2business_home="/home/ai2business"

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Configure apt and install packages
RUN apt-get update \
    && apt-get -y install --no-install-recommends apt-utils dialog 2>&1 \
    #
    # Verify git, process tools, lsb-release (common in install instructions for CLIs) installed
    && apt-get -y install git iproute2 procps iproute2 lsb-release \
    #
    # cleanup
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=dialog

# Clone pandas repo
RUN mkdir "$ai2business_home" \
    && git clone "https://github.com/$gh_username/ai2business.git" "$ai2business_home" \
    && cd "$ai2business_home" \
    && git remote add upstream "https://github.com/AI2Business/ai2business.git" \
    && git pull upstream master

# Update pip, setuptools, and wheel
RUN python -m pip install --upgrade pip setuptools wheel 


# Build ai2business
SHELL ["/bin/bash", "-c"]
RUN python -m pip install -r ./requirements.txt
RUN python -m pip install -r ./dev-requirements.txt
RUN python -m pip install -r docs/doc-requirements.txt
RUN python -m pip install -e .