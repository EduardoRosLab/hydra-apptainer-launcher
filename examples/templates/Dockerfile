# Template Dockerfile for building an Apptainer-compatible container
# Adapt this to your project's dependencies

FROM python:3.10-slim

WORKDIR /app


RUN apt-get update \
&& apt-get install -y --no-install-recommends git \
 && apt-get purge -y --auto-remove \
 && rm -rf /var/lib/apt/lists/*
# Install Python dependencies
#COPY requirements.txt /app/
#RUN pip install --no-cache-dir -r requirements.txt

# Install the Hydra Apptainer launcher plugin INSIDE the container.
# This is REQUIRED: submitit deserializes the launcher object on the compute
# node, so the plugin must be importable inside the container.

RUN pip install git+https://github.com/EduardoRosLab/hydra-apptainer-launcher.git

# Copy and install your project
COPY . /app/

# Here u install your project 
#RUN pip install -e .

CMD ["/bin/bash"]
