# Template Dockerfile for building an Apptainer-compatible container
# Adapt this to your project's dependencies

FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Install the Hydra Apptainer launcher plugin INSIDE the container.
# This is REQUIRED: submitit deserializes the launcher object on the compute
# node, so the plugin must be importable inside the container.

RUN pip install -

# Copy and install your project
COPY . /app/
RUN pip install -e .

CMD ["/bin/bash"]
