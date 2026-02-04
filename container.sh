#!/bin/bash
# Build pipeline: Dockerfile → Docker image → tar archive → Apptainer .sif
#
# Usage:
#   bash container.sh
#
# The resulting .sif file can be copied to the cluster's shared filesystem.
# Compute nodes will access it via the "python" field in the launcher YAML.

set -e
IMAGE_NAME="my_project"
echo "==> Building Docker image..."
docker build . -t "${IMAGE_NAME}"

echo "==> Exporting to tar archive..."
docker save -o "${IMAGE_NAME}.tar" "${IMAGE_NAME}:latest"

echo "==> Converting to Apptainer .sif..."
apptainer build "${IMAGE_NAME}.sif" "docker-archive://${IMAGE_NAME}.tar"

echo "==> Done: ${IMAGE_NAME}.sif"