#!/bin/bash
# usage: sh setup.sh <CONTAINER_NAME>
# Example: sh setup.sh inrsteg
CONTAINER_NAME=$1

echo "CONTAINER_NAME: $CONTAINER_NAME"

docker exec -u 0 $CONTAINER_NAME python3 -m pip install --no-cache-dir opencv-python
docker exec -u 0 $CONTAINER_NAME bash -c "apt-get update -y && apt-get install -y libgl1-mesa-glx"
docker exec -u 0 $CONTAINER_NAME bash -c "apt-get install -y libglib2.0-0"
docker exec -u 0 $CONTAINER_NAME pip install matplotlib
docker exec -u 0 $CONTAINER_NAME pip install scipy
docker exec -u 0 $CONTAINER_NAME pip install scikit-image
docker exec -u 0 $CONTAINER_NAME pip install scikit-video
docker exec -u 0 $CONTAINER_NAME pip install h5py
docker exec -u 0 $CONTAINER_NAME pip install gdown
docker exec -u 0 $CONTAINER_NAME pip install cmapy
docker exec -u 0 $CONTAINER_NAME pip install tensorboard
docker exec -u 0 $CONTAINER_NAME pip install torchmetrics
docker exec -u 0 $CONTAINER_NAME pip install einops
docker exec -u 0 $CONTAINER_NAME python3 -m pip install --no-cache-dir configargparse
docker exec -u 0 $CONTAINER_NAME pip install --upgrade numpy==1.23