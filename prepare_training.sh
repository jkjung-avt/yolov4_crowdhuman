#!/bin/bash

set -e

CROWDHUMAN=crowdhuman-512x512

if [[ ! -f data/${CROWDHUMAN}/train.txt || ! -f data/${CROWDHUMAN}/test.txt ]]; then
  echo "ERROR: missing txt file in data/${CROWDHUMAN}/"
  exit 1
fi

echo "** Install requirements"
# "gdown" is for downloading files from GoogleDrive
pip3 install --user gdown > /dev/null
export PATH=${HOME}/.local/bin:${PATH}

echo "** Copy files for training"
ln -sf $(readlink -f data/${CROWDHUMAN}) darknet/data/
cp data/${CROWDHUMAN}.data darknet/data/
cp data/crowdhuman.names darknet/data/
cp cfg/yolov4-${CROWDHUMAN}.cfg darknet/cfg/

if [[ ! -f darknet/yolov4.conv.137 ]]; then
  pushd darknet > /dev/null
  echo "** Download pre-trained yolov4 weights"
  gdown https://drive.google.com/uc?id=1JKF-bdIklxOOVy-2Cr5qdvjgGpmGfcbp
  popd > /dev/null
fi

echo "** Done."
