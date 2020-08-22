#!/bin/bash

set -e

if [[ ! -f data/crowdhuman/train.txt || ! -f data/crowdhuman/test.txt ]]; then
  echo "ERROR: missing txt file in data/crowdhuman/"
  exit 1
fi

echo "** Install requirements"
# "gdown" is for downloading files from GoogleDrive
pip3 install --user gdown > /dev/null

echo "** Copy files for training"
ln -sf $(readlink -f data/crowdhuman) darknet/data/
cp data/crowdhuman.data darknet/data/
cp data/crowdhuman.names darknet/data/
cp cfg/yolov4-crowdhuman-608.cfg darknet/cfg/

if [[ ! -f darknet/yolov4.conv.137 ]]; then
  pushd darknet > /dev/null
  echo "** Download pre-trained yolov4 weights"
    gdown https://drive.google.com/uc?id=1JKF-bdIklxOOVy-2Cr5qdvjgGpmGfcbp
  popd > /dev/null
fi

echo "** Done."
