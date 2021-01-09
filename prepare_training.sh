#!/bin/bash

set -e

# check argument
if [[ -z $1 || ! $1 =~ [[:digit:]]x[[:digit:]] ]]; then
  echo "ERROR: This script requires 1 argument, \"input dimension\" of the YOLO model."
  echo "The input dimension should be {width}x{height} such as 608x608 or 416x256.".
  exit 1
fi

CROWDHUMAN=crowdhuman-$1

if [[ ! -f data/${CROWDHUMAN}/train.txt || ! -f data/${CROWDHUMAN}/test.txt ]]; then
  echo "ERROR: missing txt file in data/${CROWDHUMAN}/"
  exit 1
fi

echo "** Install requirements"
# "gdown" is for downloading files from GoogleDrive
pip3 install --user gdown > /dev/null

echo "** Copy files for training"
ln -sf $(readlink -f data/${CROWDHUMAN}) darknet/data/
cp data/${CROWDHUMAN}.data darknet/data/
cp data/crowdhuman.names darknet/data/
cp cfg/*.cfg darknet/cfg/

if [[ ! -f darknet/yolov4.conv.137 ]]; then
  pushd darknet > /dev/null
  echo "** Download pre-trained yolov4 weights"
  python3 -m gdown.cli https://drive.google.com/uc?id=1JKF-bdIklxOOVy-2Cr5qdvjgGpmGfcbp
  popd > /dev/null
fi

echo "** Done."
