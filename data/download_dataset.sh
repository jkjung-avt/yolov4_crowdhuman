#!/bin/bash

set -e

# "gdown" is for downloading files from GoogleDrive
pip3 install --user gdown > /dev/null

# make sure to download dataset files to where this script is located
# (yolov4_crowdhuman/data/)
pushd $(dirname $0) > /dev/null

get_file()
{
  # do download only if the file does not exist
  if [[ -f $2 ]];  then
    echo Skipping $2
  else
    echo Downloading $2...
    gdown $1
  fi
}

get_file https://drive.google.com/uc?id=134QOvaatwKdy0iIeNqA_p-xkAhkV4F8Y CrowdHuman_train01.zip
get_file https://drive.google.com/uc?id=17evzPh7gc1JBNvnW1ENXLy5Kr4Q_Nnla CrowdHuman_train02.zip
get_file https://drive.google.com/uc?id=1tdp0UCgxrqy1B6p8LkR-Iy0aIJ8l4fJW CrowdHuman_train03.zip
get_file https://drive.google.com/uc?id=18jFI789CoHTppQ7vmRSFEdnGaSQZ4YzO CrowdHuman_val.zip
get_file https://drive.google.com/uc?id=1tQG3E_RrRI4wIGskorLTmDiWHH2okVvk CrowdHuman_test.zip
get_file https://drive.google.com/u/0/uc?id=1UUTea5mYqvlUObsC1Z8CFldHJAtLtMX3 annotation_train.odgt
get_file https://drive.google.com/u/0/uc?id=10WIRwu8ju8GRLuCkZ_vT6hnNxs5ptwoL annotation_val.odgt

popd > /dev/null

echo Done.
