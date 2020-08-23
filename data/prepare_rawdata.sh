#!/bin/bash

set -e

echo "** Install requirements"
# "gdown" is for downloading files from GoogleDrive
pip3 install --user gdown > /dev/null

# make sure to download dataset files to "yolov4_crowdhuman/data/raw/"
mkdir -p $(dirname $0)/raw
pushd $(dirname $0)/raw > /dev/null

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

echo "** Download dataset files"
get_file https://drive.google.com/uc?id=134QOvaatwKdy0iIeNqA_p-xkAhkV4F8Y CrowdHuman_train01.zip
get_file https://drive.google.com/uc?id=17evzPh7gc1JBNvnW1ENXLy5Kr4Q_Nnla CrowdHuman_train02.zip
get_file https://drive.google.com/uc?id=1tdp0UCgxrqy1B6p8LkR-Iy0aIJ8l4fJW CrowdHuman_train03.zip
get_file https://drive.google.com/uc?id=18jFI789CoHTppQ7vmRSFEdnGaSQZ4YzO CrowdHuman_val.zip
get_file https://drive.google.com/uc?id=1tQG3E_RrRI4wIGskorLTmDiWHH2okVvk CrowdHuman_test.zip
get_file https://drive.google.com/u/0/uc?id=1UUTea5mYqvlUObsC1Z8CFldHJAtLtMX3 annotation_train.odgt
get_file https://drive.google.com/u/0/uc?id=10WIRwu8ju8GRLuCkZ_vT6hnNxs5ptwoL annotation_val.odgt

# unzip image files (ignore CrowdHuman_test.zip for now)
echo "** Unzip dataset files"
for f in CrowdHuman_train01.zip CrowdHuman_train02.zip CrowdHuman_train03.zip CrowdHuman_val.zip ; do
  unzip ${f}
done

echo "** Create the crowdhuman/ subdirectory"
rm -rf ../crowdhuman/
mv Images ../crowdhuman

# the crowdhuman/ subdirectory now contains all train/val jpg images

popd > /dev/null

echo "** Done."
