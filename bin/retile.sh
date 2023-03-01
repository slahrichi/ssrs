#!/bin/bash
# This bash script iterates through the files and runs a gdal utility to retile
# the images into the desired sizes.
# For each of the files output, generate retiled images of size 224 x 224

large_images="ls /data/users/sl636/output_images/*.tif"
target="/shared/data/sl636/retiled_climate+_images/"
mkdir -p $target

for image in $large_images; do
	echo Working on ${image}
	gdal_retile.py -targetDir $target -ps 224 224 $image
done
