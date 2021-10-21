#!/bin/bash

# Download large files from google drive while confirming virus scan warning
# https://gist.github.com/guysmoilov/ff68ef3416f99bd74a3c431b4f4c739a
function gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://drive.google.com/uc?export=download&id=$1" -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1/p')
  wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -f /tmp/cookies.txt
}

# data
# mkdir -p data
# gdrive_download 1gPeUHtGzTcVlbnyThNRC3ShuAELpFsC8 data/images.npy
# gdrive_download 1mY6QKYCzwzAqPGAh6ppubvD_zUNJd9B0 data/labels.npy
# gdrive_download 1g8f-7ftONOSih4ssxeUcTUCVcnqmRxn4 data/bounding_boxes.npy
# gdrive_download 1yu4bNjPsXdJb8lWVzRlfh6mjwtiDdZWL data/masks.npy

# models
mkdir -p models
gdrive_download 1uXsgpJSOiKfIe1haqoRchx-AMRF9ormK models/utility.ckpt
gdrive_download 1FEy61tSQzYF10e0N8xNENs0a0Rv0UMPv models/unoise_small.ckpt
gdrive_download 11_rTHLkB56QIlPXTlRb7ln9WURbbAUDD models/unoise_medium.ckpt
gdrive_download 1evV2daEgnfbyctwCkQ5PHXhZT8LNsidr models/unoise_large.ckpt

gdrive_download 1kzR1I_lgynPtqEQqwaHiVdmQKghSi2bv models/unoise_small_pretrained.ckpt
gdrive_download 1xdJH9jcRZoVa6i_mdKCfbjyPQQrMbLac models/unoise_medium_pretrained.ckpt
gdrive_download 1834JqlUcxeS3ifAnTjiCGHvP3GYST7Bl models/unoise_large_pretrained.ckpt
