#!/bin/bash
# Download the 15 high-dimensional datasets (scikit-feature repository) into data/hd/
mkdir -p data/hd
for n in leukemia colon lung lymphoma GLIOMA Carcinom nci9 COIL20 ORL Yale warpPIE10P warpAR10P USPS Isolet PCMAC; do
  [ -f data/hd/$n.mat ] || curl -sL -o data/hd/$n.mat https://raw.githubusercontent.com/jundongl/scikit-feature/master/skfeature/data/$n.mat
  echo "data/hd/$n.mat $(stat -c %s data/hd/$n.mat 2>/dev/null) bytes"
done
