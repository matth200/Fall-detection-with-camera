#!/bin/bash

echo "Starting configuration..."
cd yolo
git clone https://github.com/pjreddie/darknet.git
cd darknet
make
cd ..
wget https://pjreddie.com/media/files/yolov3.weights
cd ..