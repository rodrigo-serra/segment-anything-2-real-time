#!/bin/bash

# docker run --gpus all -it --rm --net=host \
#     -e DISPLAY=$DISPLAY \
#     -v /tmp/.X11-unix:/tmp/.X11-unix \
#     -v /home/socrob/Development/segment-anything-2-real-time:/workspace/segment-anything-2-real-time \
#     sam2_realtime

docker run --gpus all -it --rm --net=host \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /home/socrob/Development/segment-anything-2-real-time:/workspace/segment-anything-2-real-time \
    sam2_realtime_saved:v2
