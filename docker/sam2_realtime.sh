#!/bin/bash

# INITIAL RUN
# docker run --gpus all -it --rm --net=host \
#     -e DISPLAY=$DISPLAY \
#     -v /tmp/.X11-unix:/tmp/.X11-unix \
#     -v /home/socrob/Development/segment-anything-2-real-time:/workspace/segment-anything-2-real-time \
#     sam2_realtime_saved:v2


# FOR THE LIVE CAMERA TEST (Laptop's camera)
# docker run --gpus all -it --rm --net=host \
#     -e DISPLAY=$DISPLAY \
#     -v /tmp/.X11-unix:/tmp/.X11-unix \
#     -v /home/socrob/Development/segment-anything-2-real-time:/workspace/segment-anything-2-real-time \
#     --device /dev/video0:/dev/video0 \
#     sam2_realtime_saved:v2


# TESTING WITH REALSENSE
docker run --gpus all -it --rm --net=host \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /home/socrob/Development/segment-anything-2-real-time:/workspace/segment-anything-2-real-time \
    -v /etc/udev/rules.d:/etc/udev/rules.d \
    --device=/dev/video0 --device=/dev/video1 \
    --device=/dev/usbmon0 --device=/dev/usbmon1 \
    --privileged \
    sam2_realtime_v2
