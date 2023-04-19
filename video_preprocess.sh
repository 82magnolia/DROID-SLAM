#!/bin/bash

VIDEO_DIR=$1
RESULT_DIR=$2
VID_H=$3
VID_W=$4
FPS=$5

NEW_VIDEO_DIR="${VIDEO_DIR/".mp4"/"_resize.mp4"}"

mkdir $RESULT_DIR
ffmpeg -i ${VIDEO_DIR} -r ${FPS:-"60"} -s ${VID_W:-"640"}x${VID_H:-"360"} -c:a copy ${NEW_VIDEO_DIR} -sws_flags neighbor
ffmpeg -i ${NEW_VIDEO_DIR} -q:v 2 ${RESULT_DIR}/image-%3d.jpg

