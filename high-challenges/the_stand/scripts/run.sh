#!/bin/bash
IMAGE="the_stand"
PORT=5000
docker run -p ${PORT}:${PORT} ${IMAGE}
