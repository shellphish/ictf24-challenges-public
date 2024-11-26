#!/bin/bash
IMAGE="ictf_welcome"
PORT=12567
docker run -p ${PORT}:${PORT} ${IMAGE}
