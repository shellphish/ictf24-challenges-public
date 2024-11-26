#!/bin/sh

socat \
  -dd \
  -T300 \
  TCP-LISTEN:25932,reuseaddr,fork \
  EXEC:"timeout 6000 ./main.py"
