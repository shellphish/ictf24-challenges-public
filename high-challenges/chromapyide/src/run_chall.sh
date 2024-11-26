#!/bin/bash

echo "Running challenge..."
export XDG_CACHE_HOME=/home/challenge/.cache
timeout 300 python3 /home/challenge/src/chall.py
