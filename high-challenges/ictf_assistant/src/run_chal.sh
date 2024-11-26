#!/bin/bash

if ! /home/challenge/src/pow.py ask 10222; then
  echo 'Proof-of-work failed'
  exit 1
fi

python3 /home/challenge/src/chal.py