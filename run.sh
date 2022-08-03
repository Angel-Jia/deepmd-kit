#!/bin/bash
rm /root/code/deepmd-kit-pytorch-dev/source/build/op/libop_abi.so
cd /root/code/deepmd-kit-pytorch-dev/source/build
make -j
cd /root/code/deepmd-kit-pytorch-dev
python test.py
