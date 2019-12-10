#!/bin/bash
python3 -B lab4.py -bs 32 --epochs 2 --gpu-id 0 |& tee logs/Q1/bs_32.txt &
python3 -B lab4.py -bs 128 --epochs 2 --gpu-id 1 |& tee logs/Q1/bs_128.txt &
python3 -B lab4.py -bs 512 --epochs 2 --gpu-id 2 |& tee logs/Q1/bs_512.txt &
