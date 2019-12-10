#!/bin/bash
python3 -B lab4.py -bs 32 --epochs 2 --gpu-id 0 |& tee logs/Q2/bs_32_1g.txt &
python3 -B lab4.py -bs $((32*2)) --epochs 2 --gpu-id 1,2 |& tee logs/Q2/bs_32_2g.txt &
python3 -B lab4.py -bs $((32*4)) --epochs 2 --gpu-id 3,4,5,6 |& tee logs/Q2/bs_32_4g.txt &

