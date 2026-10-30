#!/bin/bash

salloc --nodes 1 -q coe-grade --cpus-per-task=6 --job-name bash --gpus-per-node=RTX_6000:1 --mem-per-gpu=24G --time 16:00:00
salloc --nodes 1 -q coe-grade --cpus-per-task=12 --job-name bash --gpus-per-node=rtx_pro_6000_blackwell:1 --mem-per-gpu=24G --time 16:00:00 --mail-type=BEGIN,END,FAIL --mail-user=jkamohara3@gatech.edu