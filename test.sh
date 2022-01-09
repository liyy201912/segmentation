#!/usr/bin/env bash
# bisenetv1 cityscapes
export CUDA_VISIBLE_DEVICES=3
python tools/evaluate.py --config configs/bisenetv2_city.py --weight-path model_final_v2_city.pth 

