#!/bin/bash

echo "[INFO] Starting general AE"

python train.py --model autoencoder --data other --data_path data/gold_standard/synthetic_batch_1_raw.h5ad --output AE_synthetic_batch_1_raw_v1

python train.py --model autoencoder --data other --data_path data/gold_standard/synthetic_batch_2_raw.h5ad --output AE_synthetic_batch_2_raw_v1

echo "[INFO] Finished job script"