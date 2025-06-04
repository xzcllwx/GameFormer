#!/usr/bin/env bash
# -------------------------------------------------- #
# GPUS=$1                                              #    
# -------------------------------------------------- #
# GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))

MASTER_PORT=${MASTER_PORT:-28596}

CUDA_VISIBLE_DEVICES=0 \
python3 -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=$MASTER_PORT \
        /root/xzcllwx_ws/GameFormer/train.py \
        --batch_size=64 \
        --training_epochs=1 \
        --train_set=/root/xzcllwx_ws/womd_process/womd_1M \
        --valid_set=/root/xzcllwx_ws/womd_process/womd_val_10K \
        --workers=8 \
        --name=Exp1 \
        --load_dir epochs_29.pth \