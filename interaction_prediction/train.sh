#!/usr/bin/env bash
# -------------------------------------------------- #
# GPUS=$1                                              #    
# -------------------------------------------------- #
# GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))

MASTER_PORT=${MASTER_PORT:-28596}

python3 -m torch.distributed.launch \
        --nproc_per_node=4 \
        --master_port=$MASTER_PORT \
        /root/xzcllwx_ws/GameFormer/interaction_prediction/train.py \
        --batch_size=64 \
        --training_epochs=30 \
        --train_set=/root/xzcllwx_ws/womd_process/womd_1M \
        --valid_set=/root/xzcllwx_ws/womd_process/validation_interactive_proces \
        --workers=8 \
        --name=Exp7 \