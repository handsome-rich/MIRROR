#!/bin/bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

GPU_NUM="${GPU_NUM:-4}"

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1

torchrun \
    --nproc_per_node=$GPU_NUM \
    --master_port=29513 \
    main_finetune.py \
    --model MIRROR \
    --memory_path "${MEMORY_PATH:-./weight/phase1/mirror_phase1_epoch_100.pth}" \
    --batch_size 32 \
    --blr 1e-4 \
    --epochs 2000 \
    --data_path "${TRAIN_DATA_PATH:-./datasets/train}" \
    --eval_data_path "${EVAL_DATA_PATH:-./datasets/val}" \
    "$@"
