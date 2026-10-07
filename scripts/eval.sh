#!/bin/bash
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

GPU_NUM="${GPU_NUM:-4}"

BASE_DATA_PATH="${BASE_DATA_PATH:-./datasets}"
CHECKPOINT="${CHECKPOINT:-./weight/checkpoint-h-cur.pth}"

benchmarks=(

    "Chameleon:Chameleon/test"

)

for item in "${benchmarks[@]}"
do
    IFS=":" read -r NAME DATA_PATH <<< "$item"
    echo "============================================================"
    echo "Starting Evaluation on: $NAME"
    echo "Data Path: $BASE_DATA_PATH/$DATA_PATH"
    echo "============================================================"
    torchrun \
        --nproc_per_node=$GPU_NUM \
        --master_port=29517 \
        main_finetune.py \
        --model MIRROR \
        --eval True \
        --resume "$CHECKPOINT" \
        --batch_size 128 \
        --num_workers 16 \
        --blr 1e-4 \
        --epochs 5 \
        --multi_eval "$NAME" \
        --data_path "$BASE_DATA_PATH/AIGC_bm" \
        --eval_data_path "$BASE_DATA_PATH/$DATA_PATH" \
        "$@"
        echo "Finished $NAME. Results saved to CSV."
done
