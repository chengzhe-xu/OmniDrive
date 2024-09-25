#!/usr/bin/env bash

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

LD_LIBRARY_PATH=/data1/chengzhex/envs/TensorRT-10.4.0.11/cu118/x86_64/TensorRT-10.4.0.11//lib/:$LD_LIBRARY_PATH PYTHONPATH="./":$PYTHONPATH \
python3 -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --use_env \
    --nproc_per_node=1 \
    --master_port=$PORT \
    ./deploy/test.py \
    --engine_pth ./engines/eva_base_tinyllama_.engine \
    --eval bbox \
    --config ./projects/configs/OmniDrive/eva_base_tinyllama.py \
    --launcher pytorch \
    --llm_engine_pth ./deploy/llm_lib/llm_engine_0.13_new/x86_1gpu_afp16_wi4/ \
    --tokenizer_pth ../omnidrive_tiny_pretrain/pretrain_tiny/ \
    --qa_save_path ./results_planning_only_vision_fp32_llm_fp16/
