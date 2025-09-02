#!/bin/bash
# NOTE: replace ... with actual paths
# export LD_LIBRARY_PATH=...
# export PATH=...
echo "conda location: $(which conda)"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"

# export HF_DATASETS_CACHE=...
# export HF_HOME=...
# export WANDB_DISABLED=false
# export WANDB_PROJECT=...
# export WANDB_API_KEY=...
# export HUGGING_FACE_HUB_TOKEN=...
# export WANDB_RUN_GROUP=...
export EXP_NAME=exp_debug

# export WANDB_NAME=$EXP_NAME
# export EXP_DIR=output/$EXP_NAME
# export WANDB_DIR=$EXP_DIR
# echo $EXP_DIR

# mkdir -p $EXP_DIR/wandb
# rm -rf $EXP_DIR/wandb/*

# cd PATH_TO_VLM2VEC_REPO
cmd="python main.py $EXP_NAME --train_phase unet"

echo $cmd
eval $cmd