#!/bin/bash

ROOT_DIR="logs/ckpt"
SIMULATOR="mujoco"
CONFIG_FILE="humanoidverse/config/base_eval_multi.yaml"
num_envs=$([ "$SIMULATOR" = "mujoco" ] && echo 1 || echo 10)
devices=1

# directory structure: ROOT_DIR/training_platform/checkpoint_dir
for dir1 in "$ROOT_DIR"/*/; do
    for dir2 in "$dir1"*/; do
        dir2="${dir2%/}"
        echo "Evaluating checkpoint dir: $dir2"
        # Update config file with the new checkpoint directory and number of environments
        sed -i "s|^checkpointdir:.*|checkpointdir: \"$dir2\"|" $CONFIG_FILE
        sed -i "s|^num_envs:.*|num_envs: $num_envs|" $CONFIG_FILE
        CUDA_VISIBLE_DEVICES="$devices" python humanoidverse/eval_agent_multi.py +simulator="$SIMULATOR"
    done
done
