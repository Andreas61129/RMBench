#!/bin/bash
# Client-side launcher: runs in the RMBench conda env, talks to a running model_server.sh over
# a socket (Pattern B -- see rmbench_onboarding.md Part 2.2 and deploy_policy.py's module
# docstring). Start model_server.sh first (in sir_baseline's own env), then run this.

policy_name=SIR
task_name=${1}
task_config=${2}
ckpt_setting=${3}
seed=${4}
gpu_id=${5}
port=${6:-9999}
use_graph=${7:-0}  # 1 to enable graph obs (Tier 2, best-effort) -- must match the checkpoint

export CUDA_VISIBLE_DEVICES=${gpu_id}
export SIR_USE_GRAPH=${use_graph}
echo -e "\033[33mgpu id (to use): ${gpu_id}, port: ${port}, use_graph: ${use_graph}\033[0m"

cd ../.. # move to RMBench root

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy_client.py --config policy/$policy_name/deploy_policy.yml \
    --port ${port} \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --seed ${seed} \
    --policy_name ${policy_name}
