#!/bin/bash
# Server-side launcher: runs in sir_baseline's OWN env/process (NOT the RMBench conda env),
# hosting the trained model behind RMBench/script/policy_model_server.py's generic socket server
# -- that script itself only imports socket/json/yaml/importlib at module scope, so it runs fine
# from sir_baseline's env; get_model() (called inside it) is where the heavy torch/sir_baseline
# imports actually happen. See deploy_policy.py's module docstring and
# sir_baseline/rmbench_adapter.py.
#
# Usage: ./model_server.sh <checkpoint_dir> [port] [use_ema] [execute_horizon] [checkpoint_epoch]
# checkpoint_epoch: optional epoch number (e.g. 100) to load action_generator_epoch_<N>.pth etc.
#                    instead of the *_final.pth files -- see deploy_policy.yml's checkpoint_epoch.

checkpoint=${1:?checkpoint dir required, e.g. logs/rmbench/2026-08-01/12-00-00}
port=${2:-9999}
use_ema=${3:-true}
execute_horizon=${4:-64}
checkpoint_epoch=${5:-}

cd ../.. # move to RMBench root (policy_model_server.py resolves task_config paths relative to cwd)

extra_overrides=()
if [ -n "${checkpoint_epoch}" ]; then
    extra_overrides=(--checkpoint_epoch ${checkpoint_epoch})
fi

python script/policy_model_server.py --config policy/SIR/deploy_policy.yml \
    --port ${port} \
    --overrides \
    --policy_name SIR \
    --checkpoint ${checkpoint} \
    --use_ema ${use_ema} \
    --execute_horizon ${execute_horizon} \
    "${extra_overrides[@]}"
