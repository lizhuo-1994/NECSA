#!/usr/bin/env bash

declare algo_alias="rcs"
declare algo="GEM"

declare root="$HOME/workspace/GEM/gem_mujoco"
declare max_steps=1000


declare -a gpus=(0 1 2)
declare -a envs=("Hopper-v3" "Walker2d-v3" "Swimmer-v3" "InvertedDoublePendulum-v2" "InvertedPendulum-v2" "Humanoid-v3")
declare -a steps=(400001 400001 400001 100001 100001)
declare -a envs_alias=("hopper" "walker" "swimmer" "doublependulum" "pendulum" "humanoid")
export PYTHONPATH=$root

OPENAI_LOGDIR=./log_gem/mujoco/rcs_1 python3 $root/run/train.py --num-timesteps=1000001 --max_steps=1000 --agent=RCS --comment=swimmer_rcs_0 --seed 0 --env-id=Humanoid-v3 --order 1 --grid_num 10 --decay 0.1 --state_dim 16 --state_min -6 --state_max 6 --mode state_action --reduction
OPENAI_LOGDIR=./log_gem/mujoco/rcs_1 python3 $root/run/train.py --num-timesteps=1000001 --max_steps=1000 --agent=RCS --comment=swimmer_rcs_1 --seed 1 --env-id=Humanoid-v3 --order 1 --grid_num 10 --decay 0.1 --state_dim 16 --state_min -6 --state_max 6 --mode state_action --reduction
OPENAI_LOGDIR=./log_gem/mujoco/rcs_1 python3 $root/run/train.py --num-timesteps=1000001 --max_steps=1000 --agent=RCS --comment=swimmer_rcs_2 --seed 2 --env-id=Humanoid-v3 --order 1 --grid_num 10 --decay 0.1 --state_dim 16 --state_min -6 --state_max 6 --mode state_action --reduction
OPENAI_LOGDIR=./log_gem/mujoco/rcs_1 python3 $root/run/train.py --num-timesteps=1000001 --max_steps=1000 --agent=RCS --comment=swimmer_rcs_3 --seed 3 --env-id=Humanoid-v3 --order 1 --grid_num 10 --decay 0.1 --state_dim 16 --state_min -6 --state_max 6 --mode state_action --reduction
OPENAI_LOGDIR=./log_gem/mujoco/rcs_1 python3 $root/run/train.py --num-timesteps=1000001 --max_steps=1000 --agent=RCS --comment=swimmer_rcs_4 --seed 4 --env-id=Humanoid-v3 --order 1 --grid_num 10 --decay 0.1 --state_dim 16 --state_min -6 --state_max 6 --mode state_action --reduction
