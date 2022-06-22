#!/usr/bin/env bash

declare algo_alias="rcs"
declare algo="GEM"

declare root="$HOME/workspace/GEM/gem_mujoco"
declare max_steps=1000


declare -a gpus=(0 1 2)
declare -a envs=("Hopper-v3" "Walker2d-v3" "Swimmer-v3" "InvertedDoublePendulum-v2" "InvertedPendulum-v2")
declare -a steps=(400001 400001 400001 100001 100001)
declare -a envs_alias=("hopper" "walker" "swimmer" "doublependulum" "pendulum")
export PYTHONPATH=$root

OPENAI_LOGDIR=./log_gem/mujoco/rcs_3 python3 $root/run/train.py --num-timesteps=400001 --max_steps=1000 --agent=RCS --comment=swimmer_rcs_0 --seed 0 --env-id=Swimmer-v3 --order 3 --grid_num 10 --decay 0.1 --state_min -10 --state_max 10 --mode state_action 
OPENAI_LOGDIR=./log_gem/mujoco/rcs_3 python3 $root/run/train.py --num-timesteps=400001 --max_steps=1000 --agent=RCS --comment=swimmer_rcs_1 --seed 1 --env-id=Swimmer-v3 --order 3 --grid_num 10 --decay 0.1 --state_min -10 --state_max 10 --mode state_action 
OPENAI_LOGDIR=./log_gem/mujoco/rcs_3 python3 $root/run/train.py --num-timesteps=400001 --max_steps=1000 --agent=RCS --comment=swimmer_rcs_2 --seed 2 --env-id=Swimmer-v3 --order 3 --grid_num 10 --decay 0.1 --state_min -10 --state_max 10 --mode state_action 
OPENAI_LOGDIR=./log_gem/mujoco/rcs_3 python3 $root/run/train.py --num-timesteps=400001 --max_steps=1000 --agent=RCS --comment=swimmer_rcs_3 --seed 3 --env-id=Swimmer-v3 --order 3 --grid_num 10 --decay 0.1 --state_min -10 --state_max 10 --mode state_action 
OPENAI_LOGDIR=./log_gem/mujoco/rcs_3 python3 $root/run/train.py --num-timesteps=400001 --max_steps=1000 --agent=RCS --comment=swimmer_rcs_4 --seed 4 --env-id=Swimmer-v3 --order 3 --grid_num 10 --decay 0.1 --state_min -10 --state_max 10 --mode state_action 
OPENAI_LOGDIR=./log_gem/mujoco/rcs_3 python3 $root/run/train.py --num-timesteps=400001 --max_steps=1000 --agent=RCS --comment=swimmer_rcs_5 --seed 5 --env-id=Swimmer-v3 --order 3 --grid_num 10 --decay 0.1 --state_min -10 --state_max 10 --mode state_action 
OPENAI_LOGDIR=./log_gem/mujoco/rcs_3 python3 $root/run/train.py --num-timesteps=400001 --max_steps=1000 --agent=RCS --comment=swimmer_rcs_6 --seed 6 --env-id=Swimmer-v3 --order 3 --grid_num 10 --decay 0.1 --state_min -10 --state_max 10 --mode state_action 
OPENAI_LOGDIR=./log_gem/mujoco/rcs_3 python3 $root/run/train.py --num-timesteps=400001 --max_steps=1000 --agent=RCS --comment=swimmer_rcs_7 --seed 7 --env-id=Swimmer-v3 --order 3 --grid_num 10 --decay 0.1 --state_min -10 --state_max 10 --mode state_action 
OPENAI_LOGDIR=./log_gem/mujoco/rcs_3 python3 $root/run/train.py --num-timesteps=400001 --max_steps=1000 --agent=RCS --comment=swimmer_rcs_8 --seed 8 --env-id=Swimmer-v3 --order 3 --grid_num 10 --decay 0.1 --state_min -10 --state_max 10 --mode state_action 
OPENAI_LOGDIR=./log_gem/mujoco/rcs_3 python3 $root/run/train.py --num-timesteps=400001 --max_steps=1000 --agent=RCS --comment=swimmer_rcs_9 --seed 9 --env-id=Swimmer-v3 --order 3 --grid_num 10 --decay 0.1 --state_min -10 --state_max 10 --mode state_action

