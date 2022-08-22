#!/usr/bin/env bash

declare algo_alias="rcs"
declare algo="GEM"

declare root="$HOME/NECSA/GEM/gem_mujoco"
declare max_steps=1000


declare -a gpus=(0 1 2)
declare -a envs=("Hopper-v3" "Walker2d-v3" "Swimmer-v3" "InvertedDoublePendulum-v2" "InvertedPendulum-v2" "Humanoid-v3")
declare -a steps=(400001 400001 400001 100001 100001 500001)
declare -a envs_alias=("hopper" "walker" "swimmer" "doublependulum" "pendulum" "humanoid")
export PYTHONPATH=$root


OPENAI_LOGDIR=./log_gem/mujoco/gem+tbp python3 $root/run/train.py --num-timesteps=500001 --max_steps=1000 --agent=GEM --comment=humanoid_gem+tbp_0 --seed 0 --env-id=Humanoid-v3
OPENAI_LOGDIR=./log_gem/mujoco/gem+tbp python3 $root/run/train.py --num-timesteps=500001 --max_steps=1000 --agent=GEM --comment=humanoid_gem+tbp_1 --seed 1 --env-id=Humanoid-v3
OPENAI_LOGDIR=./log_gem/mujoco/gem+tbp python3 $root/run/train.py --num-timesteps=500001 --max_steps=1000 --agent=GEM --comment=humanoid_gem+tbp_2 --seed 2 --env-id=Humanoid-v3
OPENAI_LOGDIR=./log_gem/mujoco/gem+tbp python3 $root/run/train.py --num-timesteps=500001 --max_steps=1000 --agent=GEM --comment=humanoid_gem+tbp_3 --seed 3 --env-id=Humanoid-v3
OPENAI_LOGDIR=./log_gem/mujoco/gem+tbp python3 $root/run/train.py --num-timesteps=500001 --max_steps=1000 --agent=GEM --comment=humanoid_gem+tbp_4 --seed 4 --env-id=Humanoid-v3