# Neural Episodic Control with State Abstraction
This repo is based on [tianshou](https://tianshou.readthedocs.io/en/master/index.html) platform. Please refer the original repo for installation.

## 0 Introduction

  NECSA is implemented in a highly supplementary way. Please refer to tianshou/data/necsa_collector.py for details.

## 1 requirements

  refer to env.yaml

## 2 Anaconda

  * wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
  * `bash ./Anaconda3-2020.11-Linux-x86_64.sh
  * (should be changed)echo 'export PATH="$pathToAnaconda/anaconda3/bin:$PATH"' >> ~/.bashrc
  * (optional) conda config --set auto_activate_base false

## 3 Execution:
  
  * Example:
         
         python necsa_td3.py --task Walker2d-v3 --epoch 1000 --order 3 --grid_num 5 --decay 2.0 --mode state_action

  * Execute the scripts:
         
         bash scripts/HalfCheetah-v3/train_NECSA_TD3.sh

## 4 Experiment results:

  * In ./results

  
  
