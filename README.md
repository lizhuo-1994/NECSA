# Neural Episodic Control with State Abstraction
[NECSA](https://sites.google.com/view/drl-necsa) is based on [tianshou](https://tianshou.readthedocs.io/en/master/index.html) platform. Please refer the original repo for installation.

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
         
         python necsa_td3.py --task Walker2d-v3 --epoch 1000 --step 3 --grid_num 5 --epsilon 0.2 --mode state_action

  * Execute the scripts:
         
         bash scripts/HalfCheetah-v3/train_NECSA_TD3.sh

## 4 Experiment results:

  * In ./results

## 5 Citing and Thanks 

Our work is highly depending on tianshou, thanks to the efforts by the developers. Please kindly cite the paper if you reference our repo.

```latex
@article{tianshou,
  title={Tianshou: A Highly Modularized Deep Reinforcement Learning Library},
  author={Weng, Jiayi and Chen, Huayu and Yan, Dong and You, Kaichao and Duburcq, Alexis and Zhang, Minghao and Su, Yi and Su, Hang and Zhu, Jun},
  journal={arXiv preprint arXiv:2107.14171},
  year={2021}
}
```

Our work NECSA is inspired by 2 state-of-the-art episodic control algorithms: [EMAC](https://github.com/schatty/EMAC) and [GEM](https://github.com/MouseHu/GEM). Please refer to the corresponding repo for details.

```latex
@article{kuznetsov2021solving,
  title={Solving Continuous Control with Episodic Memory},
  author={Kuznetsov, Igor and Filchenkov, Andrey},
  journal={arXiv preprint arXiv:2106.08832},
  year={2021}
}
```

```latex
@article{hu2021generalizable,
  title={Generalizable episodic memory for deep reinforcement learning},
  author={Hu, Hao and Ye, Jianing and Zhu, Guangxiang and Ren, Zhizhou and Zhang, Chongjie},
  journal={arXiv preprint arXiv:2103.06469},
  year={2021}
}
```
  
