# Neural Episodic Control with State Abstraction
  * [NECSA](https://sites.google.com/view/drl-necsa) is based on [tianshou](https://tianshou.readthedocs.io/en/master/index.html) platform. Please refer the original repo for installation.

## 0 Introduction

  * NECSA is implemented in a highly supplementary way. Please refer to tianshou/data/necsa_collector.py and necsa_atari_collector.py for details.

## 1 requirements

  * refer to requirements.txt

## 2 Anaconda and Python

  * wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
  * bash ./Anaconda3-2020.11-Linux-x86_64.sh
  * (should be changed)echo 'export PATH="$pathToAnaconda/anaconda3/bin:$PATH"' >> ~/.bashrc
  * (optional) conda config --set auto_activate_base false
  * conda env create -f env.yaml
  * conda create -n necsa python=3.8.5
  * conda activate necsa
  * pip3 install -r requirements.txt

## 3 Install Atari and MuJoCo

  * Download the [ROM files](http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html) for Atari, unzip and execute:
  * python -m atari_py.import_roms <path to folder> 
  * wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
  * tar xvf mujoco210-linux-x86_64.tar.gz && mkdir -p ~/.mujoco && mv mujoco210 ~/.mujoco/mujoco210
  * wget https://www.roboti.us/file/mjkey.txt -O  ~/.mujoco/mjkey.txt
  * echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin" >> ~/.bashrc
         

## 4 Execution:
  
  * Example:
         
         python necsa_td3.py --task Walker2d-v3 --epoch 1000 --step 3 --grid_num 5 --epsilon 0.2 --mode state_action

  * Execute the scripts:
         
         bash scripts/HalfCheetah-v3/train_NECSA_TD3.sh

## 5 Experiment results:

  * Data will be automatically saved into ./results

## 6 Citing and Thanks 

  * Our program is highly depending on tianshou, thanks to the efforts by the developers. Please kindly cite the paper if you referenced our repo.

  ```latex
  @article{tianshou,
    title={Tianshou: A Highly Modularized Deep Reinforcement Learning Library},
    author={Weng, Jiayi and Chen, Huayu and Yan, Dong and You, Kaichao and Duburcq, Alexis and Zhang, Minghao and Su, Yi and Su, Hang and Zhu, Jun},
    journal={arXiv preprint arXiv:2107.14171},
    year={2021}
  }
  ```

  * Our work NECSA is also inspired by 3 state-of-the-art episodic control algorithms: [EMAC](https://github.com/schatty/EMAC), [EVA](https://github.com/AnnaNikitaRL/EVA) and [GEM](https://github.com/MouseHu/GEM). Please refer to the corresponding repo for details.

  ```
  @article{kuznetsov2021solving,
    title={Solving Continuous Control with Episodic Memory},
    author={Kuznetsov, Igor and Filchenkov, Andrey},
    journal={arXiv preprint arXiv:2106.08832},
    year={2021}
  }
  ```

  ```
 @article{hansen2018fast,
  title={Fast deep reinforcement learning using online adjustments from the past},
  author={Hansen, Steven and Pritzel, Alexander and Sprechmann, Pablo and Barreto, Andr{\'e} and Blundell, Charles},
  journal={Advances in Neural Information Processing Systems},
  volume={31},
  year={2018}
}
  ```

  ```
  @article{hu2021generalizable,
    title={Generalizable episodic memory for deep reinforcement learning},
    author={Hu, Hao and Ye, Jianing and Zhu, Guangxiang and Ren, Zhizhou and Zhang, Chongjie},
    journal={arXiv preprint arXiv:2103.06469},
    year={2021}
  }
  ```
  
