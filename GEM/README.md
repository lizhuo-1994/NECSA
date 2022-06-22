
Ubuntu 20.04

## 1 Miniconda

  * wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  * chmod +x Miniconda3-latest-Linux-x86_64.sh
  * ./Miniconda3-latest-Linux-x86_64.sh
  * echo 'export PATH="$pathToMiniconda/anaconda3/bin:$PATH"' >> ~/.bashrc
  * source ~/.bashrc
  * (optional) conda config --set auto_activate_base false

## 2 Install & activate environment:  
  download [gem.tar.gz]([https://1drv.ms/u/s!Aj44OX1lWGicc35GmcDsOfg8SDE?e=up9Vjf](https://drive.google.com/file/d/1bL69C_r0lvmh1lniaH9keAAPIeAAZmMe/view?usp=sharing)

  * tar -zxzf gem.tar.gz 
  * mv drl ~/conda/envs/
  * conda activate drl

## 3 Install Mujoco:

  * download [mujoco.tar.gz](https://drive.google.com/file/d/1Pi3HWx5ZPe92WxtJ8lEZzI3tPBQ15J8-/view?usp=sharing)
  * tar -zxzf mujoco.tar.gz 
  * mv .mujoco /home/YOURACCOUNT/
  * echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/YOURACCOUNT/.mujoco/mujoco210/bin' >> ~/.bashrc
  * echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia' >> ~/.bashrc
  * source ~/.bashrc

## 4 Execution:

  * To run MuJoCO , please refer to ./gem_mujoco/run/run.sh
  * To run Atari , please refer to /gem_atari/run_gem.sh

## 5 Results

   For example, MuJoCo experiments are in GEM/gem_mujoco/log_gem/mujoco/gem+tbp


