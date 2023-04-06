# Generative Model-Based Testing on Decision-Making Policies
  * This work is based on the implementation of paper [MDPFuzz: Testing Models Solving Markov Decision Processes](https://github.com/Qi-Pang/MDPFuzz). Please refer the original repo for installation.

## 0 Introduction

  * This work is uses a 1D diffusion model as test case generators for decision-making models.

## 1 Anaconda

  * wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
  * bash ./Anaconda3-2020.11-Linux-x86_64.sh
  * (should be changed)echo 'export PATH="$pathToAnaconda/anaconda3/bin:$PATH"' >> ~/.bashrc
  * (optional) conda config --set auto_activate_base false
  * conda env create -f env.yaml
  * conda activate necsa

## 2 Install

## 3 Execution:
  
  * Run test by our method in BipelWalker:
         
         python test_gen.py --method generative+novelty --hour 12

  * Run repair based on the detected failures:
         
         python retrain.py --alg tqc --env BipedalWalkerHardcore-v3 --folder rl-trained-agents/

## 4 Experiment results:

  * Data will be automatically saved into ./results

## 5 Citing and Thanks 

  * Our program is highly depending on MDPFuzz, thanks to the efforts by the developers. Please kindly cite the paper if you referenced our repo.

  ```latex
  @inproceedings{10.1145/3533767.3534388,
  author = {Pang, Qi and Yuan, Yuanyuan and Wang, Shuai},
  title = {MDPFuzz: Testing Models Solving Markov Decision Processes},
  year = {2022},
  isbn = {9781450393799},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3533767.3534388},
  doi = {10.1145/3533767.3534388},
  booktitle = {Proceedings of the 31st ACM SIGSOFT International Symposium on Software Testing and Analysis},
  pages = {378–390},
  numpages = {13},
  keywords = {Markov decision procedure, Deep learning testing},
  location = {Virtual, South Korea},
  series = {ISSTA 2022}
  }
  ```
