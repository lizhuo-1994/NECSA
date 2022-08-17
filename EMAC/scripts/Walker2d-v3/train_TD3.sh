python train.py --policy TD3 --env Walker2d-v3 --k 2 --max_timesteps 1000001 --exp_name base --device cuda:0 --eval_freq 1000 --seed 0
python train.py --policy TD3 --env Walker2d-v3 --k 2 --max_timesteps 1000001 --exp_name base --device cuda:0 --eval_freq 1000 --seed 1
python train.py --policy TD3 --env Walker2d-v3 --k 2 --max_timesteps 1000001 --exp_name base --device cuda:0 --eval_freq 1000 --seed 2
python train.py --policy TD3 --env Walker2d-v3 --k 2 --max_timesteps 1000001 --exp_name base --device cuda:0 --eval_freq 1000 --seed 3
python train.py --policy TD3 --env Walker2d-v3 --k 2 --max_timesteps 1000001 --exp_name base --device cuda:0 --eval_freq 1000 --seed 4

python train.py --policy EMAC --env Walker2d-v3 --k 2 --max_timesteps 1000001 --exp_name k2_alpha_01_beta_05 --alpha 0.1 --device cuda:0 --eval_freq 1000 --prioritized --beta 0.5 --seed 0
python train.py --policy EMAC --env Walker2d-v3 --k 2 --max_timesteps 1000001 --exp_name k2_alpha_01_beta_05 --alpha 0.1 --device cuda:0 --eval_freq 1000 --prioritized --beta 0.5 --seed 1
python train.py --policy EMAC --env Walker2d-v3 --k 2 --max_timesteps 1000001 --exp_name k2_alpha_01_beta_05 --alpha 0.1 --device cuda:0 --eval_freq 1000 --prioritized --beta 0.5 --seed 2
python train.py --policy EMAC --env Walker2d-v3 --k 2 --max_timesteps 1000001 --exp_name k2_alpha_01_beta_05 --alpha 0.1 --device cuda:0 --eval_freq 1000 --prioritized --beta 0.5 --seed 3
python train.py --policy EMAC --env Walker2d-v3 --k 2 --max_timesteps 1000001 --exp_name k2_alpha_01_beta_05 --alpha 0.1 --device cuda:0 --eval_freq 1000 --prioritized --beta 0.5 --seed 4
