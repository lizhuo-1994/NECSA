python necsa_dqn.py --task AlienNoFrameskip-v4  --epoch 500 --step 1 --epsilon 0.01 --mode hidden --reduction
killall -9 python
python necsa_dqn.py --task AlienNoFrameskip-v4  --epoch 500 --step 1 --epsilon 0.05 --mode hidden --reduction
killall -9 python
python necsa_dqn.py --task AlienNoFrameskip-v4  --epoch 500 --step 1 --epsilon 0.2 --mode hidden --reduction
killall -9 python
python necsa_dqn.py --task AlienNoFrameskip-v4  --epoch 500 --step 1 --epsilon 0.4 --mode hidden --reduction
killall -9 python
python necsa_dqn.py --task AlienNoFrameskip-v4  --epoch 500 --step 1 --epsilon 0.5 --mode hidden --reduction
killall -9 python
python necsa_dqn.py --task AlienNoFrameskip-v4  --epoch 500 --step 1 --epsilon 1.0 --mode hidden --reduction
killall -9 python
python necsa_dqn.py --task AlienNoFrameskip-v4  --epoch 500 --step 1 --epsilon 2.0 --mode hidden --reduction
killall -9 python


