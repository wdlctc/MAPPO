#!/bin/sh
env="abr_5G"
algo="ppo"
exp="check"
seed_max=1
ulimit -n 22222

echo "env is ${env}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python3 train/train_abr_5G.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --seed ${seed} --n_training_threads 1 --n_rollout_threads 32 \
    --num_mini_batch 1 --episode_length 100 --num_env_steps 500000 --ppo_epoch 15 \
    --gain 0.01 --lr 1e-4 --critic_lr 1e-4 --hidden_size 512 --layer_N 2 --entropy_coef 0.015 
    echo "training is done!"
done
