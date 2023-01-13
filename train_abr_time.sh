#!/bin/sh
env="abr_sat"
abr="time"
algo="ppo"
exp="torch_lr_scheduler"
seed_max=1
ulimit -n 22222

num_agents=2

echo "env is ${env}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python3 train/train_abr_time.py --env_name ${env} --abr_name ${abr} --algorithm_name ${algo} --experiment_name ${exp} \
    --num_agents ${num_agents} --seed ${seed} --n_training_threads 1 --n_rollout_threads 1 \
    --num_mini_batch 1 --episode_length 100 --num_env_steps 50000 --ppo_epoch 15 \
    --gain 0.01 --lr 3e-4 --critic_lr 1e-4 --hidden_size 512 --layer_N 2 --entropy_coef 0.015 --use_wandb
    echo "training is done!"
done