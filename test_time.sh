#!/bin/sh
env="abr_sat"
abr="time"
algo="ppo"
exp="train&test_tight"
seed_max=1
model_dir=""
ulimit -n 22222

num_agents=$1

echo "env is ${env}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python3 train/test_time.py --env_name ${env} --abr_name ${abr} --algorithm_name ${algo} --experiment_name ${exp} \
    --num_agents ${num_agents} --seed ${seed} --n_training_threads 1 --n_rollout_threads 1 \
    --num_mini_batch 1 --episode_length 100 --num_env_steps 20000000 --ppo_epoch 15 \
    --gain 0.01 --lr 1e-4 --critic_lr 1e-4 --hidden_size 512 --layer_N 2 --entropy_coef 0.015 --use_wandb \
    --use_eval --eval_interval 1000 --eval_type "all" --save_interval 1000
    echo "training is done!"
done
