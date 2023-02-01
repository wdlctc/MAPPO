#!/bin/sh
env="abr_sat"
abr="time"
algo="ppo"
exp="train&test10states"
seed_max=1
#model_dir=$2
ulimit -n 22222

num_agents=$1
#logfile=$3

echo "env is ${env}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=3 python3 train/test_time.py --env_name ${env} --abr_name ${abr} --algorithm_name ${algo} --experiment_name ${exp} \
    --num_agents ${num_agents} --seed ${seed} --n_training_threads 1 --n_rollout_threads 1 \
    --num_mini_batch 1 --episode_length 100 --num_env_steps 2000000 --ppo_epoch 15 \
    --gain 0.01 --lr 1e-4 --critic_lr 1e-3 --hidden_size 128 --layer_N 2 --entropy_coef 0.015 --use_wandb \
    --use_eval --eval_interval 10 --eval_type "all" --save_interval 1000
    echo "training is done!"
done
