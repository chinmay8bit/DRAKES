set -euo pipefail

# arg1: cuda device number
# arg2: kl weight (alpha)

# export WANDB_CACHE_DIR="./wandb_cache"
export CUDA_VISIBLE_DEVICES=$1
export LD_LIBRARY_PATH=

# for final_strategy in multinomial; do
#     for guidance_scale in 5000 10000 50000; do
#         for n_particles in 1 2 4 8 16 32; do
#             python test_smc_grad.py --n_particles $n_particles --final_strategy $final_strategy \
#                 --log_dir "logs/reward_bp_results_final/base/grad_smc/guidance_${guidance_scale}" \
#         done
#     done
# done

final_strategy="argmax_rewards"
kl_weight=$2
for lambda_one_at in 100; do
    for n_particles in 1 2 4 8 16 32; do
        python smc/test_smc_grad.py smc.num_particles=$n_particles smc.final_strategy=$final_strategy \
            smc.batch_p=8192 smc.kl_weight=$kl_weight smc.lambda_tempering.one_at=$lambda_one_at \
            smc.resampling.frequency=8 smc.resampling.ess_threshold=null \
            smc.proposal_type="ft_model" \
            smc.use_ft_model_for_expected_reward=True \
            smc.ft_model_ckpt="/home/zo122/CHINMAY/papers_with_code/DRAKES/drakes_dna/log_variance_ft/model_weights_final/20250923/152254/ckpt_326/model.pth" \
            smc.ft_label="_ft_swept-glitter-52_ckpt-326"
    done
done
