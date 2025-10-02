export CUDA_VISIBLE_DEVICES=$1
export LD_LIBRARY_PATH=

python log_variance_ft/train_wandb.py \
    finetuning.alpha=$2 \
    finetuning.kl_weight=$3 \
    finetuning.kl_method=$4 \
    finetuning.entropy_weight=$5
