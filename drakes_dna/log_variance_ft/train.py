import sys
sys.path.append('../')

import os

import math

from rich import print
import argparse

import diffusion_gosai_update
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import oracle
from grelu.lightning import LightningModel
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import set_seed, get_metadata, save_metadata_json
set_seed(0, use_cuda=True)


def main(args):
    BASE_PATH = '/home/zo122/CHINMAY/papers_with_code/DRAKES/data_and_model'
    # pretrained model
    CKPT_PATH = os.path.join(BASE_PATH, 'mdlm/outputs_gosai/pretrained.ckpt')

    # reinitialize Hydra
    GlobalHydra.instance().clear()

    # Initialize Hydra and compose the configuration|
    initialize(config_path="../configs_gosai", job_name="load_model")
    cfg = compose(config_name="config_gosai.yaml")
    cfg.eval.checkpoint_path = CKPT_PATH
    
    
    p_ref = diffusion_gosai_update.Diffusion.load_from_checkpoint(CKPT_PATH, config=cfg)
    p_ref.eval()
    
    
    reward_model_ft = oracle.get_gosai_oracle(mode='train')
    reward_model_ft.eval()

    # This is what DRAKES has used - 0.001 (Appendix F.2)
    kl_weight = args.kl_weight
    kl_weight_annealing = args.kl_weight_annealing

    @torch.no_grad()
    def compute_rewards(tokens) -> torch.Tensor:
        """
        takes integer tokens directly
        """
        onehot_tokens = F.one_hot(tokens, num_classes=4).float()
        preds = reward_model_ft(onehot_tokens.float().transpose(1, 2)).squeeze()
        return preds[:, 0]

    def compute_rewards_with_kl_weight(tokens, anneal_factor=1.0):
        rewards = compute_rewards(tokens)
        return (rewards / kl_weight) * anneal_factor
    
    @torch.no_grad()
    def estimate_reward(probs, num_samples, method='mean', kl_anneal_factor=1.0):
        B = probs.shape[0]
        dist = torch.distributions.Categorical(probs=probs)
        samples = dist.sample((num_samples,)).reshape(num_samples * B, -1) # type: ignore
        rewards = compute_rewards_with_kl_weight(samples, anneal_factor=kl_anneal_factor).reshape(num_samples, B)
        if method == 'mean':
            return rewards.mean(dim=0) # E[r(x)/alpha]
        elif method == 'logmeanexp':
            return rewards.logsumexp(dim=0) - math.log(num_samples) # log E[exp(r(x)/alpha)]
        else:
            raise ValueError(f"Unknown method: {method}")
        
    atac_acc_model = LightningModel.load_from_checkpoint(os.path.join(BASE_PATH, 'mdlm/gosai_data/binary_atac_cell_lines.ckpt'), map_location='cuda')
    atac_acc_model.eval()

    @torch.no_grad()
    def cal_atac_acc_fast(tokens):
        """
        tokens: list of sequences (tokenized)
        """
        onehot_tokens = F.one_hot(tokens, num_classes=4).float()
        preds = atac_acc_model(onehot_tokens.float().transpose(1, 2)).detach().cpu().numpy()
        preds = preds.squeeze() # numpy array with shape [n_seqs, 7]
        return (preds[:,1]>0.5).sum()/len(preds)
            
    q_phi = diffusion_gosai_update.Diffusion.load_from_checkpoint(CKPT_PATH, config=cfg)
    if args.start_ckpt is not None:
        start_ckpt = args.start_ckpt
        q_phi.load_state_dict(torch.load(start_ckpt))
    q_phi.eval()
    num_timesteps = q_phi.config.sampling.steps
    f_psi = torch.nn.Parameter(torch.zeros(num_timesteps, device=q_phi.device))
    
    
    batch_size = args.batch_size
    lr = args.lr
    optimizer = torch.optim.AdamW(list(q_phi.parameters()) + [f_psi], lr=lr)
    num_epochs = args.num_epochs
    batches_per_epoch = args.batches_per_epoch
    patience = args.patience
    sample_onpolicy = args.sample_onpolicy
    num_samples_for_reward_estimate = args.num_samples_for_reward_estimate
    reward_estimate_method = args.reward_estimate_method
    timesteps_for_loss = args.timesteps_for_loss
    regularization_strength = args.regularization_strength

    base_dir = 'model_weights'  # keep base folder
    timestamp = datetime.now().strftime("%Y%m%d/%H%M%S")  # e.g. 20250818/004927
    model_save_dir = os.path.join(base_dir, timestamp)

    os.makedirs(model_save_dir, exist_ok=True)
    ckpt_path_best_loss = f'{model_save_dir}/best_loss.pth'
    ckpt_path_best_reward = f'{model_save_dir}/best_reward.pth'
    ckpt_path_best_atac_acc = f'{model_save_dir}/best_atac_acc.pth'
    
    # Save config and metadata files
    OmegaConf.save(config=cfg, f=f'{model_save_dir}/config.yaml')

    metadata = get_metadata(dict(locals()), ignore_hidden=True)
    print(metadata)
    save_metadata_json(metadata, model_save_dir)
    
    
    loss_trace = []
    reward_trace = []
    atac_acc_trace = []
    
    
    L = q_phi.config.model.length
    eps=1e-5
    timesteps = torch.linspace(1, eps, num_timesteps + 1, device=q_phi.device)
    dt = (1 - eps) / num_timesteps

    # Training loop
    for epoch in range(num_epochs):
        total_epoch_loss = 0.0
        total_epoch_rewards = 0.0
        total_epoch_atac_acc = 0.0
        for batch_idx in range(batches_per_epoch):
            q_phi.train()
            
            rewards_prev = None
            log_prob_p_ref = None
            log_prob_q_phi = None
            loss = torch.tensor(0.0, device=q_phi.device)
            kld_regularization = torch.tensor(0.0, device=q_phi.device)
            
            # We select only #timesteps_for_loss timesteps randomly for loss calculation to fit in memory
            is_selected_timestep = torch.zeros(
                num_timesteps, dtype=torch.bool
            ).scatter_(0, torch.randperm(num_timesteps)[:timesteps_for_loss], True)
            
            # Generate batch_size samples from q_phi
            z_t = q_phi._sample_prior(batch_size, L).to(q_phi.device)
            for i in range(num_timesteps, 0, -1):
                t = timesteps[num_timesteps - i] * torch.ones(z_t.shape[0], 1, device=q_phi.device)
                # Invoke pretrained and finetune models
                with torch.enable_grad() if is_selected_timestep[i-1] else torch.no_grad():
                    q_phi_zs_given_zt, q_phi_z0_given_zt = q_phi._sample_step(z_t, t, dt)
                with torch.no_grad():
                    p_ref_zs_given_zt, p_ref_z0_given_zt = p_ref._sample_step(z_t, t, dt)
                    
                if is_selected_timestep[i-1]:
                    assert q_phi_z0_given_zt.requires_grad
                    kld_batch = torch.where(
                        p_ref_z0_given_zt > 0,
                        p_ref_z0_given_zt * (torch.log(p_ref_z0_given_zt) - torch.log(q_phi_z0_given_zt.clamp_min(1e-12))),
                        torch.zeros_like(p_ref_z0_given_zt)
                    ).sum(dim=(1, 2))
                    kld_regularization += kld_batch.mean(dim=0) # take mean across batch dimension
                
                # Only calculate the rewards, if it will be used for loss calculation 
                is_rewards_needed = (
                    (i < num_timesteps and is_selected_timestep[i]) or # is needed in this step
                    is_selected_timestep[i-1] # is needed for the next step
                )
                # Estimate rewards
                if is_rewards_needed:
                    if kl_weight_annealing:
                        anneal_factor = (num_timesteps - i) / num_timesteps
                    else:
                        anneal_factor = 1.0
                    rewards = estimate_reward(p_ref_z0_given_zt, num_samples_for_reward_estimate, method=reward_estimate_method, kl_anneal_factor=anneal_factor)
                else:
                    rewards = None
                
                if i < num_timesteps and is_selected_timestep[i]:
                    # Sanity checks
                    assert rewards is not None and rewards_prev is not None
                    assert log_prob_p_ref is not None and log_prob_q_phi is not None
                    assert log_prob_q_phi.requires_grad
                    
                    log_w = (rewards - rewards_prev) + (log_prob_p_ref - log_prob_q_phi) # Shape: (batch-size,)
                    log_variance = (log_w - f_psi[i]) ** 2
                    loss += log_variance.mean(dim=0) # take mean across batch dimension
                
                q_phi_dist = torch.distributions.Categorical(probs=q_phi_zs_given_zt)
                p_ref_dist = torch.distributions.Categorical(probs=p_ref_zs_given_zt)
                
                if sample_onpolicy:
                    z_s = q_phi_dist.sample()
                else:
                    z_s = p_ref_dist.sample()
                    
                log_prob_q_phi = q_phi_dist.log_prob(z_s).sum(dim=1)
                log_prob_p_ref = p_ref_dist.log_prob(z_s).sum(dim=1)
                
                # Update for next step
                z_t = z_s
                rewards_prev = rewards
                
            z_0 = z_t
            if q_phi.config.sampling.noise_removal:
                with torch.no_grad():
                    t = timesteps[-1] * torch.ones(z_0.shape[0], 1, device=q_phi.device)
                    unet_conditioning = q_phi.noise(t)[0]
                    logits = q_phi.forward(z_0, unet_conditioning)
                    z_0 = logits[:, :, :-1].argmax(dim=-1)
            
            # Compute rewards
            rewards = compute_rewards_with_kl_weight(z_0)
            total_epoch_rewards += rewards.sum(dim=0).item() * kl_weight # because the rewards we have is with kl weight r(x)/kl_weight
            
            if is_selected_timestep[0]:
                assert rewards_prev is not None and log_prob_p_ref is not None and log_prob_q_phi is not None
                log_w = (rewards - rewards_prev) + (log_prob_p_ref - log_prob_q_phi) # Shape: (batch-size,)
                log_variance = (log_w - f_psi[0]) ** 2
                loss += log_variance.mean(dim=0) # take mean across batch dimension
            
            # Add KL regularization
            loss += regularization_strength * kld_regularization
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            atac_acc = cal_atac_acc_fast(z_0) * 100.0 # in percentage
            total_epoch_atac_acc += atac_acc.item()

            total_epoch_loss += loss.item()
            print((f"Batch {batch_idx+1}/{batches_per_epoch}, "
               f"Loss: {loss.item()}, Reward (avg): {rewards.mean(dim=0).item() * kl_weight} "
               f"KL Regularization: {kld_regularization.item()} "
               f"ATAC Accuracy: {atac_acc.item()}"))
        
        q_phi.eval()
        avg_loss = total_epoch_loss / batches_per_epoch
        avg_rewards = total_epoch_rewards / (batches_per_epoch * batch_size)
        avg_atac_acc = total_epoch_atac_acc / batches_per_epoch
        
        print(f"Epoch {epoch+1}/{num_epochs},  Loss (avg): {avg_loss}, Reward (avg): {avg_rewards}, ATAC acc (avg): {avg_atac_acc}")
        loss_trace.append(avg_loss)
        reward_trace.append(avg_rewards)
        atac_acc_trace.append(avg_atac_acc)
        
        if loss_trace[-1] == min(loss_trace):
            # store model weights
            torch.save(q_phi.state_dict(), ckpt_path_best_loss)
            print(f"Best loss yet! Saved model weights to {ckpt_path_best_loss}")
        if reward_trace[-1] == max(reward_trace):
            # store model weights
            torch.save(q_phi.state_dict(), ckpt_path_best_reward)
            print(f"Best reward yet! Saved model weights to {ckpt_path_best_reward}")
        if atac_acc_trace[-1] == max(atac_acc_trace):
            # store model weights
            torch.save(q_phi.state_dict(), ckpt_path_best_atac_acc)
            print(f"Best ATAC accuracy yet! Saved model weights to {ckpt_path_best_atac_acc}")
            
        # If BOTH loss and reward stop imporving, then stop training
        if (
            min(loss_trace) < min(loss_trace[-patience:]) and 
            max(reward_trace) > max(reward_trace[-patience:]) and 
            max(atac_acc_trace) > max(atac_acc_trace[-patience:])
        ):
            break
        
    # loss_trace and reward_trace are 1D lists (or 1D arrays) of the same length
    epochs = range(1, len(loss_trace) + 1)

    fig, ax1 = plt.subplots()
    ax1.plot(epochs, loss_trace, label='Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')

    ax2 = ax1.twinx()
    ax2.plot(epochs, reward_trace, label='Reward', color='green')
    ax2.set_ylabel('Reward')

    # place legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.title('Loss and Reward vs. Epoch')
    plt.savefig(f'{model_save_dir}/loss_reward_trace.png')
    plt.close()
        
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--batches_per_epoch", type=int, default=5)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--sample_onpolicy", action='store_true', default=False)
parser.add_argument("--num_samples_for_reward_estimate", type=int, default=10)
parser.add_argument("--reward_estimate_method", type=str, default='logmeanexp', choices=['mean', 'logmeanexp'])
parser.add_argument("--timesteps_for_loss", type=int, default=10)
parser.add_argument("--kl_weight", type=float, default=0.001)
parser.add_argument("--kl_weight_annealing", action='store_true', default=False)
parser.add_argument("--regularization_strength", type=float, default=1.0)
parser.add_argument("--start_ckpt", type=str, default=None)


# 3. Parse the arguments
args = parser.parse_args()
main(args)
