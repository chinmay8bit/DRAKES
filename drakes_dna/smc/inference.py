import os
import sys
sys.path.append('.')

import torch
import torch.nn.functional as F
import hydra

import diffusion_gosai_update
import oracle
from smc.pipeline import Pipeline
from smc.scheduler import MDLMScheduler
from smc.resampling import resample

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")



def compute_rewards(tokens, reward_model) -> torch.Tensor:
    """
    takes integer tokens directly
    """
    if tokens.ndim == 2:
        onehot_tokens = F.one_hot(tokens, num_classes=4).float()
    elif tokens.ndim == 3:
        onehot_tokens = tokens[:, :, :4]
    else:
        raise ValueError("Tokens must be 2 or 3 dimensional")
    preds = reward_model(onehot_tokens.transpose(1, 2)).squeeze()
    return preds[:, 0]



@hydra.main(config_path="configs", config_name="inference")
def main(config):
    old_path = os.path.join('/home/zo122/CHINMAY/papers_with_code/DRAKES/data_and_model', 'mdlm/outputs_gosai/pretrained.ckpt')
    old_model = diffusion_gosai_update.Diffusion.load_from_checkpoint(old_path, config=config)
    old_model.eval()

    reward_model = oracle.get_gosai_oracle(mode='train')
    reward_model.eval()
    
    scheduler = MDLMScheduler(model=old_model)
    pipe = Pipeline(old_model, scheduler, device, model_dtype=torch.float)
    
    if config.smc.lambda_tempering.enabled:
        lambdas = torch.cat([torch.linspace(0, 1, config.smc.lambda_tempering.one_at + 1), torch.ones(config.smc.num_inference_steps - config.smc.lambda_tempering.one_at)])
    else:
        lambdas = None
    
    samples = pipe(
        resample_fn=lambda log_w: resample(log_w, ess_threshold=config.smc.resampling.ess_threshold, partial=config.smc.resampling.partial),
        reward_fn=lambda x: compute_rewards(x, reward_model),
        batches=config.smc.batches,
        num_particles=config.smc.num_particles,
        batch_p=config.smc.batch_p,
        resample_frequency=config.smc.resampling.frequency,
        num_inference_steps=config.smc.num_inference_steps,
        proposal_type=config.smc.proposal_type,
        use_continuous_formulation=config.smc.use_continuous_formulation,
        kl_weight=config.smc.kl_weight,
        lambdas=lambdas,
        phi=config.smc.phi,
        tau=config.smc.tau,
        final_strategy=config.smc.final_strategy,
    )
    print(samples.shape)
    print(samples)

if __name__ == '__main__':
    main()
