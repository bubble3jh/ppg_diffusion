import time
import torch
from denoising_diffusion_pytorch.model import Regressor, MLPRegressor, Unet1DEncoder
from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d_guided import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
import data
import paths
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import argparse
from denoising_diffusion_pytorch.resample import create_named_schedule_sampler
import wandb
import numpy as np
import pickle


def cycle(dl):
    while True:
        for data in dl:
            yield data

def main(args):
    device = torch.device("cuda")
    batch_size = args.train_batch_size
    diffuse_time_step = args.diffusion_time_steps #TODO: 2000step이면 weight 고려되려면 trainstep이 엄청 많아야할듯. 100일때 weight 변화 확인
    epochs=args.train_epochs
    if not args.ignore_wandb:
        wandb.init(entity="ppg-diffusion" ,project="ppg_regressor", config=args)
        wandb.run.name=f"{args.t_scheduling}_epoch_{epochs}_diffuse_{diffuse_time_step}"
    training_seq, label = data.get_data(sampling_method='first_k',
                                    num_samples=5,
                                    data_root=paths.DATA_ROOT,
                                    benchmark='bcg')
        
    schedule_sampler = create_named_schedule_sampler(
        args.t_scheduling, diffuse_time_step=diffuse_time_step,total_epochs=epochs, init_bias=args.init_bias, final_bias=args.final_bias
    ) 
    
    model_path = f"./reg_model/reg_{args.t_scheduling}_epoch_{epochs}_diffuse_{diffuse_time_step}.pt"

    dataset = Dataset1D(training_seq, label=label, normalize=True)
    dl = DataLoader(dataset, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = 0)
    dl = cycle(dl)

    regressor = Unet1DEncoder(
            dim = args.seq_length,
            dim_mults = (1, 2, 4, 8),
            channels = 1
        ).to(device)


    optimizer = optim.Adam(regressor.parameters(), lr=args.init_lr)
    if args.load_checkpoint:
        checkpoint = torch.load(model_path)
        model_state_dict = checkpoint['model_state_dict']
        optim_state_dict = checkpoint['optimizer_state_dict']

        regressor.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optim_state_dict)

    # regressor.load_state_dict(weight)
    diffusion = GaussianDiffusion1D(
        model = regressor,
        seq_length = 625,
        timesteps = diffuse_time_step,
        objective = 'pred_v'
    ).to(device)
    with tqdm(initial = 0, total = epochs) as pbar:
        for i in range(epochs):
            batch, spdp = next(dl)
            batch = batch.to(device); spdp=spdp.to(device)
            # t = torch.randint(0, num_timesteps, (batch.size(0),), device=device).long()
            t, weight = schedule_sampler.sample(batch.size(0), device)
            batch = diffusion.q_sample(batch, t)
            optimizer.zero_grad()
            out, emb = regressor(batch, t)
            loss = F.mse_loss(out, spdp, reduction="none")
            if args.t_scheduling == "loss-second-moment":
                schedule_sampler.update_with_local_losses(t,loss)
            elif args.t_scheduling == "train-step":
                schedule_sampler.set_epoch(i+1)
            loss = loss.mean()
            t_mean = t.sum().item()/len(t)
            pbar.set_description(f'loss: {loss:.4f} / t: {t_mean:.1f}')
            if not args.ignore_wandb :
                wandb.log({"loss": loss.item(), "t_mean": t_mean})
            loss.backward()
            optimizer.step()
            pbar.update(1)
    torch.save({'model_state_dict': regressor.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}
            , model_path)


if __name__ == '__main__':

    ## COMMON --------------------------------------------------
    parser = argparse.ArgumentParser(description="generate ppg with regressor guidance")
    parser.add_argument("--seed", type=int, default=1000, help="random seed (default: 1000)")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--ignore_wandb", action='store_true',
        help = "Stop using wandb (Default : False)")

    ## DATA ----------------------------------------------------
    parser.add_argument("--seq_length", type=int, default=625)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--min_max", action='store_false',
        help = "Min-Max normalize data (Default : True)")
    parser.add_argument("--benchmark", type=str, default='bcg')

    ## Model ---------------------------------------------------
    parser.add_argument("--load_checkpoint", action='store_true',
        help = "Resume model training (Default : False)")
    
    ## Training ------------------------------------------------
    parser.add_argument("--diffusion_time_steps", type=int, default=2000)
    parser.add_argument("--train_epochs", type=int, default=1000)
    parser.add_argument("--init_lr", type=float, default=0.0001)
    parser.add_argument("--init_bias", type=float, default=0.2)
    parser.add_argument("--final_bias", type=float, default=1)
    parser.add_argument("--optim", type=str, default='adam')
    parser.add_argument("--t_scheduling", type=str, default="uniform",  choices=["loss-second-moment", "uniform", "train-step"])

    ## Sampling ------------------------------------------------
    parser.add_argument("--target_group", type=str, default="normal", choices=['all', 'hypo', 'normal','prehyper', 'hyper2', 'crisis'])

    args = parser.parse_args()

    main(args)