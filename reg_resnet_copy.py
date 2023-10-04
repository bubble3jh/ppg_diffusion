import time
import torch
from denoising_diffusion_pytorch.model import *
from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d_guided import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
import paths
from utils import *
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import argparse
from denoising_diffusion_pytorch.resample import create_named_schedule_sampler
import wandb
import numpy as np
import pickle
from torch.optim.lr_scheduler import CosineAnnealingLR

def calculate_group_losses(out, spdp, g, args):
    group = same_to_group(g)
    group_losses = {}
    for g_i in torch.unique(group):
        indices = torch.where(group == g_i)[0]
        model_output_group = out[indices]
        ground_truth_group = spdp[indices]
        g_loss = F.mse_loss(model_output_group, ground_truth_group, reduction="mean")
        group_losses[g_i.item()] = g_loss
    
    total_loss = sum(group_losses.values())
    group_avg_loss = total_loss / len(group_losses)
    worst_group_loss = max(group_losses.values())
    import pdb;pdb.set_trace()
    if not args.ignore_wandb:
        for g_i in torch.unique(group):
            wandb.run.summary[f"val_best_group_{g_i}_mae_sbp"] = group_losses[g_i.item()]
            wandb.run.summary[f"val_best_group_{g_i}_mae_dbp"] = sum(val_mae_dbp_lists[group]) / len(val_mae_dbp_lists[group])
            wandb.run.summary[f"val_best_group_{g_i}_mae_tot"] = (sum(val_mae_sbp_lists[group]) / len(val_mae_sbp_lists[group])) + (sum(val_mae_dbp_lists[group]) / len(val_mae_dbp_lists[group]))
    
    return group_avg_loss, worst_group_loss

def train_one_epoch(batch, spdp, g, regressor, optimizer, args, diffusion, schedule_sampler, tr_dataset, epoch, device):
    batch = batch.to(device)
    spdp = spdp.to(device)
    g = g.to(device)
    t, _ = schedule_sampler.sample(batch.size(0), device)
    batch = diffusion.q_sample(batch, t)
    optimizer.zero_grad()
    out = regressor(batch, t, g)
    loss = F.mse_loss(out, spdp, reduction="none")
    group_avg_loss = calculate_group_losses(out, spdp, g, args)
    overall_mae_sbp_list, overall_mae_dbp_list, mae_sbp_lists, mae_dbp_lists = calculate_batch_mae(out, spdp, tr_dataset, g, {}, {}, [], [])
    overall_mae_sbp, overall_mae_dbp, _, _ = log_global_metrics(args, overall_mae_sbp_list, overall_mae_dbp_list, mae_sbp_lists, mae_dbp_lists, "train")
    if args.loss == "ERM":
        loss = loss.mean()
        loss.backward()
    elif args.loss == "group_average_loss":
        group_avg_loss.backward()
    optimizer.step()
    t_mean = t.sum().item() / len(t)
    
    # Time scheduling update
    if args.t_scheduling == "loss-second-moment":
        schedule_sampler.update_with_local_losses(t, loss)
    elif args.t_scheduling == "train-step":
        schedule_sampler.set_epoch(epoch + 1)
        
    return loss, overall_mae_sbp, overall_mae_dbp, t_mean

def validate_all(val_dl, regressor, diffusion, val_t_all, val_dataset, args, device):
    val_mae_sbp_lists = {}
    val_mae_dbp_lists = {}
    val_overall_mae_sbp_list = []
    val_overall_mae_dbp_list = []
    val_loss_sbp = 0
    val_loss_dbp = 0
    start_idx = 0
    group_losses = []
    worst_group_losses = []

    with torch.no_grad():
        for data in val_dl:
            val_batch, val_spdp, val_g = data
            val_batch = val_batch.to(device)
            val_spdp = val_spdp.to(device)
            val_g = val_g.to(device)
            val_t = val_t_all[start_idx:start_idx + val_batch.size(0)].to(device)
            val_batch = diffusion.q_sample(val_batch, val_t)
            val_out = regressor(val_batch, val_t, val_g)
            
            group_avg_loss, worst_group_loss = calculate_group_losses(val_out, val_spdp, val_g)
            group_losses.append(group_avg_loss.item())
            worst_group_losses.append(worst_group_loss.item())

            val_overall_mae_sbp_list, val_overall_mae_dbp_list, val_mae_sbp_lists, val_mae_dbp_lists = calculate_batch_mae(val_out, val_spdp, val_dataset, val_g, val_mae_sbp_lists, val_mae_dbp_lists, val_overall_mae_sbp_list, val_overall_mae_dbp_list)
            start_idx += val_batch.size(0)
        val_loss_sbp, val_loss_dbp, _, _ = log_global_metrics(args, val_overall_mae_sbp_list, val_overall_mae_dbp_list, val_mae_sbp_lists, val_mae_dbp_lists, "valid")

    return val_loss_sbp, val_loss_dbp, sum(group_losses)/len(group_losses), max(worst_group_losses)


def load_checkpoint(model_path, regressor, optimizer):
    checkpoint = torch.load(model_path+"_resnet.pt")
    model_state_dict = checkpoint['model_state_dict']
    optim_state_dict = checkpoint['optimizer_state_dict']

    regressor.load_state_dict(model_state_dict)
    optimizer.load_state_dict(optim_state_dict)
    
def save_checkpoint(model_path, regressor, optimizer, suffix=""):
    torch.save({
        'model_state_dict': regressor.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, model_path+f"_resnet_{suffix}.pt")
    
def update_best_model(loss, val_loss_sbp, val_loss_dbp, best_val_loss_sbp, best_val_loss_dbp, regressor, optimizer, model_path, mode="ERM"):
    updated = False
    if val_loss_sbp + val_loss_dbp < best_val_loss_sbp + best_val_loss_dbp:
        best_val_loss_sbp = val_loss_sbp
        best_val_loss_dbp = val_loss_dbp
        save_checkpoint(model_path, regressor, optimizer, f"resnet_{args.loss}_{mode}")
        updated = True

    return best_val_loss_sbp, best_val_loss_dbp, updated


def cycle(dl):
    while True:
        for data in dl:
            yield data

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda")
    batch_size = args.train_batch_size
    diffuse_time_step = args.diffusion_time_steps 
    epochs=args.train_epochs
    if not args.ignore_wandb:
        wandb.init(entity="ppg-diffusion" ,project="ppg_regressor", config=args)
        wandb.run.name=f"fold_{args.train_fold}_{args.t_scheduling}_epoch_{epochs}_diffuse_{diffuse_time_step}_wd_{args.weight_decay}_eta_{args.eta_min}_lr_{args.init_lr}_nblock_{args.n_block}_{args.final_layers}-layer-clf"
    data = get_data(sampling_method='first_k',
                                    num_samples=5,
                                    data_root=paths.DATA_ROOT,
                                    benchmark='bcg',
                                    train_fold=args.train_fold)
    
    schedule_sampler = create_named_schedule_sampler(
        args.t_scheduling, diffuse_time_step=diffuse_time_step,total_epochs=epochs, init_bias=args.init_bias, final_bias=args.final_bias
    ) 
    
    model_path = f"/mlainas/ETRI_2023/reg_model/fold_{args.train_fold}/{args.t_scheduling}_epoch_{epochs}_diffuse_{diffuse_time_step}_wd_{args.weight_decay}_eta_{args.eta_min}_lr_{args.init_lr}_nblock_{args.n_block}_{args.final_layers}-layer-clf"

    tr_dataset = Dataset1D(data['train']['ppg'], label=data['train']['spdp'], groups=data['train']['group_label'] ,normalize=True)
    val_dataset = Dataset1D(data['valid']['ppg'], label=data['valid']['spdp'], groups=data['valid']['group_label'] ,normalize=True)
    val_every = int(len(tr_dataset) / args.train_batch_size); print(f'eval every {val_every} epochs')
    tr_dl = DataLoader(tr_dataset, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = 0)
    tr_dl = cycle(tr_dl)
    
    val_dl = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, pin_memory = True, num_workers = 0)
    # regressor = Unet1DEncoder(
    #         dim = args.seq_length,
    #         dim_mults = (1, 2, 4, 8),
    #         channels = 1
    #     ).to(device)
    regressor = ResNet1D(output_size=2, final_layers=args.final_layers, n_block=args.n_block).to(device)
    optimizer = optim.Adam(regressor.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.eta_min)

    # 전체 validation set 크기만큼 val_t_all을 미리 샘플링
    total_val_size = len(val_dl.dataset)
    val_t_all, _ = schedule_sampler.sample(total_val_size, device)

    if args.load_checkpoint:
        load_checkpoint(model_path, regressor, optimizer)

    diffusion = GaussianDiffusion1D(
        model = regressor,
        seq_length = 625,
        timesteps = diffuse_time_step,
        objective = 'pred_v'
    ).to(device)
    # Initialize loss variables
    best_train_loss = float('inf')
    best_val_train_loss = float('inf')

    # For standard validation
    best_val_loss_sbp = float('inf')
    best_val_loss_dbp = float('inf')

    # For group average loss validation
    best_gal_val_loss_sbp = float('inf')
    best_gal_val_loss_dbp = float('inf')

    # For worst group validation
    best_worst_g_val_loss_sbp = float('inf')
    best_worst_g_val_loss_dbp = float('inf')
    with tqdm(initial = 0, total = epochs) as pbar:
        for i in range(epochs):
            batch, spdp, g = next(tr_dl)
            loss, overall_mae_sbp, overall_mae_dbp, t_mean = train_one_epoch(batch, spdp, g, regressor, optimizer, args, diffusion, schedule_sampler, tr_dataset, i, device)
            
            if args.t_scheduling == "loss-second-moment":
                schedule_sampler.update_with_local_losses(t, loss)
            elif args.t_scheduling == "train-step":
                schedule_sampler.set_epoch(i+1)
            
            if not args.ignore_wandb:
                wandb.log({"train_loss": loss.item(), "t_mean": t_mean})

            if (i + 1) % val_every == 0:
                val_loss_sbp, val_loss_dbp, avg_group_val_loss, worst_group_val_loss = validate_all(val_dl, regressor, diffusion, val_t_all, val_dataset, args, device)
                
                # Update Best Model using ERM loss
                best_val_loss_sbp, best_val_loss_dbp, updated = update_best_model(loss, val_loss_sbp, val_loss_dbp, best_val_loss_sbp, best_val_loss_dbp, regressor, optimizer, model_path, mode="ERM")
                if updated:
                    print(f"best ERM val loss updated: sbp {best_val_loss_sbp:.4f} dbp {best_val_loss_dbp:.4f}")
                
                # Update Best Model using group average loss
                best_gal_val_loss_sbp, best_gal_val_loss_dbp, updated = update_best_model(avg_group_val_loss, val_loss_sbp, val_loss_dbp, best_gal_val_loss_sbp, best_gal_val_loss_dbp, regressor, optimizer, model_path, mode="GAL")
                if updated:
                    print(f"best GAL val loss updated: sbp {best_gal_val_loss_sbp:.4f} dbp {best_gal_val_loss_dbp:.4f}")
                
                # Update Best Model using worst group loss
                best_worst_g_val_loss_sbp, best_worst_g_val_loss_dbp, updated = update_best_model(worst_group_val_loss, val_loss_sbp, val_loss_dbp, best_worst_g_val_loss_sbp, best_worst_g_val_loss_dbp, regressor, optimizer, model_path, mode="WORST")
                if updated:
                    print(f"best WORST val loss updated: sbp {best_worst_g_val_loss_sbp:.4f} dbp {best_worst_g_val_loss_dbp:.4f}")
            pbar.set_description(f'tr_loss: {loss:.4f} | tr_sbp_mae: {overall_mae_sbp:.4f} | tr_dbp_mae: {overall_mae_dbp:.4f} | val_sbp_mae: {val_loss_sbp:.4f} | val_dbp_mae: {val_loss_dbp:.4f} | t: {t_mean:.1f}')
            pbar.update(1)
    # Save the final model
    save_checkpoint(model_path, regressor, optimizer, "last_resnet_gal")
    if not args.ignore_wandb:
        wandb.run.summary["best_train_loss"] = loss
        wandb.run.summary["best_val_train_loss"] = best_val_train_loss
        wandb.run.summary["best_val_loss_sbp"] = best_val_loss_sbp
        wandb.run.summary["best_val_loss_dbp"] = best_val_loss_dbp
        wandb.run.summary["best_val_loss_tot"] = best_val_loss_tot
        wandb.run.summary["best_train_loss_sbp"] = overall_mae_sbp
        wandb.run.summary["best_train_loss_dbp"] = overall_mae_dbp
    print(f'best_val_loss_sbp: {best_val_loss_sbp:.4f} | best_val_loss_dbp: {best_val_loss_dbp:.4f} | tr_mae_sbp: {overall_mae_sbp:.4f} | tr_mae_dbp: {overall_mae_dbp:.4f}')
            
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
    parser.add_argument("--train_fold", type=int, default=0)

    ## Model ---------------------------------------------------
    parser.add_argument("--load_checkpoint", action='store_true',
        help = "Resume model training (Default : False)")
    parser.add_argument("--final_layers", type=int, default=3)
    parser.add_argument("--n_block", type=int, default=8)
    
    ## Training ------------------------------------------------
    parser.add_argument("--diffusion_time_steps", type=int, default=2000)
    parser.add_argument("--train_epochs", type=int, default=2000)
    parser.add_argument("--init_lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--init_bias", type=float, default=0.2)
    parser.add_argument("--final_bias", type=float, default=1)
    parser.add_argument("--optim", type=str, default='adam')
    parser.add_argument("--loss", type=str, default='ERM',  choices=["ERM", "group_average_loss"])
    parser.add_argument("--t_scheduling", type=str, default="uniform",  choices=["loss-second-moment", "uniform", "train-step"])
    parser.add_argument("--T_max", type=int, default=2000)  
    parser.add_argument("--eta_min", type=float, default=0)  

    ## Sampling ------------------------------------------------
    parser.add_argument("--target_group", type=int, default=1, choices=[0,1,2,3,4], 
                        help="0(hyp0) 1(normal) 2(perhyper) 3(hyper2) 4(crisis) (Default : 1 (normal))")
    args = parser.parse_args()

    main(args)

