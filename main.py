import time
import wandb
import argparse
import pickle
import torch
import os 
import pandas as pd
from denoising_diffusion_pytorch.model import Classifier, Regressor
from denoising_diffusion_pytorch.cond_fn import classifier_cond_fn, regressor_cond_fn
from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d_guided import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
from vis import visualize
import data
import paths

def main(args):
    if args.device == 'cuda' and torch.cuda.is_available():
        device_name = 'cuda:0'
    else:
        device_name = 'cpu'

    device = torch.device(device_name)
    print(f"Device : {device_name}")

    train_setting = f'seed_{args.seed}_sampling_method_{args.sampling_method}_num_samples_{args.num_samples}-'\
                      f'diffusion_time_steps_{args.diffusion_time_steps}-train_num_steps_{args.train_num_steps}'
    if not args.ignore_wandb:
        wandb.init(project='check_train_time',
                   entity='ppg-diffusion')
        wandb_run_name = train_setting
        wandb.run.name = wandb_run_name

    #------------------------------------ Load Data --------------------------------------
    train_set_root = paths.TRAINSET_ROOT
    train_set_name = f'seed_{args.seed}-sampling_method_{args.sampling_method}-num_samples_{args.num_samples}'
    print(f"data sampling started, sampling method: {args.sampling_method}, num_samples for each patient: {args.num_samples}")
    data_sampling_start = time.time()
    training_seq, len_seq = data.get_data(sampling_method=args.sampling_method,
                                 num_samples=args.num_samples,
                                 data_root=paths.DATA_ROOT)
    data_sampling_time = time.time() - data_sampling_start
    if not args.ignore_wandb:
        wandb.log({'n_sample': args.num_samples})
        wandb.log({'data_sampling_time': data_sampling_time})
    print(f"data sampling finished, collapsed time: {data_sampling_time}")
    os.makedirs(train_set_root, exist_ok=True)
    with open(os.path.join(train_set_root, train_set_name), 'wb') as f:
        pickle.dump(training_seq, f)
    

    dataset = Dataset1D(training_seq)

    #----------------------------------- Create Model ------------------------------------

    model = Unet1D(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        channels = 1
    )

    diffusion = GaussianDiffusion1D(
        model,
        seq_length = args.seq_length,
        timesteps = args.diffusion_time_steps,
        objective = 'pred_v'
    )

    result_path = os.path.join(paths.WEIGHT_ROOT, train_setting)
    # trainer = Trainer1D(
    #     diffusion,
    #     dataset = dataset,
    #     train_batch_size = 64,
    # )
    # classifier = Classifier(image_size=seq_length, num_classes=2, t_dim=1)
    regressor = Regressor(seq_len=args.seq_length, num_classes=2, t_dim=1).to(device)

    #------------------------------------- Training --------------------------------------
    
    trainer = Trainer1D(
        diffusion,
        dataset = dataset,
        train_batch_size = args.train_batch_size,
        train_lr = 8e-5,
        train_num_steps = args.train_num_steps * len(dataset),          # total training steps
        gradient_accumulate_every = 2,                                  # gradient accumulation steps
        ema_decay = 0.995,                                              # exponential moving average decay
        amp = True,                                                     # turn on mixed precision
        results_folder = result_path
    )
    trainer.save('last')
    if not args.sample_only:
        trainer.train()

    # TODO: 가중치 불러오는 코드 짜기


    #------------------------------------- Sampling --------------------------------------


    sampling_root = paths.SAMPLING_ROOT
    if args.reg_guidance:
        train_setting = "guidance_" + train_setting
    sampling_name = train_setting + f'_sampling_batch_size_{args.sampling_batch_size}'
    sampling_dir = os.path.join(sampling_root, sampling_name)

    if args.reg_guidance:
        sampled_seq = diffusion.sample(
        batch_size = args.sample_batch_size,
        return_all_timesteps = False,
        cond_fn = regressor_cond_fn,
        guidance_kwargs={
            "regressor":regressor,
            "y":torch.fill(torch.zeros(args.sample_batch_size, 2), 1).long().to(device),
            "regressor_scale":1,
        }
    )
    else:
        sampled_seq = diffusion.sample(batch_size = 16)

    os.makedirs(sampling_dir, exist_ok=True)
    if args.save_seq:
        with open(f'{sampling_dir}/sample.pkl', 'wb') as f:
            pickle.dump(sampled_seq, f)
    #------------------------------------- Visualize --------------------------------------

    if args.visualize:
        visualize(sampling_dir)

if __name__ == '__main__':

    ## COMMON --------------------------------------------------
    parser = argparse.ArgumentParser(description="gp-regression for the confirmation and dead prediction")
    parser.add_argument("--seed", type=int, default=1000, help="random seed (default: 1000)")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--ignore_wandb", action='store_true',
        help = "Stop using wandb (Default : False)")
    parser.add_argument("--visualize", action='store_true',
        help = "Visualize results (Default : False)")

    ## DATA ----------------------------------------------------
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--seq_length", type=int, default=1000)
    parser.add_argument("--sampling_method", type=str, default='first_k')
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--sample_batch_size", type=int, default=32)

    ## Model ---------------------------------------------------
    parser.add_argument("--reg_guidance", action='store_false',
        help = "Stop using guidance (Default : True)")

    ## Training ------------------------------------------------
    parser.add_argument("--diffusion_time_steps", type=int, default=1000)
    parser.add_argument("--train_num_steps", type=int, default=15)
    parser.add_argument("--init_lr", type=float, default=0.1)
    parser.add_argument("--optim", type=str, default='adam')
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--tolerance", type=int, default=500)
    parser.add_argument("--eval_only", action='store_true',
        help = "Stop using wandb (Default : False)")

    ## Sampling
    parser.add_argument("--save_seq", action='store_true',
        help = "Stop using wandb (Default : False)")
    parser.add_argument("--sample_only", action='store_true',
        help = "Stop using wandb (Default : False)")
    parser.add_argument("--sampling_batch_size", type=int, default=16)

    args = parser.parse_args()

    main(args)