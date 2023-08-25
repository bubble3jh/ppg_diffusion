import time
import wandb
import argparse
import pickle
import torch
import os 
import pandas as pd
from denoising_diffusion_pytorch.model import Classifier, Regressor, Unet1DEncoder
from denoising_diffusion_pytorch.cond_fn import classifier_cond_fn, regressor_cond_fn
from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d_guided import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
from utils import visualize, sample_sbp_dbp, get_data
import paths

def main(args):
    if args.device == 'cuda' and torch.cuda.is_available():
        device_name = 'cuda:0'
    else:
        device_name = 'cpu'

    device = torch.device(device_name)
    print(f"Device : {device_name}")

    train_set_root = paths.TRAINSET_ROOT
    train_setting = f'fold_{args.train_fold}/seed_{args.seed}_sampling_method_{args.sampling_method}_num_samples_{args.num_samples}-'\
                      f'diffusion_time_steps_{args.diffusion_time_steps}-train_num_steps_{args.train_num_steps}'
    if not args.ignore_wandb:
        wandb.init(project='check_train_time',
                   entity='ppg-diffusion')
        wandb_run_name = train_setting
        wandb.run.name = wandb_run_name

    result_path = os.path.join(paths.WEIGHT_ROOT, train_setting)
    sampling_root = paths.SAMPLING_ROOT
    if not args.disable_guidance:
        train_setting = train_setting + "_guided"
    sampling_name = train_setting + f'_sampling_batch_size_{args.sampling_batch_size}'
    sampling_dir = os.path.join(sampling_root, sampling_name)

    #------------------------------------ Load Data --------------------------------------

    if args.benchmark == "bcg":
        print("data sampling started")
    else:
        print(f"data sampling started, sampling method: {args.sampling_method}, num_samples for each patient: {args.num_samples}")
    data_sampling_start = time.time()
    # ppg, label = get_data(sampling_method=args.sampling_method,
    #                              num_samples=args.num_samples,
    #                              data_root=paths.DATA_ROOT,
    #                              benchmark=args.benchmark)
    data = get_data(sampling_method='first_k',
                                    num_samples=5,
                                    data_root=paths.DATA_ROOT,
                                    benchmark='bcg',
                                    train_fold=0)
    data_sampling_time = time.time() - data_sampling_start
    if not args.ignore_wandb:
        wandb.log({'n_sample': args.num_samples})
        wandb.log({'data_sampling_time': data_sampling_time})
    print(f"data sampling finished, collapsed time: {data_sampling_time:.5f}")
    os.makedirs(train_set_root, exist_ok=True)
    # TODO : ppg 저장 파일/label도 같이/tr, val 따로?
    # with open(os.path.join(train_set_root, train_set_name), 'wb') as f:
    #     pickle.dump(ppg, f)
    
    tr_dataset = dataset = Dataset1D(data['train']['ppg'], label=data['train']['spdp'], groups=data['train']['group_label'] ,normalize=True)
    val_dataset = Dataset1D(data['valid']['ppg'], label=data['valid']['spdp'], groups=data['valid']['group_label'] ,normalize=True)

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
        objective = 'pred_v',
        normalized = args.min_max,             
        denorm_func = dataset.undo_normalization if args.min_max else None
    )
    # trainer = Trainer1D(
    #     diffusion,
    #     dataset = dataset,
    #     train_batch_size = 64,
    # )
    # classifier = Classifier(image_size=seq_length, num_classes=2, t_dim=1)
    regressor = Unet1DEncoder(
        dim = args.seq_length,
        dim_mults = (1, 2, 4, 8),
        channels = 1
    ).to(device)
    if args.train_fold == 0:
        best_eta=0.0; best_lr=0.001; args.regressor_epoch=2000; args.diffusion_time_steps=2000
    elif args.train_fold == 1:
        best_eta=0.001; best_lr=0.001; args.regressor_epoch=2000; args.diffusion_time_steps=2000
    elif args.train_fold == 2:
        best_eta=0.01; best_lr=0.0001; args.regressor_epoch=2000; args.diffusion_time_steps=2000
    elif args.train_fold == 3:
        best_eta=0.0; best_lr=0.0001; args.regressor_epoch=2000; args.diffusion_time_steps=2000
    elif args.train_fold == 4:
        best_eta=0.0; best_lr=0.0001; args.regressor_epoch=2000; args.diffusion_time_steps=2000

    model_path = f"/mlainas/ETRI_2023/reg_model/fold_{args.train_fold}/epoch_{args.regressor_epoch}_diffuse_{args.diffusion_time_steps}_eta_{best_eta}_lr_{best_lr}.pt" # best model로 변경
    if not args.disable_guidance:
        model_state_dict = torch.load(model_path)['model_state_dict']
        regressor.load_state_dict(model_state_dict)
    regressor.eval()

    #------------------------------------- Training --------------------------------------
    
    trainer = Trainer1D(
        diffusion,
        dataset = dataset,
        train_batch_size = args.train_batch_size,
        train_lr = args.train_lr,
        train_num_steps = args.train_num_steps * len(dataset),          # total training steps
        gradient_accumulate_every = 2,                                  # gradient accumulation steps
        ema_decay = 0.995,                                              # exponential moving average decay
        amp = True,                                                     # turn on mixed precision
        results_folder = result_path
    )
    
    if not args.sample_only:
        trainer.train()
        trainer.save('last')
    else:
        trainer.load('last')

    #------------------------------------- Sampling --------------------------------------

    if not args.disable_guidance:
        print("sampling with guidance")
        y = sample_sbp_dbp(args.target_group, args.sample_batch_size)
        y = dataset._min_max_normalize(y, dataset.label_max, dataset.label_min).to(device)
        sampled_seq = diffusion.sample(
        batch_size = args.sample_batch_size,
        return_all_timesteps = False,
        cond_fn = regressor_cond_fn,
        guidance_kwargs={
            "regressor":regressor,
            # "y":torch.fill(torch.zeros(args.sample_batch_size, 2), args.target_label).long().to(device), 
            "y":y,
            "g":torch.fill(torch.zeros(args.sample_batch_size,), args.target_group).long().to(device),
            "regressor_scale":args.regressor_scale,
        }
    )
    else:
        sampled_seq = diffusion.sample(batch_size = 16)

    os.makedirs(sampling_dir, exist_ok=True)
    with open(f'{sampling_dir}/sample_{args.target_group}.pkl', 'wb') as f:
        pickle.dump(sampled_seq, f)
    print(f"Data sampled at {sampling_dir}")
    #------------------------------------- Visualize --------------------------------------

    if args.visualize:
        visualize(sampling_dir, args.target_group, min_value=-0.1, max_value=0.5)

if __name__ == '__main__':

    ## COMMON --------------------------------------------------
    parser = argparse.ArgumentParser(description="generate ppg with regressor guidance")
    parser.add_argument("--seed", type=int, default=1000, help="random seed (default: 1000)")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--ignore_wandb", action='store_true',
        help = "Stop using wandb (Default : False)")
    parser.add_argument("--visualize", action='store_true',
        help = "Visualize results (Default : False)")

    ## DATA ----------------------------------------------------
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--seq_length", type=int, default=625)
    parser.add_argument("--sampling_method", type=str, default='first_k')
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--sample_batch_size", type=int, default=32)
    parser.add_argument("--min_max", action='store_false',
        help = "Min-Max normalize data (Default : True)")
    parser.add_argument("--benchmark", type=str, default='bcg')
    parser.add_argument("--train_fold", type=int, default=0)

    ## Model ---------------------------------------------------
    parser.add_argument("--disable_guidance", action='store_true',
        help = "Stop using guidance (Default : False)")

    ## Training ------------------------------------------------
    parser.add_argument("--diffusion_time_steps", type=int, default=2000)
    parser.add_argument("--train_num_steps", type=int, default=16)
    parser.add_argument("--train_lr", type=float, default=8e-5)
    parser.add_argument("--optim", type=str, default='adam')
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--eval_only", action='store_true',
        help = "Stop using wandb (Default : False)")

    ## Sampling ------------------------------------------------
    parser.add_argument("--sample_only", action='store_true',
        help = "Stop Training (Default : False)")
    parser.add_argument("--sampling_batch_size", type=int, default=16)
    # parser.add_argument("--target_label", type=float, default=1) # deprecated
    parser.add_argument("--target_group", type=int, default=1, choices=[0,1,2,3,4], 
                        help="0(hyp0) 1(normal) 2(perhyper) 3(hyper2) 4(crisis) (Default : 1 (normal))")
    parser.add_argument("--t_scheduling", type=str, default="uniform",  choices=["loss-second-moment", "uniform", "train-step"])
    parser.add_argument("--regressor_scale", type=float, default=1.0)
    parser.add_argument("--regressor_epoch", type=int, default=2000)
    args = parser.parse_args()

    main(args)