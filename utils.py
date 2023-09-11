import os
import torch
import pickle
import numpy as np
import tabulate
from collections import defaultdict, Counter
import scipy.io
from torch import nn
from matplotlib import pyplot as plt
import wandb
import random

def visualize(root, target_label, min_value, max_value):
    plot_root = os.path.join(root, f'plot_results_target_{target_label}')
    os.makedirs(plot_root, exist_ok=True)
    sample_path = os.path.join(root, f'sample_{target_label}.pkl')
    try:
        with open(sample_path, 'rb') as f:
            sample = pickle.load(f)
            samples = sample.squeeze(1)
            samples = samples.detach().cpu().numpy()
            for i, sample in enumerate(samples):
                plt_path = f'{plot_root}/{i}.png'
                plt.plot(sample)
                plt.ylim(min_value, max_value)  # Set the y-axis limits
                plt.savefig(plt_path)
                plt.cla()
    except Exception as e:
        print(e)


def sample_sbp_dbp(target_group, batch_size, mode = "sample_each"):
    size = batch_size if mode == "sample_each" else 1
    if target_group == 0:#"hypo":
        sbp = np.random.uniform(80, 90, size=size)
        dbp = np.random.uniform(40, 60, size=size)
    elif target_group == 1:#"normal":
        sbp = np.random.uniform(90, 120, size=size)
        dbp = np.random.uniform(60, 80, size=size)
    elif target_group == 2:#"prehyper":
        sbp = np.random.uniform(120, 140, size=size)
        dbp = np.random.uniform(80, 90, size=size)
    elif target_group == 3:#"hyper2":
        sbp = np.random.uniform(140, 180, size=size)
        dbp = np.random.uniform(90, 120, size=size)
    elif target_group == 4:#"crisis":
        sbp = np.random.uniform(180, 200, size=size)
        dbp = np.random.uniform(120, 130, size=size)
    else:
        raise ValueError("Invalid target group")
    print(f"Target ({target_group}) Batch ({sbp.shape[0]}) : [sbp {sbp.mean().item():.2f}, dbp {dbp.mean().item():.2f}]")
    if mode == "same":
        return torch.tensor([[sbp, dbp]] * batch_size)
    return torch.tensor(np.array([sbp, dbp]).T)

def assign_group_label_old(sbp, dbp):
    if 80 <= sbp < 90 and 40 <= dbp < 60:
        return 1 #"hypo"
    elif 90 <= sbp < 120 and 60 <= dbp < 80:
        return 2 #"normal"
    elif 120 <= sbp < 140 and 80 <= dbp < 90:
        return 3 #"prehyper"
    elif 140 <= sbp < 180 and 90 <= dbp < 120:
        return 4 #"hyper2"
    elif 180 <= sbp < 200 and 120 <= dbp < 130:
        return 5 #"crisis"
    else:
        return 0 #"undefined"
    
def assign_group_label(sbp, dbp, mode, r=10):
    if (sbp <= 80) or (sbp >= 200) or (dbp >= 130) or (dbp <= 40):
        return None # eliminate from dataset
    if mode=="etri":
        if (180 <= sbp) or (120 <= dbp):
            return 4 #"crisis"
        elif (140 <= sbp) or (90 <= dbp):
            return 3 #"hyper2"
        elif (120 <= sbp) or (80 <= dbp):
            return 2 #"prehyper"
        elif (90 <= sbp) or (60 <= dbp):
            return 1 #"normal"
        elif (80 <= sbp) or (40 <= dbp):
            return 0 #"hypo"
    elif mode=="same":
        sbp_label = (sbp - 80) // 10
        dbp_label = (dbp - 40) // 10
        return (sbp_label, dbp_label)
    
def print_group_counts(group_labels):
    group_names = {0: "undefined", 1: "hypo", 2: "normal", 3: "prehyper", 4: "hyper2", 5: "crisis"}
    group_counts = defaultdict(int, Counter(group_labels))
    print("Group       | Count")
    print("------------|-------")
    for group_id, group_name in group_names.items():
        count = group_counts[group_id]
        print(f"{group_name.ljust(12)}| {count}")

def get_data(sampling_method='first_k',
             num_samples=5,
             data_root='./',
             benchmark='bcg',
             train_fold = 0,
             group_mode="same",
             d500_idx=0):
    if benchmark=='bcg':
        if train_fold == 0:
            fold_nums = [0,1,4]
            val_fold = 3
        elif train_fold == 1:
            fold_nums = [1,2,4]
            val_fold = 0
        elif train_fold == 2:
            fold_nums = [2,3,4]
            val_fold = 1
        elif train_fold == 3:
            fold_nums = [0,2,3]
            val_fold = 4
        elif train_fold == 4:
            fold_nums = [0,1,3]
            val_fold = 2
        data = {"train": {},
                "valid": {} }
        # train
        ppgs_tensor, spdps_tensor, group_labels = fold_data(fold_nums, group_mode)
        data['train']['ppg'] = ppgs_tensor
        data['train']['spdp'] = spdps_tensor
        data['train']['group_label'] = group_labels
        # valid
        ppgs_tensor, spdps_tensor, group_labels = fold_data([val_fold], group_mode)
        data['valid']['ppg'] = ppgs_tensor
        data['valid']['spdp'] = spdps_tensor
        data['valid']['group_label'] = group_labels
        return data

    #--------------------------------------------------------------------------------------
    
    elif benchmark == 'etri':
        assert sampling_method in ['first_k']

        col_names = ['time', 'PPG', 'abp']
        sample_list = []

        if sampling_method == 'first_k':
            for patient_id in sorted(os.listdir(data_root)):
                patient_dir = os.path.join(data_root, patient_id)

                for signal_name in sorted(os.listdir(patient_dir))[:num_samples]:
                    sample = pd.read_csv(f'{os.path.join(patient_dir, signal_name)}', names=col_names)
                    sample_tensor = torch.tensor(sample['PPG'].values)
                    sample_list.append(sample_tensor)

        len_seq = len(sample_list)
        training_seq = torch.stack(sample_list).unsqueeze(1).half()
        
        return training_seq, len_seq
    
    #--------------------------------------------------------------------------------------
    
    elif benchmark == 'd500':
        return torch.load(f'/data1/bubble3jh/ppg/data/six_ch/d500_sliced_{d500_idx}.pth')
    
def fold_data(fold_nums, group_mode):
    ppgs, spdps, group_labels = [], [], []
    count = 0
    for fold_num in fold_nums:
        ppg, spdp = load_fold_np(fold_num=fold_num)
        ppg = torch.permute(ppg, (1,0,2))
    
        sbp = spdp[0, :, 0].squeeze()
        dbp = spdp[1, :, 0].squeeze()
        # Assign group labels based on sbp and dbp values
        for s, d, p, sp in zip(sbp, dbp, ppg, spdp.squeeze().T):
            group_label = assign_group_label(s, d, group_mode)
            if group_label is not None:
                ppgs.append(p)
                spdps.append(sp)
                group_labels.append(group_label)
            else:
                count = count + 1
    print_group_counts(group_labels)
    print(f"{count} datas eliminated.\n")
    return torch.cat(ppgs, dim=0).unsqueeze(1).half(), torch.stack(spdps, dim=0).float(), torch.tensor(group_labels)

def load_fold_np(fold_num, root='../data/bcg_dataset'):
    if os.path.exists(f'{root}/signal_fold_{fold_num}_ppg.npy'):
        return torch.from_numpy(np.load(f'{root}/signal_fold_{fold_num}_ppg.npy')), torch.from_numpy(np.load(f'{root}/signal_fold_{fold_num}_spdp.npy'))
    else:
        mat = scipy.io.loadmat(f'{root}/signal_fold_{fold_num}.mat')
        data_dict = {key: mat[key] for key in ['SP', 'DP']}
        spdp_numpy_data = np.array([value for value in data_dict.values()])
        np.save(f'{root}/signal_fold_{fold_num}_spdp.npy', spdp_numpy_data)
        data_dict = {key: mat[key] for key in ['signal']}
        ppg_numpy_data = np.array([value for value in data_dict.values()])
        np.save(f'{root}/signal_fold_{fold_num}_ppg.npy', ppg_numpy_data)   
        return torch.from_numpy(ppg_numpy_data), torch.from_numpy(spdp_numpy_data)

def get_sample_batch_size(data, target_group):
    unique_elements, counts = torch.unique(data['train']['group_label'], return_counts=True)

    max_count = torch.max(counts).item()
    
    target_count = counts[unique_elements == target_group].item() if target_group in unique_elements else 0
    print(f"max count : {max_count}, target count : {target_count}")

    return max_count - target_count

class Lambda(nn.Module):
    def __init__(self, func):
        super(Lambda, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)
    

# Batch-wise MAE 계산 함수
def calculate_batch_mae(model_output, ground_truth, dataset):
    model_output = dataset.undo_normalization_label(model_output)
    ground_truth = dataset.undo_normalization_label(ground_truth)
    import pandas as pd
    # Save model_output to CSV
    df_output = pd.DataFrame(model_output.detach().cpu().numpy(), columns=['Output_SBP', 'Output_DBP'])
    df_output.to_csv('/data1/bubble3jh/ppg/denoising-diffusion-pytorch/check_outs/model_output.csv', index=False)
    
    # Save ground_truth to CSV
    df_truth = pd.DataFrame(ground_truth.detach().cpu().numpy(), columns=['True_SBP', 'True_DBP'])
    df_truth.to_csv('/data1/bubble3jh/ppg/denoising-diffusion-pytorch/check_outs/ground_truth.csv', index=False)
    
    loss_batch = torch.abs(model_output - ground_truth)
    mae_sbp = loss_batch[:, 0].mean().item()
    mae_dbp = loss_batch[:, 1].mean().item()
    
    return mae_sbp, mae_dbp

# Global Metrics Logging 함수
def log_global_metrics(args, total_mae_sbp, total_mae_dbp, total_count, phase):
    avg_mae_sbp = total_mae_sbp / total_count
    avg_mae_dbp = total_mae_dbp / total_count
    
    if not args.ignore_wandb:
        wandb.log({
            f"{phase}_avg_mae_sbp": avg_mae_sbp,
            f"{phase}_avg_mae_dbp": avg_mae_dbp
        })
    return avg_mae_sbp, avg_mae_dbp

def set_seed(random_seed=1000):
    '''
    Set Seed for Reproduction
    '''
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)