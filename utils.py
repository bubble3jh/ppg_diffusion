import os
import torch
import pickle
import numpy as np

from matplotlib import pyplot as plt

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
    if target_group == "hypo":
        sbp = np.random.uniform(80, 90, size=size)
        dbp = np.random.uniform(40, 60, size=size)
    elif target_group == "normal":
        sbp = np.random.uniform(90, 120, size=size)
        dbp = np.random.uniform(60, 80, size=size)
    elif target_group == "prehyper":
        sbp = np.random.uniform(120, 140, size=size)
        dbp = np.random.uniform(80, 90, size=size)
    elif target_group == "hyper2":
        sbp = np.random.uniform(140, 180, size=size)
        dbp = np.random.uniform(90, 120, size=size)
    elif target_group == "crisis":
        sbp = np.random.uniform(180, 200, size=size)
        dbp = np.random.uniform(120, 130, size=size)
    else:
        raise ValueError("Invalid target group")
    print(f"Target ({target_group}) : [sbp {sbp.mean().item():.2f}, dbp {dbp.mean().item():.2f}]")
    if mode == "same":
        return torch.tensor([[sbp, dbp]] * batch_size)
    return torch.tensor(np.array([sbp, dbp]).T)
