import os
import torch
import pickle

from matplotlib import pyplot as plt

def visualize(root):
    plot_root = os.path.join(root, 'plot_results')
    os.makedirs(plot_root, exist_ok=True)
    sample_path = os.path.join(root, 'sample.pkl')
    try:
        with open(sample_path, 'rb') as f:
            sample = pickle.load(f)
            samples = sample.squeeze(1)
            samples = samples.detach().cpu().numpy()
            for i, sample in enumerate(samples):
                plt_path = f'{plot_root}/{i}.png'
                plt.plot(sample)
                plt.savefig(plt_path)
                plt.cla()
    except Exception as e:
        print(e)