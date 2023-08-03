import time
import torch
from denoising_diffusion_pytorch.model import Regressor
from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d_guided import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
from vis import visualize
import data
import paths
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

def cycle(dl):
    while True:
        for data in dl:
            yield data

device = torch.device("cuda")
batch_size = 32

train_set_root = paths.TRAINSET_ROOT
train_set_name = f'seed_1000-sampling_method_first_k-num_samples_5'
print(f"data sampling started, sampling method: first_k, num_samples for each patient: 5")
data_sampling_start = time.time()
training_seq, len_seq = data.get_data(sampling_method='first_k',
                                num_samples=5,
                                data_root=paths.DATA_ROOT)
data_sampling_time = time.time() - data_sampling_start

print(f"data sampling finished, collapsed time: {data_sampling_time}")

model_path = "./clf_model/clf.pt"

dataset = Dataset1D(training_seq)
dl = DataLoader(dataset, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = 0)
dl = cycle(dl)

regressor = Regressor(1000,2,1).to(device)

model = Unet1D(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        channels = 1
    ).to(device)

diffusion = GaussianDiffusion1D(
    model = model,
    seq_length = 1000,
    timesteps = 2000,
    objective = 'pred_v'
).to(device)

optimizer = optim.Adam(regressor.parameters(), lr=0.001)

timesteps, = diffusion.betas.shape
num_timesteps = int(timesteps)

loss_values = []
for _ in tqdm(range(100), desc="Training", unit="epoch"):
    batch = next(dl).to(device)
    t = torch.randint(0, num_timesteps, (batch.size(0),), device=device).long()
    label = torch.full((batch.size(0), 2), 1).to(device).float()

    batch = diffusion.q_sample(batch, t)

    optimizer.zero_grad()
    out = regressor(batch, t)

    loss = F.mse_loss(out, label)
    loss.backward()
    optimizer.step()

    loss_values.append(loss.item())

# Print the loss values
for i, loss in enumerate(loss_values):
    if i % 10 == 0 :
        print(f"Epoch {i+1}: Loss = {loss}")

torch.save(regressor.state_dict(), model_path)

        