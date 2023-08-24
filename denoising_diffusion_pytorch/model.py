import torch
from torch import nn, einsum
import torch.nn.functional as F
from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d_guided import default, LinearAttention, Residual, PreNorm, Downsample, Upsample, partial, Attention, RandomOrLearnedSinusoidalPosEmb, SinusoidalPosEmb, ResnetBlock
from denoising_diffusion_pytorch.nn import normalization, AttentionPool2d,zero_module, conv_nd
class Classifier(nn.Module):
    def __init__(self, image_size, num_classes, t_dim=1) -> None:
        super().__init__()
        self.linear_t = nn.Linear(t_dim, num_classes)
        self.linear_img = nn.Linear(image_size * image_size * 3, num_classes)
    def forward(self, x, t):
        """
        Args:
            x (_type_): [B, 3, N, N]
            t (_type_): [B,]

        Returns:
                logits [B, num_classes]
        """
        B = x.shape[0]
        t = t.view(B, 1)
        logits = self.linear_t(t.float()) + self.linear_img(x.view(x.shape[0], -1))
        return logits
    
class Regressor(nn.Module):
    def __init__(self, seq_len, num_classes=2, t_dim=1) -> None:
        super().__init__()
        self.linear_t = nn.Linear(t_dim, num_classes) 
        self.linear_seq = nn.Linear(seq_len, num_classes)  

    def forward(self, x, t):
        """
        Args:
            x (_type_): [B, 1, Seq_len]
            t (_type_): [B,]

        Returns:
                outputs [B, num_classes]
        """
        B = x.shape[0]
        t = t.view(B, 1)
        outputs = self.linear_t(t.float()) + self.linear_seq(x.view(x.shape[0], -1))
        return outputs
    
class MLPRegressor(nn.Module):
    def __init__(self, ch, dims, out_channels=2, num_head_channels=-1, pool="adaptive"):
        super().__init__()
        if pool == "adaptive":
            self.out = nn.Sequential(
                nn.SiLU(),
                nn.AdaptiveAvgPool1d(1),
                zero_module(conv_nd(1, ch, out_channels, 1)),
                nn.Flatten(),
            )
        elif pool == "attention":
            assert num_head_channels != -1
            self.out = nn.Sequential(
                nn.SiLU(),
                AttentionPool2d(
                    (4), ch, num_head_channels, out_channels
                    # (image_size // ds), ch, num_head_channels, out_channels
                ),
            )
        elif pool == "spatial":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                nn.ReLU(),
                nn.Linear(2048, self.out_channels),
            )
        elif pool == "spatial_v2":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                normalization(2048),
                nn.SiLU(),
                nn.Linear(2048, self.out_channels),
            )
        else:
            raise NotImplementedError(f"Unexpected {pool} pooling")
        
    def forward(self, x):
        return self.out(x)

class Unet1DEncoder(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        resnet_block_groups = 5,#8
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        pool = "adaptive"
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)
        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim - 1, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        num_groups=6
        self.group_emb = nn.Embedding(num_groups, time_dim)
        self.out_mlp = MLPRegressor(dims[-1], dims[-1], 2, pool=pool)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim * 2),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim * 2),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim * 2) # time_emb_dim = time_dim
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim * 2)

    def forward(self, x, time, group, x_self_cond = None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)
        x = self.init_conv(x)
        t = self.time_mlp(time)
        g = self.group_emb(group)
        tg = torch.cat((t, g), dim=-1)
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, tg)
            x = block2(x, tg)
            x = attn(x)
            x = downsample(x)

        x = self.mid_block1(x, tg)
        x = self.mid_attn(x)
        emb = self.mid_block2(x, tg)

        return self.out_mlp(emb), emb
