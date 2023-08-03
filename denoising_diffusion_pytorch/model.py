import torch
from torch import nn, einsum

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
        self.linear_t = nn.Linear(t_dim, num_classes)  # output dimension is now 2 for regression
        self.linear_seq = nn.Linear(seq_len, num_classes)  # input dimension is now seq_len, output is 2

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