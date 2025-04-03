import torch.nn as nn

class Projector(nn.Module):
    def __init__(self, in_dims, hidden_dims, out_dims, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.linear_01 = nn.Linear(in_dims, hidden_dims)
        self.linear_02 = nn.Linear(hidden_dims, out_dims)
        
    def forward(self, x, skip_first=False):
        
        if not skip_first:
            x = self.linear_01(x)
        
        x = self.linear_02(x)
        
        return x