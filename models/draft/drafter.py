from torch import nn

class AbstractDrafter(nn.Module):
    def score(self, **kwargs):
        raise NotImplementedError