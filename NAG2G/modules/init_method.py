from torch.nn.init import xavier_uniform_


def init_xavier_params(module):
    for p in module.parameters():
        if p.dim() > 1:
            xavier_uniform_(p)
    for p in module.parameters():
        if p.dim() > 1:
            xavier_uniform_(p)
