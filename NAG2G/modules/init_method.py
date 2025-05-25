from torch.nn.init import xavier_uniform_


def init_xavier_params(module):
    # tmp test
    # if model_opt.param_init != 0.0:
    # for p in module.parameters():
    #     p.data.uniform_(-model_opt.param_init, model_opt.param_init)
    # if model_opt.param_init_glorot:
    for p in module.parameters():
        if p.dim() > 1:
            xavier_uniform_(p)
    for p in module.parameters():
        if p.dim() > 1:
            xavier_uniform_(p)
