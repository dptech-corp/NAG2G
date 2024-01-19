def freeze_network(network):
    for p in network.parameters():
        p.requires_grad = False
