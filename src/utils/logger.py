import os
from torch.utils.tensorboard import SummaryWriter

def tb_logger(args, root, name):
    path = args['rel_path']
    tag = args['tag']
    log_path = os.path.join(root, path, f'run_{name}_{tag}')
    logger = SummaryWriter(log_path)

    return logger
