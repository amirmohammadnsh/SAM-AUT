import torch.backends.cudnn as cudnn
import random
import numpy as np

import torch
def set_seed(random_state = 42):
    random.seed(random_state)
    np.random.seed(random_state)
    

    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)

    cudnn.deterministic = True
    cudnn.benchmark = False
 