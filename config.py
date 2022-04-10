import torch
import os
from tensorboardX import SummaryWriter

# BATCH_SIZE = 8 # V100 16gb
BATCH_SIZE = 16 # A100 80gb

class Config():
    '''
    Config class
    '''
    def __init__(self):
        self.dataset_root = '../sealwatch_data'
        self.device       = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.lr           = BATCH_SIZE*1e-5
        self.batch_size   = BATCH_SIZE
        self.epochs       = 2000
        self.checkpoints  = './checkpoints'
        self.writer       = SummaryWriter()

        self.__mkdir(self.checkpoints)

    def __mkdir(self, path):
        '''
        create directory while not exist
        '''
        if not os.path.exists(path):
            os.makedirs(path)
            print('create dir: ',path)