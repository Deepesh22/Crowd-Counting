import torch.nn as nn
from torch.autograd import Variable
import torch

from .layers import Gaussianlayer
from .SCAR import SCAR
from .CSRNet import CSRNet
from .MCNN import MCNN
from .Res101_SFCN import Res101_SFCN
import os


models = {"SCAR": [SCAR, 'pre/SCAR.pth'],
            "CSRNet2": [CSRNet, "pre/CSRNet2.pth"],
            "CSRNet": [CSRNet, "pre/CSRNet.pth"],
            "MCNN":[MCNN, "pre/MCNN.pth"],
            "SFCN":[Res101_SFCN, "pre/SFCN.pth"]
            }


class CrowdCounter(nn.Module):
    def __init__(self, model_name, use_gpu = False):
        super(CrowdCounter, self).__init__()  
        ccnet, model_weight_path = models[model_name]
        self.CCN = ccnet()
        if model_name not in ["CSRNet2", "SDCNet"] :
            self.gs = Gaussianlayer()
            if use_gpu:
                self.gs = self.gs.cuda()
        if use_gpu:
            self.CCN = self.CCN.cuda()
        self.model_weight_path = model_weight_path
        print(f"LOADED {model_name} MODEL")

    def test_forward(self, img):                               
        density_map = self.CCN(img)
        return density_map