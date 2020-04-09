import torch.nn as nn
from torch.autograd import Variable
import torch

from .layers import Gaussianlayer
from .SCAR import SCAR
from .CSRNet import CSRNet
from .MCNN import MCNN
import os

models = {"SCAR": [SCAR, 'pre/SCAR.pth'],
            "CSRNet": [CSRNet, "pre/CSRNet2.pth"],
            "MCNN":[MCNN, "pre/MCNN.pth"]
            }


class CrowdCounter(nn.Module):
    def __init__(self, model_name):
        super(CrowdCounter, self).__init__()  
        ccnet, model_weight_path = models[model_name]
        self.CCN = ccnet()
        if model_name != "CSRNet":
            self.gs = Gaussianlayer()
        self.model_weight_path = model_weight_path
        print("LOADED MODEL")

    def test_forward(self, img):                               
        density_map = self.CCN(img)
        return density_map