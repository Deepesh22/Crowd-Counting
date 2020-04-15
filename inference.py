from matplotlib import pyplot as plt
import torch
import torchvision.transforms as standard_transforms
import numpy as np
from torch.autograd import Variable

from model.CC import CrowdCounter, models
from config import cfg
from PIL import Image, ImageOps

import os
import time

plt.set_cmap("jet")

mean_std = cfg.DATA.MEAN_STD
img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])  


class Infer:

    def __init__(self, gpu = False):
        print("LOADING MODELS INSTANCE")
        
        self.models = dict()
        self.use_gpu = gpu

        if gpu:
            device = torch.device("cuda")
            location = "cuda:0"
            print("RUNNING ON GPU INSTANCE")
        else:
            device = torch.device("cpu")
            location = "cpu"
            print("RUNNING ON CPU INSTANCE")

        for model in models.keys():
            try:
                net = CrowdCounter(model_name = model)
                net.load_state_dict(torch.load(net.model_weight_path, map_location = location))
                print(f"LOADED {model} WEIGHTS")
                self.models[model] = net
            except Exception as e:
                print(f"Couldn't load weights of {model}")
                print(e)

    def getAvailableModels(self):
        return list(self.models.keys())

    def infer(self, imgname, model):

        net = self.models[model]

        net.eval()

        try:
            img = Image.open(imgname)
        except Exception as e:
            print("Couldn't load Image")
            print(e)
            raise Exception

        if img.mode == 'L':
                img = img.convert('RGB')

        print("IMAGE SIZE is :", img.size)

        img = img_transform(img)
        with torch.no_grad():
            if self.use_gpu:
                img = Variable(img[None,:,:,:]).cuda()
            else:
                img = Variable(img[None,:,:,:])
            print("STARTING INFERENCE")
            start = time.time()
            pred_map = net.test_forward(img)
            if self.use_gpu:
                pred_map = pred_map.cpu()
            end = time.time()
            print("Inference took ", end-start, "seconds")
            
        predicted_density_map = pred_map.data.numpy()[0,0,:,:]
        prediction = np.sum(predicted_density_map)/100.0
        print("Approximate people :-", int(prediction))
        path = f'static/images/{imgname.split("/")[-1].split(".")[0]}__{model}.png'
        plt.imsave(path, predicted_density_map)
        filename = path.split("/")[-1]

        return prediction, filename

if __name__ == '__main__':
    model = "MCNN"
    imgpath ="test_images/word-image.jpeg"
    Infer().infer(imgpath, model)