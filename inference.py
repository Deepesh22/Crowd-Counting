from matplotlib import pyplot as plt
import torch
import torchvision.transforms as standard_transforms
import numpy as np
from torch.autograd import Variable

from model.CC import CrowdCounter
from config import cfg
from PIL import Image, ImageOps

import os
import time

mean_std = cfg.DATA.MEAN_STD
img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])  


def infer(imgname, model):

    try:
        net = CrowdCounter(model_name = model)
        net.load_state_dict(torch.load(net.model_weight_path, map_location="cpu"))
        print("LOADED WEIGHTS")
    except Exception as e:
        print("Couldn't load weights")
        print(e)
        raise Exception

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
        img = Variable(img[None,:,:,:])
        print("STARTING INFERENCE")
        start = time.time()
        pred_map = net.test_forward(img)
        end = time.time()
        print("Inference took ", end-start, "seconds")
        
    predicted_density_map = pred_map.data.numpy()[0,0,:,:]
    prediction = np.sum(predicted_density_map)/100.0
    print("Approximate people :-", int(prediction))

    plt.imshow(predicted_density_map, 'jet')

    plt.savefig(f'{imgname}:-'+str(int(prediction))+"::"+model+'.png')

if __name__ == '__main__':
    model = "SFCN"
    imgpath ="test_images/test.jpg"
    infer(imgpath, model)