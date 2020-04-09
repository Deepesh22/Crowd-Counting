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
    # os.environ['KMP_DUPLICATE_LIB_OK']='True'
    # os.environ["OMP_NUM_THREADS"] = "4"
    # os.environ["MKL_NUM_THREADS"] = "4"

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

    # print("CPU Threads : ", torch.get_num_threads())
    # torch.set_num_threads(4)
    # print("CPU Threads are now : ", torch.get_num_threads())

    if img.mode == 'L':
            img = img.convert('RGB')

    print("IMAGE SIZE is :", img.size)

    # wd_1, ht_1 = img.size
    # if wd_1 < cfg.DATA.INPUT_SIZE[1]:
    #         dif = cfg.DATA.INPUT_SIZE[1] - wd_1
    #         img = ImageOps.expand(img, border=(0,0,dif,0), fill=0)
            
    # if ht_1 < cfg.DATA.INPUT_SIZE[0]:
    #     dif = cfg.DATA.INPUT_SIZE[0] - ht_1
    #     img = ImageOps.expand(img, border=(0,0,0,dif), fill=0)

    # print("Modified IMAGE SIZE is :", img.size)

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
    model = "CSRNet"
    imgpath ="test_images/test.jpg"
    infer(imgpath, model)

 
#export NUM_CORES=4
#export MKL_NUM_THREADS=$NUM_CORES OMP_NUM_THREADS=$NUM_CORES
