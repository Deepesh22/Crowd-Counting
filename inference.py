from matplotlib import pyplot as plt
import torch
import torchvision.transforms as standard_transforms
import numpy as np
from torch.autograd import Variable

from model.CC import CrowdCounter
from config import cfg
from PIL import Image, ImageOps

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
        raise Exception

    net.eval()

    img = Image.open(imgname)

    print("CPU Threads : ", torch.get_num_threads())

    if img.mode == 'L':
            img = img.convert('RGB')

    print("IMAGE SIZE is :", img.size)

    wd_1, ht_1 = img.size
    if wd_1 < cfg.DATA.INPUT_SIZE[1]:
            dif = cfg.DATA.INPUT_SIZE[1] - wd_1
            img = ImageOps.expand(img, border=(0,0,dif,0), fill=0)
            
    if ht_1 < cfg.DATA.INPUT_SIZE[0]:
        dif = cfg.DATA.INPUT_SIZE[0] - ht_1
        img = ImageOps.expand(img, border=(0,0,0,dif), fill=0)

    print("Modified IMAGE SIZE is :", img.size)

    img = img_transform(img)
    with torch.set_grad_enabled(False):
        img = Variable(img[None,:,:,:])
        pred_map = net.test_forward(img)
        
    predicted_density_map = pred_map.data.numpy()[0,0,:,:]
    prediction = np.sum(predicted_density_map)/100.0
    print("Approximate people :-", int(prediction))

    plt.imshow(predicted_density_map, 'jet')

    plt.savefig(f'{imgname}:-'+str(int(prediction))+"::"+model+'.png')

if __name__ == '__main__':
    model = "CSRNet"
    imgpath ="test_images/images/test.jpg"
    infer(imgpath, model)