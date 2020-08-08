'''
工具包
'''
from __future__ import print_function
import torch

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms

#加载图像




def image_loader(img_path,img_size = 128):
    transfrom= transforms.Compose([
    transforms.Resize((img_size,img_size)),
    transforms.ToTensor()
    ])
    img = Image.open(img_path)
    img = transfrom(img).unsqueeze(0)
    print(img.size())
    return img

def imshow(inp,title = None):
    inp = inp.squeeze(0)
    inp = inp.numpy().transpose((1,2,0))
    inp = np.clip(inp,0,1)
    if title is not None:
        plt.title(title)
    plt.imshow(inp)
    plt.show()
    plt.pause(0.001)

if __name__ == '__main__':
    img_size = 512 if torch.cuda.is_available() else 128    # 如果使用gpu,则用512x512大小，否则使用128x128
    style_img = image_loader("c:/Users/Mr.fei/pytorch-learn/data/images/picasso.jpg")
    content_img = image_loader("c:/Users/Mr.fei/pytorch-learn/data/images/dance.jpg")
    imshow(style_img,'style_img')
    imshow(content_img,'content_img')









