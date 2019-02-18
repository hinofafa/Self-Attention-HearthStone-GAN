from PIL import Image
import os, sys

path = "/home/hinova/Desktop/cv/Self-Attention-GAN/data/hearthstone-card-images/rel/"
files = os.listdir(path)
for name in files:
    print(name)
    im = Image.open(path+name)
    rgb_im = im.convert('RGB')
    rgb_im.save("/home/hinova/Desktop/cv/Self-Attention-GAN/data/hearthstone-card-images/jpg/"+name.split('.')[0]+'.jpg')
