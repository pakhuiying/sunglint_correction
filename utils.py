import numpy as np
import PIL.Image
import cv2
import glob
from os import listdir
from os.path import join
from math import ceil
import matplotlib.pyplot as plt
import random
import shutil

def pad_images(img):
    """
    img (1024 x m)
    """
    nrow,ncol = img.shape[0], img.shape[1]
    if ncol > 512:
        raise ValueError("Img col is > 512")
    if img.ndim == 3:
        pad_img = np.zeros((512,512,3))
        pad_img[:,:ncol,:] = img
        pad_img = pad_img.astype(np.uint8)
    else:
        pad_img - np.zeros((512,512))
        pad_img[:,:ncol] = img
        pad_img = pad_img.astype(np.uint8)
    
    return pad_img

def cut_into_512(img):
    """
    img (1024 x m)
    """
    nrow,ncol = img.shape[0], img.shape[1]
    if ncol > 512:
        cut_images = []
        for i in range(0,nrow,512):
            for j in range(0,ncol,512):
                if img.ndim == 3:
                    if j+512 <= ncol:
                        cut_images.append(img[i:i+512,j:j+512,:])
                    else:
                        padded_img = pad_images(img[i:i+512,j:ncol,:])
                        cut_images.append(padded_img)
                else:
                    if j+512 <= ncol:
                        cut_images.append(img[i:i+512,j:j+512])
                    else:
                        padded_img = pad_images(img[i:i+512,j:ncol])
                        cut_images.append(padded_img)
        
    elif ncol == 512:
        cut_images = [img]
        
    else:
        cut_images = [pad_images(img)]

    return cut_images

def save_imgs(cut_img_list,directory):
    for i, img in enumerate(cut_img_list):
        img_name = join(directory, "output{}.png".format(i))
        img = PIL.Image.fromarray(img)
        img.save(img_name)
    
    return

def preview_cut_img(cut_img_list):
    ncol = 4
    nrow = int(ceil(len(cut_img_list)/ncol))
    fig,axes = plt.subplots(nrow,ncol,figsize=(15,15))
    for img, ax in zip(cut_img_list,axes.flatten()):
        if img.ndim == 3:
            ax.imshow(img)
        else:
            ax.imshow(img,cmap='grey',vmin=0,vmax=255)
    
    return