import numpy as np
import PIL.Image
import cv2
import glob
from os import listdir
from os.path import join
from math import ceil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import shutil
import cv2 as cv
from os.path import join, basename
from os import listdir
import json
from skimage.color import rgb2gray

import sys
sys.path.insert(0,r'D:\PAKHUIYING\Image_processing\F3_raw_images\utils')
import importlib
import utils
importlib.reload(utils)

def view_selection(fp,plot=True):
    """ 
    fp (str): filepath to sunglint.txt file from select_glint_GUI.py
    plot (bool): to plot the figures or not
    returns spectra_dict: mean of DN pixels for each channel for glint and non-glint image
    """
    with open(fp,'r') as cf:
        bbox = json.load(cf)

    for g in list(bbox):
        ((x1,y1),(x2,y2)) = bbox[g]['bbox']
        if y1 > y2:
            y1, y2 = y2, y1
        if x1 > x2:
            x1, x2 = x2, x1
        bbox[g]['bbox'] = ((x1,y1),(x2,y2))
        # print(((x1,y1),(x2,y2)))
        h = y2 - y1
        w = x2 - x1
        bbox[g]['h'] = h
        bbox[g]['w'] = w
        img = np.array(PIL.Image.open(bbox[g]['fp']))
        bbox[g]['img'] = img
        if g == 'glint':
            bbox[g]['patch'] = patches.Rectangle((x1,img.shape[0] - y1),w,-h,linewidth=1,edgecolor='r',facecolor='none')
        else:
            bbox[g]['patch'] = patches.Rectangle((x1,img.shape[0] - y1),w,-h,linewidth=1,edgecolor='purple',facecolor='none')

    spectra_dict = {i: None for i in list(bbox)}

    if bbox['glint']['fp'] == bbox['non_glint']['fp']:
        img = bbox['glint']['img']
        # create figure plot, image on top with 2 line plots at the bottom
        ## add image with selected patches
        fig = plt.figure(figsize=(12,7),constrained_layout=True)
        spec = fig.add_gridspec(2,3)
        ax0 = fig.add_subplot(spec[0,:])
        ax0.imshow(img)
        
        ax12 = fig.add_subplot(spec[1,2]) # add spectra

        for i,(g,d) in enumerate(bbox.items()):
            
            ax0.add_patch(d['patch'])
            ax0.axis('off')
            ((x1,y1),(x2,y2)) = d['bbox']
            # print(d['bbox'])
            im = d['img']
            nrow = im.shape[0]
            im = im[nrow-y2:nrow-y1,x1:x2,:]
            ax = fig.add_subplot(spec[1,i]) #display cropped im
            ax.imshow(im)
            ax.set_title(g)
            ax.axis('off')
            # add spectra
            spectra = np.mean(im,axis=(0,1))
            if g == 'glint':
                ax12.plot(np.arange(3),spectra,'r')
            else:
                ax12.plot(np.arange(3),spectra,'purple')
            ax12.set_ylabel('DN')
            ax12.set_ylabel('RGB spectra')
            spectra_dict[g] = spectra
        
    
    else:
        fig = plt.figure(figsize=(12,7),constrained_layout=True)
        spec = fig.add_gridspec(3,3)
        ax22 = fig.add_subplot(spec[2,2]) # add spectra
        for i,(g,d) in enumerate(bbox.items()):
            ax = fig.add_subplot(spec[i,:])
            im = d['img']
            ax.imshow(im)
            ax.axis('off')
            ax.add_patch(d['patch'])
            nrow = im.shape[0]
            im = im[nrow-y2:nrow-y1,x1:x2,:]
            ax = fig.add_subplot(spec[2,i]) #display cropped im
            ax.imshow(im)
            ax.axis('off')
            ax.set_title(g)
            # add spectra
            spectra = np.mean(im,axis=(0,1))
            spectra_dict[g] = spectra
            if g == 'glint':
                ax22.plot(np.arange(3),spectra,'r')
            else:
                ax22.plot(np.arange(3),spectra,'purple')
            ax22.set_ylabel('DN')
            ax22.set_ylabel('RGB spectra')


    plt.show()
    return spectra_dict

def glint_ratio(spectra_dict_list):
    """
    spectra_dict_list (list of dict): keys: glint, non_glint
    """
    glint_array = np.array([d['glint'] for d in spectra_dict_list])
    non_glint_array = np.array([d['non_glint'] for d in spectra_dict_list])
    glint_ratio = glint_array/non_glint_array
    glint_ratio = np.mean(glint_ratio,axis=0)

    return glint_ratio

def add_glint(non_glint,non_glint_mask,thresh,glint_ratio,plot=True):
    
    #mask object
    mask = np.repeat(non_glint_mask[:,:,np.newaxis],3,axis=2)
    masked_non_glint = mask*non_glint
    # utils.plot_an_image(masked_non_glint)
    
    # greyscale then normalise
    std_grey_im = utils.normalise_img(rgb2gray(masked_non_glint)) #already masked, greyscaled, then normalised
    
    #identify bright regions based on threshold
    glint_mask = std_grey_im > thresh #identify brightspots with DN > 0.8. brightspots will have DN = 1
    # utils.plot_an_image(glint_mask)

    # on glint mask, increase glint ratio 
    simulated_glint = non_glint.copy()#.astype(np.uint16)
    sim_glint = non_glint[glint_mask == 1]*(glint_ratio)
    sim_glint = np.where(sim_glint>255,255,sim_glint)
    simulated_glint[glint_mask == 1] = sim_glint
    # utils.plot_an_image(simulated_glint)
    
    #increase brightness for glint mask (unnatural)
    # simulated_glint = non_glint.copy()#.astype(np.uint16)
    # brightened_region = increase_brightness(non_glint[glint_mask == 1],alpha=0.5,beta=10)
    # simulated_glint[glint_mask == 1] = brightened_region
    # utils.plot_an_image(simulated_glint)

    # bin_upward,bin_downward = boundary_binary(glint_mask)
    # utils.plot_an_image(bin_upward)
    # utils.plot_an_image(bin_downward)

    # sim_glint = add_rainbow(simulated_glint,bin_upward,bin_downward,alpha)
    # utils.plot_an_image(sim_glint)
    # apply gaussian smoothing
    # colors_gauss = scipy.ndimage.gaussian_filter(sim_glint,sigma=(1,1,0))
    # utils.plot_an_image(colors_gauss)
    if plot is True:
        fig, ax = plt.subplots(1,2,figsize=(8,4))
        ax[0].imshow(non_glint)
        ax[0].set_title('Original image')
        ax[1].imshow(simulated_glint)
        ax[1].set_title('simulated glint')
        ax[0].axis('off')
        ax[1].axis('off')
        
    return simulated_glint

