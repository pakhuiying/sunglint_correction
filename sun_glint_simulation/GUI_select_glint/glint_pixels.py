from matplotlib import cm
import cv2 as cv
import scipy.signal
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
from os.path import join, basename, isdir, isfile
from os import listdir, mkdir
from skimage.color import rgb2gray
from random import randrange
from math import ceil

import sys
# sys.path.insert(0,r'D:\PAKHUIYING\Image_processing\F3_raw_images\utils')
# import importlib
import utils
# importlib.reload(utils)

def boundary_binary(bin_im):
    """
    bin_im (np.array): 2d binary image, where 1 = bright pixels, 0 = non-bright pixels
    mode (str): upward or downward, where upward is [0,1] kernel, downward is [1,0] kernel
    """
    nrow,ncol = bin_im.shape
    # pad_im = np.pad(bin_im,pad_width=1,mode='edge')
    # print(pad_im.shape)
    bin_upward = np.zeros(bin_im.shape)
    bin_downward = np.zeros(bin_im.shape)
    for i in range(nrow-1):
        for j in range(ncol):
            if np.sum(bin_im[i:i+2,j]) == 1:
                if bin_im[i,j] == 0: #if top pixel == 0, then the edge is upward
                    bin_upward[i+1,j] = 1
                else:
                    bin_downward[i,j] = 1
            else:
                continue
    return bin_upward,bin_downward

def add_rainbow(simulated_glint,bin_upward,bin_downward,alpha=0.5):
    beta = 1- alpha
    sunglint_pixels = create_sun_glint_pixels(4,sigma=5)
    nrow,ncol = sunglint_pixels.shape[0], sunglint_pixels.shape[1]
    sim_glint = np.pad(simulated_glint.copy(),pad_width=[(nrow,nrow),(nrow,nrow),(0,0)],mode='edge')
    print(sim_glint.shape)
    for i in range(simulated_glint.shape[0]):
        for j in range(simulated_glint.shape[1]):
            padded_i = i + nrow
            padded_j = j+ nrow
            if bin_upward[i,j] == 1:
                crop_sim = sim_glint[padded_i-nrow//2:padded_i,padded_j:padded_j+ncol,:]
                # print(crop_sim.shape,nrow)
                dst = cv.addWeighted(crop_sim,alpha,sunglint_pixels[:nrow//2,:,:],beta,0) #where = = gamma, a constant to add to the image
                sim_glint[padded_i-nrow//2:padded_i,padded_j:padded_j+ncol,:] = dst
                # sim_glint[padded_i-nrow//2:padded_i,padded_j:padded_j+ncol,:] = sunglint_pixels[:nrow//2,:,:]
            if bin_downward[i,j] == 1:
                crop_sim = sim_glint[padded_i:padded_i+nrow//2,padded_j:padded_j+ncol,:]
                # print(crop_sim.shape,nrow//2,ncol)
                dst = cv.addWeighted(crop_sim,alpha,sunglint_pixels[-nrow//2:,:,:],beta,0) #where = = gamma, a constant to add to the image
                sim_glint[padded_i:padded_i+nrow//2,padded_j:padded_j+ncol,:] = dst
                # sim_glint[padded_i:padded_i+nrow//2,padded_j:padded_j+ncol,:] = sunglint_pixels[-nrow//2:,:,:]

    return sim_glint[nrow:-nrow,nrow:-nrow,:]

def create_sun_glint_pixels(h,alpha=50,w=7,sigma=5,noiseExtent=50):
    
    r = np.array([[252,60,66]])
    o = np.array([[255,112,58]])
    y = np.array([[255,220,101]])
    white = np.array([[255,255,255]])
    # white = np.repeat(white,h,axis=0)
    # print(white.shape)
    g = np.array([[146,255,200]])
    b = np.array([[98,235,243]])
    p = np.array([[38,88,130]])
    color_list = [np.repeat(i,h,axis=0) for i in [r,o,y,white,g,b,p]]
    colors = np.row_stack(color_list)
    colors = np.repeat(colors[:,:,np.newaxis],w,axis=2)
    colors = np.swapaxes(colors,1,2).astype(np.uint8)
    # utils.plot_an_image(colors)
    
    # apply gaussian smoothing
    colors_gauss = scipy.ndimage.gaussian_filter(colors,sigma=(sigma,sigma,0))

    # introduce noises to color
    num_pixels = (colors.shape[0])*colors.shape[1]
    ran_int = np.array(sorted([randrange(0,num_pixels) for i in range(num_pixels//2)]))
    ran_indices = np.unravel_index(ran_int,(colors.shape[0],colors.shape[1]))
    colors_noise = colors_gauss.copy().astype(np.uint16)
    colors_noise[ran_indices] = colors_noise[ran_indices]+randrange(0,noiseExtent)
    colors_noise = np.where(colors_noise>255,255,colors_noise).astype(np.uint8)
    utils.plot_an_image(colors_noise)

    # add/reduce random brightness
    # colors = colors_noise.copy().astype(np.uint16)
    # colors = colors + randrange(-alpha,alpha)
    # colors = utils.normalise_img(colors)*255
    # colors = colors.astype(np.uint8)
    # utils.plot_an_image(colors)

    return colors_noise

def add_rainbow(simulated_glint,bin_upward,bin_downward,alpha=0.5):
    beta = 1- alpha
    sunglint_pixels = create_sun_glint_pixels(4,sigma=5)
    nrow,ncol = sunglint_pixels.shape[0], sunglint_pixels.shape[1]
    sim_glint = np.pad(simulated_glint.copy(),pad_width=[(nrow,nrow),(nrow,nrow),(0,0)],mode='edge')
    print(sim_glint.shape)
    for i in range(simulated_glint.shape[0]):
        for j in range(simulated_glint.shape[1]):
            padded_i = i + nrow
            padded_j = j+ nrow
            if bin_upward[i,j] == 1:
                crop_sim = sim_glint[padded_i-nrow//2:padded_i,padded_j:padded_j+ncol,:]
                # print(crop_sim.shape,nrow)
                dst = cv.addWeighted(crop_sim,alpha,sunglint_pixels[:nrow//2,:,:],beta,0) #where = = gamma, a constant to add to the image
                sim_glint[padded_i-nrow//2:padded_i,padded_j:padded_j+ncol,:] = dst
                # sim_glint[padded_i-nrow//2:padded_i,padded_j:padded_j+ncol,:] = sunglint_pixels[:nrow//2,:,:]
            if bin_downward[i,j] == 1:
                crop_sim = sim_glint[padded_i:padded_i+nrow//2,padded_j:padded_j+ncol,:]
                # print(crop_sim.shape,nrow//2,ncol)
                dst = cv.addWeighted(crop_sim,alpha,sunglint_pixels[-nrow//2:,:,:],beta,0) #where = = gamma, a constant to add to the image
                sim_glint[padded_i:padded_i+nrow//2,padded_j:padded_j+ncol,:] = dst
                # sim_glint[padded_i:padded_i+nrow//2,padded_j:padded_j+ncol,:] = sunglint_pixels[-nrow//2:,:,:]

    return sim_glint[nrow:-nrow,nrow:-nrow,:]

def interpolate_colors(color,spectra,steps,plot=True):
    """
    linear interpolation from one spectra to another color
    """
    # nrow = ncol = steps#int(np.sqrt(steps))
    nrow = steps
    steps_colors = (color - spectra)/steps
    # color_sq = np.zeros((nrow*ncol,1,3))
    color_sq = np.zeros((nrow,1,3))
    # fig,ax = plt.subplots(1,2,figsize=(7,3))
    for i in range(steps):
        spectra = spectra + steps_colors
        spectra_cmap = np.where(spectra>255,255,spectra).flatten()/255
        # ax[1].plot(np.arange(3),spectra.flatten(),color = tuple(spectra_cmap))
        spectra = spectra.reshape(1,1,3)
        # row_i = int(i/nrow)
        # col_i = i%nrow
        color_sq[i,0,:] = spectra
    # for i in range(3):
    #     [spectra[i] + steps[i] for i in range(3)]
    # utils.plot_an_image(color_sq.astype(np.uint8))
    
    # ax[0].imshow(color_sq.astype(np.uint8))
    # ax[0].axis('off')
    if plot is True:
        utils.plot_an_image(color_sq.astype(np.uint8))

    return color_sq.astype(np.uint8)

def water_palette(non_glint,color_step=100,spectra_step=25,cut_off=0.15,extend_extreme=3,plot=True):
    """ 
    non_glint (np.array): image with no glint
    color_step (int): how many colors to sample from cm's Spectral palette
    spectra_step (int): sq number. Number of interpolation steps between spectra to color
    cut_off (float): [0,1]: How much of the red and blue to remove from cm.Spectral
    extend_extreme (int): how much of the red and blue to interpolate to the target spectrum as a multiple of spectra_step
    returns glint palette
    >>> water_palette(non_glint,color_step=100,spectra_step=25,cut_off=0.15,extend_extreme=3,plot=False)
    """
    spectra = np.mean(non_glint,axis=(0,1))
    x = np.linspace(0,1,color_step)
    c = cm.get_cmap('Spectral')(x)[:,:3] #remove the alpha channel
    c = np.swapaxes(c[:,:,np.newaxis],1,2)*255
    c = c.astype(np.uint8)
    row,col = c.shape[0], c.shape[1]
    c = c[int(cut_off*row):int((1-cut_off+0.05)*row),:,:]
    # utils.plot_an_image(c)
    interp_colors = np.zeros((c.shape[0],spectra_step,3))
    for row in range(c.shape[0]):
        interp_colors[row,:,:] = interpolate_colors(c[row,0,:],spectra,spectra_step,plot=False).reshape(1,-1,3)
    
    interp_colors = interp_colors.astype(np.uint8)

    c_list = []
    c1_list = []
    for i in range(spectra_step):
        c = interpolate_colors(interp_colors[0,i,:],spectra,spectra_step*extend_extreme,plot=False)
        c_list.append(c)
        c1 = interpolate_colors(spectra,interp_colors[-1,i,:],spectra_step*extend_extreme,plot=False)
        c1_list.append(c1)

    interp_top = np.column_stack(c_list)
    interp_bottom = np.column_stack(c1_list)
    
    c = np.row_stack((interp_top,interp_colors,interp_bottom))
    if plot is True:
        utils.plot_an_image(c.astype(np.uint8))

    return c.astype(np.uint8)

def glint_sampling(glint_palette,brightness,saturation,contrast,h,w=7,sample_n=10,sigma=1,plot=True):
    """ 
    glint_palette (np.array): output from water_palette
    brightness (float): refers to how bright the pixels are 
    
    saturation (float):[0,1]: how close the colors are to the water spectra
    contrast (float): the differentiation between colors
    h (int): how many white pixels sandwiched between
    w (int): width of glint pixels generated
    sample_n (int): how many colors to sample
    sigma (int): how much gaussian smoothing to apply
    >>> glint_sampling(glint_palette,brightness=0,saturation=0.4,contrast=1,h=0,w=7,sample_n=12,sigma=1,plot=False)
    """
    brightened_palette = cv.convertScaleAbs(glint_palette,alpha = contrast, beta=brightness)#alpha is the gain, beta is the bias to control contrast and brightness respectively
    nrow,ncol = glint_palette.shape[0], glint_palette.shape[1]
    saturation_col = int(saturation*ncol)
    if saturation_col == ncol:
        saturation_col = ncol -1
    sampled_pixels = brightened_palette[np.arange(0,nrow,nrow//sample_n),saturation_col,:]
    sampled_pixels = sampled_pixels.reshape(-1,1,3)
    r,c = sampled_pixels.shape[0], sampled_pixels.shape[1]
    white_pixels = np.tile(np.array([255]),(h,1,3))
    glint_pixel = np.row_stack((sampled_pixels[:r//2,:,:],white_pixels,sampled_pixels[-r//2:,:,:]))
    glint_pixel = np.tile(glint_pixel,(1,w,1))
    # apply gaussian smoothing
    if sigma > 0:
        glint_pixel = scipy.ndimage.gaussian_filter(glint_pixel,sigma=(sigma,sigma,0))
    # utils.plot_an_image(sampled_pixels)
    # utils.plot_an_image(glint_pixel)
    if plot is True:
        utils.plot_an_image(glint_pixel)

    return glint_pixel

def add_glint(non_glint,non_glint_mask,thresh,glint_ratio,plot=True):
    """
    *outdated*
    """
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
    outputs = {'Original_image':non_glint,'Glint_mask':glint_mask,'Simulated_glint':simulated_glint}
    if plot is True:
        fig, axes = plt.subplots(1,len(list(outputs)),figsize=(10,4))
        for ax, (name,im) in zip(axes.flatten(),outputs.items()):
            if im.ndim == 2:
                ax.imshow(im,cmap='gray')
            else:
                ax.imshow(im)
            ax.set_title(name.replace('_',' '))
            ax.axis('off')
        plt.show()
        
        
    return non_glint, glint_mask.astype(np.uint8), simulated_glint.astype(np.uint8)

def sin_curve(amplitude,period,x_translate,y_translate,n=50,plot=True):
    x = np.linspace(-2*np.pi,2*np.pi,n)
    fx = amplitude*np.sin(x*2*np.pi/period+x_translate) + amplitude
    if plot is True:
        plt.figure()
        plt.plot(x,fx)
        plt.title(f'Amplitude: {amplitude}, period: {period}, x_translate: {x_translate}, y_translate: {y_translate}')
        plt.show()
    return fx.astype(int)

def randomize_tiles_shuffle_blocks(a, M, N):
    """ 
    A function to shuffle randomly 3D blocks wrt each other, while conserving the pattern within the block across all axes
    Ensure that shape of a is multiples of M & n
    a (np.array): 3 channels
    M , N (int): dimensions of the block
    """    
    m,n,c = a.shape
    b = a.reshape(m//M,M,n//N,N,c).swapaxes(1,2).reshape(-1,M*N,c)
    np.random.shuffle(b)
    return b.reshape(m//M,n//N,M,N,c).swapaxes(1,2).reshape(a.shape)

def roll_column(a,shift_array,width):
    """ 
    this function shifts the columns
    shift_array (1D np.array): any shape. As it will be broadcasted to account for the width and shape of 'a'
    width (int): shift blocks of column size width
    >>> roll_column(a,np.arange(-2,2),width=2)
    """
    # shift_array can be any size, because it is repeated by the width and then cropped to fit the size of column
    shift_array = np.repeat(shift_array,width)[:a.shape[1]]
    # get the row and column indices
    row_indices,column_indices = np.ogrid[:a.shape[0],:a.shape[1]]
    shift_array[shift_array<0] += a.shape[0] #because we are shifting the rows of the pixels around
    row_indices = row_indices - shift_array[:,np.newaxis].T
    result = a[row_indices,column_indices,:]
    return result

def apply_glint_mask(non_glint,glint_mask_list,params_dict,palette_dict = None,plot=True,plot_gui=False):
    """
    non_glint (np.array)
    glint_mask_list (dictionary. keys: threshold, value: 2D np.array of mask):in increasing order of threshold that is used to create the mask
    """

    # initialise plot
    if plot is True:
        fig, axes = plt.subplots(2,len(glint_mask_list.keys())+1,figsize=(10,5))

    # initialise glint palette    
    if palette_dict is None:
        glint_palette = water_palette(non_glint,color_step=125,spectra_step=25,plot=False)
    else:
        glint_palette = water_palette(non_glint,palette_dict['-COLOR_STEP-'],
                                                            palette_dict['-SPECTRA_STEP-'],
                                                            palette_dict['-CUT_OFF-'],
                                                            palette_dict['-EXTEND_EXTREME-'],
                                                            plot=False)

    c_list = []

    # include assert to check if params_dict's element's list is == len(glint_mask_list)
    c0_row,c0_col = 512,512
    for i in range(len(glint_mask_list.keys())):
        c = glint_sampling(glint_palette,
                                        params_dict['-BRIGHTNESS-'][i],
                                        params_dict['-SATURATION-'][i],
                                        params_dict['-CONTRAST-'][i],
                                        params_dict['-H-'][i],
                                        params_dict['-W-'][i],
                                        params_dict['-SAMPLE_N-'][i],
                                        params_dict['-SIGMA-'][i],
                                        plot=False)
        c = c[::1+i]
        if c.shape[0] < c0_row:
            c0_row = c.shape[0]
        if c.shape[1] < c0_col:
            c0_col = c.shape[1]
        c_list.append(c)
        if plot is True:
            axes[0,i].imshow(c_list[i])
            axes[0,i].axis('off')
            axes[0,i].set_title('B:{}, S: {:.3f}, C: {}'.format(params_dict['-BRIGHTNESS-'][i],
                                                    params_dict['-SATURATION-'][i],
                                                    params_dict['-CONTRAST-'][i]))

    nrow,ncol = glint_mask_list[list(glint_mask_list)[0]].shape
    # print(nrow,ncol)
    # c0_row,c0_col = c_list[-1].shape[0],c_list[-1].shape[1]
    mrow,mcol = ceil(nrow/c0_row), ceil(ncol/c0_col)
    # print(mrow,mcol)
    
    
    simulated_glint = non_glint.copy()
    for i,(c,(t,mask)) in enumerate(zip(c_list,glint_mask_list.items())):
        shuffle_column = np.array([randrange(-c0_row,c0_row) for i in range(ceil(ncol/params_dict['-W-'][0]))])
        c_tile = np.tile(c,(mrow,mcol,1))
        # shuffle the columns randomly
        c_tile = roll_column(c_tile,shuffle_column,width=c0_col)
        c_tile = c_tile[:nrow,:ncol,:] *np.repeat(mask[:,:,np.newaxis],3,axis=2)
        simulated_glint[mask==1] = c_tile[mask==1]
        if plot is True:
            axes[1,i].imshow(simulated_glint)
            axes[1,i].set_title(f'Thresh: {t}')
            axes[1,i].axis('off')
            # utils.plot_an_image(simulated_glint)
    if plot is True:
        axes[1,-1].imshow(non_glint)
        axes[1,-1].axis('off')
        axes[1,-1].set_title('Original image')
        fig.delaxes(axes[0][-1])
        if plot_gui is True:
            plt.show(block=False)
        else:
            plt.show()

    return simulated_glint

def create_masks(non_glint,non_glint_mask,thresh_array,plot=True,plot_gui=False):
    """ 
    non_glint (3D np.array): non glint image
    non_glint_mask (2D np.array): a mask that masks non-water objects (water = 1, non-water = 0)
    thresh_array (1D np.array, or tuple, or list): thresh in ascending order
    """
    #mask object
    mask = np.repeat(non_glint_mask[:,:,np.newaxis],3,axis=2)
    masked_non_glint = mask*non_glint
    # utils.plot_an_image(masked_non_glint)
    
    # greyscale then normalise
    std_grey_im = utils.normalise_img(rgb2gray(masked_non_glint)) #already masked, greyscaled, then normalised
    if plot is True:
        fig,axes = plt.subplots(1,thresh_array.shape[0]+1,figsize=(8,4))
    #identify bright regions based on various thresholds
    mask_list = {}
    for i,t in enumerate(thresh_array):
        glint_mask = std_grey_im > t
        # mask_list.append(glint_mask)
        mask_list['{:.2f}'.format(t)] = glint_mask
        if plot is True:
            axes[i].imshow(glint_mask,cmap='gray')
            axes[i].set_title('thresh = {:.2f}'.format(t))
            axes[i].axis('off')

    if plot is True:
        axes[-1].imshow(non_glint)
        axes[-1].axis('off')
        if plot_gui is True:
            plt.show(block=False)

    return non_glint,mask_list