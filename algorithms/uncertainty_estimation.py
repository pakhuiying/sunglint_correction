import cv2
import os, glob
import json
from tqdm import tqdm
# import pickle #This library will maintain the format as well
import importlib
import radiometric_calib_utils
import mutils
importlib.reload(radiometric_calib_utils)
importlib.reload(mutils)
import radiometric_calib_utils as rcu
import mutils
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy import ndimage
from scipy import stats
import algorithms.SUGAR as sugar

class UncertaintyEst:
    def __init__(self,corrected_im_background, corrected_im):
        """
        :param corrected_im_background (np.ndarray): image accounted for 
        :param corrected_im (np.ndarray)

        """
        self.corrected_im_background = corrected_im_background
        self.corrected_im = corrected_im

    @classmethod
    def get_corrected_image(glint_image,iter=3,bounds = [(1,2)]*10):

        def correction_iterative(glint_image,iter,bounds,estimate_background):
            gm_list = []
            bg_list = []
            for i in range(iter):
                HM = sugar.SUGAR(glint_image,iter,bounds,estimate_background)
                corrected_bands = HM.get_corrected_bands()
                glint_image = np.stack(corrected_bands,axis=2)
                
                b_list = HM.b_list
                bounds = [(1,b*1.2) for b in b_list]
                glint_mask = HM.glint_mask
                est_background = HM.est_background

                gm_list.append(glint_mask)
                bg_list.append(est_background)
            return glint_image, gm_list, bg_list
        
        corrected_im_background, gm_list_background, bg_list_background = correction_iterative(glint_image,iter=iter,bounds = bounds,estimate_background=True)
        corrected_im, gm_list, bg_list = correction_iterative(glint_image,iter=iter,bounds = bounds,estimate_background=False)
        

def plot_glint_contour(im_aligned,glint_mask,NIR_band=3,add_weights=False):
    """ 
    :param im_aligned (np.ndarray) band-aligned image from:
        RI = espect.ReflectanceImage(cap)
        im_aligned = RI.get_aligned_reflectance()
    :param glint_mask (np.ndarray): where 1 is glint, 0 is non-glint
    :param NIR_band (int): band number e.g. NIR band to extract the glint mask
    """
    # use the NIR band
    gm = glint_mask[:,:,NIR_band]
    
    nrow, ncol = gm.shape
    y = np.linspace(0,nrow-1,nrow,dtype=int)[::5]
    x = np.linspace(0,ncol-1,ncol,dtype=int)[::5]
    X, Y = np.meshgrid(x,y)
    # Y = np.flipud(Y)
    xy = np.vstack([X.ravel(), Y.ravel()])
    print(xy.shape)
    idx = np.argwhere(gm==1)
    # Xtrain = np.vstack([idx[:,0], idx[:,1]])
    Xtrain = np.vstack([idx[:,1], idx[:,0]])

    if add_weights is True:
        im = im_aligned[:,:,NIR_band]
        weights = im[(idx[:,0], idx[:,1])]#.reshape(-1,1)
        # weights = np.flipud(weights)
        weights[weights<0] = 0

        # uses Scott's Rule to implement bandwidth selection
        kernel = stats.gaussian_kde(Xtrain,weights=weights)
    else:
        kernel = stats.gaussian_kde(Xtrain)

    Z = kernel(xy).T
    print(Z.shape)
    Z = np.reshape(Z, X.shape)
    Z = np.flipud(Z)
    # plot contours of the density
    levels = np.linspace(0, Z.max(), 25)

    fig, axes = plt.subplots(1,2,figsize=(10,5),sharex=True)
    axes[0].imshow(np.take(im_aligned,[2,1,0],axis=2))
    im = axes[1].contourf(X, Y, Z, levels=levels, cmap=plt.cm.RdGy_r)
    axes[1].set_aspect('equal')
    fig.colorbar(im,ax=axes[1])
    plt.show()
    return 