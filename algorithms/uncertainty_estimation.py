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
from skimage.transform import resize
import algorithms.SUGAR as sugar

class UncertaintyEst:
    def __init__(self,im_aligned,corrected_im_background, corrected_im,glint_mask=None):
        """
        :param im_aligned (np.ndarray): band aligned and calibrated & corrected reflectance image
        :param corrected_im_background (np.ndarray): image accounted for background spectra
        :param corrected_im (np.ndarray)
        :param glint_mask (np.ndarray): where 1 is glint pixel and 0 is non-glint
        """
        self.im_aligned = im_aligned
        self.corrected_im_background = corrected_im_background
        self.corrected_im = corrected_im
        self.glint_mask = glint_mask

    @classmethod
    def get_corrected_image(cls,glint_image,iter=3,bounds = [(1,2)]*10):
        
        im_aligned = glint_image.copy()
        
        def correction_iterative(glint_image,iter,bounds,estimate_background):
            
            for i in range(iter):
                HM = sugar.SUGAR(glint_image,iter,bounds,estimate_background)
                corrected_bands = HM.get_corrected_bands()
                glint_image = np.stack(corrected_bands,axis=2)
                if i == 0:
                    glint_mask = np.stack(HM.glint_mask,axis=2)
                
            return glint_image, glint_mask
        
        corrected_im_background, glint_mask = correction_iterative(glint_image,iter=iter,bounds = bounds,estimate_background=True)
        corrected_im, _ = correction_iterative(glint_image,iter=iter,bounds = bounds,estimate_background=False)
        
        return cls(im_aligned,glint_image,corrected_im_background,corrected_im,glint_mask)

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
        weights = im[(idx[:,1], idx[:,0])]#.reshape(-1,1)
        # weights = np.flipud(weights)
        weights[weights<0] = 0

        # uses Scott's Rule to implement bandwidth selection
        kernel = stats.gaussian_kde(Xtrain,weights=weights)
    else:
        kernel = stats.gaussian_kde(Xtrain)

    Z = kernel(xy).T
    # print(Z.shape)
    Z = np.reshape(Z, X.shape)
    # plot contours of the density
    levels = np.linspace(0, Z.max(), 25)

    fig, axes = plt.subplots(1,2,figsize=(10,5),sharex=True)
    axes[0].imshow(np.take(im_aligned,[2,1,0],axis=2))
    axes[0].set_title('Original RGB image')
    im = axes[1].contourf(X, Y, np.flipud(Z), levels=levels, cmap=plt.cm.RdGy_r)
    axes[1].set_aspect('equal')
    axes[1].set_title('Glint mask KDE')
    fig.colorbar(im,ax=axes[1])
    for ax in axes:
        ax.axis('off')
    plt.show()
    return resize(Z,(nrow,ncol),anti_aliasing=True)

