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
import matplotlib.colors as colors
import matplotlib.cm as cmx
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
        :param corrected_im_background (list of np.ndarray): image accounted for background spectra, where each element is from each iteration
        :param corrected_im (list of np.ndarray): image not accounting for background spectra
        :param glint_mask (np.ndarray or None): where 1 is glint pixel and 0 is non-glint
        """
        self.im_aligned = im_aligned
        self.corrected_im_background = corrected_im_background
        self.corrected_im = corrected_im
        self.glint_mask = glint_mask
        wavelengths = mutils.sort_bands_by_wavelength()
        self.wavelength_dict = {i[0]:i[1] for i in wavelengths}
        self.n_bands = len(list(self.wavelength_dict))

    @classmethod
    def get_corrected_image(cls,im_aligned,iter=3,bounds = [(1,2)]*10):
        
        corrected_im_background, glint_mask = sugar.correction_iterative(im_aligned,iter=iter, bounds = bounds,estimate_background=True,get_glint_mask=True)
        corrected_im = sugar.correction_iterative(im_aligned,iter=iter, bounds = bounds,estimate_background=False,get_glint_mask=False)
        
        return cls(im_aligned,corrected_im_background,corrected_im,glint_mask)

    def get_glint_kde(self,NIR_band=3,add_weights=False, plot=True):
        """ 
        :param im_aligned (np.ndarray) band-aligned image from:
            RI = espect.ReflectanceImage(cap)
            im_aligned = RI.get_aligned_reflectance()
        :param glint_mask (np.ndarray): where 1 is glint, 0 is non-glint
        :param NIR_band (int): band number e.g. NIR band to extract the glint mask
        returns KDE map of glint mask to estimate the spatial probability distribution of glint. sum of KDE map = 1
        """
        if self.glint_mask is not None:
            # use the NIR band
            gm = self.glint_mask[:,:,NIR_band]
            
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
                im = self.im_aligned[:,:,NIR_band]
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

            if plot is True:
                # plot contours of the density
                levels = np.linspace(0, Z.max(), 25)

                fig, axes = plt.subplots(1,2,figsize=(10,5),sharex=True)
                axes[0].imshow(np.take(self.im_aligned,[2,1,0],axis=2))
                axes[0].set_title('Original RGB image')
                im = axes[1].contourf(X, Y, np.flipud(Z), levels=levels, cmap=plt.cm.RdGy_r)
                axes[1].set_aspect('equal')
                axes[1].set_title('Glint mask KDE')
                fig.colorbar(im,ax=axes[1])
                for ax in axes:
                    ax.axis('off')
                plt.show()
            return resize(Z,(nrow,ncol),anti_aliasing=True)
        
        else:
            return None
        
    def get_uncertainty_bounds(self,get_upper_and_lower_bounds = False, plot = True):
        """
        returns lower, upper bound and total variance of all iterations in band order i.e. 0,1,2,3,4,5,6,7,8,9
        returns np.ndarray 
        """
        upper_bound_uncertainty = []
        lower_bound_uncertainty = []
        total_uncertainty = []

        for band_idx in range(self.n_bands):
            im_background = np.stack([self.corrected_im_background[img_idx][:,:,band_idx] for img_idx in range(len(self.corrected_im_background))],axis=2)
            im = np.stack([self.corrected_im[img_idx][:,:,band_idx] for img_idx in range(len(self.corrected_im))],axis=2)
            total_im = np.concatenate([im_background,im],axis=2)

            if get_upper_and_lower_bounds is True:
                lower_bound_uncertainty.append(np.var(im,axis=2))
                upper_bound_uncertainty.append(np.var(im_background,axis=2))
            total_uncertainty.append(np.var(total_im,axis=2))
        
        total_uncertainty = np.stack(total_uncertainty,axis=2)

        if plot is True:
            fig, axes = plt.subplots(len(total_uncertainty),1,figsize=(5,20))
            for i,(band_number,wavelength) in zip(range(len(total_uncertainty)),self.wavelength_dict.items()):
                # im = axes[i].imshow(normalise(total_uncertainty[i]))
                im = axes[i].contourf(total_uncertainty[:,:,band_number],levels=7,origin='upper')
                axes[i].axis('off')
                axes[i].set_title(f'Total uncertainty ({wavelength} nm)')
                fig.colorbar(im,ax=axes[i])
            plt.show()
        return total_uncertainty if get_upper_and_lower_bounds is False else (lower_bound_uncertainty, upper_bound_uncertainty, total_uncertainty)

