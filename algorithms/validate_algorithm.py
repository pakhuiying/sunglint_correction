import numpy as np
import os
import pickle #This library will maintain the format as well
import micasense.imageutils as imageutils
import micasense.capture as capture
import cv2
import matplotlib.pyplot as plt
import PIL.Image as Image
import matplotlib.patches as patches
import json
import glob
import shutil
import mutils
import extract_spectral as espect
import algorithms.Hedley as Hedley
import algorithms.Hedley_multiband as HedleyMulti
from scipy.ndimage import gaussian_filter,laplace, gaussian_laplace
from scipy.optimize import minimize_scalar

class ValidateCorrection:
    def __init__(self,im_aligned, background_spectral=None):
        self.im_aligned = im_aligned
        self.background_spectral = background_spectral
        self.n_bands = im_aligned.shape[-1]
        if self.background_spectral is not None:
             assert background_spectral.shape == (1,1,self.n_bands)
        wavelengths = mutils.sort_bands_by_wavelength()
        self.wavelength_dict = {i[0]:i[1] for i in wavelengths}
    
    def otsu_thresholding(self,im):
        """
        otsu thresholding with Brent's minimisation of a univariate function
        returns the value of the threshold for input
        """
        count,bin,_ = plt.hist(im.flatten(),bins='auto')
        plt.close()
        
        hist_norm = count/count.sum() #normalised histogram
        Q = hist_norm.cumsum() # CDF function ranges from 0 to 1
        N = count.shape[0]
        bins = np.arange(N)
        
        def otsu_thresh(x):
            x = int(x)
            p1,p2 = np.hsplit(hist_norm,[x]) # probabilities
            q1,q2 = Q[x],Q[N-1]-Q[x] # cum sum of classes
            b1,b2 = np.hsplit(bins,[x]) # weights
            # finding means and variances
            m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
            v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
            # calculates the minimization function
            fn = v1*q1 + v2*q2
            return fn
        
        # brent method is used to minimise an univariate function
        # bounded minimisation
        res = minimize_scalar(otsu_thresh, bounds=(1, N), method='bounded')
        thresh = bin[int(res.x)]
        
        return thresh
    
    def get_glint_mask(self, plot = True):
        """
        get glint mask using laplacian of image. 
        We assume that water constituents and features follow a smooth continuum, 
        but glint pixels vary a lot spatially and in intensities
        Note that for very extensive glint, this method may not work as well <--:TODO use U-net to identify glint mask
        returns a list of np.ndarray
        """
        glint_mask_list = []
        for i in range(self.im_aligned.shape[-1]):
            im_copy = self.im_aligned[:,:,i].copy()
            # find the laplacian of gaussian first
            # take the absolute value of laplacian because the sign doesnt really matter, we want all edges
            im_smooth = np.abs(gaussian_laplace(im_copy,sigma=1))
            im_smooth = im_smooth/np.max(im_smooth)

            #threshold mask
            thresh = self.otsu_thresholding(im_smooth)
            glint_mask = np.where(im_smooth>thresh,1,0)
            glint_mask_list.append(glint_mask)

        return glint_mask_list
         
    def get_background_spectral(self):
        """
        get the average reflectance across the whole image from non-glint regions
        """
        glint_mask = self.get_glint_mask()
        background_spectral = []
        for i in range(self.n_bands):
            im_copy = self.im_aligned[:,:,i].copy()
            gm = glint_mask[i]
            background_spectral.append(np.mean(im_copy[gm == 0]))
        
        background_spectral = np.array(background_spectral).reshape(1,1,self.n_bands)
        return background_spectral

    def simulate_glint(self, plot = True):

        if self.background_spectral is None:
            background_spectral = self.get_background_spectral()
        else:
            assert self.background_spectral.shape == (1,1,10)
            background_spectral = self.background_spectral

        glint_mask = self.get_glint_mask(plot=False)
        nrow, ncol, n_bands = self.im_aligned.shape

        # simulate back ground shape with known spectral curves
        background_im = np.tile(background_spectral,(nrow,ncol,1))

        im_aligned = self.im_aligned.copy()
        for i in range(n_bands):
            background_im[:,:,i][glint_mask[i]==1] = im_aligned[:,:,i][glint_mask[i]==1]

        if plot is True:
            plt.figure()
            plt.imshow(np.take(background_im,[2,1,0],axis=2))
            plt.title('Simulated Glint RGB image')
            plt.axis('off')
            plt.show()

        return background_im
    
    def validate_correction(self, save_dir=None, filename = None, plot = True):
        """
        validate sun glint correction algorithm with simulated image
        """
        if self.background_spectral is None:
            background_spectral = self.get_background_spectral()
        else:
            assert background_spectral.shape == (1,1,10)
        

        simulated_glint = self.simulate_glint(plot=False)
        background_spectral = np.tile(background_spectral,(simulated_glint.shape[0],simulated_glint.shape[1],1))
        
        # get corrected_bands
        HM = HedleyMulti.HedleyMulti(simulated_glint,None)
        corrected_bands = HM.get_corrected_bands(plot=False)
        corrected_bands = np.stack(corrected_bands,axis=2)

        rgb_bands = [2,1,0]
        fig = plt.figure(figsize=(10, 8), layout="constrained")
        spec = fig.add_gridspec(3, 3)

        plot_dict = {"Simulated background":background_spectral,"Simulated glint":simulated_glint,"Corrected Image":corrected_bands}

        ax0 = fig.add_subplot(spec[:, 1:])
        ax_plots = [fig.add_subplot(spec[i,0]) for i in range(3)]
        for (title, im), ax in zip(plot_dict.items(),ax_plots):
            ax0.plot(list(self.wavelength_dict.values()),[im[:,:,i].mean() for i in list(self.wavelength_dict)],label=title)
            ax.imshow(np.take(im,rgb_bands,axis=2))
            ax.set_title(title)
            ax.axis('off')
        
        ax0.set_xlabel("Wavelength (nm)")
        ax0.set_ylabel("Reflectance")
        ax0.legend(loc='upper right')

        if save_dir is None:
            #create a new dir to store plot images
            save_dir = os.path.join(os.getcwd(),"validate_corrected_images")
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
        else:
            save_dir = os.path.join(save_dir,"validate_corrected_images")
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

        filename = mutils.get_all_dir(filename,iter=4)
        filename = os.path.splitext(filename)[0]
        full_fn = os.path.join(save_dir,filename)

        fig.suptitle(filename)
        fig.savefig('{}.png'.format(full_fn))

        if plot is True:
            plt.show()
        else:
            plt.close()
        return
        
        return