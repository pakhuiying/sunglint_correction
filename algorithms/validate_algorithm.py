import numpy as np
import os
import pickle #This library will maintain the format as well
import micasense.imageutils as imageutils
import micasense.capture as capture
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import PIL.Image as Image
import json
import glob
import shutil
import mutils
import extract_spectral as espect
import algorithms.Hedley as Hedley
import algorithms.Hedley_multiband as HedleyMulti
from scipy.ndimage import gaussian_filter,laplace, gaussian_laplace
from scipy.optimize import minimize_scalar

class SimulateGlint:
    def __init__(self,im_aligned, bbox=None, background_spectral=None):
        """
        :param im_aligned (np.ndarray) band-aligned image from:
            RI = espect.ReflectanceImage(cap)
            im_aligned = RI.get_aligned_reflectance()
        :param bbox (tuple) bounding boxes ((x1,y1),(x2,y2))
        :param background_spectral (np.ndarray): spectra that determines the ocean colour for the simulated background
        """
        self.im_aligned = im_aligned
        self.background_spectral = background_spectral
        self.n_bands = im_aligned.shape[-1]
        self.bbox = bbox
        if bbox is not None:
            self.bbox = mutils.sort_bbox(bbox)
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
    
    def get_glint_mask(self):
        """
        get glint mask using laplacian of image. 
        We assume that water constituents and features follow a smooth continuum, 
        but glint pixels vary a lot spatially and in intensities
        Note that for very extensive glint, this method may not work as well <--:TODO use U-net to identify glint mask
        returns a list of np.ndarray
        """
        if self.bbox is not None:
            ((x1,y1),(x2,y2)) = self.bbox

        glint_mask_list = []
        for i in range(self.im_aligned.shape[-1]):
            if self.bbox is not None:
                im_copy = self.im_aligned[y1:y2,x1:x2,i].copy()
            else:
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
        if self.bbox is not None:
            ((x1,y1),(x2,y2)) = self.bbox

        glint_mask = self.get_glint_mask()
        
        background_spectral = []
        for i in range(self.n_bands):
            if self.bbox is not None:
                im_copy = self.im_aligned[y1:y2,x1:x2,i].copy()
            else:
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

        glint_mask = self.get_glint_mask()
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
    
    def validate_correction(self, sigma=1, save_dir=None, filename = None, plot = True):
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
        HM = HedleyMulti.HedleyMulti(simulated_glint,None, sigma=sigma)
        corrected_bands = HM.get_corrected_bands(plot=False)
        corrected_bands = np.stack(corrected_bands,axis=2)

        rgb_bands = [2,1,0]
        fig = plt.figure(figsize=(12, 9), layout="constrained")
        spec = fig.add_gridspec(3, 3)

        plot_dict = {"Simulated background":background_spectral,"Simulated glint":simulated_glint,"Corrected Image":corrected_bands}

        # calculate differences spatially between original and corrected image
        residual_im = corrected_bands - background_spectral
        ax0 = fig.add_subplot(spec[0,1])
        im0 = ax0.imshow(np.take(residual_im,rgb_bands,axis=2))
        ax0.axis('off')
        ax0.set_title(r"Corrected - original $\rho$")
        # divider = make_axes_locatable(ax0)
        # cax = divider.append_axes('right', size='5%', pad=0.05)
        # fig.colorbar(im0,cax=cax,orientation='vertical')

        # calculate differences in reflectance between original and corrected image
        ax1 = fig.add_subplot(spec[0,2])
        ax1.plot(list(self.wavelength_dict.values()),[residual_im[:,:,i].mean() for i in list(self.wavelength_dict)])
        ax1.set_xlabel("Wavelength (nm)")
        ax1.set_title(r"Corrected - original $\rho$")

        ax2 = fig.add_subplot(spec[1:, 1:])
        ax_plots = [fig.add_subplot(spec[i,0]) for i in range(3)]
        for (title, im), ax in zip(plot_dict.items(),ax_plots):
            ax2.plot(list(self.wavelength_dict.values()),[im[:,:,i].mean() for i in list(self.wavelength_dict)],label=title)
            ax.imshow(np.take(im,rgb_bands,axis=2))
            ax.set_title(title)
            ax.axis('off')
        
        ax2.set_xlabel("Wavelength (nm)")
        ax2.set_ylabel("Reflectance")
        ax2.legend(loc='upper right')

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
            plt.tight_layout()
            plt.show()
        else:
            plt.close()
        return
    
class SimulateBackground:
    """
    :param background_fp (str): filepath to a pickle file of water spectra
    :param turbid_fp (str): filepath to a pickle file of turbid spectra
    :param glint_fp (str): filepath to a pickle file of extracted glint
    :param bboxes_list (list of tuple): where each tuple is ((x1,y1),(x2,y2)) to superimpose turbid spectra over water spectra
    :param y_line (float): get spectra on a horizontal cross-section
    :param sigma (int): sigma for gaussian filtering to blend turbid into background spectra
    :param x_range (tuple or list): set x_limit for displaying spectra
    :TODO randomly generate integers within the shape of im_aligned to randomly generate tuples
    :TODO add glint spectra on top of background spectra instead of replacing it
    """
    def __init__(self,background_fp,
                 turbid_fp,
                 glint_fp,
                 bboxes_list,y_line=None,sigma=20,x_range=None):
        
        self.background_fp = background_fp
        self.turbid_fp = turbid_fp
        self.glint_fp = glint_fp
        self.bboxes_list = bboxes_list
        self.y_line = y_line
        self.sigma = sigma
        self.x_range = x_range
    
    def correction_iterative(self,glint_image,iter=3):
        fig, axes = plt.subplots(iter+1,1)
        axes[0].imshow(np.take(glint_image,[2,1,0],axis=2))
        axes[0].set_title(f'Original image (var: {np.var(glint_image):.4f})')
        axes[0].axis('off')
        for i in range(iter):
            HM = HedleyMulti.HedleyMulti(glint_image,None)
            corrected_bands = HM.get_corrected_bands(plot=False)
            glint_image = np.stack(corrected_bands,axis=2)
            axes[i+1].set_title(f'after var: {np.var(glint_image):.4f}')
            axes[i+1].imshow(np.take(glint_image,[2,1,0],axis=2))
            axes[i+1].axis('off')
        plt.show()
        return glint_image

    def simulation(self,iter=1):
        """
        :param iter (int): number of iterations to run the correction
        returns simulated_background, simulated_glint, and corrected_img
        """
        water_spectra = mutils.load_pickle(self.background_fp)
        turbid_spectra = mutils.load_pickle(self.turbid_fp)
        glint = mutils.load_pickle(self.glint_fp)

        nrow,ncol = glint.shape[0],glint.shape[1]

        water_spectra = np.tile(water_spectra,(nrow,ncol,1))
        
        for bbox in self.bboxes_list:
            ((x1,y1),(x2,y2)) = bbox
            water_spectra[y1:y2,x1:x2,:] = turbid_spectra
        
        for i in range(water_spectra.shape[-1]):
            water_spectra[:,:,i] = gaussian_filter(water_spectra[:,:,i],sigma=self.sigma)
        
        # add simulated glint, add the signal from glint + background water spectra
        simulated_glint = water_spectra.copy()
        
        simulated_glint[glint>0] = glint[glint>0] + water_spectra[glint>0]

        # apply SUGAR on simulated glint
        # get corrected_bands
        corrected_bands = self.correction_iterative(simulated_glint,iter=iter)
        # HM = HedleyMulti.HedleyMulti(simulated_glint,None, sigma=1)
        # corrected_bands = HM.get_corrected_bands(plot=False)
        # corrected_bands = np.stack(corrected_bands,axis=2)

        im_list = {'Simulated background':water_spectra,
                'Simulated glint': simulated_glint,
                'Corrected for glint': corrected_bands}
        
        rgb_bands = [2,1,0]

        fig, axes = plt.subplots(3,2,figsize=(11,13))

        for i,(title, im) in enumerate(im_list.items()):
            im_cropped = np.take(im,rgb_bands,axis=2)
            axes[i,0].imshow(im_cropped)
            axes[i,0].plot([0,ncol-1],[self.y_line]*2,c='r',ls='--',alpha=0.5)
            for j,c in zip(rgb_bands,['r','g','b']):
                # plot reflectance for each band along red line
                axes[i,1].plot(list(range(ncol)),im[self.y_line,:,j],c=c,alpha=0.5)
            for ax in axes[i,:]:
                ax.set_title(title)

        # for ax in axes[:,0]:
        #     ax.axis('off')
        for ax in axes[:,1]:
            ax.set_xlabel("Pixels along red line")
            ax.set_ylabel("Reflectance")
            ax.set_ylim(0,1)
            if self.x_range is not None:
                ax.set_xlim(self.x_range[0],self.x_range[1])
                

        plt.tight_layout()
        plt.show()

        return im_list

        