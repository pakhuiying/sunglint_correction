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
import algorithms.SUGAR as sugar
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
        HM = sugar.SUGAR(simulated_glint,None, sigma=sigma)
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
    """
    def __init__(self,background_fp,
                 turbid_fp,
                 glint_fp,
                 bboxes_list,
                 estimate_background=True,
                 iter=3,
                 y_line=None,sigma=20,x_range=None):
        
        self.background_fp = background_fp
        self.turbid_fp = turbid_fp
        self.glint_fp = glint_fp
        self.bboxes_list = bboxes_list
        self.estimate_background = estimate_background
        self.iter = iter
        self.y_line = y_line
        self.sigma = sigma
        self.x_range = x_range
    
    def correction_iterative(self,glint_image,bounds = [(1,2)]*10,plot=False):
        for i in range(self.iter):
            HM = sugar.SUGAR(glint_image,bounds,estimate_background=self.estimate_background)
            corrected_bands = HM.get_corrected_bands()
            glint_image = np.stack(corrected_bands,axis=2)
            if plot is True:
                plt.figure()
                plt.title(f'after var: {np.var(glint_image):.4f}')
                plt.imshow(np.take(glint_image,[2,1,0],axis=2))
                plt.axis('off')
                plt.show()
            b_list = HM.b_list
            bounds = [(1,b*1.2) for b in b_list]
        
        return glint_image

    def simulate_background(self):
        """
        :param iter (int): number of iterations to run the correction
        returns simulated_background, simulated_glint
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

        return {'Simulated background':water_spectra,
                'Simulated glint': simulated_glint}
    
    def simulation(self):
        """
        :param iter (int): number of iterations to run the correction
        returns simulated_background, simulated_glint, and corrected_img
        """
        simulated_im = self.simulate_background()
        water_spectra = simulated_im['Simulated background']
        simulated_glint = simulated_im['Simulated glint']

        nrow,ncol = simulated_glint.shape[0],simulated_glint.shape[1]
        # apply SUGAR on simulated glint
        # get corrected_bands
        corrected_bands = self.correction_iterative(simulated_glint)
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


class EvaluateCorrection:
    def __init__(self,glint_im,corrected_glint,glint_mask=None,no_glint=None):
        """
        :param glint_im (np.ndarray): original image with glint
        :param corrected_glint (np.ndarray): image corrected for glint
        :param glint_mask (np.ndarray): glint mask where 1 is glint and 0 is non-glint
        :param no_glint (np.ndarray): Optional ground-truth input, only if the images are simulated
        """
        self.glint_im = glint_im
        self.corrected_glint = corrected_glint
        self.glint_mask = glint_mask
        self.no_glint = no_glint
        self.n_bands = glint_im.shape[-1]
        # import wavelengths for each band
        wavelengths = mutils.sort_bands_by_wavelength()
        self.wavelength_dict = {i[0]:i[1] for i in wavelengths}

    def correction_stats(self,bbox):
        """
        :param bbox (tuple): ((x1,y1),(x2,y2)), where x1,y1 is the upper left corner, x2,y2 is the lower right corner
        Show corrected and original rgb image, mean reflectance
        """
        ((x1,y1),(x2,y2)) = bbox
        
        rgb_bands = [2,1,0]

        fig, axes = plt.subplots(2,4,figsize=(14,10))
        #non-corrected images and reflectance for bbox
        rgb_im = np.take(self.glint_im,rgb_bands,axis=2)
        glint_area = self.glint_im[y1:y2,x1:x2,:]
        avg_reflectance = [np.mean(glint_area[:,:,band_number]) for band_number in range(self.n_bands)]
        
        #corrected images and reflectance for bbox
        rgb_im_corrected = np.stack([self.corrected_glint[:,:,i] for i in rgb_bands],axis=2)
        avg_reflectance_corrected = [np.mean(self.corrected_glint[y1:y2,x1:x2,band_number]) for band_number in range(self.n_bands)]
        
        # plot original rgb
        axes[0,0].imshow(rgb_im)
        axes[0,0].set_title('Original RGB')
        coord, w, h = mutils.bboxes_to_patches(bbox)
        rect = patches.Rectangle(coord, w, h, linewidth=1, edgecolor='red', facecolor='none')
        axes[0,0].add_patch(rect)
        # plot corrected rgb
        axes[0,1].imshow(rgb_im_corrected)
        axes[0,1].set_title('Corrected RGB')
        rect = patches.Rectangle(coord, w, h, linewidth=1, edgecolor='red', facecolor='none')
        axes[0,1].add_patch(rect)
        # reflectance
        axes[0,2].plot(list(self.wavelength_dict.values()),[avg_reflectance[i] for i in list(self.wavelength_dict)], label='Original')
        axes[0,2].plot(list(self.wavelength_dict.values()),[avg_reflectance_corrected[i] for i in list(self.wavelength_dict)], label='Corrected')
        axes[0,2].set_xlabel('Wavelengths (nm)')
        axes[0,2].set_ylabel('Reflectance')
        axes[0,2].legend(loc='upper right')
        axes[0,2].set_title(r'$R_T(\lambda)$ and $R_T(\lambda)\prime$')

        residual = [og-cor for og,cor in zip(avg_reflectance,avg_reflectance_corrected)]
        axes[0,3].plot(list(self.wavelength_dict.values()),[residual[i] for i in list(self.wavelength_dict)])
        axes[0,3].set_xlabel('Wavelengths (nm)')
        axes[0,3].set_ylabel('Residual in Reflectance')
        axes[0,3].set_title(r'$R_T(\lambda) - R_T(\lambda)\prime$')

        for ax in axes[0,:2]:
            ax.axis('off')
        
        h = y2 - y1
        w = x2 - x1

        axes[1,0].imshow(rgb_im[y1:y2,x1:x2,:])
        axes[1,1].imshow(rgb_im_corrected[y1:y2,x1:x2,:])
        axes[1,0].set_title('Original Glint')
        axes[1,1].set_title('Corrected Glint')

        axes[1,0].plot([0,w],[h//2,h//2],color="red",linewidth=3,alpha=0.5)
        axes[1,1].plot([0,w],[h//2,h//2],color="red",linewidth=3,alpha=0.5)
        
        # for ax in axes[1,0:2]:
        #     ax.axis('off')

        for i,c in zip(range(3),['r','g','b']):
            # plot for original
            axes[1,2].plot(list(range(w)),rgb_im[y1:y2,x1:x2,:][h//2,:,i],c=c,alpha=0.5,label=c)
            # plot for corrected reflectance
            axes[1,3].plot(list(range(w)),rgb_im_corrected[y1:y2,x1:x2,:][h//2,:,i],c=c,alpha=0.5,label=c)

        for ax in axes[1,2:]:
            ax.set_xlabel('Width of image')
            ax.set_ylabel('Reflectance')
            ax.legend(loc="upper right")

        axes[1,2].set_title(r'$R_T(\lambda)$ along red line')
        axes[1,3].set_title(r'$R_T(\lambda)\prime$ along red line')

        plt.tight_layout()
        plt.show()

        return
    
    def glint_vs_corrected(self):
        """ plot scatter plot of glint correction vs glint magnitude"""
        n = self.glint_im.shape[-1]
        simulated_background = self.no_glint
        simulated_glint = self.glint_im
        
        if self.no_glint is None:
            fig, axes = plt.subplots(n,1,figsize=(7,20))
        else:
            fig, axes = plt.subplots(n,4,figsize=(15,25))
            
        for i in range(n):
            if self.glint_mask is not None:
                gm = self.glint_mask[:,:,i]
            # check how much glint has been removed
            correction_mag = simulated_glint[:,:,i] - self.corrected_glint[:,:,i]
            if self.glint_mask is None:
                # glint + water background
                extracted_glint = simulated_glint[:,:,i].flatten()
                extracted_correction = correction_mag.flatten()
            else:
                extracted_glint = simulated_glint[:,:,i][gm!=0]
                extracted_correction = correction_mag[gm!=0]

            if self.no_glint is not None:
                # actual glint contribution
                glint_original_glint = simulated_glint[:,:,i] - simulated_background[:,:,i]
                # how much glint is under/overcorrected i.e. ground truth vs corrected
                residual_glint = self.corrected_glint[:,:,i] - simulated_background[:,:,i]
                if self.glint_mask is None:
                    extracted_original_glint = glint_original_glint.flatten()
                    extracted_residual_glint = residual_glint.flatten()
                else:
                    extracted_original_glint = glint_original_glint[gm!=0]
                    extracted_residual_glint = residual_glint[gm!=0]
                # colors are indicated by residual glint
                # check how much glint has been removed
                im = axes[i,0].scatter(extracted_glint,extracted_correction,c=extracted_residual_glint,alpha=0.3,s=1)
                fig.colorbar(im, ax=axes[i,0])
                # check how much glint has been removed vs actual glint contribution
                axes[i,1].scatter(extracted_glint,extracted_original_glint,c=extracted_residual_glint,alpha=0.3,s=1)
                axes[i,1].set_xlabel('Glint magnitude')
                axes[i,1].set_ylabel('Glint contribution')

                axes[i,2].imshow(simulated_glint[:,:,i])
                axes[i,2].set_title(f'Original image (Band {i})')

                im = axes[i,3].imshow(residual_glint,interpolation='none')
                fig.colorbar(im, ax=axes[i,3])
                axes[i,3].set_title(f'Residual glint (Band {i})')

                axes[i,0].set_title(f'Band {i}')
                axes[i,0].set_xlabel('Glint magnitude')
                axes[i,0].set_ylabel('Glint Correction')
            
            else:
                axes[i].scatter(extracted_glint,extracted_correction,s=1)
                axes[i].set_title(f'Band {i}')
                axes[i].set_xlabel('Glint magnitude')
                axes[i].set_ylabel('Glint Correction')
            
        plt.tight_layout()
        plt.show()
        return 
    
def compare_correction_algo(im_aligned,bbox,iter=3):
    """
    :param im_aligned (np.ndarray): reflectance image
    :param bbox (tuple): bbox over glint area for Hedley algorithm
    :param iter (int): number of iterations for SUGAR algorithm
    compare SUGAR and Hedley algorithm
    """
    HH = Hedley.Hedley(im_aligned,bbox,smoothing=False,glint_mask=False)
    corrected_Hedley = HH.get_corrected_bands(plot=False)
    corrected_Hedley = np.stack(corrected_Hedley,axis=2)

    corrected_bands = sugar.correction_iterative(im_aligned,iter=iter,plot=False)
    corrected_SUGAR = corrected_bands[-1]

    rgb_bands = [2,1,0]
    fig, axes = plt.subplots(1,3,figsize=(12,5))
    im_list = [im_aligned,corrected_Hedley,corrected_SUGAR]
    title_list = ['Original','Hedley',f'SUGAR (iters: {iter})']
    for im, title, ax in zip(im_list,title_list,axes.flatten()):
        ax.imshow(np.take(im,rgb_bands,axis=2))
        ax.set_title(title + r'($\sigma^2_T$' + f': {np.var(im):.4f})')
        ax.axis('off')

    coord, w, h = mutils.bboxes_to_patches(bbox)
    rect = patches.Rectangle(coord, w, h, linewidth=1, edgecolor='red', facecolor='none')
    axes[1].add_patch(rect)
    plt.show()
    return