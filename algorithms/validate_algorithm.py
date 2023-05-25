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
from algorithms.GLORIA import GloriaSimulate
from skimage.transform import resize
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
    :param glint_fp (str): filepath to a pickle file of extracted glint
    # image fine-tuning
    :param fp_rrs (str): folder path to GLORIA Rrs dataset (multiply by pi to get surface reflectance)
    :param fp_meta (str): folder path to metadata
    :param water_type (list of int): where...
        1: sediment-dominated
        2: chl-dominated
        3: CDOM-dominated
        4: Chl+CDOM-dominated
        5: Moderate turbid coastal (e.g., 0.3<TSS<1.2 & 0.5 <Chl<2.0)
        6: Clear (e.g., TSS<0.3 & 0.1<Chl<0.7)
    :param sigma (int): sigma for gaussian filtering to blend background spectra
    :param n_rrs (int): number of distinct Rrs observation
    :param scale (float): scale Rrs by a factor
    :param set_seed (bool): to ensure replicability if needed
    # image distortion
    :param rotation (float): Additional rotation applied to the image.
    :param strength (float): The amount of swirling applied.
    :param radius (float): The extent of the swirl in pixels. The effect dies out rapidly beyond radius.
    # SUGAR parameters
    :param estimate_background (bool): parameter in SUGAR, whether to estimate the underlying background
    :param iter (int): number of iterations to run SUGAR algorithm
    # plotting parameters
    :param y_line (float): get spectra on a horizontal cross-section
    :param x_range (tuple or list): set x_limit for displaying spectra
    :TODO randomly generate integers within the shape of im_aligned to randomly generate tuples
    """
    def __init__(self,
                 glint_fp,
                 fp_rrs, fp_meta, water_type, sigma=10, n_rrs=5, scale=5, set_seed=False,# image fine-tuning
                 rotation=90,strength=10,radius=120,# image distortion
                 estimate_background=True, iter=3, # parameters in SUGAR
                 y_line=None,x_range=None):
        
        self.glint_fp = glint_fp
        self.fp_rrs = fp_rrs
        self.fp_meta = fp_meta
        self.water_type = water_type
        self.sigma = sigma
        # image fine-tuning
        self.n_rrs = n_rrs
        self.scale = scale
        self.set_seed = set_seed
        # image distortion
        self.rotation = rotation
        self.strength = strength
        self.radius = radius
        # SUGAR parameters
        self.estimate_background = estimate_background
        self.iter = iter
        # plotting parameters
        self.y_line = y_line
        self.x_range = x_range
        # wavelengths
        self.rgb_bands = [2,1,0]
        wavelengths = mutils.sort_bands_by_wavelength()
        self.wavelength_dict = {i[0]:i[1] for i in wavelengths}
    
    def correction_iterative(self,glint_image,bounds = [(1,2)],plot=False):
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
            # b_list = HM.b_list
            # bounds = [(1,b*1.2) for b in b_list]
        
        return glint_image

    def simulate_background(self, plot = False):
        """
        returns simulated_background, simulated_glint
        """
        glint = mutils.load_pickle(self.glint_fp)

        nrow,ncol,n_bands = glint.shape

        G = GloriaSimulate(self.fp_rrs,self.fp_meta,self.water_type,self.sigma)

        if self.set_seed:
            np.random.seed(1)

        n_rrs = np.random.randint(self.n_rrs)
        scale = np.random.randint(self.scale)
        rotation = np.random.randint(self.rotation)
        strength = np.random.randint(self.strength)
        radius = np.random.randint(self.radius)
        print(f"n_rrs:{n_rrs}, scale:{scale}, rotation: {rotation}, strength: {strength}, radius: {radius}")
        
        im = G.get_image(n_rrs=n_rrs,scale=scale,plot=False,set_seed=self.set_seed)
        im = G.image_distortion(im,rotation=rotation,strength=strength,radius=radius,plot=False)
        water_spectra = resize(im,(nrow,ncol,n_bands),anti_aliasing=True) # water spectra in wavelength order

        #change to band order
        band_idx = {b:i for i,b in enumerate(self.wavelength_dict.keys())}
        band_idx = [band_idx[i] for i in range(len(band_idx.values()))]
        water_spectra = np.take(water_spectra,band_idx,axis=2)

        if plot is True:
            plt.figure()
            plt.imshow(np.take(water_spectra,[2,1,0],axis=2))
            plt.axis('off')
            plt.show()
        return water_spectra
    
    def simulate_glint(self,water_spectra):
        """
        add glint on top of background spectra
        """
        glint = mutils.load_pickle(self.glint_fp)

        nrow,ncol = glint.shape[0],glint.shape[1]

        assert (nrow == water_spectra.shape[0]) and (ncol == water_spectra.shape[1])
        # add simulated glint, add the signal from glint + background water spectra
        simulated_glint = water_spectra.copy()
        
        simulated_glint[glint>0] = glint[glint>0] + water_spectra[glint>0]

        return {'Simulated background':water_spectra,
                'Simulated glint': simulated_glint}
    
    def simulation(self,iter=3,bounds= [(1,2)]):
        """
        :param iter (int): number of iterations to run the correction
        returns simulated_background, simulated_glint, and corrected_img
        """
        simulated_im = self.simulate_background()
        simulated_im = self.simulate_glint(simulated_im)
        water_spectra = simulated_im['Simulated background']
        simulated_glint = simulated_im['Simulated glint']

        nrow,ncol = simulated_glint.shape[0],simulated_glint.shape[1]
        corrected_bands = sugar.correction_iterative(simulated_glint, iter=iter, bounds = bounds,estimate_background=False,get_glint_mask=False)
        corrected_bands_background = sugar.correction_iterative(simulated_glint, iter=iter, bounds = bounds,estimate_background=True,get_glint_mask=False)

        im_list = {'R_BG':water_spectra,
                'R_T': simulated_glint,
                'R_prime_T': corrected_bands[-1],
                'R_prime_T_BG':corrected_bands_background[-1]}

        fig, axes = plt.subplots(3,4,figsize=(12,8))

        titles = [r'$R_{BG}$',r'$R_T$',r'$R_T\prime$',r'$R_{T,BG}\prime$']
        x = list(self.wavelength_dict.values())
        # water spectra
        og_avg_reflectance = [np.mean(water_spectra[:,:,band_number]) for band_number in range(simulated_glint.shape[-1])]
        og_y = [og_avg_reflectance[i] for i in list(self.wavelength_dict)]
        # simulated_glint
        sim_avg_reflectance = [np.mean(simulated_glint[:,:,band_number]) for band_number in range(simulated_glint.shape[-1])]
        sim_y = [sim_avg_reflectance[i] for i in list(self.wavelength_dict)]

        for i,(title, im) in enumerate(zip(titles,im_list.values())):
            rgb_im = np.take(im,self.rgb_bands,axis=2)
            axes[0,i].imshow(rgb_im)
            axes[0,i].plot([0,ncol-1],[self.y_line]*2,c='r',linewidth=3,alpha=0.5)
            axes[0,i].set_title(title + r'($\sigma^2_T$' + f': {np.var(im):.4f})')
            # plot original reflectance
            axes[1,i].plot(x,og_y,label=r'$R_{BG}(\lambda)$')
            # plot simulated glint reflectance
            axes[1,i].plot(x,sim_y,label=r'$R_T(\lambda)$')
            # plot corrected reflectance
            if i > 1:
                avg_reflectance = [np.mean(im[:,:,band_number]) for band_number in range(im.shape[-1])]
                y = [avg_reflectance[i] for i in list(self.wavelength_dict)]
                axes[1,i].plot(x,y,label=r'$R_T(\lambda)\prime$')

            for j,c in zip(self.rgb_bands,['r','g','b']):
                # plot reflectance for each band along red line
                axes[2,i].plot(list(range(ncol)),im[self.y_line,:,j],c=c,alpha=0.5,label=c)
                

        y1,y2 = axes[1,1].get_ylim()
        for i,ax in enumerate(axes[1,:]):
            ax.set_title(titles[i] + " (Mean Reflectance)")
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Reflectance")
            ax.set_ylim(y1,y2)
            ax.legend(loc='upper right')

        y1,y2 = axes[2,1].get_ylim()
        for i,ax in enumerate(axes[2,:]):
            ax.set_title(titles[i]+" (Pixels along red line)")
            ax.set_xlabel("Image position")
            ax.set_ylabel("Reflectance")
            ax.set_ylim(y1,y2)
            ax.legend(loc='upper right')

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
        coord, w, h = mutils.bboxes_to_patches(bbox)
        rgb_bands = [2,1,0]

        fig, axes = plt.subplots(2,4,figsize=(14,10))
        for im, title, ax in zip([self.glint_im,self.corrected_glint],['Original RGB','Corrected RGB'],axes[0,:2]):
            ax.imshow(np.take(im,rgb_bands,axis=2))
            ax.set_title(title)
            ax.axis('off')
            rect = patches.Rectangle(coord, w, h, linewidth=1, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

        if self.no_glint is not None:
            avg_reflectance = [np.mean(self.no_glint[y1:y2,x1:x2,band_number]) for band_number in range(self.glint_im.shape[-1])]
            axes[0,2].plot(list(self.wavelength_dict.values()),[avg_reflectance[i] for i in list(self.wavelength_dict)],label=r'$R_{BG}$')

        for im,label in zip([self.glint_im,self.corrected_glint],[r'$R_T$',r'$R_T\prime$']):
            avg_reflectance = [np.mean(im[y1:y2,x1:x2,band_number]) for band_number in range(self.glint_im.shape[-1])]
            axes[0,2].plot(list(self.wavelength_dict.values()),[avg_reflectance[i] for i in list(self.wavelength_dict)],label=label)

        axes[0,2].set_xlabel('Wavelengths (nm)')
        axes[0,2].set_ylabel('Reflectance')
        axes[0,2].legend(loc='upper right')
        axes[0,2].set_title(r'$R_T(\lambda)$ and $R_T(\lambda)\prime$')

        residual = self.glint_im - self.corrected_glint
        residual = np.mean(residual[y1:y2,x1:x2,:],axis=(0,1))
        axes[0,3].plot(list(self.wavelength_dict.values()),[residual[i] for i in list(self.wavelength_dict)])
        axes[0,3].set_xlabel('Wavelengths (nm)')
        axes[0,3].set_ylabel('Reflectance difference')
        axes[0,3].set_title(r'$R_T(\lambda) - R_T(\lambda)\prime$')
        
        h = y2 - y1
        w = x2 - x1

        for i, (im, title, ax) in enumerate(zip([self.glint_im,self.corrected_glint],[r'$R_T$',r'$R_T\prime$'],axes[1,:2])):
            rgb_cropped = np.take(im[y1:y2,x1:x2,:],rgb_bands,axis=2)
            ax.imshow(rgb_cropped)
            ax.set_title(title)
            # ax.axis('off')
            ax.plot([0,w-1],[h//2,h//2],color="red",linewidth=3,alpha=0.5)
            for j,c in enumerate(['r','g','b']):
                axes[1,i+2].plot(list(range(w)),rgb_cropped[h//2,:,j],c=c,alpha=0.5,label=c)
            axes[1,i+2].set_xlabel('Image position')
            axes[1,i+2].set_ylabel('Reflectance')
            axes[1,i+2].legend(loc="upper right")
            axes[1,i+2].set_title(f'{title} along red line')
        
        y1,y2 = axes[1,2].get_ylim()
        axes[1,3].set_ylim(y1,y2)
        plt.tight_layout()
        plt.show()

        return
    
    def glint_vs_corrected(self):
        """ plot scatter plot of glint correction vs glint magnitude"""
        simulated_background = self.no_glint
        simulated_glint = self.glint_im
        
        if self.no_glint is None:
            fig, axes = plt.subplots(self.n_bands,1,figsize=(7,20))
        else:
            fig, axes = plt.subplots(self.n_bands,4,figsize=(15,25))
            
        for i, (band_number, wavelength) in enumerate(self.wavelength_dict.items()):
            if self.glint_mask is not None:
                gm = self.glint_mask[:,:,band_number]
            # check how much glint has been removed
            correction_mag = simulated_glint[:,:,band_number] - self.corrected_glint[:,:,band_number]
            if self.glint_mask is None:
                # glint + water background
                extracted_glint = simulated_glint[:,:,band_number].flatten()
                extracted_correction = correction_mag.flatten()
            else:
                extracted_glint = simulated_glint[:,:,band_number][gm!=0]
                extracted_correction = correction_mag[gm!=0]

            if self.no_glint is not None:
                # actual glint contribution
                glint_original_glint = simulated_glint[:,:,band_number] - simulated_background[:,:,band_number]
                # how much glint is under/overcorrected i.e. ground truth vs corrected
                residual_glint = self.corrected_glint[:,:,band_number] - simulated_background[:,:,band_number]
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
                axes[i,1].set_title(f'Glint contribution ({wavelength} nm)')
                axes[i,1].set_xlabel(r'$R_T$')
                axes[i,1].set_ylabel(r'$R_T - R_{BG}$')

                axes[i,2].imshow(simulated_glint[:,:,band_number])
                axes[i,2].set_title(r'$R_T$' + f' ({wavelength} nm)')
                axes[i,2].axis('off')

                im = axes[i,3].imshow(residual_glint,interpolation='none')
                fig.colorbar(im, ax=axes[i,3])
                axes[i,3].set_title(r'$R_T\prime - R_{BG}$' + f' ({wavelength} nm)')
                axes[i,3].axis('off')

                axes[i,0].set_title(f'Glint correction ({wavelength} nm)')
                axes[i,0].set_xlabel(r'$R_T$')
                axes[i,0].set_ylabel(r'$R_T - R_T\prime$')
            
            else:
                axes[i].scatter(extracted_glint,extracted_correction,s=1)
                axes[i].set_title(f'{wavelength} nm')
                axes[i].set_xlabel('Glint magnitude')
                axes[i].set_ylabel(r'$R_T - R_T\prime$')
            
        plt.tight_layout()
        plt.show()
        return 

def compare_plots(im_list, title_list, bbox=None):
    """
    :param im_list (list of np.ndarray): where the first item is always the original image
    :param title_list (list of str): the title for the first row
    :param bbox (tuple): bbox over glint area for Hedley algorithm
    """
    rgb_bands = [2,1,0]
    wavelengths = mutils.sort_bands_by_wavelength()
    wavelength_dict = {i[0]:i[1] for i in wavelengths}
    nrow, ncol, n_bands = im_list[0].shape

    plot_width = len(im_list)*3
    if bbox is not None:
        ((x1,y1),(x2,y2)) = bbox
        coord, w, h = mutils.bboxes_to_patches(bbox)
        plot_height = 14
        plot_row = 4
    else:
        plot_height = 11
        plot_row = 3

    fig, axes = plt.subplots(plot_row,len(im_list),figsize=(plot_width,plot_height))

    og_avg_reflectance = [np.mean(im_list[0][y1:y2,x1:x2,band_number]) for band_number in range(n_bands)]
    x = list(wavelength_dict.values())
    og_y = [og_avg_reflectance[i] for i in list(wavelength_dict)]

    for i,(im, title) in enumerate(zip(im_list,title_list)): #iterate acoss column
        # plot image
        rgb_im = np.take(im,rgb_bands,axis=2)
        axes[0,i].imshow(rgb_im)
        axes[0,i].set_title(title + r'($\sigma^2_T$' + f': {np.var(im):.4f})')
        axes[0,i].axis('off')
        if bbox is not None:
            rect = patches.Rectangle(coord, w, h, linewidth=1, edgecolor='red', facecolor='none')
            axes[0,i].add_patch(rect)
        # plot original reflectance
        axes[1,i].plot(x,og_y,label=r'$R_T(\lambda)$')
        # plot corrected reflectance
        if i > 0:
            if bbox is not None:
                avg_reflectance = [np.mean(im[y1:y2,x1:x2,band_number]) for band_number in range(n_bands)]
            else:
                avg_reflectance = [np.mean(im) for band_number in range(n_bands)]
            y = [avg_reflectance[i] for i in list(wavelength_dict)]
            axes[1,i].plot(x,y,label=r'$R_T(\lambda)\prime$')
        axes[1,i].legend(loc="upper right")
        axes[1,i].set_title(r'$R_T(\lambda)\prime$'+' in AOI')
        axes[1,i].set_xlabel('Wavelengths (nm)')
        axes[1,i].set_ylabel('Reflectance')

        # plot cropped rgb
        rgb_cropped = rgb_im[y1:y2,x1:x2,:] if bbox is not None else rgb_im
        if bbox is not None:
            axes[2,i].imshow(rgb_cropped)
            axes[2,i].set_title('AOI')
            axes[2,i].plot([0,w-1],[abs(h)//2,abs(h)//2],color="red",linewidth=3,alpha=0.5)
        
        row_idx = 3 if bbox is not None else 2
        h = nrow if bbox is None else h
        # plot reflectance along red line
        for j,c in enumerate(['r','g','b']):
            axes[row_idx,i].plot(list(range(w)),rgb_cropped[h//2,:,j],c=c,alpha=0.5,label=c)
        axes[row_idx,i].set_xlabel('Image position')
        axes[row_idx,i].set_ylabel('Reflectance')
        axes[row_idx,i].legend(loc="upper right")
        axes[row_idx,i].set_title(r'$R_T(\lambda)\prime$'+' along red line')
    
    y1,y2 = axes[row_idx,0].get_ylim()
    for i in range(len(im_list)):
        axes[row_idx,i].set_ylim(y1,y2)
    
    plt.tight_layout()
    plt.show()
    return


def compare_sugar_algo(im_aligned,bbox=None,corrected = None, corrected_background = None, iter=3, bounds=[(1,2)]):
    """
    :param im_aligned (np.ndarray): reflectance image
    :param bbox (tuple): bbox over glint area for Hedley algorithm
    :param corrected (np.ndarray): corrected for glint without taking into account of background
    :param corrected_background (np.ndarray): corrected for glint taking into account of background
    :param iter (int): number of iterations for SUGAR algorithm
    compare SUGAR algorithm, whether to take into account of background spectra
    returns a tuple (corrected, corrected_background)
    """
    if corrected is None:
        corrected = sugar.correction_iterative(im_aligned, iter=iter, bounds = bounds,estimate_background=False,get_glint_mask=False)
    if corrected_background is None:
        corrected_background = sugar.correction_iterative(im_aligned, iter=iter, bounds = bounds,estimate_background=True,get_glint_mask=False)

    im_list = [im_aligned,corrected[-1],corrected_background[-1]]
    title_list = [r'$R_T$',r'$R_T\prime$',r'$R_{T,BG}\prime$']
    
    compare_plots(im_list, title_list, bbox)
    return (corrected,corrected_background)

def compare_correction_algo(im_aligned,bbox,corrected_Hedley = None, corrected_SUGAR = None,  iter=3):
    """
    :param im_aligned (np.ndarray): reflectance image
    :param bbox (tuple): bbox over glint area for Hedley algorithm
    :param iter (int): number of iterations for SUGAR algorithm
    compare SUGAR and Hedley algorithm
    """
    if corrected_Hedley is None:
        HH = Hedley.Hedley(im_aligned,bbox,smoothing=False,glint_mask=False)
        corrected_Hedley = HH.get_corrected_bands(plot=False)
        corrected_Hedley = np.stack(corrected_Hedley,axis=2)

    if corrected_SUGAR is None:
        corrected_bands = sugar.correction_iterative(im_aligned,iter=iter,bounds = [(1,2)],estimate_background=True,get_glint_mask=False,plot=False)
        corrected_SUGAR = corrected_bands[-1]

    rgb_bands = [2,1,0]
    ((x1,y1),(x2,y2)) = bbox
    coord, w, h = mutils.bboxes_to_patches(bbox)
    wavelengths = mutils.sort_bands_by_wavelength()
    wavelength_dict = {i[0]:i[1] for i in wavelengths}

    
    im_list = [im_aligned,corrected_Hedley,corrected_SUGAR]
    title_list = ['Original','Hedley',f'SUGAR (iters: {iter})']

    og_avg_reflectance = [np.mean(im_aligned[y1:y2,x1:x2,band_number]) for band_number in range(im_aligned.shape[-1])]
    x = list(wavelength_dict.values())
    og_y = [og_avg_reflectance[i] for i in list(wavelength_dict)]
        
    fig, axes = plt.subplots(4,len(im_list),figsize=(12,14))
    for i,(im, title) in enumerate(zip(im_list,title_list)): #iterate acoss column
        # plot image
        rgb_im = np.take(im,rgb_bands,axis=2)
        axes[0,i].imshow(rgb_im)
        axes[0,i].set_title(title + r'($\sigma^2_T$' + f': {np.var(im):.4f})')
        axes[0,i].axis('off')
        rect = patches.Rectangle(coord, w, h, linewidth=1, edgecolor='red', facecolor='none')
        axes[0,i].add_patch(rect)
        # plot original reflectance
        axes[1,i].plot(x,og_y,label=r'$R_T(\lambda)$')
        # plot corrected reflectance
        if i > 0:
            avg_reflectance = [np.mean(im[y1:y2,x1:x2,band_number]) for band_number in range(im_aligned.shape[-1])]
            y = [avg_reflectance[i] for i in list(wavelength_dict)]
            axes[1,i].plot(x,y,label=r'$R_T(\lambda)\prime$')
        axes[1,i].legend(loc="upper right")
        axes[1,i].set_title(r'$R_T(\lambda)\prime$'+' in AOI')
        axes[1,i].set_xlabel('Wavelengths (nm)')
        axes[1,i].set_ylabel('Reflectance')

        # plot cropped rgb
        rgb_cropped = rgb_im[y1:y2,x1:x2,:]
        axes[2,i].imshow(rgb_cropped)
        axes[2,i].set_title('AOI')
        axes[2,i].plot([0,w-1],[abs(h)//2,abs(h)//2],color="red",linewidth=3,alpha=0.5)
        # plot reflectance along red line
        for j,c in enumerate(['r','g','b']):
            axes[3,i].plot(list(range(w)),rgb_cropped[h//2,:,j],c=c,alpha=0.5,label=c)
        axes[3,i].set_xlabel('Image position')
        axes[3,i].set_ylabel('Reflectance')
        axes[3,i].legend(loc="upper right")
        axes[3,i].set_title(r'$R_T(\lambda)\prime$'+' along red line')
    
    y1,y2 = axes[3,0].get_ylim()
    for i in range(len(im_list)):
        axes[3,i].set_ylim(y1,y2)
    
    plt.tight_layout()
    plt.show()

    return