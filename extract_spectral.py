import micasense.imageset as imageset
import micasense.capture as capture
import cv2
import micasense.imageutils as imageutils
import micasense.plotutils as plotutils
import os, glob
import json
import tqdm
import pickle #This library will maintain the format as well
import importlib
import radiometric_calib_utils
import mutils
importlib.reload(radiometric_calib_utils)
importlib.reload(mutils)
import radiometric_calib_utils as rcu
import mutils
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import numpy as np
from math import ceil


class ExtractSpectral:
    def __init__(self,dir):
        """ 
        :param dir (str): directory of where all the bboxes (.txt) are stored i.e. saved_bboxes/
        """
        self.dir = dir
        # initialise categories
        self.button_names = ['turbid_glint','water_glint','turbid','water','shore']
        # intialise colours
        self.colors = ['orange','cyan','saddlebrown','blue','yellow']
        self.color_mapping = {categories:colors for categories,colors in zip(self.button_names,self.colors)}
        # import wavelengths for each band
        wavelengths = mutils.sort_bands_by_wavelength()
        self.wavelength_dict = {i[0]:i[1] for i in wavelengths}
        self.n_bands = len(wavelengths)
        self.wavelengths_idx = np.array([i[0] for i in wavelengths])
        self.wavelengths = np.array([i[1] for i in wavelengths])
        # aligning band images
        self.warp_mode = cv2.MOTION_HOMOGRAPHY
        self.img_type = "reflectance"

    def store_bboxes(self):
        """ 
        get all the bboxes txt files are store the info in a dictionary with keys:
        parent_directory (e.g. 'F:/surveys_10band/10thSur24Aug/F1/RawImg')
            img_names (e.g. 'IMG_0004_1.tif')
        """
        fp_list = [os.path.join(self.dir,fp) for fp in os.listdir(self.dir)]
        store_dict = dict()
        for fp in fp_list:
            with open(fp, 'r') as fp:
                data = json.load(fp)
            basename,file_name = os.path.split(list(data)[0])
            store_dict[basename] = dict()
        
        for fp in fp_list:
            with open(fp, 'r') as fp:
                data = json.load(fp)
            basename,file_name = os.path.split(list(data)[0])
            bboxes = {k: v for d in data.values() for k,v in d.items() if v is not None}
            if bool(bboxes) is True:
                store_dict[basename][file_name] = bboxes
                # store_dict[basename] = {file_name: bboxes}
            
        return store_dict

    def get_warp_matrices(self,current_fp):
        """ 
        from current_fp, import captures and output warp_matrices and cropped_dimensions
        """
        cap = mutils.import_captures(current_fp)
        warp_matrices = cap.get_warp_matrices()
        cropped_dimensions,_ = imageutils.find_crop_bounds(cap,warp_matrices)
        return warp_matrices, cropped_dimensions

    def plot_bboxes(self,show_n = 6,figsize=(8,20)):
        """ 
        :param dir (str): directory of where all the bboxes (.txt) are stored i.e. saved_bboxes/
        :param show_n (int): show how many plots. if number of images exceeds show_n, plot only show_n
        plots bboxes for greyscale images only
        """
        store_dict = self.store_bboxes()

        for flight_fp, img_dict in store_dict.items():
            images_names = list(img_dict)
            n_images = len(images_names)
            if n_images > 0:
                current_fp = os.path.join(flight_fp,images_names[0])
                warp_matrices, cropped_dimensions = self.get_warp_matrices(current_fp)

                if show_n is None:
                    fig, axes = plt.subplots(ceil(n_images/2),2)

                elif n_images < show_n:
                    fig, axes = plt.subplots(ceil(n_images/2),2,figsize=figsize)
                else:
                    fig, axes = plt.subplots(ceil(show_n/2),2,figsize=figsize)
                    img_dict = {i:img_dict[i] for i in list(images_names)[:show_n]}
                
                for (image_name,bboxes),ax in zip(img_dict.items(),axes.flatten()):
                    current_fp = os.path.join(flight_fp,image_name)
                    cap = mutils.import_captures(current_fp)
                    rgb_image = mutils.aligned_capture_rgb(cap, warp_matrices, cropped_dimensions)
                    ax.imshow(rgb_image)
                    ax.set_title(image_name)
                    ax.axis('off')
                    for categories, bbox in bboxes.items():
                        coord, w, h = mutils.bboxes_to_patches(bbox)
                        rect = patches.Rectangle(coord, w, h, linewidth=1, edgecolor=self.color_mapping[categories], facecolor='none')
                        patch = ax.add_patch(rect)
                fig.suptitle(flight_fp)
                plt.tight_layout()
                plt.show()
    
    def get_multispectral_bboxes(self,parent_dir,img_fp,img_bboxes=None,warp_matrices=None, cropped_dimensions=None):
        """ 
        :param parent_dir (str): folder which contains raw images (keys of store_bboxes()) e.g. F:/surveys_10band/10thSur24Aug/F1/RawImg
        :param img_fp (str): image name e.g. 'IMG_0004_1.tif'
        :param img_bboxes (dict): keys are categories e.g. turbid_glint, turbid, water_glint, water and shore, and values are the corresponding bboxes
        :param warp_matrices (list of arrays): to align band images
        :param cropped_dimensions (tuple): to cropp images for band images alignment
        returns multispectral reflectance (im_aligned), and bboxes of categories
        """
        fp = os.path.join(parent_dir,img_fp)
        cap = mutils.import_captures(fp)
        if warp_matrices is None and cropped_dimensions is None:
            warp_matrices = cap.get_warp_matrices()
            cropped_dimensions,_ = imageutils.find_crop_bounds(cap,warp_matrices)

        im_aligned = imageutils.aligned_capture(cap, warp_matrices, self.warp_mode, cropped_dimensions, None, img_type=self.img_type)
        if img_bboxes is None:
            img_bboxes = self.store_bboxes()
        bboxes = img_bboxes[parent_dir][img_fp]
        return im_aligned, bboxes

    def plot_multispectral(self,parent_dir,img_fp,img_bboxes=None,warp_matrices=None, cropped_dimensions=None):
        """ 
        :param parent_dir (str): folder which contains raw images (keys of store_bboxes()) e.g. F:/surveys_10band/10thSur24Aug/F1/RawImg
        :param img_fp (str): image name e.g. 'IMG_0004_1.tif'
        :param img_bboxes (dict): keys are categories e.g. turbid_glint, turbid, water_glint, water and shore, and values are the corresponding bboxes
        :param warp_matrices (list of arrays): to align band images
        :param cropped_dimensions (tuple): to cropp images for band images alignment
        returns a plot of the rgb_image, drawn bboxes, and spectral reflectance
        """
        im_aligned, bboxes = self.get_multispectral_bboxes(parent_dir,img_fp,img_bboxes,warp_matrices, cropped_dimensions)
        
        n_cats = len(list(bboxes)) # number of categories based on bboxes drawn

        # initialise plot
        fig = plt.figure(figsize=(6, 2*(1+n_cats)), layout="constrained")
        spec = fig.add_gridspec(1+n_cats, 2) #first row is for the image
        # plot rgb
        rgb_image = mutils.get_rgb(im_aligned,plot=False)
        ax0 = fig.add_subplot(spec[0, :])
        ax0.imshow(rgb_image)
        ax0.set_title('Image')
        ax0.set_axis_off()

        for i, (category,bbox) in enumerate(bboxes.items()):
            ((x1,y1),(x2,y2)) = bbox
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1,y2 = y2, y1

            ax_idx = i+1 #axis index
            ax_im = fig.add_subplot(spec[ax_idx, 0]) # axis for adding cropped image based on bbox
            ax_line = fig.add_subplot(spec[ax_idx, 1]) # axis for adding spectral reflectances
            
            # crop rgb image based on bboxes drawn
            im_cropped = rgb_image[y1:y2,x1:x2,:]
            # get multispectral reflectances from bboxes
            # flatten the image such that it has shape (m x n,c), where c is the number of bands
            spectral_flatten = im_aligned[y1:y2,x1:x2,:].reshape(-1,im_aligned.shape[-1])
            # sort the image by wavelengths instead of band numbers
            wavelength_flatten = spectral_flatten[:,self.wavelengths_idx]
            wavelength_mean = np.mean(wavelength_flatten,axis=0)
            wavelength_var = np.sqrt(np.var(wavelength_flatten,axis=0)) #std dev
            
            # add patches to plots
            coord, w, h = mutils.bboxes_to_patches(bbox)
            c = self.color_mapping[category]
            rect = patches.Rectangle(coord, w, h, linewidth=1, edgecolor=c, facecolor='none')
            patch = ax0.add_patch(rect)
            
            ax_im.imshow(im_cropped)
            ax_im.set_title(category)
            ax_im.set_axis_off()
            ax_line.plot(self.wavelengths,wavelength_mean,color=c)
            eb = ax_line.errorbar(self.wavelengths,wavelength_mean,yerr=wavelength_var,color=c)
            eb[-1][0].set_linestyle('--')
            ax_line.set_title(f'Spectra ({category})')
            ax_line.set_xlabel('Wavelength (nm)')
            ax_line.set_ylabel('Reflectance (%)')
        
        plt.show()
        
        return

    def plot_multiline(self,im_aligned,bbox):
        """ 
        plot multispectral reflectance of the image cropped by bbox
        """
        ((x1,y1),(x2,y2)) = bbox
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1,y2 = y2, y1

        def multiline(xs, ys, c, ax=None, **kwargs):
            """Plot lines with different colorings

            Parameters
            ----------
            xs : iterable container of x coordinates
            ys : iterable container of y coordinates
            c : iterable container of numbers mapped to colormap
            ax (optional): Axes to plot on.
            kwargs (optional): passed to LineCollection

            Notes:
                len(xs) == len(ys) == len(c) is the number of line segments
                len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

            Returns
            -------
            lc : LineCollection instance.
            """

            # find axes
            ax = plt.gca() if ax is None else ax

            # create LineCollection
            segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
            lc = LineCollection(segments, **kwargs)

            # set coloring of line segments
            #    Note: I get an error if I pass c as a list here... not sure why.
            lc.set_array(np.asarray(c))

            # add lines to axes and rescale 
            #    Note: adding a collection doesn't autoscalee xlim/ylim
            ax.add_collection(lc)
            ax.autoscale()
            return lc

        # m, n, channels = im_aligned.shape
        spectral_flatten = im_aligned[y1:y2,x1:x2,:].reshape(-1,im_aligned.shape[-1])
        wavelength_flatten = spectral_flatten[:,self.wavelengths_idx]
        wavelength_mean = np.mean(wavelength_flatten,axis=0)
        n_lines = wavelength_flatten.shape[0]
        x_array = np.array(self.wavelengths)
        x = np.repeat(x_array[np.newaxis,:],n_lines,axis=0)
        
        assert x.shape == wavelength_flatten.shape, "x and y should have the same shape"
        c = wavelength_flatten[:,-1] # select the last column which corresponds to NIR band
        
        fig, ax = plt.subplots()
        lc = multiline(x, wavelength_flatten, c, cmap='bwr',alpha=0.3 ,lw=2)
        ax.plot(self.wavelengths,wavelength_mean,color='yellow',label='Mean reflectance')
        axcb = fig.colorbar(lc)
        axcb.set_label('NIR reflectance')
        ax.set_title('Spectra of glint area')
        plt.legend()
        plt.show()
        return (np.min(c),np.mean(c),np.max(c)) #where np.min(c) is the background NIR

    def identify_glint(self,im_aligned,bbox,percentile_threshold=90,percentile_method='nearest',mode="rgb"):
        """ 
        :param im_aligned (np.ndarray): 10 bands in band order i.e. band 1,2,3,4,5,6,7,8,9,10
        :param bbox (tuples or np.ndarray): e.g. ((x1,y1),(x2,y2))
        :param percentile_threshold (float): value between 0 and 100
        :param percentile_method (str): 'linear'(continuous) or 'nearest' (discontinuous method). 
            Note that for numpy version > 1.22, 'interpolation' argument is replaced with 'method', 'interpolation' is deprecated
        :param mode (str): 'rgb' or 'nir' which determines what mode is used for glint detection
            'rgb' means that all rgb bands will be used to detect glint
            'nir' means that only nir band is only used to detect glint
        TODO 
            1. automate finding the percentile_threshold using CDF method
        """
        ((x1,y1),(x2,y2)) = bbox
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1,y2 = y2, y1
        
        rgb_bands = [2,1,0] #668, 560, 475 nm
        # rgb image
        rgb_image = mutils.get_rgb(im_aligned[y1:y2,x1:x2,:],plot=False)
        nrow, ncol, _ = rgb_image.shape

        if mode == 'rgb':
            # rgb band images
            rgb_images = [rgb_image[:,:,i] for i in range(3)] #bands 2,1,0
            # use percentile to identify glint threshold
            rgb_images_flatten = [i.flatten() for i in rgb_images]
            glint_percentile = [np.percentile(i,percentile_threshold,interpolation=percentile_method) for i in rgb_images_flatten]
            glint_idxes = [np.argwhere(im>p) for p,im in zip(glint_percentile,rgb_images_flatten)]
            glint_idxes = [np.unravel_index(i,(nrow,ncol)) for i in glint_idxes]
            # combine the indices for each band
            unique_glint_idxes = np.unique(np.vstack([np.column_stack(i) for i in glint_idxes]),axis=0) #first column is row idx, 2nd column is col idx
            y, x = unique_glint_idxes[:,0], unique_glint_idxes[:,1]
            # use the combined indices to mask rgb image
            rgb_masked = rgb_image.copy()
            rgb_masked[y,x] = 0

            fig, axes = plt.subplots(2,3,figsize=(10,5))
            for i in range(3):
                w = self.wavelength_dict[rgb_bands[i]]
                glint_idx = glint_idxes[i]
                image_copy = rgb_images[i].copy()
                image_copy[glint_idx] = 1
                im = axes[0,i].imshow(image_copy)
                plt.colorbar(im,ax=axes[0,i])
                axes[0,i].set_axis_off()
                axes[0,i].set_title(f'{w} nm \nPercentile value ({glint_percentile[i]:.3f})')

            axes[1,0].imshow(rgb_image)
            axes[1,0].set_title('RGB image')
            axes[1,0].set_axis_off()

            axes[1,1].imshow(rgb_masked)
            axes[1,1].set_title('RGB masked image')
            axes[1,1].set_axis_off()

            extracted_glint = rgb_image - rgb_masked
            axes[1,2].imshow(extracted_glint)
            axes[1,2].set_title('Glint extracted')
            axes[1,2].set_axis_off()

            fig.suptitle(f'Glint detection using {percentile_threshold}th percentile ({percentile_method})')
            plt.tight_layout()
            plt.show()
        
        else:
            # NIR image to identify glint
            NIR_band = 3
            NIR_wavelength =self.wavelength_dict[NIR_band]
            NIR_image = im_aligned[y1:y2,x1:x2,NIR_band]
            nir_flattened = NIR_image.flatten()
            glint_percentile = np.percentile(nir_flattened,percentile_threshold,interpolation=percentile_method)
            # identify indices where NIR is higher than percentile threshold
            glint_idxes = np.argwhere(nir_flattened>glint_percentile)
            glint_idxes = np.unravel_index(glint_idxes,(nrow,ncol))
            # mask images
            rgb_masked = rgb_image.copy()
            rgb_masked[glint_idxes] = 0
            # mask on NIR_image
            NIR_masked = NIR_image.copy()
            NIR_masked[glint_idxes] = 1
            # plot
            fig, axes = plt.subplots(2,3,figsize=(10,5))
            # plot NIR band
            im = axes[0,0].imshow(NIR_image)
            plt.colorbar(im,ax=axes[0,0])
            axes[0,0].set_title(f'NIR image ({NIR_wavelength} nm)')
            # plot glint detected using NIR band
            im = axes[0,1].imshow(NIR_masked)
            plt.colorbar(im,ax=axes[0,1])
            axes[0,1].set_title(f'{NIR_wavelength} nm \nPercentile value ({glint_percentile:.3f})')

            axes[1,0].imshow(rgb_image)
            axes[1,0].set_title('RGB image')
            axes[1,0].set_axis_off()

            axes[1,1].imshow(rgb_masked)
            axes[1,1].set_title('RGB masked image')
            axes[1,1].set_axis_off()

            extracted_glint = rgb_image - rgb_masked
            axes[1,2].imshow(extracted_glint)
            axes[1,2].set_title('Glint extracted')
            axes[1,2].set_axis_off()

            # turn off all axis
            for ax in axes.flatten():
                ax.set_axis_off()
            fig.suptitle(f'Glint detection using {percentile_threshold}th percentile ({percentile_method})')
            plt.tight_layout()
            plt.show()
        
        # returns binary mask
        mask = np.where(extracted_glint[:,:,1]>0,1,0)
        return mask

