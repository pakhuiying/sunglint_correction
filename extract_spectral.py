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

                if n_images < show_n:
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
            spectral_flatten = im_aligned[y1:y2,x1:x2,:].reshape(-1,im_aligned.shape[-1])
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

