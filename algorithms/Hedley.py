# import micasense.imageset as imageset
import micasense.capture as capture
import cv2
import micasense.imageutils as imageutils
# import micasense.plotutils as plotutils
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
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
import numpy as np
from math import ceil
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

class Hedley:
    def __init__(self, im_aligned,bbox):
        """
        :param im_aligned (np.ndarray): band aligned and calibrated & corrected reflectance image
        :param bbox (tuple): bbox of an area e.g. water_glint
        """
        self.im_aligned = im_aligned
        self.bbox = bbox
        # initialise categories
        self.button_names = ['turbid_glint','water_glint','turbid','water','shore']
        # intialise colours
        self.colors = ['orange','cyan','saddlebrown','blue','yellow']
        self.color_mapping = {categories:colors for categories,colors in zip(self.button_names,self.colors)}
        # import wavelengths for each band
        wavelengths = mutils.sort_bands_by_wavelength()
        self.wavelength_dict = {i[0]:i[1] for i in wavelengths}
        self.n_bands = im_aligned.shape[-1]
    
    def sort_bbox(self):
        ((x1,y1),(x2,y2)) = self.bbox
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1,y2 = y2, y1

        return ((x1,y1),(x2,y2))

    def regression(self,plot = True,**kwargs):
        """ 
        Construct a linear regression of NIR reflectance versus the reflectance in the ith band using pixels from the deep water subset with glint
        returns a dict of regression slopes in band order i.e. band 0,1,2,3,4,5,6,7,8,9
        """
        ((x1,y1),(x2,y2)) = self.sort_bbox()

        regression_slopes = dict()
        NIR_band = list(self.wavelength_dict)[-1]
        self.NIR_pixels = self.im_aligned[y1:y2,x1:x2,NIR_band].flatten().reshape(-1, 1)
        self.R_min = np.percentile(self.NIR_pixels,5,interpolation='nearest')

        fig, axes = plt.subplots(self.n_bands//2,2,**kwargs)

        for band_number,ax in zip(range(self.n_bands),axes.flatten()):
            y = self.im_aligned[y1:y2,x1:x2,band_number].flatten().reshape(-1, 1)
            lm = LinearRegression().fit(self.NIR_pixels, y)
            b = lm.coef_[0][0]
            intercept = lm.intercept_[0]
            regression_slopes[band_number] = b
            ax.plot(self.NIR_pixels,y,'o')
            r2 = r2_score(y,lm.predict(self.NIR_pixels))
            ax.set_title(r'Band {}: {} nm ($R^2:$ {:.3f}, N = {})'.format(band_number, self.wavelength_dict[band_number],r2,y.shape[0]))
            ax.set_xlabel(f'NIR reflectance (Band {self.wavelength_dict[NIR_band]})')
            ax.set_ylabel(f'Band {self.wavelength_dict[band_number]} reflectance')
            x_vals = np.linspace(np.min(self.NIR_pixels),np.max(self.NIR_pixels),50)
            y_vals = intercept + b * x_vals
            ax.plot(x_vals.reshape(-1,1), y_vals.reshape(-1,1), '--')
            ax.text(0.1,ax.get_ylim()[1]*0.8,r"$y = {:.3f}x + {:.3f}$".format(b,intercept))

        if plot is True:
            plt.tight_layout()
            plt.show()
        else:
            plt.close()

        return regression_slopes
    
    def correction(self, plot= True,**kwargs):
        regression_slopes = self.regression(plot=False)
        regression_slopes = list(regression_slopes.values())

        NIR_band = list(self.wavelength_dict)[-1]

        # def hedley_correction(x,RT_NIR,b,R_min):
        #     return x - b*(RT_NIR - R_min)

        # hedley_c = np.vectorize(hedley_correction)

        hedley_c = lambda x,RT_NIR,b,R_min: x - b*(RT_NIR - R_min)

        corrected_bands = []
        avg_reflectance = []
        avg_reflectance_corrected = []
        ((x1,y1),(x2,y2)) = self.sort_bbox()

        fig, axes = plt.subplots(self.n_bands,2,figsize=kwargs['figsize'])
        for band_number in range(self.n_bands):
            b = regression_slopes[band_number]
            corrected_band = hedley_c(self.im_aligned[:,:,band_number],self.im_aligned[:,:,NIR_band],b,self.R_min)
            corrected_bands.append(corrected_band)
            avg_reflectance.append(np.mean(self.im_aligned[y1:y2,x1:x2,band_number]))
            avg_reflectance_corrected.append(np.mean(corrected_band[y1:y2,x1:x2]))
            axes[band_number,0].imshow(self.im_aligned[:,:,band_number])
            axes[band_number,1].imshow(corrected_band)
            axes[band_number,0].set_title(f'Band {self.wavelength_dict[band_number]} reflectance')
            axes[band_number,1].set_title(f'Band {self.wavelength_dict[band_number]} reflectance corrected')

        if plot is True:
            for ax in axes.flatten():
                ax.axis('off')
            plt.tight_layout()
            plt.show()

            rgb_bands = [2,1,0]
            fig, axes = plt.subplots(2,3)
            rgb_im = np.take(self.im_aligned,rgb_bands,axis=2)
            rgb_im_corrected = np.stack([corrected_bands[i] for i in rgb_bands],axis=2)
            axes[0,0].imshow(rgb_im)
            axes[0,0].set_title('Original RGB')
            coord, w, h = mutils.bboxes_to_patches(self.bbox)
            rect = patches.Rectangle(coord, w, h, linewidth=1, edgecolor='red', facecolor='none')
            axes[0,0].add_patch(rect)
            # axes[0,1].add_patch(rect)
            axes[0,1].imshow(rgb_im_corrected)
            axes[0,1].set_title('Corrected RGB')

            axes[0,2].plot(list(self.wavelength_dict.values()),[avg_reflectance[i] for i in list(self.wavelength_dict)], label='Original')
            axes[0,2].plot(list(self.wavelength_dict.values()),[avg_reflectance_corrected[i] for i in list(self.wavelength_dict)], label='Corrected')
            axes[0,2].set_xlabel('Wavelengths (nm)')
            axes[0,2].set_ylabel('Reflectance')
            
            for ax in axes[0,:2]:
                ax.axis('off')
            
            concat_image = np.hstack([rgb_im[y1:y2,x1:x2,:],rgb_im_corrected[y1:y2,x1:x2,:]])
            axes[1,0].imshow(concat_image)
            axes[1,0].set_title('Original/Corrected glint')
            # axes[1,0].plot([0,x2-x1],[(y2-y1)//2,(y2-y1)//2],color="red",linewidth=3)
            # axes[1,1].imshow(rgb_im_corrected[y1:y2,x1:x2,:])
            # axes[1,1].set_title('Corrected glint')
            # axes[1,1].plot([0,x2-x1],[(y2-y1)//2,(y2-y1)//2],color="blue",linewidth=3)
            for i,c in zip(range(3),['r','g','b']):
                axes[1,1].plot(list(range(x2-x1)),rgb_im[y1:y2,x1:x2,:][(y2-y1)//2,:,i],c=c)
            for i,c in zip(range(3),['r','g','b']):
                axes[1,2].plot(list(range(x2-x1)),rgb_im_corrected[y1:y2,x1:x2,:][(y2-y1)//2,:,i],c=c)

            plt.legend(loc='center left',bbox_to_anchor=(1.04, 0.5))
            plt.tight_layout()
            plt.show()
        else:
            plt.close()

        return corrected_bands