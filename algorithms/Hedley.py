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
from scipy.stats.stats import pearsonr

class Hedley:
    def __init__(self, im_aligned,bbox,mode="regression"):
        """
        :param im_aligned (np.ndarray): band aligned and calibrated & corrected reflectance image
        :param bbox (tuple): bbox of a glint area e.g. water_glint
        :param mode (str): modes for estimating the slopes for Hedley correction (e.g. regression, least_sq, covariance,pearson)
        """
        self.im_aligned = im_aligned
        if mode not in ['regression, least_sq, covariance,pearson']:
            self.mode = 'regression'
        else:
            self.mode = mode
        self.bbox = bbox
        self.bbox = self.sort_bbox()
        ((x1,y1),(x2,y2)) = self.bbox
        self.glint_area = self.im_aligned[y1:y2,x1:x2,:]
        # initialise categories
        self.button_names = ['turbid_glint','water_glint','turbid','water','shore']
        # intialise colours
        self.colors = ['orange','cyan','saddlebrown','blue','yellow']
        self.color_mapping = {categories:colors for categories,colors in zip(self.button_names,self.colors)}
        # import wavelengths for each band
        wavelengths = mutils.sort_bands_by_wavelength()
        self.wavelength_dict = {i[0]:i[1] for i in wavelengths}
        self.n_bands = im_aligned.shape[-1]
        self.NIR_band = list(self.wavelength_dict)[-1]
    
    def sort_bbox(self):
        ((x1,y1),(x2,y2)) = self.bbox
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1,y2 = y2, y1

        return ((x1,y1),(x2,y2))

    def regression_slope(self,NIR,band):
        """
        :param NIR (np.ndarray): flattened np.array for the glint area in NIR band
        :param band (np.ndarray): flattened np.array for the glint area in other band
        returns the slope and the intercept, and the model
        """
        lm = LinearRegression().fit(NIR, band)
        b = lm.coef_[0][0]
        intercept = lm.intercept_[0]
        return (b, intercept, lm)
    
    def covariance(self,NIR,band):
        """
        :param NIR (np.ndarray): flattened np.array for the glint area in NIR band
        :param band (np.ndarray): flattened np.array for the glint area in other band
        returns the slope
        """
        if len(NIR.shape) > 1:
            NIR = NIR.flatten()
            band = band.flatten()
        n = NIR.shape[0]
        pij = np.dot(NIR,band)/n - np.sum(NIR)/n*np.sum(band)/n
        pjj = np.dot(NIR,NIR)/n - (np.sum(NIR)/n)**2
        return pij/pjj
    
    def least_sq(self,NIR,band):
        """
        :param NIR (np.ndarray): flattened np.array for the glint area in NIR band
        :param band (np.ndarray): flattened np.array for the glint area in other band
        returns the slope
        """
        if len(NIR.shape) > 1:
            NIR = NIR.flatten()
            band = band.flatten()
        A = np.vstack([NIR,np.ones(NIR.shape[0])]).T
        m, _ = np.linalg.lstsq(A,band, rcond=None)[0]
        return m
    
    def pearson(self,NIR,band):
        """
        :param NIR (np.ndarray): flattened np.array for the glint area in NIR band
        :param band (np.ndarray): flattened np.array for the glint area in other band
        returns the slope
        """
        if len(NIR.shape) > 1:
            NIR = NIR.flatten()
            band = band.flatten()
        return pearsonr(NIR,band)[0]
        
    def plot_regression(self, plot = True):
        """ 
        Construct a linear regression of NIR reflectance versus the reflectance in the ith band using pixels from the deep water subset with glint
        returns a dict of regression slopes in band order i.e. band 0,1,2,3,4,5,6,7,8,9
        """
    
        regression_slopes = dict()
        NIR_band = list(self.wavelength_dict)[-1]
        NIR_pixels = self.glint_area[:,:,self.NIR_band].flatten().reshape(-1, 1)
        self.R_min = np.percentile(NIR_pixels,5,interpolation='nearest')

        fig, axes = plt.subplots(self.n_bands//2,2,figsize=(10,20))

        for band_number,ax in zip(range(self.n_bands),axes.flatten()):
            y = self.glint_area[:,:,band_number].flatten().reshape(-1, 1)
            b_regression, intercept, lm = self.regression_slope(NIR_pixels,y)
            b_covariance = self.covariance(NIR_pixels,y)
            b_least_sq = self.least_sq(NIR_pixels,y)
            b_pearson = self.pearson(NIR_pixels,y)
            regression_slopes[band_number] = b_regression
            ax.plot(NIR_pixels,y,'o')
            r2 = r2_score(y,lm.predict(NIR_pixels))
            ax.set_title(r'Band {}: {} nm ($R^2:$ {:.3f}, N = {})'.format(band_number, self.wavelength_dict[band_number],r2,y.shape[0]))
            ax.set_xlabel(f'NIR reflectance (Band {self.wavelength_dict[NIR_band]})')
            ax.set_ylabel(f'Band {self.wavelength_dict[band_number]} reflectance')
            x_vals = np.linspace(np.min(NIR_pixels),np.max(NIR_pixels),50)
            y_vals = intercept + b_regression * x_vals
            ax.plot(x_vals.reshape(-1,1), y_vals.reshape(-1,1), '--')
            ax.text(0.1,ax.get_ylim()[1]*0.8,r"$y = {:.3f}x + {:.3f}$".format(b_regression,intercept))
            ax.text(0.1,ax.get_ylim()[1]*0.6,f"Cov: {b_covariance:.3f}")
            ax.text(0.1,ax.get_ylim()[1]*0.4,f"Least sq: {b_least_sq:.3f}")
            ax.text(0.1,ax.get_ylim()[1]*0.2,f"Pearson: {b_pearson:.3f}")

        if plot is True:
            plt.tight_layout()
            plt.show()
        else:
            plt.close()

        return regression_slopes
    
    def correction_bands(self):
        """ 
        returns a list of slope in band order i.e. 0,1,2,3,4,5,6,7,8,9
        """
        
        NIR_pixels = self.glint_area[:,:,self.NIR_band].flatten().reshape(-1, 1)
        self.R_min = np.percentile(NIR_pixels,5,interpolation='nearest')

        b_list = []
        for band_number in range(self.n_bands):
            y = self.glint_area[:,:,band_number].flatten().reshape(-1, 1)
            if self.mode == 'regression':
                b,_,_ = self.regression_slope(NIR_pixels,y)
            elif self.mode == 'covariance':
                b = self.covariance(NIR_pixels,y)
            elif self.mode == 'least_sq':
                b = self.least_sq(NIR_pixels,y)
            else:
                b = self.pearson(NIR_pixels,y)
            b_list.append(b)

        return b_list
    
    def get_corrected_bands(self, plot = True):

        ((x1,y1),(x2,y2)) = self.bbox

        b_list = self.correction_bands()

        hedley_c = lambda x,RT_NIR,b,R_min: x - b*(RT_NIR - R_min)

        corrected_bands = []
        avg_reflectance = []
        avg_reflectance_corrected = []

        fig, axes = plt.subplots(self.n_bands,2,figsize=(10,20))
        for band_number in range(self.n_bands):
            b = b_list[band_number]
            corrected_band = hedley_c(self.im_aligned[:,:,band_number],self.im_aligned[:,:,self.NIR_band],b,self.R_min)
            corrected_bands.append(corrected_band)
            avg_reflectance.append(np.mean(self.glint_area[:,:,band_number]))
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

        else:
            plt.close()

        return corrected_bands
    
    def correction_stats(self,corrected_bands):
        """
        :param corrected_bands (list of np.ndarrays): images corrected for sunglint
        Show corrected and original rgb image, mean reflectance
        """
        ((x1,y1),(x2,y2)) = self.bbox

        rgb_bands = [2,1,0]

        fig, axes = plt.subplots(2,4,figsize=(14,10))
        rgb_im = np.take(self.im_aligned,rgb_bands,axis=2)
        rgb_im_corrected = np.stack([corrected_bands[i] for i in rgb_bands],axis=2)
        avg_reflectance = [np.mean(self.glint_area[:,:,band_number]) for band_number in range(self.n_bands)]
        avg_reflectance_corrected = [np.mean(corrected_bands[band_number][y1:y2,x1:x2]) for band_number in range(self.n_bands)]

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
        axes[0,2].legend(loc='upper right')
        axes[0,2].set_title('Corrected and original mean reflectance')

        residual = [og-cor for og,cor in zip(avg_reflectance,avg_reflectance_corrected)]
        axes[0,3].plot(list(self.wavelength_dict.values()),[residual[i] for i in list(self.wavelength_dict)])
        axes[0,3].set_xlabel('Wavelengths (nm)')
        axes[0,3].set_ylabel('Residual in Reflectance')

        for ax in axes[0,:2]:
            ax.axis('off')
        
        h = y2 - y1
        w = x2 - x1

        axes[1,0].imshow(rgb_im[y1:y2,x1:x2,:])
        axes[1,1].imshow(rgb_im_corrected[y1:y2,x1:x2,:])
        axes[1,0].set_title('Original Glint')
        axes[1,1].set_title('Corrected Glint')

        axes[1,0].plot([0,h],[h//2,h//2],color="red",linewidth=3)
        axes[1,1].plot([0,h],[h//2,h//2],color="red",linewidth=3)
        
        # for ax in axes[1,0:2]:
        #     ax.axis('off')

        for i,c in zip(range(3),['r','g','b']):
            axes[1,2].plot(list(range(w)),rgb_im[y1:y2,x1:x2,:][h//2,:,i],c=c)
            # plot for original
        for i,c in zip(range(3),['r','g','b']):
            # plot for corrected reflectance
            axes[1,2].plot(list(range(w)),rgb_im_corrected[y1:y2,x1:x2,:][h//2,:,i],c=c,ls='--')

        axes[1,2].set_xlabel('Width of image')
        axes[1,2].set_ylabel('Reflectance')
        axes[1,2].set_title('Reflectance along red line')

        lines = [Line2D([0], [0], color='black', linewidth=3, linestyle=ls) for ls in ['-','--']]
        labels = ['Original','Corrected']
        axes[1,2].legend(lines,labels,loc='upper right')

        residual = rgb_im[y1:y2,x1:x2,:][h//2,:,:] - rgb_im_corrected[y1:y2,x1:x2,:][h//2,:,:]
        for i,c in zip(range(3),['r','g','b']):
            axes[1,3].plot(list(range(w)),residual[:,i],c=c)

        axes[1,3].set_xlabel('Width of image')
        axes[1,3].set_ylabel('Residual in Reflectance')
        axes[1,3].set_title('Reflectance along red line')

        plt.tight_layout()
        plt.show()

        return