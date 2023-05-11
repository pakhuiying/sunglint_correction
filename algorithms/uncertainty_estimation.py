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
import algorithms.SUGAR as sugar

class UncertaintyEst:
    def __init__(self,corrected_im_background, corrected_im):
        """
        :param corrected_im_background (np.ndarray): image accounted for 
        :param corrected_im (np.ndarray)

        """
        self.corrected_im_background = corrected_im_background
        self.corrected_im = corrected_im

    @classmethod
    def get_corrected_image(glint_image,iter=3,bounds = [(1,2)]*10):

        def correction_iterative(glint_image,iter,bounds,estimate_background):
            gm_list = []
            bg_list = []
            for i in range(iter):
                HM = sugar.SUGAR(glint_image,iter,bounds,estimate_background)
                corrected_bands = HM.get_corrected_bands()
                glint_image = np.stack(corrected_bands,axis=2)
                
                b_list = HM.b_list
                bounds = [(1,b*1.2) for b in b_list]
                glint_mask = HM.glint_mask
                est_background = HM.est_background

                gm_list.append(glint_mask)
                bg_list.append(est_background)
            return glint_image, gm_list, bg_list
        
        corrected_im_background, gm_list_background, bg_list_background = correction_iterative(glint_image,iter=iter,bounds = bounds,estimate_background=True)
        corrected_im, gm_list, bg_list = correction_iterative(glint_image,iter=iter,bounds = bounds,estimate_background=False)
        