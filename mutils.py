import numpy as np
import os
import pickle #This library will maintain the format as well
import micasense.imageutils as imageutils
import micasense.plotutils as plotutils
import cv2
import matplotlib.pyplot as plt

panel_radiance_to_irradiance = lambda radiance,albedo: radiance*np.pi/albedo

def order_bands_from_filenames(imageNames):
    """ 
    listing images using glob.glob results in unordered band order (i.e. band 1, 10, 2,3,4,5,6,7,8,9)
    this function ensures that filenames are listed in band order i.e. band 1,2,3,4,5,6,7,8,9,10
    """
    imageNames_ordered = {i+1: None for i in range(10)}
    for fn in imageNames:
        filename = os.path.basename(fn)
        imageNames_ordered[int(filename.split('_')[-1].replace('.tif',''))] = fn
    return list(imageNames_ordered.values())

def load_pickle(fp):
    if fp.endswith('ob'):
        with open(fp, 'rb') as fp:
            data = pickle.load(fp)

        return data
    else:
        print("Not a pickle file")
        return None

def align_captures(cap,img_type = "reflectance"):
    """ 
    use rig relatives to align band images 
    """
    warp_mode = cv2.MOTION_HOMOGRAPHY
    warp_matrices = cap.get_warp_matrices()
    cropped_dimensions,edges = imageutils.find_crop_bounds(cap,warp_matrices)
    im_aligned = imageutils.aligned_capture(cap, warp_matrices, warp_mode, cropped_dimensions, None, img_type=img_type)
    return im_aligned

def get_rgb(im_aligned,plot=True):
    """
    :param im_aligned (np.ndarray): (m,n,c), where c = 10. output from align_captures
    """
    rgb_band_indices = [2,1,0]
    im_min = np.percentile(im_aligned[:,:,0:2].flatten(),  0.1)  # modify with these percentilse to adjust contrast
    im_max = np.percentile(im_aligned[:,:,0:2].flatten(), 99.9)  # for many images, 0.5 and 99.5 are good values

    im_display = np.zeros((im_aligned.shape[0],im_aligned.shape[1],len(rgb_band_indices)), dtype=np.float32)
    for i,rgb_i in enumerate(rgb_band_indices):
        im_display[:,:,i] = imageutils.normalize(im_aligned[:,:,rgb_i], im_min, im_max)

    if plot is True:
        plt.figure(figsize=(10,10))
        plt.imshow(im_display)
        plt.show()

    return im_display