import numpy as np
import os
import pickle #This library will maintain the format as well
import micasense.imageutils as imageutils
import micasense.plotutils as plotutils
import cv2
import matplotlib.pyplot as plt
import PIL.Image as Image
import matplotlib.patches as patches
import json

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

def plot_bboxes(fp):
    """ 
    :param fp (str): filepath of txt file which contains the bboxes of turbid, water, turbid_glint, water_glint, shore
    this function plot the bboxes of each category to validate the selection of bboxes with python GUI (get_training_data.py)
    """
    with open(fp, 'r') as fp:
        data = json.load(fp)

    # initialise categories
    button_names = ['turbid_glint','water_glint','turbid','water','shore']
    
    # intialise colours
    colors = ['orange','cyan','saddlebrown','blue','yellow']

    # mapping of categories and colors
    cat_colors = dict()
    for cat,c in zip(button_names,colors):
        cat_colors[cat] = c
    
    fig,ax = plt.subplots()
    ax.imshow(Image.open(list(data)[0]))
    for v in data.values():
        for k1,v1 in v.items():
            if v1 is not None:
                ((x1,y1),(x2,y2)) = v1
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
                h = y1 - y2 # negative height as the origin is on the top left
                w = x2 - x1
                # in patches, x,y coord is the bottom left corner, 
                rect = patches.Rectangle((x1, y2), w, h, linewidth=1, edgecolor=cat_colors[k1], facecolor='none')
                patch = ax.add_patch(rect)
    plt.show()
    return 