import numpy as np
import PIL.Image
import cv2
import glob
from os import listdir
from os.path import join
from math import ceil
import matplotlib.pyplot as plt
import random
import shutil
import cv2 as cv
from os.path import join, basename, splitext, exists
from os import listdir

def pad_images(img):
    """
    img (1024 x m)
    """
    nrow,ncol = img.shape[0], img.shape[1]
    if ncol > 512:
        raise ValueError("Img col is > 512")
    if img.ndim == 3:
        pad_img = np.zeros((512,512,3))
        pad_img[:,:ncol,:] = img
        pad_img = pad_img.astype(np.uint8)
    else:
        pad_img = np.zeros((512,512))
        pad_img[:,:ncol] = img
        pad_img = pad_img.astype(np.uint8)
    
    return pad_img

def cut_into_512(img):
    """
    img (1024 x m)
    """
    nrow,ncol = img.shape[0], img.shape[1]
    if ncol > 512:
        cut_images = []
        for i in range(0,nrow,512):
            for j in range(0,ncol,512):
                if img.ndim == 3:
                    if j+512 <= ncol:
                        cut_images.append(img[i:i+512,j:j+512,:])
                    else:
                        padded_img = pad_images(img[i:i+512,j:ncol,:])
                        cut_images.append(padded_img)
                else:
                    if j+512 <= ncol:
                        cut_images.append(img[i:i+512,j:j+512])
                    else:
                        padded_img = pad_images(img[i:i+512,j:ncol])
                        cut_images.append(padded_img)
        
    elif ncol == 512:
        cut_images = [img]
        
    else:
        cut_images = [pad_images(img)]

    return cut_images

def save_imgs(cut_img_list,directory):
    for i, img in enumerate(cut_img_list):
        img_name = join(directory, "output{}.png".format(i))
        img = PIL.Image.fromarray(img)
        img.save(img_name)
    
    return

def preview_cut_img(cut_img_list,gui_plot=False):
    ncol = 4
    nrow = int(ceil(len(cut_img_list)/ncol))
    fig,axes = plt.subplots(nrow,ncol,figsize=(15,15))
    for i,(img, ax) in enumerate(zip(cut_img_list,axes.flatten())):
        if img.ndim == 3:
            ax.imshow(img)
        else:
            ax.imshow(img,cmap='gray')
        ax.set_title(f'Image no. {i}')
        ax.axis('off')
    if gui_plot is True:
        plt.show(block=False)
    return

def stitch_cut_images(cut_img_list,w,h=1024):
    """
    cut_img_list (list): a list of cut images that is sequentially cut along columns then rows
    w (int): width (ncols) of the original image that was being cut
    h (int): height (nrows) of the original image that was being cut
    """
    n_imges = len(cut_img_list)
    row_imges_list = []
    nrow = cut_img_list[0].shape[0]
    n_rows = ceil(h/nrow)
    for r in range(n_rows):
        n_imges_per_row = n_imges/n_rows
        start = int(r*n_imges_per_row)
        end = int(r*n_imges_per_row + n_imges_per_row)
        row_imges = np.column_stack(cut_img_list[start:end])
        row_imges_list.append(row_imges)
    
    stitched_img = np.row_stack(row_imges_list)
    stitched_img = stitched_img[:h,:w]
    return stitched_img

def open_images_from_directory(fp):
    """
    fp (str): filepath of directory that contains images
    """
    img_list = [np.asarray(PIL.Image.open(join(fp,f))) for f in sorted(listdir(fp))]
    print(f'Number of images: {len(img_list)} in {basename(fp)}')
    return img_list

def normalise_img(img):
    im_min = np.min(img)
    im_max = np.max(img)
    return (img - im_min)/(im_max-im_min)

def uniquify(path,zfill_val=5):
    """
    detects whether a path exists, and if True, append a counter or increment the counter to the path
    """
    filename, extension = splitext(path)
    counter = 1

    while exists(path):
        if len(path.rsplit('_',1)[-1].replace(extension,'')) == 2: #identify whether the postfix is a img cut number
            path = filename + "_{}".format(str(counter).zfill(zfill_val)) + extension
            # break
        else:
            filename = path.rsplit('_',1)[0]
            # print(filename)
            path = filename + "_{}".format(str(counter).zfill(zfill_val)) + extension
        counter += 1

    return path

def save_img(img,dir,name,prefix='',postfix='',ext=".png",overwrite=False):
    """
    img (np.array)
    dir (str): directory to store the images
    name (str): name of image
    ext (str): default is png
    """
    img = PIL.Image.fromarray(img)
    fn = '{}_{}_{}{}'.format(prefix,name,postfix,ext)
    if prefix == '':
        fn = fn[1:] #remove the first underscore
    if overwrite is True:
        img.save(join(dir,fn))
    else:
        fp = uniquify(join(dir,fn))
        img.save(fp)
    return

def read_txt_into_list(fp):
    # opening the file in read mode
    my_file = open(fp, "r")

    # reading the file
    data = my_file.read()
    
    # replacing end splitting the text 
    # when newline ('\n') is seen.
    data_into_list = data.split("\n")
    data_into_list = [d for d in data_into_list if d !='']
    my_file.close()
    return data_into_list

def mask_objects(model,img):
    """
    model (unet model)
    img (np.array): RGB
    """
    img = cv.cvtColor(img,cv.COLOR_RGB2BGR)
    predicted_mask = model.predict_segmentation(img)
    # produce a binary mask, where water = 1, others = 0
    mask = np.where(predicted_mask>0,0,1).astype(np.uint8)
    mask = cv.resize(mask,(512,512))
    return mask

def scale_img(img,scale_list= [1,1.2,1.4,1.6,1.8,2]):
    """
    scales image and then crop it to 512x512
    returns a list of 5 images
    """
    # scale_list = [1.2,1.4,1.6,1.8,2]
    scaled_img = []
    nrow, ncol = img.shape[0],img.shape[1]
    for r in scale_list:
        width = int(ncol*r)
        height = int(nrow*r)
        image = cv.resize(img,(width,height),interpolation = cv.INTER_AREA)
        scaled_img.append(image[:512,:512])
    return scaled_img

def rotate_img(img):
    """
    rotates image by 90cw, 180 and 90ccw
    returns a list of 3 images
    """
    rotation_list = [cv.ROTATE_90_CLOCKWISE,cv.ROTATE_180,cv.ROTATE_90_COUNTERCLOCKWISE]
    rotated_img = [img] #include original image
    for r in rotation_list:
        image = cv.rotate(img, r)
        rotated_img.append(image)
    return rotated_img

def augment_img(img):
    """
    augment image using scale_img and rotate_img
    """
    rotated_img = rotate_img(img)
    augmented_im = []
    for im in rotated_img:
        scaled_img = scale_img(im)
        for s in scaled_img:
            augmented_im.append(s)
    return augmented_im

def plot_an_image(img):
    plt.figure()
    if img.ndim == 3:
        plt.imshow(img)
    elif img.ndim == 2:
        plt.imshow(img,cmap='gray')
        plt.colorbar()
    return