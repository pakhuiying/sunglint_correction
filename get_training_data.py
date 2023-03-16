from matplotlib import pyplot as plt
from matplotlib.widgets import Button
import matplotlib.patches as patches
import PIL.Image as Image
import json
from os.path import join, split, exists
from os import listdir, mkdir, getcwd
import numpy as np
import cv2
import glob
import micasense.imageutils as imageutils
import micasense.capture as capture
import mutils

def get_all_dir(fp,iter=3):
    """ get all parent sub directories up to 3 levels"""
    fp_temp = fp
    sub_dir_list = []
    for i in range(iter):
        base_fn, fn = split(fp_temp)
        sub_dir_list.append(fn)
        fp_temp = base_fn
    return '_'.join(reversed(sub_dir_list))

def aligned_capture(capture, img_type = 'reflectance',interpolation_mode=cv2.INTER_LANCZOS4):
    """ 
    :param capture (capture object): for 10-bands image
    :param warp_matrices (mxmx3 np.ndarray): in rgb order of [2,1,0] loaded from pickle
    :param cropped_dimensions (tuple): loaded from pickle
    align images using the warp_matrices used for aligning 10-band images and outputs an rgb image
    """

    warp_mode = cv2.MOTION_HOMOGRAPHY
    
    width, height = capture.images[0].size()

    rgb_band_indices = [2,1,0]

    im_aligned = np.zeros((height,width,len(rgb_band_indices)), dtype=np.float32 )

    for i,rgb_i in enumerate(rgb_band_indices):
        if img_type == 'reflectance':
            img = capture.images[rgb_i].undistorted_reflectance()
        else:
            img = capture.images[rgb_i].undistorted_radiance()

        if warp_mode != cv2.MOTION_HOMOGRAPHY:
            im_aligned[:,:,i] = cv2.warpAffine(img,
                                            warp_matrices[rgb_i],
                                            (width,height),
                                            flags=interpolation_mode + cv2.WARP_INVERSE_MAP)
        else:
            im_aligned[:,:,i] = cv2.warpPerspective(img,
                                                warp_matrices[rgb_i],
                                                (width,height),
                                                flags=interpolation_mode + cv2.WARP_INVERSE_MAP)
    (left, top, w, h) = tuple(int(i) for i in cropped_dimensions)
    im_cropped = im_aligned[top:top+h, left:left+w][:]

    # get normalised rgb image
    im_min = np.percentile(im_cropped.flatten(),  0.1)  # modify with these percentilse to adjust contrast
    im_max = np.percentile(im_cropped.flatten(), 99.9)  # for many images, 0.5 and 99.5 are good values

    im_display = np.zeros((im_cropped.shape[0],im_cropped.shape[1],len(rgb_band_indices)), dtype=np.float32)
    
    for i in range(len(rgb_band_indices)):
        im_display[:,:,i] = imageutils.normalize(im_cropped[:,:,i], im_min, im_max)
    
    return im_display

def get_current_img_counter(dir):
    """ 
    list all the files in saved_bboxes to track which is the last processed image wrt to current dir
    """
    if exists("saved_bboxes"):
        stored_fp = listdir("saved_bboxes") 
        selected_fp = []
        for fp in stored_fp:
            if all([i in dir for i in fp.split('_')[:2]]):
                selected_fp.append(fp)
        if len(selected_fp) > 0:
            selected_img = sorted(selected_fp)[-1]
            img_line = selected_img.split('IMG_')[1][:4]
            img_counter = int(img_line) + 1
        else:
            img_counter = 0
    else:
        img_counter = 0
    return img_counter

class LineBuilder:
    lock = "water"  # only one can be animated at a time
    def __init__(self,dict_builder,fig,ax,im,rgb_fp,img_counter):
        self.dict_builder = dict_builder
        self.categories = list(dict_builder)
        
        for k in self.categories:
            setattr(self,k + '_x', []) #store x coord
            setattr(self,k + '_y', []) #store y coord
            setattr(self,k + '_bbox', None) #store bbox
            setattr(self,k + '_line', dict_builder[k]['line']) #store line
            setattr(self,k + '_patch', dict_builder[k]['patch']) #store patch

        # self.cid = fig.canvas.mpl_connect('button_press_event', self)
        self.cid = fig.canvas.mpl_connect('button_press_event', self)
        self.ax = ax
        self.im = im
        self.rgb_fp = rgb_fp
        self.n_img = len(rgb_fp)
        self.img_counter = img_counter
        self.current_fp = rgb_fp[self.img_counter] # initialise with first fp
        


    def __call__(self, event):
        if any([event.inaxes != getattr(self,k+'_line').axes for k in self.categories]) is True:
            print("No call")
            return
        
        k = LineBuilder.lock
        getattr(self,k+'_x').append(event.xdata)
        getattr(self,k+'_y').append(event.ydata)
        xs = getattr(self,k+'_x')[-2:]
        ys = getattr(self,k+'_y')[-2:]
        print(xs,ys)
        getattr(self,k+'_line').set_data(xs,ys)
        getattr(self,k+'_line').figure.canvas.draw_idle()
        self.draw_rect(event)
        

    def draw_rect(self, _event):
        img_index = self.img_counter%self.n_img
        current_fp = self.rgb_fp[img_index] 

        k = LineBuilder.lock
        x1,x2 = getattr(self,k+'_x')[-2:]
        y1,y2 = getattr(self,k+'_y')[-2:]
        setattr(self, k+'_bbox', ((int(x1),int(y1)),(int(x2),int(y2))))
        h = y2 - y1
        w = x2 - x1
        getattr(self,k+'_patch').set_xy((x1,y1))
        getattr(self,k+'_patch').set_height(h)
        getattr(self,k+'_patch').set_width(w)
        self.current_fp = current_fp
        

    def reset(self, _event):
        """clear all points, lines and patches"""
        for k in self.categories:
            setattr(self, k+'_x', [])
            setattr(self, k+'_y', [])
            x1 = y1 = h = w = 0
            getattr(self,k+'_line').set_data([],[])
            getattr(self,k+'_patch').set_xy((x1,y1))
            getattr(self,k+'_patch').set_height(h)
            getattr(self,k+'_patch').set_width(w)
            getattr(self,k+'_line').figure.canvas.draw_idle()
        
    def turbid_glint(self,_event):
        LineBuilder.lock = "turbid_glint"

    def turbid(self,_event):
        LineBuilder.lock = "turbid"
    
    def water_glint(self,_event):
        LineBuilder.lock = "water_glint"
    
    def water(self,_event):
        LineBuilder.lock = "water"

    def shore(self,_event):
        LineBuilder.lock = "shore"
    
    def save(self, _event):
        
        save_bboxes = {self.current_fp:{k: getattr(self,k+'_bbox') for k in self.categories}}
        print(save_bboxes)
        print(self.current_fp)
        # get unique filename from current_fp
        fn = get_all_dir(self.current_fp,iter=4)
        

        #create a new dir to store bboxes
        store_dir = join(getcwd(),"saved_bboxes")
        if not exists(store_dir):
            mkdir(store_dir)

        fp_store = join(store_dir,fn)
        fp_store = fp_store.replace('.tif','')
        print(f"File saved at {fp_store}!")
        
        with open('{}.txt'.format(fp_store),'w') as cf:
            json.dump(save_bboxes,cf)
        

    def next(self, _event):
        self.reset(_event)
        self.save(_event)

        self.img_counter = self.img_counter+1
        img_index = self.img_counter%self.n_img
        current_fp = self.rgb_fp[img_index]

        cap = mutils.import_captures(current_fp)
        img = aligned_capture(cap)
        
        # img = np.asarray(Image.open(current_fp))
        # img = Image.open(current_fp)
        img_line = current_fp.split('IMG_')[1][:4] #get the current image line
        self.ax.set_title('Select T,W,TG, WG, S areas\nImage Index {}'.format(img_line))
        self.im.set_extent((0,img.shape[1],img.shape[0],0)) # left, right, bottom, top
        
        self.im.set_data(img)
        self.im.figure.canvas.draw_idle()
        
    def previous(self,_event):
        self.img_counter = self.img_counter-1
        img_index = self.img_counter%self.n_img
        current_fp = self.rgb_fp[img_index] 

        cap = mutils.import_captures(current_fp)
        img = aligned_capture(cap)
        
        # img = np.asarray(Image.open(current_fp))
        # img = Image.open(current_fp)
        self.reset(_event)
        img_line = current_fp.split('IMG_')[1][:4] #get the current image line
        self.ax.set_title('Select T,W,TG, WG, S areas\nImage Index {}'.format(img_line))
        self.im.set_extent((0,img.shape[1],img.shape[0],0)) # left, right, bottom, top
        
        self.im.set_data(img)
        self.im.figure.canvas.draw_idle()

def draw_sunglint_correction(fp_store):
    # initialise plot
    fig, ax = plt.subplots()

    # import files
    rgb_fp = [join(fp_store,f) for f in sorted(listdir(fp_store)) if f.endswith("1.tif")]
    # go to the last processed image
    img_counter = get_current_img_counter(fp_store)

    cap = mutils.import_captures(rgb_fp[img_counter])
    global warp_matrices
    global cropped_dimensions
    warp_matrices = cap.get_warp_matrices()
    cropped_dimensions,_ = imageutils.find_crop_bounds(cap,warp_matrices)

    img = aligned_capture(cap)
    im = ax.imshow(img)

    

    # set title
    img_line = rgb_fp[img_counter].split('IMG_')[1][:4]
    ax.set_title('Select T, W, TG, WG, S areas\nImage Index {}'.format(img_line))

    # initialise categories
    button_names = ['turbid_glint','water_glint','turbid','water','shore']
    
    # intialise colours
    colors = ['orange','cyan','saddlebrown','blue','yellow']
    
    # initialise dictionary to store stuff
    dict_builder = {b:None for b in button_names}
    
    # add to plot
    for button,c in zip(button_names,colors):
        line_category, = ax.plot(0,0,"o",c=c)
        rect = patches.Rectangle((0, 0), 10, 10, linewidth=1, edgecolor=c, facecolor='none')
        patch = ax.add_patch(rect)
        dict_builder[button] = {'line':line_category,'patch':patch}

    # initialise LineBuilder
    linebuilder = LineBuilder(dict_builder,fig,ax,im,rgb_fp,img_counter)

    plt.subplots_adjust(bottom=0.2,right=0.8)
    #reset button
    breset_ax = plt.axes([0.2, 0.05, 0.1, 0.075]) #left,bottom, width, height
    breset = Button(breset_ax, 'Reset')
    breset.on_clicked(linebuilder.reset)
    #prev image
    prev_ax = plt.axes([0.35, 0.05, 0.1, 0.075]) #left,bottom, width, height
    prev = Button(prev_ax, 'Prev')
    prev.on_clicked(linebuilder.previous)
    #next image
    next_ax = plt.axes([0.5, 0.05, 0.1, 0.075]) #left,bottom, width, height
    next = Button(next_ax, 'Next')
    next.on_clicked(linebuilder.next)
    
    #save coordinates
    save_ax = plt.axes([0.65, 0.05, 0.1, 0.075]) #left,bottom, width, height
    save = Button(save_ax, 'Save')
    save.on_clicked(linebuilder.save)
    # buttons to select region
    
    
    axes = [plt.axes([0.82, 0.8-i/10, 0.15, 0.075]) for i in range(len(button_names))]
    buttons = [Button(a,name) for a,name in zip(axes,button_names)]
    for k,button in zip(button_names,buttons):
        button.on_clicked(getattr(linebuilder,k)) #update this
    
    plt.show()


if __name__ == '__main__':
    
    fp_store = input("Enter directory: ")#r"C:\Users\PAKHUIYING\Downloads\test" #r"C:\Users\PAKHUIYING\Documents\image_processing\F3_processed_surveys\2021_11_10_2"
    fp_store = fp_store.replace("\\","/").replace("\"","")
    draw_sunglint_correction(fp_store)