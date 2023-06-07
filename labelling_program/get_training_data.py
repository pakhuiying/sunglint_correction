from matplotlib import pyplot as plt
from matplotlib.widgets import Button, TextBox
import matplotlib.patches as patches
import PIL.Image as Image
import json
from os.path import join, split, exists, basename, splitext
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
    align images using the warp_matrices used for aligning 10-band images and outputs a tuple (rgb image, normalised rgb image)
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
    
    return im_cropped, im_display # rgb image, normalised rgb image

def get_current_img_counter(dir, mode='raw'):
    """ 
    list all the files in saved_bboxes (raw) or QAQC_plots (rgb) to track which is the last processed image wrt to current dir
    : param mode (str): raw or rgb. Raw mode means at the stage of labelling images, rgb mode means at the stage of QAQC labelled images
    """
    if mode == 'raw':
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
    
    elif mode == 'rgb':
        last_image_fp = join("QAQC_relabel","last_image.txt")
        if exists(last_image_fp):
            with open(last_image_fp) as f:
                last_image = f.readlines()[0]

            img_counter = [i for i,fp in enumerate(dir) if last_image in fp][0]
        else:
            img_counter = 0

        return img_counter
    
    else:
        return 0

class LineBuilder:
    lock = "water"  # only one can be animated at a time
    def __init__(self,dict_builder,fig,axes,im,im_norm,rgb_fp,img_counter,mode):
        """
        :param dict_builder (dict): use it to save bboxes for different categories
        :param fig (mpl figure object)
        :param axes (mpl Axes object)
        :param im (imshow object): that plots true rgb color
        :param imnorm (imshow object): that plots normalised rgb color
        :param rgb_fp (list of str): list of filepaths
        :param img_counter (int): to keep track of the image counter/img line
        """
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
        self.close_figure = fig.canvas.mpl_connect('close_event', self.on_close)
        self.fig = fig
        self.axes = axes
        self.im = im
        self.im_norm = im_norm
        self.rgb_fp = rgb_fp
        self.n_img = len(rgb_fp)
        # keep track of current img counter
        self.img_counter = img_counter
        self.current_fp = rgb_fp[self.img_counter]
        # keep track of saved img counter, only keep track when bboxes are drawn
        self.save_counter = img_counter
        self.mode = mode

    def __call__(self, event):
        if any([event.inaxes != getattr(self,k+'_line').axes for k in self.categories]) is True:
            print("No call")
            return
        
        else:
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
        # img_index = self.img_counter%self.n_img
        # current_fp = self.rgb_fp[img_index] 
        # self.current_fp = current_fp

        # get current bbox
        k = LineBuilder.lock
        x1,x2 = getattr(self,k+'_x')[-2:]
        y1,y2 = getattr(self,k+'_y')[-2:]
        setattr(self, k+'_bbox', ((int(x1),int(y1)),(int(x2),int(y2))))

        # draw
        h = y2 - y1
        w = x2 - x1
        getattr(self,k+'_patch').set_xy((x1,y1))
        getattr(self,k+'_patch').set_height(h)
        getattr(self,k+'_patch').set_width(w)
        
        # keep track of save counter when drawing bboxes
        self.save_counter = self.img_counter%self.n_img
        

    def reset(self, _event):
        """clear all points, lines and patches."""
        self.save_counter = self.img_counter%self.n_img
        for k in self.categories:
            setattr(self, k+'_x', [])
            setattr(self, k+'_y', [])
            setattr(self, k+'_bbox', None)
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
        
        # save the last drawn bbox's fp
        save_fp = self.rgb_fp[self.save_counter] #if bbox is drawn, there is definitely a save counter
        save_bboxes = {save_fp:{k: getattr(self,k+'_bbox') for k in self.categories}}
        
        # save if bboxes is not all None
        if len([i for i in save_bboxes[list(save_bboxes)[0]].values() if i is not None]) > 0:
            # print(save_bboxes)
            print(f'IMG_index: {self.save_counter}\nFilename: {save_fp}')
            # get unique filename from current_fp

            if self.mode == 'raw':
                fn = get_all_dir(save_fp,iter=4)
                #create a new dir to store bboxes
                store_dir = join(getcwd(),"saved_bboxes")
                #create a new dir to store plot images
                plot_dir = join(getcwd(),"saved_plots")

            elif self.mode == 'rgb':
                fn = splitext(basename(save_fp))[0]
                #create a new dir to store bboxes
                store_dir = join(getcwd(),"QAQC_relabel")
                plot_dir = store_dir

            if not exists(store_dir):
                mkdir(store_dir)

            if not exists(plot_dir):
                mkdir(plot_dir)

            fp_store = join(store_dir,fn)
            fp_store = fp_store.replace('.tif','')
            print(f"File saved at {fp_store}!")

            fp_plot = join(plot_dir,fn)
            fp_plot = fp_plot.replace('.tif','')
            print(f"File saved at {fp_plot}!")
            
            with open('{}.txt'.format(fp_store),'w') as cf:
                json.dump(save_bboxes,cf)
            
            # save plt
            self.fig.savefig('{}.png'.format(fp_plot))

    def next(self, _event):
        
        self.save(_event) # save previous data first then reset
        self.reset(_event) # clear all bboxes
        self.img_counter = self.img_counter+1
        img_index = self.img_counter%self.n_img
        self.current_fp = self.rgb_fp[img_index] #update fp with current img counter

        if self.mode == 'raw':
            cap = mutils.import_captures(self.current_fp)
            img, img_norm = aligned_capture(cap)
        
        elif self.mode == 'rgb':
            img = np.asarray(Image.open(self.current_fp))
        
        # img = np.asarray(Image.open(current_fp))
        # img = Image.open(current_fp)
        img_line = self.current_fp.split('IMG_')[1][:4] #get the current image line
        # self.ax.set_title('Select T,W,TG, WG, S areas\nImage Index {}'.format(img_line))
        self.fig.suptitle('Select T,W,TG, WG, S areas\n{}\nImage Index {}'.format(self.current_fp,img_line))

        self.im.set_extent((0,img.shape[1],img.shape[0],0)) # left, right, bottom, top
        self.im.set_data(img)
        self.im.figure.canvas.draw_idle()

        if self.mode == 'raw':
            self.im_norm.set_extent((0,img_norm.shape[1],img_norm.shape[0],0)) # left, right, bottom, top
            self.im_norm.set_data(img_norm)
            self.im_norm.figure.canvas.draw_idle()

        if self.img_counter == self.n_img-1:
            self.fig.suptitle('END OF IMAGES. CONGRATULATIONS! STOP LABELLING AFTER THIS!')
        
    def previous(self,_event):
        self.img_counter = self.img_counter-1
        img_index = self.img_counter%self.n_img
        self.current_fp = self.rgb_fp[img_index]

        if self.mode == 'raw':
            cap = mutils.import_captures(self.current_fp)
            img, img_norm = aligned_capture(cap)
        
        elif self.mode == 'rgb':
            img = np.asarray(Image.open(self.current_fp))
        
        # img = np.asarray(Image.open(self.current_fp))
        # img = Image.open(self.current_fp)
        self.reset(_event)
        img_line = self.current_fp.split('IMG_')[1][:4] #get the current image line
        # self.ax.set_title('Select T,W,TG, WG, S areas\nImage Index {}'.format(img_line))
        self.fig.suptitle('Select T,W,TG, WG, S areas\n{}\nImage Index {}'.format(self.current_fp,img_line))

        self.im.set_extent((0,img.shape[1],img.shape[0],0)) # left, right, bottom, top
        self.im.set_data(img)
        self.im.figure.canvas.draw_idle()

        if self.mode == 'raw':
            self.im_norm.set_extent((0,img_norm.shape[1],img_norm.shape[0],0)) # left, right, bottom, top
            self.im_norm.set_data(img_norm)
            self.im_norm.figure.canvas.draw_idle()
    
    def submit(self,_expression):
        """
        allows user to jump directly to an image index
        """
        if self.mode == 'raw':
            img_index = int(_expression)
            # evaluate expression and update image index
            self.img_counter = img_index%self.n_img
            self.current_fp = self.rgb_fp[self.img_counter]

            # update images
            cap = mutils.import_captures(self.current_fp)
            img, img_norm = aligned_capture(cap)
        
        elif self.mode == 'rgb':
            fn = splitext(_expression)[0] #split any extension
            # evaluate expression and update image index
            self.img_counter = [i for i, fp in enumerate(self.rgb_fp) if fn in fp][0]
            self.current_fp = self.rgb_fp[self.img_counter]
            img = np.asarray(Image.open(self.current_fp))

        img_line = self.current_fp.split('IMG_')[1][:4] #get the current image line
        self.fig.suptitle('Select T,W,TG, WG, S areas\n{}\nImage Index {}'.format(self.current_fp,img_line))

        self.im.set_extent((0,img.shape[1],img.shape[0],0)) # left, right, bottom, top
        self.im.set_data(img)
        self.im.figure.canvas.draw_idle()

        if self.mode == 'raw':
            self.im_norm.set_extent((0,img_norm.shape[1],img_norm.shape[0],0)) # left, right, bottom, top
            self.im_norm.set_data(img_norm)
            self.im_norm.figure.canvas.draw_idle()

        # clear all bboxes
        for k in self.categories:
            setattr(self, k+'_x', [])
            setattr(self, k+'_y', [])
            setattr(self, k+'_bbox', None)
            x1 = y1 = h = w = 0
            getattr(self,k+'_line').set_data([],[])
            getattr(self,k+'_patch').set_xy((x1,y1))
            getattr(self,k+'_patch').set_height(h)
            getattr(self,k+'_patch').set_width(w)
            getattr(self,k+'_line').figure.canvas.draw_idle()

    def on_close(self,_event):
        #create a new dir to store plot images
        plot_dir = join(getcwd(),"QAQC_relabel")
        if not exists(plot_dir):
            mkdir(plot_dir)
        
        last_img = splitext(basename(self.current_fp))[0]
        with open(join(plot_dir,'last_image.txt'),'w') as cf:
            cf.write(last_img)

def draw_sunglint_correction(fp_store):
    if 'QAQC_plots' in fp_store:
        mode = 'rgb'
    else:
        mode = 'raw'

    # initialise categories
    button_names = ['turbid_glint','water_glint','turbid','water','shore']
    
    # intialise colours
    colors = ['orange','cyan','saddlebrown','blue','yellow']

    # import files
    rgb_fp = [join(fp_store,f) for f in sorted(listdir(fp_store)) if (f.endswith("1.tif") and mode == 'raw') or (f.endswith("1.png") and mode == 'rgb')]
    
    # initialise dictionary to store stuff
    dict_builder = {b:None for b in button_names}
    
    if mode == 'raw':
        # go to the last processed image
        img_counter = get_current_img_counter(fp_store,mode)
        # initialise plot
        fig, axes = plt.subplots(1,2,figsize=(15,8))

        cap = mutils.import_captures(rgb_fp[img_counter])
        global warp_matrices
        global cropped_dimensions
        warp_matrices = cap.get_warp_matrices()
        cropped_dimensions,_ = imageutils.find_crop_bounds(cap,warp_matrices)

        img, img_norm = aligned_capture(cap)
        im = axes[0].imshow(img)
        im_norm = axes[1].imshow(img_norm)
        axes[0].set_title('True RGB')
        axes[1].set_title('Normalised RGB')
        for ax in axes:
            ax.axis('off')
        
        # set title
        img_line = rgb_fp[img_counter].split('IMG_')[1][:4]
        fig.suptitle('Select T,W,TG, WG, S areas\n{}\nImage Index {}'.format(rgb_fp[img_counter],img_line))
        
        # add to plot
        for button,c in zip(button_names,colors):
            line_category, = axes[1].plot(0,0,"o",c=c)
            rect = patches.Rectangle((0, 0), 10, 10, linewidth=1, edgecolor=c, facecolor='none')
            patch = axes[1].add_patch(rect)
            dict_builder[button] = {'line':line_category,'patch':patch}

        # initialise LineBuilder
        linebuilder = LineBuilder(dict_builder,fig,axes,im,im_norm,rgb_fp,img_counter,mode)
        plt.subplots_adjust(bottom=0.2,right=0.8)

        # add text box
        textbox_ax = plt.axes([0.2, 0.15, 0.3, 0.075])
        text_box = TextBox(textbox_ax, "IMG_index", textalignment="center")
        text_box.on_submit(linebuilder.submit)
        text_box.set_val(f"{img_counter}")

    elif mode == 'rgb':
        # go to the last processed image
        img_counter = get_current_img_counter(rgb_fp,mode)
        # initialise plot
        fig, axes = plt.subplots(figsize=(15,10))
        fn = rgb_fp[img_counter]
        img = np.asarray(Image.open(fn))
        im = axes.imshow(img)
        axes.set_title('True RGB')
        axes.axis('off')
        # set title
        img_line = fn.split('IMG_')[1][:4]
        fig.suptitle('Check T,W,TG, WG, S areas\n{}\nImage Index {}'.format(fn,img_line))

        # add to plot
        for button,c in zip(button_names,colors):
            line_category, = axes.plot(0,0,"o",c=c)
            rect = patches.Rectangle((0, 0), 10, 10, linewidth=1, edgecolor=c, facecolor='none')
            patch = axes.add_patch(rect)
            dict_builder[button] = {'line':line_category,'patch':patch}

        # initialise LineBuilder
        linebuilder = LineBuilder(dict_builder,fig,axes,im,None,rgb_fp,img_counter,mode)
        plt.subplots_adjust(bottom=0.2,right=0.8)

        # add text box
        textbox_ax = plt.axes([0.2, 0.15, 0.3, 0.05])
        text_box = TextBox(textbox_ax, "Filename", textalignment="center")
        text_box.on_submit(linebuilder.submit)
        text_box.set_val(f"{basename(fn)}")
    
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
    axes_buttons = [plt.axes([0.82, 0.8-i/10, 0.15, 0.075]) for i in range(len(button_names))]
    buttons = [Button(a,name) for a,name in zip(axes_buttons,button_names)]
    for k,button in zip(button_names,buttons):
        button.on_clicked(getattr(linebuilder,k)) 
    plt.show()

if __name__ == '__main__':
    
    fp_store = input("Enter directory: ")#r"C:\Users\PAKHUIYING\Downloads\test" #r"C:\Users\PAKHUIYING\Documents\image_processing\F3_processed_surveys\2021_11_10_2"
    fp_store = fp_store.replace("\\","/").replace("\"","")
    draw_sunglint_correction(fp_store)