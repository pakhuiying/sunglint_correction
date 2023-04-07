from matplotlib import pyplot as plt
from matplotlib.widgets import Button, TextBox
from matplotlib.lines import Line2D
import PIL.Image as Image
import os
import numpy as np

def get_current_img_counter(rgb_fp):
    """ 
    :param filename (str): filename in last_image.txt
    :param rgb_fp (list of str): sorted list of image files from saved_plots
    """
    last_image_fp = os.path.join("QAQC_relabel","last_image.txt")
    if os.path.exists(last_image_fp):
        with open(last_image_fp) as f:
            last_image = f.readlines()[0]

        img_counter = [i for i,fp in enumerate(rgb_fp) if last_image in fp][0]
    else:
        img_counter = 0

    return img_counter

class LineBuilder:
    def __init__(self,fig,axes,im,rgb_fp,img_counter):
        """
        :param fig (mpl figure object)
        :param axes (mpl Axes object)
        :param im (imshow object): that plots true rgb color
        :param imnorm (imshow object): that plots normalised rgb color
        :param rgb_fp (list of str): list of filepaths
        :param img_counter (int): to keep track of the image counter/img line
        """
        self.cid = fig.canvas.mpl_connect('button_press_event', self)
        self.close_figure = fig.canvas.mpl_connect('close_event', self.on_close)
        self.fig = fig
        self.axes = axes
        self.im = im
        self.rgb_fp = rgb_fp
        self.n_img = len(rgb_fp)
        # keep track of current img counter
        self.img_counter = img_counter
        self.current_fp = rgb_fp[self.img_counter]
        # keep track of saved img counter, only keep track when bboxes are drawn
        self.save_counter = img_counter

    # def __call__(self, event):
    #     if any([event.inaxes != getattr(self,k+'_line').axes for k in self.categories]) is True:
    #         print("No call")
    #         return
        
    #     else:
    #         getattr(self,k+'_line').figure.canvas.draw_idle()
    #         self.draw_rect(event)
    def on_close(self,_event):
        #create a new dir to store plot images
        plot_dir = os.path.join(os.getcwd(),"QAQC_relabel")
        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)
        
        last_img = os.path.splitext(os.path.basename(self.current_fp))[0]
        with open(os.path.join(plot_dir,'last_image.txt'),'w') as cf:
            cf.write(last_img)
    
    def relabel(self, _event):
        
        #create a new dir to store plot images
        plot_dir = os.path.join(os.getcwd(),"QAQC_relabel")
        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)

        fn = os.path.basename(self.current_fp)
        fp_store = os.path.join(plot_dir,fn)
        fp_store = fp_store.replace('.png','.txt')
        print(f"File saved at {fp_store}!")
        
        with open(fp_store,'w') as cf:
            cf.write(self.current_fp)
            

    def next(self, _event):
        
        self.img_counter = self.img_counter+1
        img_index = self.img_counter%self.n_img
        self.current_fp = self.rgb_fp[img_index] #update fp with current img counter

        img = np.asarray(Image.open(self.current_fp))
        # img = Image.open(self.current_fp)
        img_line = self.current_fp.split('IMG_')[1][:4] #get the current image line
        # self.ax.set_title('Select T,W,TG, WG, S areas\nImage Index {}'.format(img_line))
        self.fig.suptitle('{}\nImage Index {}'.format(self.current_fp,img_line))

        self.im.set_extent((0,img.shape[1],img.shape[0],0)) # left, right, bottom, top
        self.im.set_data(img)
        self.im.figure.canvas.draw_idle()
        
    def previous(self,_event):
        self.img_counter = self.img_counter-1
        img_index = self.img_counter%self.n_img
        self.current_fp = self.rgb_fp[img_index]

        img = np.asarray(Image.open(self.current_fp))
        img_line = self.current_fp.split('IMG_')[1][:4] #get the current image line
        # self.ax.set_title('Select T,W,TG, WG, S areas\nImage Index {}'.format(img_line))
        self.fig.suptitle('{}\nImage Index {}'.format(self.current_fp,img_line))

        self.im.set_extent((0,img.shape[1],img.shape[0],0)) # left, right, bottom, top
        self.im.set_data(img)
        self.im.figure.canvas.draw_idle()
    
    def submit(self,_expression):
        """
        allows user to jump directly to an image index
        """
        fn = os.path.splitext(_expression)[0] #split any extension
        # evaluate expression and update image index
        self.img_counter = [i for i, fp in enumerate(self.rgb_fp) if fn in fp][0]
        self.current_fp = self.rgb_fp[self.img_counter]

        # update images
        
        img = np.asarray(Image.open(self.current_fp))
        img_line = self.current_fp.split('IMG_')[1][:4] #get the current image line
        self.fig.suptitle('{}\nImage Index {}'.format(self.current_fp,img_line))

        self.im.set_extent((0,img.shape[1],img.shape[0],0)) # left, right, bottom, top
        self.im.set_data(img)
        self.im.figure.canvas.draw_idle()

def draw_sunglint_correction(fp_store):
    # initialise plot
    fig, axes = plt.subplots(figsize=(15,12))

    # import files
    rgb_fp = [os.path.join(fp_store,f) for f in sorted(os.listdir(fp_store)) if f.endswith("1.png")]
    # go to the last processed image
    # img_counter = get_current_img_counter(fp_store)
    img_counter = get_current_img_counter(rgb_fp)
    
    fn = rgb_fp[img_counter]
    img = np.asarray(Image.open(fn))
    im = axes.imshow(img)

    # set title
    img_line = fn.split('IMG_')[1][:4]
    fig.suptitle('{}\nImage Index {}'.format(fn,img_line))

    # initialise categories
    labels = ['turbid_glint','water_glint','turbid','water','shore']
    
    # intialise colours
    colors = ['orange','cyan','saddlebrown','blue','yellow']
    
    lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in colors]

    # initialise LineBuilder
    linebuilder = LineBuilder(fig,axes,im,rgb_fp,img_counter)

    plt.subplots_adjust(bottom=0.2,right=0.8)
    #prev image
    prev_ax = plt.axes([0.35, 0.05, 0.1, 0.075]) #left,bottom, width, height
    prev = Button(prev_ax, 'Prev')
    prev.on_clicked(linebuilder.previous)
    #next image
    next_ax = plt.axes([0.5, 0.05, 0.1, 0.075]) #left,bottom, width, height
    next = Button(next_ax, 'Next')
    next.on_clicked(linebuilder.next)
    
    # relabel button to flag images that require relabelling
    relabel_ax = plt.axes([0.65, 0.05, 0.1, 0.075]) #left,bottom, width, height
    relabel = Button(relabel_ax, 'Relabel')
    relabel.on_clicked(linebuilder.relabel)
    
    # add text box
    textbox_ax = plt.axes([0.2, 0.15, 0.3, 0.075])
    text_box = TextBox(textbox_ax, "Filename", textalignment="center")
    text_box.on_submit(linebuilder.submit)
    text_box.set_val(f"{os.path.basename(fn)}")
    axes.axis('off')
    plt.legend(lines,labels,loc='center left',bbox_to_anchor=(2, 2))
    plt.show()



if __name__ == '__main__':
    
    fp_store = input("Enter directory: ")#r"C:\Users\PAKHUIYING\Downloads\test" #r"C:\Users\PAKHUIYING\Documents\image_processing\F3_processed_surveys\2021_11_10_2"
    fp_store = fp_store.replace("\\","/").replace("\"","")
    draw_sunglint_correction(fp_store)