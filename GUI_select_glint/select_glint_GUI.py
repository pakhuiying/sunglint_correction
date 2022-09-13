from matplotlib import pyplot as plt
from matplotlib.widgets import Button
import matplotlib.patches as patches
import PIL.Image as Image
import json
from os.path import join
from os import listdir
import numpy as np
from option import args
# import PySimpleGUI as sg
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
# from matplotlib.backend_bases import key_press_handler

class LineBuilder:
    """
    line_glint (mpl.lines.Line2D Line2D artist): line plot of the glint region
    line_nonglint (mpl.lines.Line2D Line2D artist): line plot of the nonglint region
    fig (plot figure)
    ax (axes figure)
    im (Axes.imshow): image plot from ax.imshow()
    prefix (str): prefix of json file to be saved
    rgb_fp (list of str): list of file paths to the image
    fp_save (str): directory of where you want to save the json file
    """
    lock = "glint"  # only one can be animated at a time i.e. either draw glint box or nonglint box
    def __init__(self,line_glint,line_nonglint,r_glint,r_nonglint,fig,ax,im,prefix,rgb_fp,fp_save):
        self.prefix = prefix
        self.line_glint = line_glint
        self.r_glint = r_glint
        self.line_nonglint = line_nonglint
        self.r_nonglint = r_nonglint
        self.xs_glint = []#list(line.get_xdata())
        self.ys_glint = []#list(line.get_ydata())
        self.xs_nonglint = []#list(line.get_xdata())
        self.ys_nonglint = []#list(line.get_ydata())
        # self.cid = fig.canvas.mpl_connect('button_press_event', self)
        self.cid = fig.canvas.mpl_connect('button_press_event', self)
        self.ax = ax
        self.im = im
        self.rgb_fp = rgb_fp 
        self.fp_save = fp_save
        self.n_img = len(rgb_fp)
        self.img_counter = 0
        #to save into json file, keep track of the last drawn bbox drawn on the img_line
        self.img_line_glint = 0
        self.img_line_nonglint = 0
        self.img_bbox_glint = None
        self.img_bbox_nonglint = None


    def __call__(self, event):
        if event.inaxes!=self.line_glint.axes or event.inaxes!=self.line_nonglint.axes:
            return
        if LineBuilder.lock == "glint":
            self.xs_glint.append(event.xdata)
            self.ys_glint.append(event.ydata)
            self.line_glint.set_data(self.xs_glint[-2:], self.ys_glint[-2:])
            self.line_glint.figure.canvas.draw_idle()
            print(self.xs_glint[-2:])
            print(self.ys_glint[-2:])
            self.draw_rect(event)
        else:
            self.xs_nonglint.append(event.xdata)
            self.ys_nonglint.append(event.ydata)
            self.line_nonglint.set_data(self.xs_nonglint[-2:], self.ys_nonglint[-2:])
            self.line_nonglint.figure.canvas.draw_idle()
            print(self.xs_nonglint[-2:])
            print(self.ys_nonglint[-2:])
            self.draw_rect(event)

    def draw_rect(self, _event):
        img_index = self.img_counter%self.n_img
        current_fp = self.rgb_fp[img_index] 
        # img_line = int(current_fp.split('line_')[1][:2]) #get the current image line
        if LineBuilder.lock == "glint":
            x1,x2 = self.xs_glint[-2:]
            y1,y2 = self.ys_glint[-2:]
            self.img_bbox_glint = ((int(x1),int(y1)),(int(x2),int(y2)))
            h = y2 - y1
            w = x2 - x1
            self.r_glint.set_xy((x1,y1))
            self.r_glint.set_height(h)
            self.r_glint.set_width(w)
            self.img_line_glint = current_fp #update img_line based don the latest rect patch drawn
        else:
            x1,x2 = self.xs_nonglint[-2:]
            y1,y2 = self.ys_nonglint[-2:]
            self.img_bbox_nonglint = ((int(x1),int(y1)),(int(x2),int(y2)))
            h = y2 - y1
            w = x2 - x1
            self.r_nonglint.set_xy((x1,y1))
            self.r_nonglint.set_height(h)
            self.r_nonglint.set_width(w)
            self.img_line_nonglint = current_fp #update img_line based don the latest rect patch drawn

    def reset(self, _event):
        self.xs_glint = []
        self.ys_glint = []
        self.xs_nonglint = []
        self.ys_nonglint = []
        x1 = y1 = h = w = 0
        self.line_glint.set_data(self.xs_glint, self.ys_glint)
        self.r_glint.set_xy((x1,y1))
        self.r_glint.set_height(h)
        self.r_glint.set_width(w)
        self.line_glint.figure.canvas.draw_idle()

        self.line_nonglint.set_data(self.xs_nonglint, self.ys_nonglint)
        self.r_nonglint.set_xy((x1,y1))
        self.r_nonglint.set_height(h)
        self.r_nonglint.set_width(w)
        self.line_nonglint.figure.canvas.draw_idle()

    def glint(self, _event):
        LineBuilder.lock = "glint"

    def non_glint(self, _event):
        LineBuilder.lock = "nonglint"
    
    def save(self, _event):
        # x1_glint,x2_glint = self.xs_glint[-2:]
        # y1_glint,y2_glint = self.ys_glint[-2:]
        # x1_nonglint,x2_nonglint = self.xs_nonglint[-2:]
        # y1_nonglint,y2_nonglint = self.ys_nonglint[-2:]
        line_glint = int(self.img_line_glint.split('line_')[1][:2]) #get the current image line
        line_nonglint = int(self.img_line_nonglint.split('line_')[1][:2]) #get the current image line
        bboxes = {'glint':{'line':line_glint,'fp':self.img_line_glint,'bbox':self.img_bbox_glint},\
            'non_glint':{'line':line_nonglint,'fp':self.img_line_nonglint,'bbox':self.img_bbox_nonglint}}
        with open(join(self.fp_save,'glint_{}.txt'.format(self.prefix)),'w') as cf:
            json.dump(bboxes,cf)

    def next(self, _event):
        self.img_counter = self.img_counter+1
        img_index = self.img_counter%self.n_img
        current_fp = self.rgb_fp[img_index] 
        img_line = int(current_fp.split('line_')[1][:2]) #get the current image line
        img = np.asarray(Image.open(current_fp))
        self.reset(_event)
        self.ax.set_title('Select glint (r) & non-glint (p) areas\nLine {}'.format(img_line))
        self.im.set_extent((0,img.shape[1],0,img.shape[0]))
        self.im.set_data(img)
        self.im.figure.canvas.draw_idle()
        
        # im.set_data

def draw_sunglint_correction(args):
    fig, ax = plt.subplots()
    # fp = r"C:\Users\PAKHUIYING\Documents\image_processing\F3_processed_surveys\2021_11_10\11_34_17\2021_10_11_11-34-17_rgb_image_line_08_15713_17196.tif"
    fp_store = args.fp_store
    rgb_fp = [join(fp_store,f) for f in listdir(fp_store) if (f.endswith(".tif"))]
    # print(rgb_fp)

    # assert os.path.exists(user_input), "I did not find the file at, "+str(user_input)
    # fp = open(user_input,'r+')
    # print("Hooray we found your file!")
    img = Image.open(rgb_fp[0])
    img_line = int(rgb_fp[0].split('line_')[1][:2])
    im = ax.imshow(img)
    ax.set_title('Select glint (r) & non-glint (p) areas\nLine {}'.format(img_line))
    line_glint, = ax.plot(0,0,"o",c="r")
    line_nonglint, = ax.plot(0,0,"o",c="purple")
    # Create a Rectangle patch
    rect_glint = patches.Rectangle((0, 0), 10, 10, linewidth=1, edgecolor='r', facecolor='none')
    r_glint = ax.add_patch(rect_glint)

    rect_nonglint = patches.Rectangle((0, 0), 10, 10, linewidth=1, edgecolor='purple', facecolor='none')
    r_nonglint = ax.add_patch(rect_nonglint)

    linebuilder = LineBuilder(line_glint,line_nonglint,r_glint,r_nonglint,fig,ax,im,args.prefix,rgb_fp,args.fp_save)

    plt.subplots_adjust(bottom=0.2)
    #reset button
    breset_ax = plt.axes([0.5, 0.05, 0.1, 0.075]) #left,bottom, width, height
    breset = Button(breset_ax, 'Reset')
    breset.on_clicked(linebuilder.reset)
    #glint areas
    glint_ax = plt.axes([0.2, 0.05, 0.1, 0.075]) #left,bottom, width, height
    glint = Button(glint_ax, 'Glint')
    glint.on_clicked(linebuilder.glint)
    #non-glint areas
    non_glint_ax = plt.axes([0.3, 0.05, 0.15, 0.075]) #left,bottom, width, height
    non_glint = Button(non_glint_ax, 'Non-glint')
    non_glint.on_clicked(linebuilder.non_glint)
    #next image
    next_ax = plt.axes([0.6, 0.05, 0.15, 0.075]) #left,bottom, width, height
    next = Button(next_ax, 'Next image')
    next.on_clicked(linebuilder.next)
    #save coordinates
    save_ax = plt.axes([0.8, 0.05, 0.1, 0.075]) #left,bottom, width, height
    save = Button(save_ax, 'Save')
    save.on_clicked(linebuilder.save)
    plt.show()

if __name__ == '__main__':
    draw_sunglint_correction(args)