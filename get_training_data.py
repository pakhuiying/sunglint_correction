from matplotlib import pyplot as plt
from matplotlib.widgets import Button
import matplotlib.patches as patches
import PIL.Image as Image
import json
from os.path import join, split
from os import listdir
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler

def get_all_dir(fp,iter=3):
    """ get all parent sub directories up to 3 levels"""
    fp_temp = fp
    sub_dir_list = []
    for i in range(iter):
        base_fn, fn = split(fp_temp)
        sub_dir_list.append(fn)
        fp_temp = base_fn
    return '_'.join(reversed(sub_dir_list))

class LineBuilder:
    lock = "water"  # only one can be animated at a time
    def __init__(self,dict_builder,fig,ax,im,rgb_fp,postfix):
        self.dict_builder = dict_builder
        self.categories = list(dict_builder)
        self.postfix = postfix
        # self.prefix = prefix
        # self.line_glint = line_glint
        # self.r_glint = r_glint
        # self.line_nonglint = line_nonglint
        # self.r_nonglint = r_nonglint
        # self.xs_glint = []#list(line.get_xdata())
        # self.ys_glint = []#list(line.get_ydata())
        # self.xs_nonglint = []#list(line.get_xdata())
        # self.ys_nonglint = []#list(line.get_ydata())
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
        self.img_counter = 0
        self.current_fp = None
        #to save into json file, keep track of the last drawn bbox drawn on the img_line
        # self.img_line_glint = 0
        # self.img_line_nonglint = 0
        # self.img_bbox_glint = None
        # self.img_bbox_nonglint = None


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
        # if event.inaxes!=self.line_glint.axes or event.inaxes!=self.line_nonglint.axes:
        #     return
        # if LineBuilder.lock == "glint":
        #     self.xs_glint.append(event.xdata)
        #     self.ys_glint.append(event.ydata)
        #     self.line_glint.set_data(self.xs_glint[-2:], self.ys_glint[-2:])
        #     self.line_glint.figure.canvas.draw_idle()
        #     print(self.xs_glint[-2:])
        #     print(self.ys_glint[-2:])
        #     self.draw_rect(event)
        # else:
        #     self.xs_nonglint.append(event.xdata)
        #     self.ys_nonglint.append(event.ydata)
        #     self.line_nonglint.set_data(self.xs_nonglint[-2:], self.ys_nonglint[-2:])
        #     self.line_nonglint.figure.canvas.draw_idle()
        #     print(self.xs_nonglint[-2:])
        #     print(self.ys_nonglint[-2:])
        #     self.draw_rect(event)

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
        # if LineBuilder.lock == "glint":
        #     x1,x2 = self.xs_glint[-2:]
        #     y1,y2 = self.ys_glint[-2:]
        #     self.img_bbox_glint = ((int(x1),int(y1)),(int(x2),int(y2)))
        #     h = y2 - y1
        #     w = x2 - x1
        #     self.r_glint.set_xy((x1,y1))
        #     self.r_glint.set_height(h)
        #     self.r_glint.set_width(w)
        #     self.img_line_glint = current_fp #update img_line based don the latest rect patch drawn
        # else:
        #     x1,x2 = self.xs_nonglint[-2:]
        #     y1,y2 = self.ys_nonglint[-2:]
        #     self.img_bbox_nonglint = ((int(x1),int(y1)),(int(x2),int(y2)))
        #     h = y2 - y1
        #     w = x2 - x1
        #     self.r_nonglint.set_xy((x1,y1))
        #     self.r_nonglint.set_height(h)
        #     self.r_nonglint.set_width(w)
        #     self.img_line_nonglint = current_fp #update img_line based don the latest rect patch drawn

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
        # self.xs_glint = []
        # self.ys_glint = []
        # self.xs_nonglint = []
        # self.ys_nonglint = []
        # x1 = y1 = h = w = 0
        # self.line_glint.set_data(self.xs_glint, self.ys_glint)
        # self.r_glint.set_xy((x1,y1))
        # self.r_glint.set_height(h)
        # self.r_glint.set_width(w)
        # self.line_glint.figure.canvas.draw_idle()

        # self.line_nonglint.set_data(self.xs_nonglint, self.ys_nonglint)
        # self.r_nonglint.set_xy((x1,y1))
        # self.r_nonglint.set_height(h)
        # self.r_nonglint.set_width(w)
        # self.line_nonglint.figure.canvas.draw_idle()

    # def glint(self, _event):
    #     LineBuilder.lock = "glint"

    # def non_glint(self, _event):
    #     LineBuilder.lock = "nonglint"
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
        with open('identify_glint_{}.txt'.format(self.postfix),'w') as cf:
            json.dump(save_bboxes,cf)
        # line_glint = int(self.img_line_glint.split('line_')[1][:2]) #get the current image line
        # line_nonglint = int(self.img_line_nonglint.split('line_')[1][:2]) #get the current image line
        # bboxes = {'glint':{'line':line_glint,'fp':self.img_line_glint,'bbox':self.img_bbox_glint},\
        #     'non_glint':{'line':line_nonglint,'fp':self.img_line_nonglint,'bbox':self.img_bbox_nonglint}}
        # with open('sunglint_correction_{}.txt'.format(self.prefix),'w') as cf:
        #     json.dump(bboxes,cf)

    def next(self, _event):
        self.img_counter = self.img_counter+1
        img_index = self.img_counter%self.n_img
        current_fp = self.rgb_fp[img_index] 
        img_line = current_fp.split('IMG_')[1][:4] #get the current image line
        img = np.asarray(Image.open(current_fp))
        self.reset(_event)
        self.ax.set_title('Select T,W,TG, WG, S areas\nImage Index {}'.format(img_line))
        self.im.set_extent((0,img.shape[1],0,img.shape[0]))
        self.im.set_data(img)
        self.im.figure.canvas.draw_idle()
        
    def previous(self,_event):
        self.img_counter = self.img_counter-1
        img_index = self.img_counter%self.n_img
        current_fp = self.rgb_fp[img_index] 
        img_line = current_fp.split('IMG_')[1][:4] #get the current image line
        img = np.asarray(Image.open(current_fp))
        self.reset(_event)
        self.ax.set_title('Select T,W,TG, WG, S areas\nImage Index {}'.format(img_line))
        self.im.set_extent((0,img.shape[1],0,img.shape[0]))
        self.im.set_data(img)
        self.im.figure.canvas.draw_idle()

def draw_sunglint_correction(postfix,fp_store):
    # initialise plot
    fig, ax = plt.subplots()

    # import files
    rgb_fp = [join(fp_store,f) for f in listdir(fp_store) if f.endswith("1.tif")]
    print(rgb_fp)
    img = Image.open(rgb_fp[0])
    img_line = rgb_fp[0].split('IMG_')[1][:4]
    print(img_line)
    im = ax.imshow(img)

    # set title
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
    linebuilder = LineBuilder(dict_builder,fig,ax,im,rgb_fp,postfix)

    # line_glint, = ax.plot(0,0,"o",c="r")
    # line_nonglint, = ax.plot(0,0,"o",c="purple")
    # # Create a Rectangle patch
    # rect_glint = patches.Rectangle((0, 0), 10, 10, linewidth=1, edgecolor='r', facecolor='none')
    # r_glint = ax.add_patch(rect_glint)

    # rect_nonglint = patches.Rectangle((0, 0), 10, 10, linewidth=1, edgecolor='purple', facecolor='none')
    # r_nonglint = ax.add_patch(rect_nonglint)

    # linebuilder = LineBuilder(line_glint,line_nonglint,r_glint,r_nonglint,fig,ax,im,prefix,rgb_fp)

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
    # #turbid glint areas
    # glint_ax = plt.axes([0.82, 0.9, 0.1, 0.075]) #left,bottom, width, height
    # glint = Button(glint_ax, 'turbid_glint')
    # glint.on_clicked(linebuilder.glint)
    # #water glint areas
    # non_glint_ax = plt.axes([0.82, 0.8, 0.15, 0.075]) #left,bottom, width, height
    # non_glint = Button(non_glint_ax, 'water_glint')
    # non_glint.on_clicked(linebuilder.non_glint)
    plt.show()


if __name__ == '__main__':
    # prefix = input("Enter prefix:")
    
    fp_store = input("Enter directory: ")#r"C:\Users\PAKHUIYING\Downloads\test" #r"C:\Users\PAKHUIYING\Documents\image_processing\F3_processed_surveys\2021_11_10_2"
    fp_store = fp_store.replace("\\","/").replace("\"","")
    postfix = get_all_dir(fp_store,iter=3)
    draw_sunglint_correction(postfix,fp_store)