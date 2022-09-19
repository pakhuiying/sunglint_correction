import PySimpleGUI as sg
import GUI_select_glint.glint_pixels as gp
import GUI_utils as gu
import utils
import cv2 as cv
import scipy.signal
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
from os.path import join, basename, isdir, isfile, exists
from os import listdir, mkdir
from skimage.color import rgb2gray
from random import randrange,sample
from math import ceil
import json
import ast


sg.ChangeLookAndFeel('Default1')

def get_main_window():

    tab1_layout = [
                [sg.Text('Image folder path:',tooltip='Folder where images are stored')],
                [sg.Input(size=(50,1), key='-IMAGE_FOLDER_PATH-',enable_events=True), sg.FolderBrowse()],
                [sg.Text('Image mask folder path:', tooltip='Folder where binary masks are stored')],
                [sg.Input(size=(50,1), key='-MASK_FOLDER_PATH-',enable_events=True), sg.FolderBrowse()],
                [sg.Text('Prefix for files:')],
                [sg.Input(size=(50,1),default_text='input_prefix',key='-PREFIX-')],
                [sg.Text('Directory to store processed images:')],
                [sg.Input(size=(50,1), key='-STORE_DIR_PATH-'), sg.FolderBrowse()],
            ]

    images_list_col = [
            [sg.Frame(layout=[
                [sg.Text(size=(45,1),key='-BASENAME_IMG-')],
                [sg.Listbox(values=[],enable_events=True,horizontal_scroll=True,size=(60,15),key='-IMAGE_LIST-')], 
            ], title='Image list',font='Arial 8 bold',size=(460,230),element_justification='center')],

            [sg.Frame(layout=[
                [sg.Image(key='-PREVIEW_WHOLE_IMAGE-')],#preview the entire image
                [sg.Button('Split',tooltip='To split the image into 512 by 512 px')], #pop up a window to show all the cut images
            ], title='Image preview',font='Arial 8 bold',size=(460,180),element_justification='center')],        
        ]

    mask_list_col = [
            [sg.Frame(layout=[
                [sg.Text(size=(45,1),key='-BASENAME_MASK-')],
                [sg.Listbox(values=[],enable_events=True,horizontal_scroll=True,size=(60,15),key='-MASK_LIST-')],
            ], title='Mask list',font='Arial 8 bold',size=(460,230),element_justification='center')],

            [sg.Frame(layout=[
                [sg.Image(key='-PREVIEW_WHOLE_MASK-')],#preview the entire image
                
            ], title='Mask preview',font='Arial 8 bold',size=(460,180),element_justification='center')],        
        ]

    tab2_layout = [
            [sg.Frame(layout=[
                [sg.Text('Colour step:',size=(12,1)), sg.Input(size=(5,1),key='-COLOR_STEP-',tooltip='How many colors to sample from mpl.cm Spectral palette (color resolution)')],
                [sg.Text('Spectra step:',size=(12,1)), sg.Input(size=(5,1),key='-SPECTRA_STEP-',tooltip='Number of interpolation steps between spectra to color')],
                [sg.Text('Cut off:',size=(12,1)), sg.Input(size=(5,1),key='-CUT_OFF-',tooltip='How much of the extreme red and blue to remove from cm.Spectral')],
                [sg.Text('Extend extreme:',size=(12,1)), sg.Input(size=(5,1),key='-EXTEND_EXTREME-',tooltip='How much of the red and blue to interpolate to the target spectrum as a multiple of spectra_step')],
            ], title='Glint palette',font='Arial 8 bold',size=(460,130))],

            [sg.Frame(layout=[
                [sg.Text('Brightness:',size=(12,1)), sg.Input(default_text='0,-50',size=(15,1),key='-BRIGHTNESS-',tooltip='[int]: How much white/dark is in the colours')],
                [sg.Text('Saturation:',size=(12,1)), sg.Input(default_text='0.25,0.4',size=(15,1),key='-SATURATION-',tooltip='[0,1] (float): Influences how obvious the rainbow spectra is. Low saturation means it is closer to the water spectra')],
                [sg.Text('Contrast:',size=(12,1)), sg.Input(default_text='1,2',size=(15,1),key='-CONTRAST-',tooltip='[float]: Influences how green/blue the colors are. Lower contrast means the greener/darker the water spectra is')],
                [sg.Text('Height:',size=(12,1)), sg.Input(default_text='0,0',size=(15,1),key='-H-',tooltip='[int]: Height of the glint pixel')],
                [sg.Text('Width:',size=(12,1)), sg.Input(default_text='7,7',size=(15,1),key='-W-',tooltip='[int]: Width of the glint pixel')],
                [sg.Text('Sample colours:',size=(12,1)), sg.Input(default_text='12,24',size=(15,1),key='-SAMPLE_N-',tooltip='[int]: How many different colors to sample from the palette')],
                [sg.Text('Sigma:',size=(12,1)), sg.Input(default_text='1,1',size=(15,1),key='-SIGMA-',tooltip='[int]: Standard deviation of the gaussian smoothing')],
                [sg.Button('Upload parameters',key='-UPLOAD-')]
            ], title='Glint customisation',font='Arial 8 bold',size=(460,240))],

            [sg.Frame(layout=[
                [sg.Text('Mask Threshold:',size=(12,1)), 
                sg.Input(default_text='0.5,0.6',size=(15,1),key='-THRESHOLD-',tooltip='[float]: Threshold for identifying levels of brightness in the image to apply glint on'),
                sg.Button('Generate')],
            ], title='Mask customisation',font='Arial 8 bold',size=(460,60))],

        ]

    tab3_layout = [
            [sg.Frame(layout=[
                [sg.Text('Brightness:',size=(12,1)), sg.Input(size=(5,1),key='-BRIGHTNESS_AUG-',tooltip='[int]: How much white/dark is in the colours')],
                [sg.Text('Saturation:',size=(12,1)), sg.Input(size=(5,1),key='-SATURATION_AUG-',tooltip='[0,1] (float): Influences how obvious the rainbow spectra is. Low saturation means it is closer to the water spectra')],
                [sg.Text('Contrast:',size=(12,1)), sg.Input(size=(5,1),key='-CONTRAST_AUG-',tooltip='[float]: Influences how green/blue the colors are. Lower contrast means the greener/darker the water spectra is')],
                [sg.Text('Sample colours:',size=(12,1)), sg.Input(size=(5,1),key='-SAMPLE_N_AUG-',tooltip='[int]: How many different colors to sample from the palette')],
                [sg.Text('Sigma:',size=(12,1)), sg.Input(size=(5,1),key='-SIGMA_AUG-',tooltip='[int]: Standard deviation of the gaussian smoothing')],
            ], title='Augmentation',font='Arial 8 bold',size=(460,180))],
    ]
    left_layout = [
                [sg.TabGroup([[
                    sg.Tab('Required Inputs',tab1_layout),
                    sg.Tab('Image Preview',images_list_col),
                    sg.Tab('Mask Preview',mask_list_col),
                    sg.Tab('Parameters', tab2_layout),
                    sg.Tab('Augmentation',tab3_layout)
                    ]],font='Arial 9 bold',pad=(5,10))],
                [sg.Button('Process all',key='-PROCESS_ALL-',tooltip='Process all and SAVE the cut images of the same image'), sg.Exit()]
            ]

    # images_list_col = [
    #             [sg.Text(size=(45,1),key='-BASENAME-')],
    #             [sg.Listbox(values=[],enable_events=True,horizontal_scroll=True,size=(40,15),key='-IMAGE_LIST-')],
    #             [sg.Image(key='-PREVIEW_WHOLE_IMAGE-')],#preview the entire image
    #             [sg.Button('Split',tooltip='To split the image into 512 by 512 px')], #pop up a window to show all the cut images
    #         ]
    
    images_col = [
                [sg.Text(size=(45,1),key='-IMAGE_NUM-')],
                [sg.Image(key='-PREVIEW_IMAGE-')],#preview just one cut image
                [sg.Push(),sg.Button('Prev'),sg.Button('Next'),sg.Push()],
                [sg.Push(),sg.Button('Process',tooltip='Only process this 512x512 image'), sg.Button('Save'),sg.Push()],
            ]
    
    layout = [[sg.Column(left_layout,element_justification='l'),
                #sg.Column(images_list_col,element_justification='c'),
                sg.Column(images_col,element_justification='c')]]
    
    return sg.Window('Simulate glint',layout,resizable=True,finalize=True)

#------------------------------EVENT HANDLING-----------------------------------
main_window = get_main_window()
#------------button states of window---------------
main_window['Split'].update(disabled=True)
main_window['Generate'].update(disabled=True)
main_window['Prev'].update(disabled=True)
main_window['Next'].update(disabled=True)
main_window['Process'].update(disabled=True)
main_window['-PROCESS_ALL-'].update(disabled=True)
main_window['Save'].update(disabled=True)
#------------initialisation-----------------------
cut_img_counter = 0


while True:
    window, event, values = sg.read_all_windows()
    print(event,values)
    if event == sg.WIN_CLOSED or event == 'Exit':
        window.close()
        if window == main_window:
            print('main window closed')
            break
    
    if event == '-IMAGE_FOLDER_PATH-':
        non_glint_basename = [f for f in sorted(listdir(values['-IMAGE_FOLDER_PATH-']))]
        # non_glint_fp = [join(values['-IMAGE_FOLDER_PATH-'],f) for f in non_glint_basename]
        window['-IMAGE_LIST-'].update(non_glint_basename)

    if event == '-IMAGE_LIST-':#when selecting an item in the list
        window['Split'].update(disabled=False)

        window['-BASENAME_IMG-'].update(values['-IMAGE_LIST-'][0])
        filename = join(values['-IMAGE_FOLDER_PATH-'],values['-IMAGE_LIST-'][0])
        window['-PREVIEW_WHOLE_IMAGE-'].update(data=gu.convert_to_bytes(filename, resize=(300,300)))
    
    if event == '-MASK_FOLDER_PATH-':
        mask_basename = [f for f in sorted(listdir(values['-MASK_FOLDER_PATH-']))]
        # mask_fp = [join(values['-IMAGE_FOLDER_PATH-'],f) for f in mask_basename]
        window['-MASK_LIST-'].update(mask_basename)

    if event == '-MASK_LIST-':#when selecting an item in the list
        window['-BASENAME_MASK-'].update(values['-MASK_LIST-'][0])
        filename = join(values['-MASK_FOLDER_PATH-'],values['-MASK_LIST-'][0])
        window['-PREVIEW_WHOLE_MASK-'].update(data=gu.array_to_bytes(filename, resize=(300,300),binary=True))

    if event == 'Split':
        filename = join(values['-IMAGE_FOLDER_PATH-'],values['-IMAGE_LIST-'][0])
        img = np.asarray(PIL.Image.open(filename))
        cut_img = utils.cut_into_512(img)
        utils.preview_cut_img(cut_img,gui_plot=True)
        # after cutting image, enable next, prev buttons, and generate buttons
        window['Prev'].update(disabled=False)
        window['Next'].update(disabled=False)
        window['Process'].update(disabled=False)
        window['Generate'].update(disabled=False)
        # update image
        cut_img_counter = 0 #everytime a cut is initiated, counter must return to 0 to ensure the first cut img of a new image is shown
        window['-PREVIEW_IMAGE-'].update(data=gu.array_to_bytes(cut_img[cut_img_counter], resize=(300,300),binary=False))

    if event == 'Generate':
        mask_threshold = values['-THRESHOLD-']
        mask_threshold = mask_threshold.replace(' ','').split(',')
        mask_threshold = np.array([ast.literal_eval(i) for i in mask_threshold])
        # # cut mask
        # filename = join(values['-MASK_FOLDER_PATH-'],values['-MASK_LIST-'][0])
        # mask = np.asarray(PIL.Image.open(filename))
        # cut_mask = utils.cut_into_512(mask)
        # try:
        #     _ = gp.create_masks(cut_img[cut_img_counter%len(cut_img)],cut_mask[cut_img_counter%len(cut_img)],mask_threshold,plot=True,plot_gui=True)
        # except Exception as E:
        #     sg.popup(f'Image has not been split yet. {E}')
        #     pass

        # need to ensure that image and mask are the SAME img
        # ensure that the image previewed in the right column has its corresponding mask, and then different thresholds are applied
        filename_img = join(values['-IMAGE_FOLDER_PATH-'],values['-IMAGE_LIST-'][0])
        img = np.asarray(PIL.Image.open(filename_img))
        basename_img = basename(filename_img).replace('.tif','')
        cut_img = utils.cut_into_512(img)
        filename_mask = [f for f in sorted(listdir(values['-MASK_FOLDER_PATH-'])) if basename_img in f]
        if len(filename_mask) != 1:
            sg.popup(f'Could not precisely locate the corresponding mask for {basename_img}. {len(filename_mask)} masks found.')
            pass
        else:
            filename_mask = filename_mask[0]
            mask = np.asarray(PIL.Image.open(join(values['-MASK_FOLDER_PATH-'],filename_mask)))
            cut_mask = utils.cut_into_512(mask)
            # make sure correct image is selected with the prev and next button
            current_img = cut_img[cut_img_counter%len(cut_img)]
            current_mask = cut_mask[cut_img_counter%len(cut_mask)]
            # just to preview the mask
            _ = gp.create_masks(current_img,current_mask,mask_threshold,plot=True,plot_gui=True)

    if event == 'Next':
        cut_img_counter += 1
        current_img = cut_img[cut_img_counter%len(cut_img)]
        # update image
        window['-IMAGE_NUM-'].update('Image number: {}'.format(str(cut_img_counter).zfill(2)))
        window['-PREVIEW_IMAGE-'].update(data=gu.array_to_bytes(current_img, resize=(300,300),binary=False))
    
    elif event == 'Prev':
        cut_img_counter -= 1
        current_img = cut_img[cut_img_counter%len(cut_img)]
        # update image
        window['-IMAGE_NUM-'].update('Image number: {}'.format(str(cut_img_counter).zfill(2)))
        window['-PREVIEW_IMAGE-'].update(data=gu.array_to_bytes(current_img, resize=(300,300),binary=False))

    if event == '-UPLOAD-':
        params_file = sg.popup_get_file("Upload parameters file")
        try:
            with open(params_file) as cf:
                params_file = json.load(cf)
            for k,v in params_file.items():
                window[k].update(v)
        
        except Exception as E:
            sg.popup(f'{E}')
            pass
            
    if event == 'Process':
        params_dict = {'-BRIGHTNESS-':values['-BRIGHTNESS-'],
                '-SATURATION-':values['-SATURATION-'],
                '-CONTRAST-':values['-CONTRAST-'],
                '-H-':values['-H-'],
                '-W-':values['-W-'],
                '-SAMPLE_N-':values['-SAMPLE_N-'],
                '-SIGMA-':values['-SIGMA-']}
        
        palette_dict = {'-COLOR_STEP-':values['-COLOR_STEP-'],
                    '-SPECTRA_STEP-':values['-SPECTRA_STEP-'],
                    '-CUT_OFF-':values['-CUT_OFF-'],
                    '-EXTEND_EXTREME-':values['-EXTEND_EXTREME-']}

        # make sure that images and mask images are loaded in
        if values['-IMAGE_FOLDER_PATH-'] == '' or values['-MASK_FOLDER_PATH-'] == '':
            sg.popup("Ensure images are loaded in.")

        if ('' in params_dict.values()) is True or (values['-THRESHOLD-'] == ''):
            sg.popup("Empty values in [Glint cutomisation]/[Mask threshold] under [Parameters] tab! Ensure all fields are filled.")
            pass

        else:
            #if no values are entered under palette dict, then use this default value
            palette_number = len([k for k,v in palette_dict.items() if v!=''])
            if palette_number == 0: #if no values are filled in
                palette_dict = {'-COLOR_STEP-':100,
                        '-SPECTRA_STEP-':25,
                        '-CUT_OFF-':0.20,
                        '-EXTEND_EXTREME-':2}
            
            elif palette_number < 4:
                sg.popup('Values in [Glint palette] are only partially filled. Fill all or clear all.')
                pass

            else: #if all values are filled in
                try:
                    for k,v in palette_dict.items():
                        palette_dict[k] = ast.literal_eval(v)
                except Exception as E:
                    sg.popup(f'Values cannot be converted to float or int. {E}')
                    palette_dict = {'-COLOR_STEP-':100,
                        '-SPECTRA_STEP-':25,
                        '-CUT_OFF-':0.20,
                        '-EXTEND_EXTREME-':2}

            # make sure number of values keyed in params is == to the number of threshold values
            mask_threshold = values['-THRESHOLD-']
            mask_threshold = mask_threshold.replace(' ','').split(',')
            mask_threshold = np.array([ast.literal_eval(i) for i in mask_threshold])

            # identify which parameters do not have the same number as mask threshold
            params_error = []
            for k,v in params_dict.items():
                v1 = v.replace(' ','').split(',')
                v1 = np.array([ast.literal_eval(i) for i in v1]) #convert str to float
                if v1.shape != mask_threshold.shape:
                    params_error.append(k)
                else:  
                    params_dict[k] = v1

            if len(params_error) > 0:
                sg.popup("make sure number of values keyed in {} are the same as the number of values in [Mask threshold].".format(params_error))
                pass
            else:
                
                # need to ensure that image and mask are the SAME img
                filename_img = join(values['-IMAGE_FOLDER_PATH-'],values['-IMAGE_LIST-'][0])
                img = np.asarray(PIL.Image.open(filename_img))
                basename_img = basename(filename_img).replace('.tif','')
                cut_img = utils.cut_into_512(img)
                # locate corresponding mask
                filename_mask = [f for f in sorted(listdir(values['-MASK_FOLDER_PATH-'])) if basename_img in f]
                if len(filename_mask) != 1:
                    sg.popup(f'Could not precisely locate the corresponding mask for {basename_img}. {len(filename_mask)} masks found.')
                else:
                    filename_mask = filename_mask[0]
                    try:
                        mask = np.asarray(PIL.Image.open(join(values['-MASK_FOLDER_PATH-'],filename_mask)))
                        cut_mask = utils.cut_into_512(mask)
                    except Exception as E:
                        sg.popup(f'Filename not found: {E}')
                        cut_img = cut_mask = None
                        pass
                    
                if cut_img is not None and cut_mask is not None:
                    current_img = cut_img[cut_img_counter%len(cut_img)]
                    current_mask = cut_mask[cut_img_counter%len(cut_mask)]
                    # create mask threshold
                    non_glint,mask_list = gp.create_masks(current_img,current_mask,mask_threshold,plot=False,plot_gui=False)
                    # apply glint
                    simulated_glint = gp.apply_glint_mask(non_glint,mask_list,params_dict,palette_dict,plot=True,plot_gui=True)
                    # preview simulated glint
                    window['-PREVIEW_IMAGE-'].update(data=gu.array_to_bytes(simulated_glint, resize=(300,300),binary=False))
                    
                    
                    window['Save'].update(disabled=False)
                    window['-PROCESS_ALL-'].update(disabled=False)
                else:
                    pass
    
    if event == 'Save': #only save one cut_image
        if values['-STORE_DIR_PATH-'] == '':
            fp_store = sg.popup_get_folder("Specify directory to store image")
        else:
            fp_store = values['-STORE_DIR_PATH-']

        # make directory of:
        # original image without glint
        dir_non_glint = 'non_glint_cut'
        # glint mask
        dir_glint_mask = 'glint_mask_cut'
        # simulated glint
        dir_sim_glint = 'sim_glint_cut'
        # params
        dir_params = 'params'

        fp_store = values['-STORE_DIR_PATH-']
        if exists(join(fp_store,dir_non_glint)) is False:
            mkdir(join(fp_store,dir_non_glint))
        if exists(join(fp_store,dir_glint_mask)) is False:
            mkdir(join(fp_store,dir_glint_mask))
        if exists(join(fp_store,dir_sim_glint)) is False:
            mkdir(join(fp_store,dir_sim_glint))
        if exists(join(fp_store,dir_params)) is False:
            mkdir(join(fp_store,dir_params))
        
        save_dict = {dir_non_glint: non_glint,
                    dir_glint_mask: mask_list[list(mask_list)[0]], 
                    dir_sim_glint:simulated_glint} #take the first mask from the list
        postfix = str(cut_img_counter%len(cut_img)).zfill(2) #append number of the cut_img
        # remember to process first before saving

        # save parameters for repeatability
        param_fp = utils.uniquify(join(fp_store,dir_params,'{}_{:02}.json'.format(basename_img,cut_img_counter%len(cut_img))))
        with open(param_fp,'w') as cf:
            params_json = {k:v.tolist() for k,v in params_dict.items()}
            params_json['-THRESHOLD-'] = mask_threshold.tolist()
            json.dump(params_json,cf)
        # save images in their respective folders
        for dir, im in save_dict.items():
            utils.save_img(im,join(fp_store,dir),basename_img,prefix='',postfix=postfix,ext=".png",overwrite=False) #do not overwrite file because of different params settings on the same img

        sg.popup(f'Files saved successfully in {fp_store}!')

        










