import PySimpleGUI as sg
import GUI_select_glint.glint_pixels as gp
import GUI_utils as gu
import utils
import cv2 as cv
import scipy.signal
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
from os.path import join, basename, isdir, isfile
from os import listdir, mkdir
from skimage.color import rgb2gray
from random import randrange,sample
from math import ceil


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
                [sg.Text('Brightness:',size=(12,1)), sg.Input(size=(15,1),key='-BRIGHTNESS-',tooltip='[int]: How much white/dark is in the colours')],
                [sg.Text('Saturation:',size=(12,1)), sg.Input(size=(15,1),key='-SATURATION-',tooltip='[0,1] (float): Influences how obvious the rainbow spectra is. Low saturation means it is closer to the water spectra')],
                [sg.Text('Contrast:',size=(12,1)), sg.Input(size=(15,1),key='-CONTRAST-',tooltip='[float]: Influences how green/blue the colors are. Lower contrast means the greener/darker the water spectra is')],
                [sg.Text('Height:',size=(12,1)), sg.Input(size=(15,1),key='-H-',tooltip='[int]: Height of the glint pixel')],
                [sg.Text('Width:',size=(12,1)), sg.Input(size=(15,1),key='-W-',tooltip='[int]: Width of the glint pixel')],
                [sg.Text('Sample colours:',size=(12,1)), sg.Input(size=(15,1),key='-SAMPLE_N-',tooltip='[int]: How many different colors to sample from the palette')],
                [sg.Text('Sigma:',size=(12,1)), sg.Input(size=(15,1),key='-SIGMA-',tooltip='[int]: Standard deviation of the gaussian smoothing')],
            ], title='Glint customisation',font='Arial 8 bold',size=(460,200))],

            [sg.Frame(layout=[
                [sg.Text('Mask Threshold:',size=(12,1)), sg.Input(size=(15,1),key='-THRESHOLD-',tooltip='[float]: Threshold for identifying levels of brightness in the image to apply glint on')],
            ], title='Mask customisation',font='Arial 8 bold',size=(460,100))],

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
main_window['Prev'].update(disabled=True)
main_window['Next'].update(disabled=True)
main_window['Process'].update(disabled=True)
main_window['-PROCESS_ALL-'].update(disabled=True)
main_window['Save'].update(disabled=True)


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
        # after cutting image, enable next, prev buttons
        window['Prev'].update(disabled=False)
        window['Next'].update(disabled=False)
        window['Process'].update(disabled=False)
        # update image
        cut_img_counter = 0 #everytime a cut is initiated, counter must return to 0 to ensure the first cut img of a new image is shown
        window['-PREVIEW_IMAGE-'].update(data=gu.array_to_bytes(cut_img[0], resize=(300,300),binary=False))

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

    if event == 'Process':
        params_dict = {'-BRIGHTNESS-':values['-BRIGHTNESS-'],
                '-SATURATION-':values['-SATURATION-'],
                '-CONTRAST-':values['-CONTRAST-'],
                '-H-':values['-H-'],
                '-W-':values['-W-'],
                '-SAMPLE_N-':values['-SAMPLE_N-'],
                '-SIGMA-':values['-SIGMA-']}

        # make sure that mask images are loaded in
        if values['-MASK_FOLDER_PATH-'] == '':
            sg.popup("Ensure masked images are loaded in.")

        mask_threshold = values['-THRESHOLD-']

        if ('' in params_dict.values()) is True or ('' == mask_threshold is True):
            sg.popup("Empty values in [Glint cutomisation]/[Mask threshold] under [Parameters] tab! Ensure all fields are filled.")
        
        # make sure number of values keyed in params is == to the number of threshold values
        mask_threshold = np.array(mask_threshold.replace(' ','').split(','))
        for k,v in params_dict.items():
            v1 = np.array(v.replace(' ','').split(','))
            if v1.shape != mask_threshold.shape:
                sg.popup("make sure number of values keyed in {} are the same as the number of values in [Mask threshold].".format(k))
            params_dict[k] = v1

        
        
        



        

        window['Save'].update(disabled=False)


        










