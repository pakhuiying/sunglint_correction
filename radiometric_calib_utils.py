import micasense.capture as capture
import os, glob
import json
import tqdm
import pickle #This library will maintain the format as well

def list_img_subdir(dir):
    """ 
    given the directory, find all the panel images in the subdirectories
    """
    img_dir = []
    for survey in os.listdir(dir):
        for flight in os.listdir(os.path.join(dir,survey)):
            if len(flight) == 2: # to exclude the folder with missing blue bands
                if os.path.isdir(os.path.join(dir,survey,flight,'RawImg')):
                    img_dir.append(os.path.join(dir,survey,flight,'RawImg'))
                else:
                    print("RawImg folder not found!")
    return img_dir

def save_fp_panels(dir):
    """ 
    given the directory, find all the panel images in the subdirectories and save the filepaths in a dictionary
    >>> save_fp_panels(r"F:\surveys_10band")
    """
    RawImg_list = list_img_subdir(dir)

    RawImg_json = {f: None for f in RawImg_list}
    for k in tqdm(RawImg_json.keys()):
        RawImg_json[k] = get_panels(dir = k,search_n=3)

    if not os.path.exists('saved_data'):
        os.mkdir('saved_data')
    with open(os.path.join('saved_data','panel_fp.json'), 'w') as fp:
        json.dump(RawImg_json, fp)
    
    return

def get_panels(dir,search_n=5):
    """
    :param dir(str): directory where images are stored
    search for the first 5 and last 5 captures to detect panel
    returns a list of captures in band order
    For panel images, efforts will be made to automatically extract the panel information, 
    panel images are not warped and the QR codes are individually detected for each band
    """
    number_of_files = len(glob.glob(os.path.join(dir,'IMG_*.tif')))//10 #divide by 10 since each capture has 10 bands
    last_file_number = number_of_files - 1 #since index of image starts from 0

    first_few_panels_fp = [glob.glob(os.path.join(dir,'IMG_000{}_*.tif'.format(str(i)))) for i in range(search_n)]
    last_few_panels_fp = [glob.glob(os.path.join(dir,'IMG_{}_*.tif'.format(str(last_file_number-i).zfill(4)))) for i in reversed(range(search_n))]
    panels_fp = first_few_panels_fp + last_few_panels_fp
    panels_list = []

    for f in panels_fp:
        cap_dict = {i+1:None for i in range(10)} # in order to order the files by band order, otherwise IMG_1 and IMG_10 are consecutive
        for cap in f:
            cap_dict[int(cap.split('_')[-1].replace('.tif',''))] = cap
        panels_list.append(list(cap_dict.values()))
    
    panelCaps = [capture.Capture.from_filelist(f) for f in panels_list] # list of captures
    detected_panels = [cap.detect_panels() for cap in panelCaps]

    detected_panels_fp = []
    for panels_n,panel_f in zip(detected_panels,panels_list):
        if panels_n == 10:
            detected_panels_fp.append(panel_f)
    return detected_panels_fp

def load_panel_fp(fp):
    """ 
    load panel_fp into dictionary, where keys are the parent directory, and keys are a list of list of panel images
    """
    with open(fp, 'r') as fp:
        data = json.load(fp)

    return data

def import_panel_reflectance(panelNames):
    """
    :param PanelNames (list of str): full file path of panelNames
    this should only be done once assuming that panel used is the same for all flights
    returns panel_reflectance_by_band
    """
    if panelNames is not None:
        panelCap = capture.Capture.from_filelist(panelNames)
    else:
        panelCap = None
    if panelCap is not None:
        if panelCap.panel_albedo() is not None:
            panel_reflectance_by_band = panelCap.panel_albedo()
        else:
            print("Panel reflectance not detected by serial number")
            panel_reflectance_by_band = None 
    else:
        panel_reflectance_by_band = None 
    
    return panel_reflectance_by_band



def import_panel_irradiance(panelNames,panel_reflectance_by_band):
    """
    :param PanelNames (list of str): full file path of panelNames
    :param panel_reflectance_by_band (list of float): reflectance values ranging from 0 to 1. 
    Import this value so we don't have to keep detecting the QR codes repeatedly if the QR panels are the same
    """
    if panelNames is not None:
        panelCap = capture.Capture.from_filelist(panelNames)
        panel_irradiance = panelCap.panel_irradiance(panel_reflectance_by_band)
        dls_irradiance = panelCap.dls_irradiance()
        return {'dls':dls_irradiance,'panel':panel_irradiance}
    else:
        print("Panels not found")
        return None

def save_dls_panel_irr(panel_fp,panel_albedo):
    """ save panel and dls irradiance for each band """
    dls_panel_irr = {k:[] for k in panel_fp.keys()}
    for k,list_of_caps in tqdm(panel_fp.items()):
        for cap in list_of_caps:
            dls_panel_dict = import_panel_irradiance(cap,panel_albedo)
            dls_panel_irr[k].append(dls_panel_dict)
    
    if not os.path.exists('saved_data'):
        os.mkdir('saved_data')
    with open(os.path.join('saved_data','dls_panel_irr.ob'), 'wb') as fp:
        pickle.dump(dls_panel_irr,fp)
    
    return dls_panel_irr

def load_pickle(fp):
    if fp.endswith('ob'):
        with open(fp, 'rb') as fp:
            data = pickle.load(fp)

        return data
    else:
        print("Not a pickle file")
        return None

def load_panel_albedo(fp):
    """
    load panel albedo (values range from 0 to 1) in band order i.e. 1,2,3,4,5,6,7,8,9,10 (note that it is not in the order of wavelength, but band order)
    """
    with open(fp, 'rb') as fp:
        panel_albedo = pickle.load(fp)

    return panel_albedo

def load_center_wavelengths(fp):
    """
    load center_wavelengths in band order i.e. 1,2,3,4,5,6,7,8,9,10 (note that it is not in the order of wavelength, but band order)
    with open(os.path.join('saved_data','center_wavelengths_by_band.ob'), 'wb') as fp:
        pickle.dump(center_wavelengths,fp)
    """
    with open(fp, 'rb') as fp:
        center_wavelengths = pickle.load(fp)

    return center_wavelengths

def load_dls_panel_irr(fp):
    with open(fp, 'rb') as fp:
        dls_panel_irr = pickle.load(fp)

    return dls_panel_irr