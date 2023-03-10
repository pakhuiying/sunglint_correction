import micasense.capture as capture
import os, glob
import json
import tqdm
import pickle #This library will maintain the format as well
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
import mutils

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
        # by calling panel_irradiance, it already accounts for all the radiometric calibration and correcting of vignetting effect and lens distortion
        panel_irradiance = panelCap.panel_irradiance(panel_reflectance_by_band)
        # by calling dls_irradiance, it already returns a list of the corrected earth-surface (horizontal) DLS irradiance in W/m^2/nm
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

class RadiometricCorrection:
    def __init__(self,dls_panel_irr,center_wavelengths,dls_panel_irr_calibration=None):
        self.dls_panel_irr = dls_panel_irr
        self.number_of_bands = 10
        self.center_wavelengths = center_wavelengths
        self.dls_panel_irr_calibration=dls_panel_irr_calibration

    def get_dls_panel_irr_by_band(self):
        """ obtain dls panel irradiance in band order i.e. 1,2,3,4,5,6,7,8,9,10"""
        panel_irr = {i:[] for i in range(self.number_of_bands)}
        dls_irr = {i:[] for i in range(self.number_of_bands)}
        for k, list_of_d in self.dls_panel_irr.items():
            for d in list_of_d:
                for i,dls in enumerate(d['dls']):
                    dls_irr[i].append(dls)
                for i,panel in enumerate(d['panel']):
                    panel_irr[i].append(panel)
        return {'dls':dls_irr,'panel':panel_irr}

    def plot(self):
        """ plot relationship between dls and panel irradiance by band order i.e. 1,2,3,4,5,6,7,8,9,10"""
        dls_panel_irr_by_band = self.get_dls_panel_irr_by_band()
        if self.dls_panel_irr_calibration is None:
            model_coeff = self.fit_curve_by_band()
        else:
            model_coeff = self.dls_panel_irr_calibration
        fig, axes = plt.subplots(self.number_of_bands//2,2,figsize=(8,15))
        for i,ax in zip(range(self.number_of_bands),axes.flatten()):
            x = dls_panel_irr_by_band['dls'][i]
            y = dls_panel_irr_by_band['panel'][i]
            ax.plot(x,y,'o')
            ax.set_title(r'Band {}: {} nm ($R^2:$ {:.3f}, N = {})'.format(i,self.center_wavelengths[i],model_coeff[i]['r2'],len(x)))
            ax.set_xlabel(r'DLS irradiance $W/m^2/nm$')
            ax.set_ylabel(r'Panel irradiance $W/m^2/nm$')
            x_vals = np.linspace(np.min(x),np.max(x),50)
            intercept = model_coeff[i]['intercept']
            slope = model_coeff[i]['coeff']
            y_vals = intercept + slope * x_vals
            ax.plot(x_vals.reshape(-1,1), y_vals.reshape(-1,1), '--')
            ax.text(0.1,ax.get_ylim()[1]*0.8,r"$y = {:.3f}x + {:.3f}$".format(slope,intercept))

        plt.tight_layout()
        return axes

    def fit_curve_by_band(self):
        """ get model coefficient for relationship between dls and panel irradiance by band order i.e. 1,2,3,4,5,6,7,8,9,10 """
        dls_panel_irr_by_band = self.get_dls_panel_irr_by_band()
        model_coeff = dict()#{i: None for i in range(10)}
        for i in range(self.number_of_bands):
            x = np.array(dls_panel_irr_by_band['dls'][i]).reshape(-1, 1)
            y = np.array(dls_panel_irr_by_band['panel'][i]).reshape(-1, 1)
            lm = LinearRegression().fit(x, y)
            r2 = r2_score(y,lm.predict(x))
            model_coeff[i] = {'coeff':lm.coef_[0][0],'intercept':lm.intercept_[0],'r2':r2}
        
        return model_coeff

class CorrectionFactor:
    def __init__(self,panel_radiance,dls_panel_irr_calibration,panel_albedo=None):
        """ 
        :param dls_panel_irr_calibration (dict): where keys (int) are band number (0 to 9), and values are dict, with keys coeff and intercept
        :param panel_albedo (list of float): list of float ranging from 0 to 1. This parameter is fixed if the same panel is used
        :param panel_radiance (list of float): radiance of panel (mission-specfic)
        """
        self.dls_panel_irr_calibration = dls_panel_irr_calibration
        if panel_albedo is not None:
            self.panel_albedo = panel_albedo
        else:
            self.panel_albedo = [0.48112499999999997,
            0.4801333333333333,
            0.4788733333333333,
            0.4768433333333333,
            0.4783016666666666,
            0.4814866666666666,
            0.48047166666666663,
            0.4790833333333333,
            0.47844166666666665,
            0.4780333333333333]
        self.panel_radiance = panel_radiance
        self.correction_factor = self.get_correction()
    
    def get_correction(self):
        """ outputs a list of float values in band order i.e. band 1,2,3,4,5,6,7,8,9,10"""
        assert len(self.panel_albedo) == len(self.panel_radiance), "panel_albedo bands must equal to panel_radiance bands"
        correction_factor = []
        for band_number,model_calib in self.dls_panel_irr_calibration.items():
            a = model_calib['coeff']
            b = model_calib['intercept']
            rho_crp = self.panel_albedo[band_number]
            L_crp = self.panel_radiance[band_number]
            cf = a/(1-b*rho_crp/(np.pi*L_crp))
            correction_factor.append(cf)
        return correction_factor

def radiometric_corrected_aligned_captures(cap,cf,img_type = "reflectance"):
    """
    :param cap (Capture object)
    :param cf (list of float): correction_factor obtained from CorrectionFactor.cf
    This function aligns the band images, then apply the correction factor to all the bands. 
    Note that the correction factor is mission-specific because it depends on the measured CRP radiance on that mission
    returns the reflectance of image in reflectance (default) or "radiance"
    """
    im_aligned = mutils.align_captures(cap,img_type = img_type)
    return np.multiply(im_aligned,np.array(cf))