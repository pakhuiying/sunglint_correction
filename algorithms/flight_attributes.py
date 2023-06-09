import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
import datetime
import os
import mutils
import cv2
from osgeo import gdal, osr
from scipy.interpolate import griddata
from tqdm import tqdm
class FlightAttributes:
    def __init__(self,df):
        """ 
        :param df (pandas DF class): output from:
            data, columns = imgset.as_nested_lists()
            df = pd.DataFrame.from_records(data, index='timestamp', columns=columns)
        """
        self.df = df

    def calculate_angle(self,x0,y0,x1,y1, ref_vec = np.array([1,0])):
        start_v = np.array([x0,y0])
        end_v = np.array([x1,y1])
        vec = end_v - start_v
        vec = vec/np.linalg.norm(vec) # normalised vector
        return np.arccos(np.dot(vec,ref_vec))#/np.pi*180 
    
    def calculate_bends(self,yaw,plot=True):
        idx_list = list(range(len(yaw)))
        yaw_diff = np.diff(yaw,append=0) # to ensure length is same as idx_list and yaw
        # yaw_diff = np.diff(yaw_diff,append=0)
        yaw_diff1 = np.rint(yaw_diff)#yaw_diff.astype(int)
        bends_idx = np.argwhere(yaw_diff1 != 0).flatten()
        bends_yaw = yaw_diff[bends_idx]

        # idx_diff = np.diff(bends_idx,append=bends_idx[-1])
        idx_diff = np.diff(bends_idx,append=0)
        idx2 = np.argwhere(idx_diff>2)
        bends_idx1 = bends_idx[idx2]
        bends_yaw1 = yaw_diff[bends_idx1]
        if plot is True:
            plt.figure()
            plt.plot(idx_list,yaw,label='yaw')
            plt.plot(idx_list,yaw_diff,label='diff')
            plt.scatter(bends_idx,bends_yaw,c='red',s=10,alpha=0.5,label='bends')
            plt.scatter(bends_idx1,bends_yaw1,c='black',s=10,alpha=0.5,label='bends')
            plt.plot()
            plt.legend()
            plt.show()
        return bends_idx1.flatten()
    
    def get_line_idx(self,pad=3):
        """ 
        :param bends_idx (np.ndarray of int): corresponds to the indices where abrupt change of angle detected
        :param pad (int): pad indices [x] after/before start/stop indices
        """
        yaw = self.df['dls-yaw']
        bends_idx = self.calculate_bends(yaw,plot=False)
        idx_list = []
        for i in range(bends_idx.shape[0]-1):
            start_idx = bends_idx[i] + pad
            stop_idx = bends_idx[i+1] -pad
            idx_list.append((start_idx,stop_idx))
        return idx_list
    
    def get_coord_yaw(self):
        """ returns a tuple of (lat_long_array (np.ndarray), yaw (np.ndarray))"""
        idx_list = self.get_line_idx()
        lat_long_list = dict()
        yaw_list = dict()
        for idx in idx_list:
            df = self.df.iloc[idx[0]:idx[1]+1,:]
            #first column is latitude, second column is longitude (y,x)
            lat_long_list[f'{idx[0]}_{idx[1]}'] = df[['latitude','longitude']].to_numpy()
            yaw_list[f'{idx[0]}_{idx[1]}'] = df['dls-yaw'].values
            
        return lat_long_list,yaw_list
    
    def calculate_flight_angle(self, ref_vec = np.array([0,1])):
        # angle with respect to east np.array([0,1]), where latitude, longitude
        coord_yaw = self.get_coord_yaw()
        lat_long_array = coord_yaw[0] #first item of the tuple
        yaw_array = coord_yaw[1] #second item of the tuple
        angle_array_list = dict()
        yaw_array_list = dict()
        # for i in range(len(lat_long_array)):
        for k,v in lat_long_array.items():
            vec = np.diff(v,axis=0)
            vec_mag = np.linalg.norm(vec,axis=1)
            vec_mag = np.tile(vec_mag.reshape(-1,1),(1,2))
            vec = vec/vec_mag #normalised vector
            angle_array = np.arccos(np.dot(vec,ref_vec))
            angle_array_list[k] = angle_array
            yaw_array_list[k] = yaw_array[k][:-1]
            # angle_array_list.append(angle_array)
        return angle_array_list, yaw_array_list
    
    def plot_flight_angle(self):
        angle_array_list, yaw_array_list = self.calculate_flight_angle()
        plt.figure()
        for k in angle_array_list.keys():
            plt.scatter(np.mean(yaw_array_list[k]),np.mean(angle_array_list[k]),s=2,alpha=0.5,label=k)
        plt.xlabel('yaw')
        plt.ylabel('Flight angle')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
        return
    
    def save_coord_yaw(self,fp=None):
        angle_array_list, yaw_array_list = self.calculate_flight_angle()
        angle = np.concatenate(list(angle_array_list.values()),axis=0)
        yaw = np.concatenate(list(yaw_array_list.values()),axis=0)
        coord_yaw = np.column_stack([angle,yaw])
        if fp is not None:
            dir = os.path.join(os.getcwd(),f'{fp}.ob')
            with open(dir,'wb') as fp:
                pickle.dump(coord_yaw,fp)
        return coord_yaw
    
def get_flight_angle(coords, ref_vec = np.array([0,1])):
    """ 
    :param coords (np.ndarray): where 1st column is latitude, 2nd column is longitude
        1st row is startpt of vector, 2nd row is endpt of vector
    :param ref_vec (np.ndarray): angle with respect to east np.array([0,1]), where latitude, longitude
    returns angle in degrees wrt to east based on vector between coord0 and coord1
    """
    vec = np.diff(coords,axis=0)
    vec_mag = np.linalg.norm(vec,axis=1)
    vec_mag = np.tile(vec_mag.reshape(-1,1),(1,2))
    vec = vec/vec_mag #normalised vector
    angle_array = np.arccos(np.dot(vec,ref_vec))
    return angle_array/np.pi*180

def get_flight_angle_fn():
    """
    returns function and params
    """
    t0 = (-np.pi,1.5)
    t1 = (-1.5,np.pi)
    t2 = (1.5,0)
    t3 = (3,1.5) #np.pi,1.5

    k0 = (t1[1]-t0[1])/(t1[0]-t0[0])
    k1 = (t2[1]-t1[1])/(t2[0]-t1[0])
    k2 = (t3[1]-t2[1])/(t3[0]-t2[0])

    c0 = t0[1] - k0*t0[0]
    c1 = t1[1] - k1*t1[0]
    c2 = t2[1] - k2*t2[0]

    x0 = -1.5
    x1 = 1.5

    fn = lambda x: k0*x+c0 if x<x0 else (k1*x+c1 if (x<x1) else k2*x+c2)

    return fn, [k0,k1,k2,c0,c1,c2]

class GeotransformImage:
    def __init__(self,im,lat,long,altitude,yaw,angle=None):
        """" 
        :param angle (float): angle in degrees relative to east (x=1,y=0)
        """
        self.im = im
        self.lat = lat
        self.long = long
        self.altitude = altitude
        self.yaw = yaw
        self.angle = angle
    
    def get_flight_angle(self):
        """returns angle in degrees"""
        fn, params = get_flight_angle_fn()
        angle_rad = fn(self.yaw)
        angle_deg = angle_rad/np.pi*180
        if angle_deg < 90:
            angle = 90 - angle_deg
        else:
            angle = 90 + angle_deg
        return angle
    
    def get_ground_resolution(self):
        """ returns ground resolution in meters in x, y direction"""
        return mutils.get_ground_resolution(height=self.altitude)
    
    def get_degrees_per_meter(self):
        """ 
        use the quick and dirty method that:
        111,111 meters (111.111 km) in the y direction is 1 degree (of latitude) and 
        111,111 * cos(latitude in degrees) meters in the x direction is 1 degree (of longitude).
        x_degrees: degree per meters in the longitude direction
        y_degrees: degree per meters in the latitude direction
        returns (y_degrees, x_degrees) which corresponds lat, long degrees resolution per meter
        """
        y_degrees = 1/111111
        x_degrees = 1/(111111*np.cos(self.lat/180*np.pi))
        return y_degrees, x_degrees 
    
    def get_degrees_per_pixel(self):
        """ 
        returns (y_degrees, x_degrees) which corresponds lat, long degrees resolution per pixel
        """
        x_meters, y_meters = self.get_ground_resolution() # resolution per pixel in meters
        y_degrees, x_degrees = self.get_degrees_per_meter() # degree resolution per meters
        avg_meter_res = (x_meters+y_meters)/2
        return avg_meter_res*y_degrees, avg_meter_res*x_degrees
    
    def geotransform(self):
        """
        # GT(0) x-coordinate of the upper-left corner of the upper-left pixel.
        # GT(1) w-e pixel resolution / pixel width.
        # GT(2) row rotation (typically zero).
        # GT(3) y-coordinate of the upper-left corner of the upper-left pixel.
        # GT(4) column rotation (typically zero).
        # GT(5) n-s pixel resolution / pixel height (negative value for a north-up image).
        geotransform = (left_extent, -lon_res_per_pixel, 0, top_extent, 0, -lat_res_per_pixel) #xmin=longitude, ymax=lat
        returns geotransformed image, and geotransform params
        """
        rot_im = self.affine_transformation(plot=False) #a north-up image
        rows, cols = rot_im.shape[0],rot_im.shape[1] 
        lat_res, lon_res = self.get_degrees_per_pixel()
        y_extent = self.lat+lat_res*int(0.5*rows)
        x_extent = self.long+lon_res*int(0.5*cols) # may need to find the upper right extent instead
        UL = (y_extent, x_extent)#lat, long
        geotransform = (UL[1],-lon_res,0,UL[0],0,-lat_res)
        # print(f'Geotransform: {geotransform}')
    
        return rot_im, geotransform
    
    def georegister(self, fp=None):
        """ 
        assign coordinates to geotransformed image
        """
        fn = f'{fp}.tif'
        rot_im, geotransform = self.geotransform()
        flipped_transformed_img = np.flipud(rot_im) #flip images because QGIS
        rows, cols = flipped_transformed_img.shape[0],flipped_transformed_img.shape[1]
        n_bands = self.im.shape[-1] if (len(self.im.shape) == 3) else 1

        if fp is not None:
            dst_ds = gdal.GetDriverByName('GTiff').Create(fn,cols, rows, n_bands, gdal.GDT_Float32)
            dst_ds.SetGeoTransform(geotransform)    # specify coords
            srs = osr.SpatialReference()            # establish encoding
            srs.ImportFromEPSG(4326)                # WGS84 lat/long
            dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
            
            # flipped_transformed_img = arr
            if len(self.im.shape) == 3: #then it has 3-bands
                for i in range(3):
                    dst_ds.GetRasterBand(i+1).WriteArray(flipped_transformed_img[:,:,i])
        
            else: #greyscale image with 1 bands
                dst_ds.GetRasterBand(1).WriteArray(flipped_transformed_img)
                # write 1-band to the raster
            dst_ds.FlushCache()                     # write to disk
            dst_ds = None
        return flipped_transformed_img

    def affine_transformation(self, plot = True):
        """angle in degrees"""
        if self.angle is None:
            angle = self.get_flight_angle()
        else:
            angle = self.angle

        rows, cols = self.im.shape[0],self.im.shape[1] 
        center = (cols//2,rows//2)
        # using cv2.getRotationMatrix2D() to get the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center,angle,1) #center, angle, scale
        # rotate the image using cv2.warpAffine
        # rotated_image = cv2.warpAffine(src=img, M=rotation_matrix, dsize=(width, height))
        cosofRotationMatrix = np.abs(rotation_matrix[0][0]) #scale*cos(angle)
        sinofRotationMatrix = np.abs(rotation_matrix[0][1]) #scale*sin(angle)
        # Now will compute new height & width of
        # an image so that we can use it in
        # warpAffine function to prevent cropping of image sides
        newImageHeight = int((cols * sinofRotationMatrix) +
                            (rows * cosofRotationMatrix))
        newImageWidth = int((cols * cosofRotationMatrix) +
                            (rows * sinofRotationMatrix))
        # After computing the new height & width of an image
        # we also need to update the values of rotation matrix
        rotation_matrix[0][2] += (newImageWidth/2) - center[0]
        rotation_matrix[1][2] += (newImageHeight/2) - center[1]

        # Now, we will perform actual image rotation WITHOUT MASK
        rotatingimage = cv2.warpAffine(self.im, rotation_matrix, (newImageWidth, newImageHeight))

        if plot is True:
            fig, axes = plt.subplots(1,2,figsize=(10,5))
            axes[0].imshow(self.im)
            axes[1].imshow(rotatingimage)
            axes[1].set_title(f'yaw: {self.yaw:.3f}'+f'\nangle: {angle:.2f}')
            plt.show()
        return rotatingimage

def interpolate_timestamp(df,milliseconds=100):
    """
    :param df (pd.DataFrame): column names must include timestamp, latitude, longitude, altitude, flight_angle
    :param milliseconds (float): timedelta in milliseconds, interpolate between two timestamps at 100ms frequency
    linear interpolation using timedelta
    returns a tuple of (original dataframe, interpolated dataframe)
    """
    column_names = ['latitude','longitude','altitude','flight_angle']
    if not all([i in df.columns for i in column_names]):
        raise NameError(f'at least one name does not exist: {column_names}')
    
    middle_idx = len(df.index)//2
    freq = datetime.timedelta(milliseconds=milliseconds)#duration//10 # timedelta object
    timedelta = freq.total_seconds()*1000
    date_range_list = []
    for i, rows in df.iterrows():
        if i == len(df.index) -1:
            pass
        else:
            daterange = pd.date_range(start=rows['timestamp'], end=df['timestamp'][i+1], freq=freq,closed='left').to_series() # date arange
            daterange = daterange.reset_index()
            date_range_list.append(daterange.iloc[:,0])
    
    interpolated_timestamp = pd.concat(date_range_list)
    # print(interpolated_timestamp)
    timedelta_series = interpolated_timestamp - df['timestamp'][0]
    timedelta_series = timedelta_series.dt.total_seconds() #returns time delta in total_seconds

    og_timestamp = df['timestamp']
    og_timedelta_series = og_timestamp - df['timestamp'][0]
    og_timedelta_series = og_timedelta_series.dt.total_seconds()
    og_dict = {'timestamp': og_timestamp, 'timedelta': og_timedelta_series}
    for names in column_names:
        og_dict[names] = df[names]
    og_df = pd.DataFrame(og_dict)
    
    interpolated_dict = {'timestamp': interpolated_timestamp,'timedelta':timedelta_series}
    for names in column_names:
        interpolated_dict[names] = griddata(og_timedelta_series, df[names], timedelta_series, method='linear')
    interpolated_df = pd.DataFrame(interpolated_dict)
    
    return og_df, interpolated_df

class InterpolateFlight:
    def __init__(self, df, column_names = ['latitude','longitude','altitude','flight_angle']):
        """ 
        :param df (pd.DataFrame): imported from the folder 'flight_attributes'
        :param column_names (list of str): column names in df
        """
        self.df = df
        self.column_names = column_names
    
    def append_flight_angle(self):
        """calculate flight angle and append to the df"""
        df = self.df.copy()
        column_idx = [i for i,c in enumerate(self.df.columns.to_list()) if c in ['latitude','longitude']]
        angle_coord_list = [np.NaN]
        for i,rows in tqdm(self.df.iterrows()):
            if (i == 0) or (i == len(self.df.index)-1):
                pass
            else:
                # estimate flight angle from 2 adjacent coordinates
                flight_att_diff = self.df.iloc[[i-1,i+1],column_idx]
                flight_att_diff = flight_att_diff.iloc[:,:2].values
                flight_angle_coord = get_flight_angle(flight_att_diff)
                angle_coord_list.append(flight_angle_coord[0])
        angle_coord_list.append(np.NaN)
        df['flight_angle'] = angle_coord_list
        df = df.ffill(axis=0).bfill(axis=0) #fill forward and fill backward
        return df
    
    def interpolate_flight(self, plot=True):
        """returns an interpolated df"""
        flight_attributes_df = self.append_flight_angle()
        # convert to datetime format
        flight_attributes_df['timestamp'] = pd.to_datetime(flight_attributes_df['timestamp'])
        # interpolate df
        df, df_interpolated = interpolate_timestamp(flight_attributes_df)
        df_interpolated = df_interpolated.merge(flight_attributes_df,how='outer',on=['timestamp']+self.column_names).ffill(axis=0)
        # assign unique index for image name
        df_interpolated['index'] = df_interpolated['image_name'].str.split('_').str[1].astype(int)
        
        if plot is True:
            fig, axes = plt.subplots(1,2,figsize=(10,5))
            axes[0].scatter(df['longitude'],df['latitude'],c=df['flight_angle'],alpha=0.5)
            im = axes[1].scatter(df_interpolated['longitude'],df_interpolated['latitude'],c=df_interpolated['flight_angle'],alpha=0.5)
            cax = plt.colorbar(im,ax=axes[1])
            cax.set_label('Flight angle (deg)')
            axes[0].set_title(f'Flight angle ({len(df.index)})')
            axes[1].set_title(f'Interpolated flight angle ({len(df_interpolated.index)})')
            for ax in axes:
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
            plt.tight_layout()
            plt.show()
        
        return df_interpolated
    

