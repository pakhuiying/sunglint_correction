import matplotlib.pyplot as plt
import pickle
import numpy as np
import os

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
        # angle with respect to east
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
    
def get_flight_angle(yaw_value):
    """ 
    :param yaw_value (np.ndarray): yaw value from dls-yaw
    returns flight angle in radians
    # bug with this piecewise function, doesnt evaluate single values
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

    f0 = lambda x: k0*x+c0
    f1 = lambda x: k1*x+c1
    f2 = lambda x: k2*x+c2

    x0 = -1.5
    x1 = 1.5

    y = np.piecewise(yaw_value, [yaw_value <= x0, (yaw_value > x0) & (yaw_value <= x1), yaw_value>x1], [f0, f1, f2])
    return y

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