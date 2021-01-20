import os
import numpy as np
from datetime import datetime
import scipy.interpolate as intp
import pandas as pd


def loadF0(data_dir,sensor=None,name='Thuillier_2002'):
    '''
    read F0 at averaged sun-earth distance
    :param name: file name
    :return: F0
    '''
    if sensor is None:
        data_shared_dir = os.path.join(data_dir,'shared')
        f0_path = os.path.join(data_shared_dir,'F0_{}.txt'.format(name))
        units,waves,f0 = None, [],[]
        if name=='Thuillier_2002':
            flag = 0
            with open(f0_path,'r',encoding='utf-8') as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line.startswith('/units'):
                    units = line.split(',')
                    continue
                if line.startswith('/end_header'):
                    flag = 1
                    continue
                if flag==1:
                    values = [float(v) for v in line.split(' ')]
                    waves.append(values[0])
                    f0.append(values[1]*0.001)  #convert mW/cm^2/um to mW/cm^2/nm
    else:
        data_shared_dir = os.path.join(data_dir, 'LUTs',sensor)
        # print(sensor)
        if name == 'F0_6SV':
            values = np.loadtxt(os.path.join(data_shared_dir,name+'.txt'), skiprows=2)
            waves,f0 = values[:,0],values[:,1]*0.0001 # convert from W/m2/mic2 to mW/cm^2/nm

    return np.asarray(waves),np.asarray(f0),'mW/cm^2/nm'
    # with open(os.path)

def calF0Ext(FO:np.ndarray,dt:datetime):
    '''
    calculate rrextrateestrial solar irradiance on a specific day considering the sun-earth distance
    (https://en.wikipedia.org/wiki/Sunlight#:~:text=If%20the%20extraterrestrial%20solar%20radiation,hitting%20the%20ground%20is%20around)
    :param FO: extraterristrial solar irradiance on the mean sun-earth distance
    :param dt: date
    :return:
    '''
    dn = dt.day_of_year
    distance_factor = 1+0.033412*np.cos(2*np.pi*(dn-3)/365.0)
    return FO* distance_factor



def mergeThuillierAndE490():
    waves,F0_Thuillier,unit = loadF0()
    s_Thuillier = np.concatenate([waves[:,np.newaxis],F0_Thuillier[:,np.newaxis]],axis=1)
    s_e490 = pd.read_csv('../data/shared/e490_00a_amo.csv')
    s_e490['Wavelength, microns'] =  s_e490['Wavelength, microns']*1000

    # s = pd.merge(s_Thuillier,s_e490)
    print(s_Thuillier.shape,s_e490.values.shape)
    d = np.concatenate([s_Thuillier,s_e490.values],axis=0)
    pd.DataFrame(data=d,columns=['waves','F0'])



if __name__ == '__main__':
    mergeThuillierAndE490()

