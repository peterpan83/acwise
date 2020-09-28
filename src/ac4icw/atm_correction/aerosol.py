import abc
import numpy as np
import os
from sklearn import linear_model
import pandas as pd
from itertools import product
import glob
import pathlib
from tqdm import tqdm
import scipy.interpolate as interp

from ..decorators import *
from ..atm_correction.interface import AerosolCalculator

import logging.config

_logger = logging.getLogger("Aerosol")

class AerosolSixS(AerosolCalculator):
    # __data_shared_dir = os.path.join(str(pathlib.Path(os.path.realpath(__file__)).parent.parent.parent),'data', 'LUTs')
    # __rhoa_file = os.path.join(__data_shared_dir, 'rhoa.txt')
    # __trana_up_file = os.path.join(__data_shared_dir, 'transa_up.txt')
    # __trana_down_file = os.path.join(__data_shared_dir, 'transa_down.txt')

    @classmethod
    def getUsrDefineAerosolType(cls):
        i = 1
        aero_coms = []
        for item in product(np.arange(0, 12, 2) * 0.1, repeat=4):
            if sum(item) == 1.0 and item[3] < 0.6 and item[0] < 0.6:
                aero_coms.append({'soot': '{:.1f}'.format(item[0]), 'water': '{:.1f}'.format(item[1]), 'oceanic': '{:.1f}'.format(item[2]), 'dust': '{:.1f}'.format(item[3])})
                i += 1
        return tuple(aero_coms)

    def __loadLUTs(self):
        import h5py
        lut_paths = glob.glob(os.path.join(self.__data_shared_dir,self.sensor,'aerosol*{}_{}_*.h5'.format(self.sensor_altitude,self.groud_altitude)))
        self.dic_values = {}
        lut_paths.sort()
        aero_names_all = self.getUsrDefineAerosolType()
        for i in tqdm(range(len(lut_paths)),desc='Loading aerosol LUTs'):
            data = h5py.File(lut_paths[i], 'r')
            aero_type_index = int (os.path.splitext(os.path.split(lut_paths[i])[-1])[0].split('_')[-1])
            _logger.debug("Loading aerosol LUT:{}".format(aero_type_index))
            if i == 0:
                self.wavelenth_lut = data['wavelength'][()]
                self.taua550_lut = data['taua_550'][()]
                self.solz_lut = data['solar_zenith'][()]
                self.vewz_lut = data['viewing_zenith'][()]
                self.phi_lut = data['relative_azimuth'][()]

            rhoa_lut = data['aerosol_reflectance'][()].reshape(
                (self.taua550_lut.shape[0], self.solz_lut.shape[0], self.vewz_lut.shape[0], self.phi_lut.shape[0], self.wavelenth_lut.shape[0]))
            albedo_lut = data['spherical_albedo'][()].reshape(
                (self.taua550_lut.shape[0], self.solz_lut.shape[0],self.vewz_lut.shape[0], self.phi_lut.shape[0], self.wavelenth_lut.shape[0]))[:,0,0,0,:]
            tran_up_lut = data['aerosol_trans_up'][()].reshape(
                (self.taua550_lut.shape[0], self.solz_lut.shape[0], self.vewz_lut.shape[0], self.phi_lut.shape[0], self.wavelenth_lut.shape[0]))[:,0,:,0,:]
            tran_down_lut = data['aerosol_trans_down'][()].reshape(
                (self.taua550_lut.shape[0], self.solz_lut.shape[0], self.vewz_lut.shape[0], self.phi_lut.shape[0], self.wavelenth_lut.shape[0]))[:,:,0,0,:]

            self.dic_values[aero_type_index] = (rhoa_lut,albedo_lut,tran_up_lut,tran_down_lut)

            self.__aerosol_types.append(aero_type_index)
            self.__aerosol_type_names.append(aero_names_all[aero_type_index])
            data.close()


    def __init__(self,sensor,**kwargs):
        super().__init__()
        _logger.info('Initializing aerosol scattering caculation based on 6SV simulation....')
        self.sensor = sensor.upper()
        self.latitude = 49 if 'latitude' not in kwargs else kwargs['latitude']
        self.groud_altitude = 0 if 'groud_altitude' not in kwargs else int(round(kwargs['groud_altitude'],-3))
        self.sensor_altitude = 3 if 'sensor_altitude' not in kwargs else kwargs['sensor_altitude']
        self.__data_shared_dir = os.path.join(kwargs['data_dir'], 'LUTs')
        self.__aerosol_types = []
        self.__aerosol_type_names = []
        self.__loadLUTs()

    def get_aerosol_type(self,index):
        return self.__aerosol_type_names[index]

    def get_aerosol_types(self):
        return self.__aerosol_type_names

    def cal_reflectance(self, iband:int, solz:np.ndarray, senz:np.ndarray, phi:np.ndarray, aerosol_type:int, taua:float):
        '''
        calculating aerosol reflectance for a given aerosol type and AOT
        :param iband:
        :param solz:
        :param senz:
        :param phi:
        :param aerosol_type: aerosol type index, int
        :param taua: aot, float
        :return:
        '''
        senz_c, solz_c, phi_c = senz.copy(), solz.copy(), phi.copy()
        valid_mask = ~np.isnan(senz)
        rhoa = np.full_like(solz,np.nan,float)
        senz_c, solz_c, phi_c = senz_c[valid_mask], solz_c[valid_mask], phi_c[valid_mask]

        senz_c[senz_c > self.vewz_lut.max()] = self.vewz_lut.max()
        solz_c[solz_c>self.solz_lut.max()] = self.solz_lut.max()
        phi_c[phi_c>self.phi_lut.max()] = self.phi_lut.max()

        rhoa_lut = self.dic_values[aerosol_type][0]

        xi = np.hstack([np.full((senz_c.size,1),taua,np.float),
                        solz_c.reshape(-1,1),senz_c.reshape(-1,1),
                        phi_c.reshape(-1,1)])

        values = interp.interpn(
            (self.taua550_lut, self.solz_lut, self.vewz_lut, self.phi_lut),
            rhoa_lut[:, :, :, :, iband], xi=xi)

        rhoa[valid_mask] = values
        return rhoa



    def cal_reflectance_candidates(self,iband,solz,senz,phi):
        '''
        calculating aerosol reflectance for all of the combinnations of aerosol type and AOT, iband,solz,senz and phi should have the same shape
        :param iband: ndarry,1D or 2D
        :param solz: ndarry,1D or 2D
        :param senz: ndarry,1D or 2D
        :param phi: ndarry,1D or 2D
        :return: rhoa, AOTs and aerosol types
        '''
        if not (solz.shape ==senz.shape == phi.shape):
            raise Exception('cal_reflectance,not solz.shape ==senz.shape == phi.shape')
        senz_c, solz_c, phi_c = senz.copy(), solz.copy(), phi.copy()

        senz_c[senz_c > self.vewz_lut.max()] = self.vewz_lut.max()
        solz_c[solz_c>self.solz_lut.max()] = self.solz_lut.max()
        phi_c[phi_c>self.phi_lut.max()] = self.phi_lut.max()

        aerosol_types = np.asarray(self.__aerosol_types)
        taua_s = np.linspace(self.taua550_lut.min(),self.taua550_lut.max(),100)

        rhoa_s = []
        for type_i in aerosol_types:
            rhoa_lut = self.dic_values[type_i][0]

            xi = np.asarray([[t,s,v,p,b] for t in taua_s for s,v,p,b in zip(solz_c.flatten(),
                                                                        senz_c.flatten(),phi_c.flatten(),self.wavelenth_lut[iband])])

            rhoa_ = interp.interpn((self.taua550_lut, self.solz_lut, self.vewz_lut, self.phi_lut,self.wavelenth_lut[iband]),
                                   rhoa_lut[:, :, :, :, iband], xi=xi)

            shape_ = taua_s.shape+senz_c.shape
            rhoa = rhoa_.reshape(shape_)
            rhoa_s.append(rhoa)
        return np.asarray(rhoa_s),taua_s,aerosol_types,np.asarray(self.__aerosol_type_names)
        # rhoa_lut = self.dic_values[aerosol_type]
        # xi = np.hstack([taua,solz_c.reshape(-1,1),senz_c.reshape(-1,1),phi_c.reshape(-1,1)])
        #
        # rhoa = rhoa_.reshape(senz_c.shape)
        # rhoa[invalid_mask] = np.nan

    def cal_spherical_albedo(self,iband:int,aerosol_type:int,taua:float):
        '''
        calculate spherical albedo for a given aerosol type and AOT
        :param iband:
        :param aerosol_type:
        :param taua:
        :return:
        '''
        albedo_lut = self.dic_values[aerosol_type][1]
        albeo = interp.interp1d(self.taua550_lut,albedo_lut[:,iband])(taua)
        return albeo

    def cal_trans_down(self,iband,solz,aerosol_type,taua):
        solz_c= solz.copy()
        valid_mask = ~np.isnan(solz_c)
        trans_down = np.full_like(solz,np.nan,float)
        solz_c = solz_c[valid_mask]
        solz_c[solz_c>self.solz_lut.max()] = self.solz_lut.max()

        trans_down_lut = self.dic_values[aerosol_type][3]

        xi = np.hstack([np.full((solz_c.size,1),taua,np.float),
                        solz_c.reshape(-1,1)])

        values = interp.interpn(
            (self.taua550_lut, self.solz_lut),
            trans_down_lut[:, :, iband], xi=xi)

        trans_down[valid_mask] = values
        return trans_down

    def cal_trans_up(self,iband,senz,aerosol_type,taua):
        senz_c= senz.copy()
        valid_mask = ~np.isnan(senz_c)
        senz_c = senz_c[valid_mask]
        trans_up = np.full_like(senz,np.nan,float)
        senz_c[senz_c>self.vewz_lut.max()] = self.vewz_lut.max()

        trans_up_lut = self.dic_values[aerosol_type][2]

        xi = np.hstack([np.full((senz_c.size,1),taua,np.float),
                        senz_c.reshape(-1,1)])

        values = interp.interpn(
            (self.taua550_lut, self.vewz_lut),
            trans_up_lut[:, :, iband], xi=xi)

        trans_up[valid_mask] = values
        return trans_up


