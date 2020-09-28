import abc
import os
import numpy as np
import scipy.interpolate as interp

from ..decorators import *
from ..atm_correction.interface import RayleighCalculator

import logging.config

_logger = logging.getLogger("Rayleigh")


class RayleighSixS(RayleighCalculator):
    '''
    implementation of Rayleigh scattering calculation based on the LUTs generated by 6SV
    '''

    @classmethod
    def GetAtmMode(cls,month:int,latitude):
        '''
        get atmosphere mode based on the month and latitude ,
        this is a copy of Py6S
        :param month:  from 1 to 12
        :param latitude: positive and negative value for northern and southern hemisphere,respectively
        :return: one of 'subarcwinter','subarcsummer','midlatsummer','midlatwinter', 'tropical'
        '''
        rounded_lat = round(latitude, -1)

        # Data from Table 2-2 in http://www.exelisvis.com/docs/FLAASH.html
        SAW,SAS,MLS,MLW,T = 'subarcwinter','subarcsummer','midlatsummer','midlatwinter', 'tropical'

        ap_JFMA = {80: SAW,70: SAW,60: MLW,50: MLW,40: SAS,30: MLS,20: T,10: T,0: T,-10: T,-20: T,-30: MLS,-40: SAS,-50: SAS,-60: MLW,-70: MLW,-80: MLW
        }

        ap_MJ = {80:SAW,70: MLW,60: MLW,50: SAS,40: SAS,30: MLS,20: T,10: T,0: T,-10: T,-20: T,-30: MLS,-40: SAS,-50: SAS,-60: MLW,-70: MLW,-80: MLW
        }

        ap_JA = {80:MLW,70: MLW,60: SAS,50: SAS,40: MLS,30: T,20: T,10: T,0: T,-10: T,-20: MLS,-30: MLS,-40: SAS,-50: MLW,-60: MLW,-70: MLW,-80: SAW
        }

        ap_SO = {80:  MLW,70:  MLW,60:  SAS,50:  SAS,40:  MLS,30:  T,20:  T,10:  T,0:   T,-10: T,-20: MLS,-30: MLS,-40: SAS,-50: MLW,-60: MLW,-70: MLW,-80: MLW
        }

        ap_ND = {80: SAW,70: SAW,60: MLW,50: SAS,40: SAS,30: MLS,20: T,10: T,0: T,-10: T,-20: T,-30: MLS,-40: SAS,-50: SAS,-60: MLW,-70: MLW,-80: MLW
        }

        ap_dict = {1: ap_JFMA,2: ap_JFMA,3: ap_JFMA,4: ap_JFMA,5: ap_MJ,6: ap_MJ,7: ap_JA,8: ap_JA,9: ap_SO,10: ap_SO,11: ap_ND,12: ap_ND
        }

        return ap_dict[month][rounded_lat]

    def __loadLUTs(self):
        '''
        load rayleigh scattering lookup table
        :return:
        '''
        import h5py
        lut_path = os.path.join(self.__data_shared_dir,self.sensor,'rayleigh_{}_{}_{}.h5'.format(self.atm_mode,self.sensor_altitude,self.groud_altitude))
        _logger.debug(lut_path)

        lut_data = h5py.File(lut_path,'r')
        self.wavelenth_lut = lut_data['wavelength'][()]
        _logger.debug(str(self.wavelenth_lut))
        self.solz_lut = lut_data['solar_zenith'][()]
        _logger.debug(str(self.solz_lut))
        self.vewz_lut = lut_data['viewing_zenith'][()]
        _logger.debug(str(self.vewz_lut))
        self.phi_lut = lut_data['relative_azimuth'][()]
        _logger.debug(str(self.phi_lut))

        self.rhor_lut = lut_data['raleigh_reflectance'][()].reshape(
            (self.solz_lut.shape[0], self.vewz_lut.shape[0], self.phi_lut.shape[0], self.wavelenth_lut.shape[0]))

        self.spherical_albedo_lut = lut_data['spherical_albedo'][0]

        ## upwelling transmittance just depends on veiwing zenith and wavelength
        self.tran_up_lut = lut_data['raleigh_trans_up'][()].reshape(
            (self.solz_lut.shape[0],  self.vewz_lut.shape[0], self.phi_lut.shape[0], self.wavelenth_lut.shape[0]))[0,:,0,:]

        ## downwelling transmittance just depends on solar zenith and wavelength
        self.tran_down_lut = lut_data['raleigh_trans_down'][()].reshape(
            (self.solz_lut.shape[0],  self.vewz_lut.shape[0], self.phi_lut.shape[0], self.wavelenth_lut.shape[0]))[:,0,0,:]

        lut_data.close()

    def __init__(self,sensor,**kwargs):
        '''
        initilize WISE level1
        :param sensor: sensor's name
        :param kwargs: month,latitude, groud_altitude, etc.
        '''
        _logger.info('initializing rayleigh scattering caculation based on 6SV simulation....')
        print(kwargs)
        self.sensor = sensor.upper()
        self.month = 7 if 'month' not in kwargs else kwargs['month']
        self.latitude = 45 if 'latitude' not in kwargs else kwargs['latitude']
        self.sensor_altitude = 3 if 'sensor_altitude' not in  kwargs else kwargs['sensor_altitude']
        self.groud_altitude = 0 if 'groud_altitude' not in kwargs else int(round(kwargs['groud_altitude'],-3))
        self.atm_mode = self.GetAtmMode(self.month,self.latitude)
        self.__data_shared_dir = os.path.join(kwargs['data_dir'], 'LUTs')
        _logger.debug('the atmosphere mode is {}'.format(self.atm_mode))
        _logger.info('loading LUTs....')
        self.__loadLUTs()
        self._cal_shape = None
        self._solz_valid, self._senz_valid,self._phi_valid, self._valid_mask = None, None,None, None



    def set_obs_geo(self, solz, senz,phi):
        solz_c, senz_c,phi_c = solz.copy(), senz.copy(),phi.copy()
        valid_mask = ~np.isnan(senz_c)
        solz_c, senz_c ,phi_c= solz_c[valid_mask], senz_c[valid_mask],phi_c[valid_mask]

        solz_c[solz_c > self.solz_lut.max()] = self.solz_lut.max()
        solz_c[solz_c < self.solz_lut.min()] = self.solz_lut.min()

        senz_c[senz_c > self.vewz_lut.max()] = self.vewz_lut.max()
        senz_c[senz_c < self.vewz_lut.min()] = self.vewz_lut.min()

        phi_c[phi_c > self.phi_lut.max()] = self.phi_lut.max()
        phi_c[phi_c < self.phi_lut.min()] = self.phi_lut.min()

        self._cal_shape = senz.shape
        self._solz_valid, self._senz_valid,self._phi_valid, self._valid_mask = solz_c, senz_c,phi_c, valid_mask

    def get_reflectance(self,iband):
        rhor = np.full(self._cal_shape,np.nan,float)
        xi = np.hstack([self._solz_valid.reshape(-1,1),self._senz_valid.reshape(-1,1),self._phi_valid.reshape(-1,1)])

        values = interp.interpn((self.solz_lut,self.vewz_lut,self.phi_lut),self.rhor_lut[:,:,:,iband],xi=xi)
        rhor[self._valid_mask] = values
        return rhor

    def get_trans_up(self,iband):
        tranr_up = np.full(self._cal_shape, np.nan, float)
        f = interp.interp1d(self.vewz_lut,self.tran_up_lut[:,iband])
        values = f(self._senz_valid.flatten())
        tranr_up[self._valid_mask] = values
        return tranr_up

    def get_trans_down(self,iband):
        tranr_down = np.full(self._cal_shape,np.nan,float)
        f = interp.interp1d(self.solz_lut,self.tran_down_lut[:,iband])
        values = f(self._solz_valid.flatten())
        tranr_down[self._valid_mask] = values
        return tranr_down

    # @processing_info
    def cal_reflectance(self,iBandIndex,solz,senz,phi):
        senz_c,solz_c,phi_c = senz.copy(),solz.copy(),phi.copy()

        valid_mask = ~np.isnan(senz)
        senz_c,solz_c,phi_c = senz_c[valid_mask],solz_c[valid_mask],phi_c[valid_mask]

        senz_c[senz_c > self.vewz_lut.max()] = self.vewz_lut.max()
        solz_c[solz_c>self.solz_lut.max()] = self.solz_lut.max()
        phi_c[phi_c>self.phi_lut.max()] = self.phi_lut.max()

        rhor = np.full_like(solz,np.nan,float)
        xi = np.hstack([solz_c.reshape(-1,1),senz_c.reshape(-1,1),phi_c.reshape(-1,1)])

        values = interp.interpn((self.solz_lut,self.vewz_lut,self.phi_lut),self.rhor_lut[:,:,:,iBandIndex],xi=xi)
        rhor[valid_mask] = values
        return rhor


    def cal_spherical_albedo(self,iBandIndex):
        return self.spherical_albedo_lut[iBandIndex]

    def cal_trans_down(self,iBandIndex,solz):
        solz_c = solz.copy()
        tranr_down = np.full_like(solz,np.nan,float)
        valid_mask = ~np.isnan(solz_c)
        solz_c = solz_c[valid_mask]
        solz_c[solz_c > self.solz_lut.max()] = self.solz_lut.max()


        f = interp.interp1d(self.solz_lut,self.tran_down_lut[:,iBandIndex])
        values = f(solz_c.flatten())
        tranr_down[valid_mask] = values
        return tranr_down

    def cal_trans_up(self,iBandIndex,senz):
        senz_c = senz.copy()
        valid_mask = ~np.isnan(senz_c)
        tranr_up = np.full_like(senz,np.nan,float)

        senz_c = senz_c[valid_mask]
        senz_c[senz_c > self.vewz_lut.max()] = self.vewz_lut.max()

        f = interp.interp1d(self.vewz_lut,self.tran_up_lut[:,iBandIndex])
        values = f(senz_c.flatten())
        tranr_up[valid_mask] = values
        return tranr_up

    # def calculate(self,iBandIndex,solz,senz,phi):
    #     '''
    #     calculate raleigh scattering reflectance, and transmittance at upward and downward
    #     :param iBandIndex: band index
    #     :param solz:  solar zenith angles
    #     :param senz:  viewing zenith angles
    #     :param phi:  relativae azimuch angles
    #     :return: rhor,tranr_up,tranr_down
    #     '''
    #     senz_c,solz_c,phi_c = senz.copy(),solz.copy(),phi.copy()
    #     senz_c[senz_c > self.vewz_lut.max()] = self.vewz_lut.max()
    #     solz_c[solz_c>self.solz_lut.max()] = self.solz_lut.max()
    #     phi_c[phi_c>self.phi_lut.max()] = self.phi_lut.max()
    #
    #     # wavelength = self.wavelenth_lut[iBandIndex]
    #
    #     invalid_mask = np.isnan(senz)
    #     senz_c[invalid_mask] = 0
    #     phi_c[invalid_mask] = 0
    #
    #     xi = np.hstack([solz_c.reshape(-1,1),senz_c.reshape(-1,1),phi_c.reshape(-1,1)])
    #     _logger.debug("rayleigh calculating")
    #
    #     rhor_ = interp.interpn((self.solz_lut,self.vewz_lut,self.phi_lut),self.rhor_lut[:,:,:,iBandIndex],xi=xi)
    #     spherical_albedo_ =  interp.interpn((self.solz_lut,self.vewz_lut,self.phi_lut),self.spherical_albedo_lut[:,:,:,iBandIndex],xi=xi)
    #     tranr_up_ = interp.interpn((self.solz_lut, self.vewz_lut, self.phi_lut), self.tran_up_lut[:,:,:,iBandIndex], xi=xi)
    #     tranr_down_ = interp.interpn((self.solz_lut, self.vewz_lut, self.phi_lut), self.tran_down_lut[:,:,:,iBandIndex], xi=xi)
    #
    #     rhor,tranr_up,tranr_down,spherical_albedo = rhor_.reshape(senz_c.shape),tranr_up_.reshape(senz_c.shape),tranr_down_.reshape(senz_c.shape), spherical_albedo_.reshape(senz_c.shape)
    #
    #     rhor[invalid_mask] = np.nan
    #     spherical_albedo[invalid_mask] = np.nan
    #     tranr_up[invalid_mask] = np.nan
    #     tranr_down[invalid_mask] = np.nan
    #
    #     return rhor,spherical_albedo,tranr_up,tranr_down






