import os
import numpy as np
import logging
import scipy.interpolate as interp

from .interface import GlintCalcultor

_logger = logging.getLogger("Glint")
class GlintCoxMunk(GlintCalcultor):
    '''
    glint estimation is based on LUTs that were generated based on the CoxMunk mode
    '''

    def __loadLUTs(self):
        import h5py
        lut_path = os.path.join(self.__data_shared_dir, 'glint_Cox_Munk.h5')
        try:
            data = h5py.File(lut_path, 'r')
        except IOError as e:
            _logger.error("can not read {}".format(lut_path))
            raise e
        self.solz_lut = data['solar_zenith'][()]
        self.vewz_lut = data['viewing_zenith'][()]
        self.phi_lut = data['relative_azimuth'][()]
        self.wind_speed_lut = data['wind_speed'][()]
        self.glint_lut = data['nLg'][()].reshape((self.solz_lut.shape[0], self.vewz_lut.shape[0], self.phi_lut.shape[0], self.wind_speed_lut.shape[0]))

        data.close()


    def __init__(self,sensor,**kwargs):
        _logger.info('initializing glint caculation based on Cox&Munk model....')
        print(kwargs)
        self.sensor = sensor.upper()
        self.__data_shared_dir = os.path.join(kwargs['data_dir'], 'LUTs')
        _logger.info('loading LUTs....')
        try:
            self.__loadLUTs()
        except Exception as e:
            _logger.error("failed to initialize GlintCoxMunk")
            raise e

        self._solz_valid,self._senz_valid,self._phi_valid = None,None,None
        self._windspeed_valid = None
        self._nLg_valid = None
        self._valid_mask = None
        self._cal_shape = None

    def _cal_LGN(self,solz,senz,phi,windspeed):
        xi = np.hstack([solz.reshape(-1, 1), senz.reshape(-1, 1), phi.reshape(-1, 1),windspeed.reshape(-1,1)])
        values = interp.interpn((self.solz_lut, self.vewz_lut, self.phi_lut,self.wind_speed_lut), self.glint_lut, xi=xi)
        return values

    def get_nlg(self):
        nLg = np.full(self._cal_shape,np.nan,np.float)
        xi = np.hstack([self._solz_valid.reshape(-1, 1), self._senz_valid.reshape(-1, 1), self._phi_valid.reshape(-1, 1),self._windspeed_valid.reshape(-1,1)])
        values = interp.interpn((self.solz_lut, self.vewz_lut, self.phi_lut,self.wind_speed_lut), self.glint_lut, xi=xi)
        nLg[self._valid_mask] = values
        return nLg

    def cal_reflectance(self,solz,senz,phi,**kwargs):
        # T(θv, λ)Lg(θv, λ) = Fo(λ)T(θs, λ)T(θv, λ)LGN
        # rho_g(θs,θv, λ) = pi* T(θs, λ)*LGN/cos(θs)
        '''
        to do....
        :param solz:
        :param senz:
        :param phi:
        :param kwargs:
        :return:
        '''
        pass

    def set_ancillary(self,**kwargs):
        '''
        set ancillary data like windspeed
        :param kwargs:
        :return:
        '''
        if 'windspeed' not in kwargs:
            _logger.error("needs wind speed")
        windspeed = kwargs['windspeed']
        windspeed_c = windspeed.copy()
        if self._valid_mask is None:
            _logger.error("vaild mask is required")
        windspeed_c = windspeed_c[self._valid_mask]
        windspeed_c[windspeed_c > self.wind_speed_lut.max()] = self.wind_speed_lut.max()
        windspeed_c[windspeed_c < self.wind_speed_lut.min()] = self.wind_speed_lut.min()
        self._windspeed_valid = windspeed_c

    def set_obs_geo(self,solz,senz,phi):
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

    def get_reflectance(self,Trans_s):

        if self._nLg_valid is None:
            if self._solz_valid is None or self._senz_valid is None or self._phi_valid is None or self._cal_shape is None:
                _logger.error("obs geo or wind speed not given...")
                raise Exception("obs geo or wind speed not given...")

            self._nLg_valid = self._cal_LGN(self._solz_valid,self._senz_valid,self._phi_valid,self._windspeed_valid)

        rhog = np.full(self._cal_shape,np.nan,np.float)
            # pi * T(θs, λ) * LGN / cos(θs)
        rhog_valid=  np.pi* Trans_s[self._valid_mask]*self._nLg_valid/np.cos(np.deg2rad(self._solz_valid))
        rhog[self._valid_mask] = rhog_valid
        return rhog
