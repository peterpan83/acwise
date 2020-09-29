import os
import numpy as np
import scipy.interpolate as interp

from ..atm_correction.interface import GasCalculator
import logging.config

_logger = logging.getLogger("Gas")


class GasSixS(GasCalculator):
    '''
    gas absorption calculator based on 6SV simulations
    '''
    def __loadLUTs(self):
        '''
        load lookup table, the dimension order is [zenith,ozone,water_vapor,wavelength]
        :return:
        '''
        import h5py
        lut_path = os.path.join(self.__data_shared_dir,self.sensor,'gas_absorption_{}_{}.h5'.format(self.sensor_altitude,self.groud_altitude))
        data = h5py.File(lut_path, 'r')
        self.wavelenth_lut = data['wavelength'][()]
        self.solz_lut = data['solar_zenith'][()]
        self.vewz_lut = data['viewing_zenith'][()]
        self.ozone_lut = data['ozone'][()]
        self.wv_lut = data['water_vapor'][()]

        self.tran_up_lut = data['gas_trans_up'][()].reshape(
            (self.vewz_lut.shape[0], self.ozone_lut.shape[0], self.wv_lut.shape[0], self.wavelenth_lut.shape[0]))
        self.tran_down_lut = data['gas_trans_down'][()].reshape(
            (self.solz_lut.shape[0], self.ozone_lut.shape[0], self.wv_lut.shape[0], self.wavelenth_lut.shape[0]))

        data.close()
    def __init__(self,sensor,**kwargs):
        self.sensor = sensor
        _logger.info('Initilizing gas aborsption correction based on 6SV...')
        self.groud_altitude = 0 if 'groud_altitude' not in kwargs else int(round(kwargs['groud_altitude'],-3))
        self.sensor_altitude = 3 if 'sensor_altitude' not in kwargs else kwargs['sensor_altitude']
        self.__data_shared_dir = os.path.join(kwargs['data_dir'], 'LUTs')
        self.__loadLUTs()
        self._valid_mask = None
        self._solz_valid,self._senz_valid  = None,None
        self._cal_shape = None

    def set_obs_geo(self, solz: np.ndarray, senz: np.ndarray):
        solz_c, senz_c = solz.copy(), senz.copy()
        valid_mask = ~np.isnan(senz_c)
        solz_c, senz_c = solz_c[valid_mask], senz_c[valid_mask]

        solz_c[solz_c > self.solz_lut.max()] = self.solz_lut.max()
        solz_c[solz_c < self.solz_lut.min()] = self.solz_lut.min()

        senz_c[senz_c > self.vewz_lut.max()] = self.vewz_lut.max()
        senz_c[senz_c < self.vewz_lut.min()] = self.vewz_lut.min()

        self._cal_shape = senz.shape
        self._solz_valid, self._senz_valid, self._valid_mask = solz_c, senz_c, valid_mask

    def set_ozone_watervapor(self, ozone: np.array, water_vapor: np.array):
        if self._valid_mask is None:
            _logger.error("valid mask is not avaliable\n")
            raise Exception("valid mask is not avaliable")
        ozone_c, water_vapor_c = ozone.copy(), water_vapor.copy()
        ozone_c, water_vapor_c = ozone_c[self._valid_mask], water_vapor_c[self._valid_mask]

        ozone_c[ozone_c > self.ozone_lut.max()] = self.ozone_lut.max()
        ozone_c[ozone_c < self.ozone_lut.min()] = self.ozone_lut.min()
        water_vapor_c[water_vapor_c > self.wv_lut.max()] = self.wv_lut.max()
        water_vapor_c[water_vapor_c < self.wv_lut.min()] = self.wv_lut.min()
        self._water_vapor_valid, self._ozone_valid = water_vapor_c, ozone_c

    def get_trans_down(self,iBandIndex):
        '''
        get downwelling trans when geo and ozone and water vapor are all set
        :param iBandIndex:
        :return:
        '''
        if self._solz_valid is None or self._ozone_valid is None or self._water_vapor_valid is None:
            _logger.error("No valid solar zenith")
            raise Exception("No valid solar zenith")
        trans_down = np.full(self._cal_shape, np.nan, float)
        trans_down_lut = self.tran_down_lut[:, :, :, iBandIndex]

        xi = np.hstack([self._solz_valid.reshape(-1, 1), self._ozone_valid.reshape(-1, 1), self._water_vapor_valid.reshape(-1, 1)])
        values = interp.interpn((self.solz_lut, self.ozone_lut, self.wv_lut), trans_down_lut, xi)
        trans_down[self._valid_mask] = values
        return  trans_down

    def get_trans_up(self,iBandIndex):
        '''
        get upwelling trans when geo and ozone and water vapor are all set
        :param iBandIndex:
        :return:
        '''
        if self._senz_valid is None or self._ozone_valid is None or self._water_vapor_valid is None:
            _logger.error("No valid solar zenith")
            raise Exception("No valid solar zenith")
        trans_up = np.full(self._cal_shape, np.nan, float)
        trans_up_lut = self.tran_up_lut[:, :, :, iBandIndex]

        xi = np.hstack([self._senz_valid.reshape(-1, 1), self._ozone_valid.reshape(-1, 1), self._water_vapor_valid.reshape(-1, 1)])
        values = interp.interpn((self.solz_lut, self.ozone_lut, self.wv_lut), trans_up_lut, xi)
        trans_up[self._valid_mask] = values
        return  trans_up

    def cal_trans_down(self, iBandIndex,solz:np.ndarray,ozone:np.ndarray,water_vapor:np.ndarray):
        '''
        :param iBandIndex: band index
        solz,ozone,water_vapors should be ndarray and have the same dimension
        :param solz:
        :param ozone:
        :param water_vapor:
        :return:
        '''
        solz_c, ozone_c, water_vapor_c = solz.copy(), ozone.copy(), water_vapor.copy()

        valid_mask = ~np.isnan(solz_c)
        trans_down = np.full_like(solz_c,np.nan,float)

        solz_c,ozone_c,water_vapor_c = solz_c[valid_mask],ozone_c[valid_mask],water_vapor_c[valid_mask]
        solz_c[solz_c > self.vewz_lut.max()] = self.vewz_lut.max()
        ozone_c[ozone_c>self.ozone_lut.max()] = self.ozone_lut.max()
        ozone_c[ozone_c<self.ozone_lut.min()]  =  self.ozone_lut.min()
        water_vapor_c[water_vapor_c>self.wv_lut.max()] = self.wv_lut.max()
        water_vapor_c[water_vapor_c<self.wv_lut.min()] = self.wv_lut.min()

        trans_down_lut = self.tran_down_lut[:,:,:,iBandIndex]

        xi = np.hstack([solz_c.reshape(-1, 1), ozone_c.reshape(-1, 1), water_vapor_c.reshape(-1, 1)])
        values = interp.interpn((self.solz_lut, self.ozone_lut, self.wv_lut), trans_down_lut, xi)
        trans_down[valid_mask]  = values
        return trans_down

    def cal_trans_up(self,iBandIndex,senz:np.ndarray,ozone:np.ndarray,water_vapor:np.ndarray):
        '''
        calculating total upwelling transmittance due to gas absorption
        :param iBandIndex:
        :param senz:
        :param ozone:
        :param water_vapor:
        :return:
        '''
        senz_c, ozone_c, water_vapor_c = senz.copy(), ozone.copy(), water_vapor.copy()

        trans_up = np.full_like(senz,np.nan,float)
        valid_mask = ~np.isnan(senz_c)

        senz_c,ozone_c,water_vapor_c = senz_c[valid_mask],ozone_c[valid_mask],water_vapor_c[valid_mask]
        senz_c[senz_c > self.vewz_lut.max()] = self.vewz_lut.max()
        ozone_c[ozone_c>self.ozone_lut.max()] = self.ozone_lut.max()
        ozone_c[ozone_c<self.ozone_lut.min()]  =  self.ozone_lut.min()
        water_vapor_c[water_vapor_c>self.wv_lut.max()] = self.wv_lut.max()
        water_vapor_c[water_vapor_c<self.wv_lut.min()] = self.wv_lut.min()


        trans_up_lut = self.tran_up_lut[:, :, :, iBandIndex]

        xi = np.hstack([senz_c.reshape(-1, 1), ozone_c.reshape(-1, 1), water_vapor_c.reshape(-1, 1)])
        values = interp.interpn((self.vewz_lut, self.ozone_lut, self.wv_lut), trans_up_lut, xi)
        trans_up[valid_mask] = values

        return trans_up








