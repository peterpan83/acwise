import abc
import numpy as np
from sklearn import linear_model
import pandas as pd
from ..atm_correction.interface import PathCalculator

import logging.config

_logger = logging.getLogger("Aerosol")

class PathAlgorithm():
    def __init__(self):
        self._rhot = [] # save  rhot spectrum
        self._bandindexs = [] #save band index corresponding to  self._rhoaw
        self._aot = None
        self._aerosol_type = None
        self._aerosol_type_name = None

        #-------the reference wavelengh and band index of AOT--------#
        self._aot_wavelength = None
        self._aot_bandindex = None

    def setPathCalculator(self,path_cal:PathCalculator):
        self._path_cal = path_cal

    @abc.abstractmethod
    def push_rhot(self,iBandIndex,rhot:np.ndarray):
        '''
        push a tile of rhot at a certain band into the algorihtm to obtain the entire spectrum,
        because most of the algorihtms need the spectrum curve of rhoaw
        :param iBandIndex: band index
        :param rhoaw: sum of aerosol and water reflectance
        :return:
        '''
        pass

    def clear(self):
        self._rhot.clear()
        self._bandindexs.clear()

    def set_obs_geo(self,solz,senz,phi):
        self._solz, self._senz, self._phi = solz, senz, phi
        invalidmask = np.isnan(self._senz)

        if np.all(invalidmask):
            self.__center_solz, self.__center_senz, self.__center_phi = 0,0,0
        else:
            self._validmask = ~invalidmask
            self.__center_solz, self.__center_senz, self.__center_phi = np.percentile(solz[self._validmask],50), \
                                                                    np.percentile(senz[self._validmask], 50), \
                                                                    np.percentile(phi[self._validmask], 50)

    @abc.abstractmethod
    def retrieve(self):
        '''
        implementation of the aerosol retrival algorithm
        aerosol type and AOT are retrived
        :return:
        '''
        pass

    def get_results(self, iBandIndex=0):
        '''
        return the aerosol results including reflectrance,albedo,upwelling and downwelling transmittance
        based on the AOT and aerosol type retrieved by the algorihtm
        :param iBandIndex:
        :return:
        '''
        pass

    def get_aerosol(self):
        return self._aot,self._aerosol_type,self._aot_wavelength




class PathFIXAEROSOL(PathAlgorithm):

    def __init__(self,sensor,**kwargs):
        _logger.info("Initializing FixAerosol algorithm...")
        super().__init__()

        ## save the darkest pixel's observing geometry
        self._dark_pixel_geo = []
        self._aot_wavelength = 550

        if 'path_cal' not in kwargs:
            raise Exception()
        if 'AERO_INDEX' not in kwargs:
            raise Exception()

        if not issubclass(kwargs['path_cal'],PathCalculator):
            raise Exception()

        # Instantiate AerosolCalculator
        self._path_cal = kwargs['path_cal'](sensor,**kwargs)

        if 'AOT_550' not in kwargs:
            raise Exception
        self._aot = float(kwargs['AOT_550'])
        self._aerosol_type = int(kwargs['AERO_INDEX'])

    
    def get_results(self, iBandIndex=0):
        aerosol_type = 2 if self._aerosol_type is None else self._aerosol_type
        taua = 0.005 if self._aot is None else self._aot
        rhop_t = self._path_cal.cal_reflectance_t(iband=iBandIndex,solz=self._solz,senz=self._senz,phi=self._phi,aerosol_type=aerosol_type,taua=taua)
        albeo = self._path_cal.cal_spherical_albedo(iband=iBandIndex,aerosol_type=aerosol_type,taua=taua)
        trans_up = self._path_cal.cal_trans_up(iband=iBandIndex,senz=self._senz,aerosol_type=aerosol_type,taua=taua)
        trans_down = self._path_cal.cal_trans_down(iband=iBandIndex, solz=self._solz, aerosol_type=aerosol_type, taua=taua)

        return (rhop_t,albeo,trans_up,trans_down)



