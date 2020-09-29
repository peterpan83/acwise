import abc
import numpy as np
from sklearn import linear_model
import pandas as pd
from ..atm_correction.interface import AerosolCalculator

import logging.config

_logger = logging.getLogger("Aerosol")

class AerosolAlgorithm():
    def __init__(self):
        self._rhoaw = [] # save  rhoaw spectrum
        self._bandindexs = [] #save band index corresponding to  self._rhoaw
        self._aot = None
        self._aerosol_type = None
        self._aerosol_type_name = None

        #-------the reference wavelengh and band index of AOT--------#
        self._aot_wavelength = None
        self._aot_bandindex = None

    def setAeroCalculator(self,aero_cal:AerosolCalculator):
        self._aero_cal = aero_cal

    @abc.abstractmethod
    def push_rhoaw(self,iBandIndex,rhoaw:np.ndarray):
        '''
        push a tile of rhoaw at a certain band into the algorihtm to obtain the entire spectrum,
        because most of the algorihtms need the spectrum curve of rhoaw
        :param iBandIndex: band index
        :param rhoaw: sum of aerosol and water reflectance
        :return:
        '''
        pass

    def clear(self):
        self._rhoaw.clear()
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



class AerosolDSF(AerosolAlgorithm):
    __AEROSOL_REFLECTANCE_MIN = 1E-5

    def __init__(self,sensor,**kwargs):
        _logger.info("Initilaizing DSF algorithm (Vanhellemont et al.,2018.)...")
        super().__init__()


        ## save the darkest pixel's observing geometry
        self._dark_pixel_geo = []
        self._aot_wavelength = 550


        usefitting = bool(int(kwargs['usefitting'])) if 'usefitting' in kwargs else False

        self.__dark_piexl_num = 20 if 'DARK_PIXEL_NUM' not in kwargs else int(kwargs['DARK_PIXEL_NUM'])
        self.__dark_spectrum_num = 5 if 'DARK_SPECTRUM_NUM' not in kwargs['DARK_SPECTRUM_NUM'] else int(kwargs['DARK_SPECTRUM_NUM'])
        self.__usefitting = usefitting
        self.__sensor = sensor


        self.__find_dark_pixl = self.__find_dark_pixel_directly
        _logger.info('using fitting:{}\n darkest spectrum number:{}'.format(usefitting,self.__dark_spectrum_num))
        if self.__usefitting:
            self.__find_dark_pixl = self.__find_dark_pixl_by_fitting
            _logger.info('darkest pixel number:{}'.format(self.__dark_piexl_num))

        if 'aero_cal' not in kwargs:
            raise Exception()

        if not issubclass(kwargs['aero_cal'],AerosolCalculator):
            raise Exception()

        # Instantiate AerosolCalculator
        self._aero_cal = kwargs['aero_cal'](sensor,**kwargs)

    def clear(self):
        super().clear()
        self._dark_pixel_geo.clear()

    def __find_dark_pixl_by_fitting(self,iBandIndex,rhoaw):
        '''
        save the darkest rhoaw values to self._rhoaw based on linear fitting
        :param iBandIndex: band index starts from 0
        :param rhoaw:  reflectane of water and aerosol
        :return:
        '''

        self._dark_pixel_geo.append([self.__center_solz, self.__center_senz, self.__center_phi])

        rhoaw_temp = rhoaw[rhoaw > self.__AEROSOL_REFLECTANCE_MIN].flatten()
        if rhoaw_temp.shape[0] == 0:
            return
        if rhoaw_temp.shape[0] < 3:
            self._rhoaw.append(rhoaw_temp[0])
            self._bandindexs.append(iBandIndex)
            return
        rhoaw_temp.sort()
        slice_end = self.__dark_piexl_num if self.__dark_piexl_num < rhoaw_temp.shape[0] else -1
        rhoaw_temp_dark = rhoaw_temp[0:slice_end]
        # print(rhoaw_temp_dark.shape)

        ## get the darkest through linear fitting
        regr = linear_model.LinearRegression()
        regr.fit(np.arange(rhoaw_temp_dark.shape[0]).reshape(-1, 1), rhoaw_temp_dark.reshape(-1, 1))
        # print(regr.coef_)
        darkest = regr.predict([[0]]) if regr.predict([[0]]) > 0 else [rhoaw_temp_dark[0]]

        # print(rhoaw_temp_dark[0],darkest,regr.coef_,regr.intercept_)
        # import matplotlib.pyplot as plt
        # plt.plot(rhoaw_temp_dark,'o')
        # plt.plot([0,rhoaw_temp_dark.shape[0]],[darkest[0],regr.predict([[rhoaw_temp_dark.shape[0]]])[0]],'--')
        # plt.show()

        self._rhoaw.append(darkest[0])
        self._bandindexs.append(iBandIndex)


    def __find_dark_pixel_directly(self,iBandIndex,rhoaw):
        '''
        find the darkest pixel directly, without fitting
        :param iBandIndex:
        :param rhoaw:
        :return:
        '''

        rhoaw_ = rhoaw[~np.isnan(rhoaw)]
        if rhoaw_.shape[0] == 0:
            self._dark_pixel_geo.append([0, 0, 0])
            return
        rhoaw_ = rhoaw_[rhoaw_ > self.__AEROSOL_REFLECTANCE_MIN]
        if rhoaw_.shape[0] == 0:
            self._dark_pixel_geo.append([0,0,0])
            return
        darkest = rhoaw_.min()
        row_,col_ = np.where(rhoaw==darkest)
        # position = np.unravel_index(rhoaw_.argmin(),rhoaw_.shape)
        self._rhoaw.append(darkest)
        self._bandindexs.append(iBandIndex)
        self._dark_pixel_geo.append([self._solz[row_[0], col_[0]], self._senz[row_[0], col_[0]], self._phi[row_[0], col_[0]]])


    def push_rhoaw(self,iBandIndex,rhoaw:np.ndarray):
        self.__find_dark_pixl(iBandIndex,rhoaw)

    def __cal_reflectance_lut(self,iBandIndex):
        self._aero_cal.cal_reflectance(iBandIndex,self._solz,self._senz,self._phi,aerosol_type=None,taua=None)


    def retrieve(self):
        '''
        implementation of the aerosol retrival algorithm
        aerosol type and AOT are retrived
        :return:
        '''
        rhoaw_dark = self._rhoaw
        if len(rhoaw_dark)<2:
            _logger.debug("The number of darkest spectrums is less than two...")
            self._aot=None
            self._aerosol_type = None
            return

        slice_t = len(rhoaw_dark) if len(rhoaw_dark) < self.__dark_spectrum_num else self.__dark_spectrum_num
        s = pd.Series(self._rhoaw,index=self._bandindexs)
        s.sort_values(inplace=True)
        ## bandindex of the darkest sepectrum
        darkest_bandindex =  sorted(s.index.tolist()[0:slice_t])
        darkest_values = s[darkest_bandindex].values
        _logger.debug("darkest index:{}".format(str(darkest_bandindex)))
        _logger.debug("")
        darkest_geo = np.asarray(self._dark_pixel_geo)[darkest_bandindex,:]

        rhoa_for_dark,taua_s,aerosol_type_s,aerosol_type_name_s = self._aero_cal.cal_reflectance_candidates(iband=darkest_bandindex,solz=darkest_geo[:,0],senz=darkest_geo[:,1],phi=darkest_geo[:,2])
        rhoa_for_dark_ = rhoa_for_dark.reshape(-1,darkest_values.shape[0])
        sum_square = np.asarray(list(map(lambda x: np.sum(np.power(x-darkest_values,2)), rhoa_for_dark_)))
        closest_index = sum_square.argmin()
        type_i,taua_i = np.unravel_index(closest_index,aerosol_type_s.shape+taua_s.shape)

        self._aot = taua_s[taua_i]
        self._aerosol_type = aerosol_type_s[type_i]
        self._aerosol_type_name = aerosol_type_name_s[type_i]

        _logger.info('Aerosol type:{},{},AOT550:{:.3f}'.format(self._aerosol_type,self._aerosol_type_name,self._aot))


    def get_results(self, iBandIndex=0):
        '''
        calculating rhoa,albedo,upwelling and downwelling transmittance based on the retrieved AOT and aerosol type
        :param iBandIndex:
        :return: (rhoa,albeo,trans_up,trans_down)
        '''
        aerosol_type = 2 if self._aerosol_type is None else self._aerosol_type
        taua = 0.005 if self._aot is None else self._aot
        rhoa = self._aero_cal.cal_reflectance(iband=iBandIndex,solz=self._solz,senz=self._senz,phi=self._phi,aerosol_type=aerosol_type,taua=taua)
        albeo = self._aero_cal.cal_spherical_albedo(iband=iBandIndex,aerosol_type=aerosol_type,taua=taua)
        trans_up = self._aero_cal.cal_trans_up(iband=iBandIndex,senz=self._senz,aerosol_type=aerosol_type,taua=taua)
        trans_down = self._aero_cal.cal_trans_down(iband=iBandIndex, solz=self._solz, aerosol_type=aerosol_type, taua=taua)

        return (rhoa,albeo,trans_up,trans_down)




