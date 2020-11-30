import abc
from ..level1 import Level1_Base
from ..level2 import Level2

class AtmCorrectionInterface(metaclass=abc.ABCMeta):
    '''
    interface of atmospheric correction
    '''
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'Run') and
                callable(subclass.Run) and
                hasattr(subclass, 'setLevel_1') and
                callable(subclass.setLevel_1) and
                hasattr(subclass,'setLevel_2') and
                callable(subclass.setLevel_2)or
                NotImplemented)

    @abc.abstractmethod
    def Run(self)->int:
        '''
        atmospheric correction main entrance

        :return:
        '''
        pass

    @abc.abstractmethod
    def setLevel_1(self,level_1:Level1_Base):
        '''
        set level 1 image
        :param level_1:
        :return:
        '''
        pass

    @abc.abstractmethod
    def setLevel_2(self,level_2:Level2):
        '''
        set level 2 image
        :param level_2:
        :return:
        '''
        pass

    def setGasCalculator(self,*args,**kwargs):
        pass

    def setRayleighCalculator(self,*args,**kwargs):
        pass

    def setAeroAlgorithm(self,*args,**kwargs):
        pass

    def setGlintCalculator(self,*args,**kwargs):
        pass

    def setAdjAlgorithm(self,*args,**kwargs):
        pass



class RayleighCalculator():
    '''
    interface of scattering calculation
    '''
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'cal_reflectance') and
                callable(subclass.cal_reflectance) and
                hasattr(subclass, 'cal_trans_up') and
                callable(subclass.cal_trans_up) and
                hasattr(subclass,'cal_trans_down') and
                callable(subclass.cal_trans_down) and
                hasattr(subclass, 'cal_spherical_albedo') and
                callable(subclass.cal_spherical_albedo) or
                NotImplemented)

    @abc.abstractmethod
    def set_obs_geo(self, solz, senz,phi):
        pass

    @abc.abstractmethod
    def get_reflectance(self,iband):
        pass

    @abc.abstractmethod
    def get_trans_up(self,iband):
        pass

    @abc.abstractmethod
    def get_trans_down(self,iband):
        pass


    @abc.abstractmethod
    def cal_reflectance(self,iband,solz,senz,phi):
        '''
        calculateing aerosol reflectance
        :param iband: band index
        :param solz: solar zenith angle (degree)
        :param senz:viewing zenith angle (degree)
        :param phi:  relative zenith angle (degree)
        :return: reflectance
        '''
        pass

    @abc.abstractmethod
    def cal_trans_up(self,iband,senz):
        '''
        calculating upwelling transmittance due to rayleigh scattering
        :param iband:
        :param senz:
        :return:
        '''
        pass

    @abc.abstractmethod
    def cal_trans_down(self,iband,solz):
        '''
        calculating downwelling transmittance due to rayleigh scattering
        :param iband:
        :param solz:
        :return:
        '''
        pass

    @abc.abstractmethod
    def cal_spherical_albedo(self,iband):
        '''
        calculating sperical albedo
        :param iband:
        :return:
        '''
        pass


class GasCalculator():
    '''
    interface of gas absorption calculation
    '''
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'cal_trans_down') and
                callable(subclass.cal_trans_down) and
                hasattr(subclass, 'cal_trans_up') and
                callable(subclass.cal_trans_up) or
                NotImplemented)

    @abc.abstractmethod
    def get_trans_up(self, iBandIndex):
        pass

    @abc.abstractmethod
    def get_trans_down(self,iBandIndex):
        pass

    @abc.abstractmethod
    def set_ozone_watervapor(self, ozone, water_vapor):
        pass

    @abc.abstractmethod
    def set_obs_geo(self, solz, senz):
        pass


    @abc.abstractmethod
    def cal_trans_down(self, iBandIndex, solz,ozone,water_vapor):
        '''
        total downwelling transmittance,which only depends on solar zenith
        as well as ozone and water_vapor
        :param iBandIndex: band index,int
        :param solz: solar zenith, degree
        :param ozone: total volum ozone concentration ()
        :param water_vapor:total volum water vapor concentration ()
        :return: total downwelling transmittance
        '''
        pass

    @abc.abstractmethod
    def cal_trans_up(self,iBandIndex,senz,ozone,water_vapor):
        '''
         total upwelling transmittance,which only depends on solar zenith
        :param iBandIndex: band index
        :param senz: viewing zenith angle, degree
        :param ozone: total volum ozone concentration ()
        :param water_vapor:total volum water vapor concentration ()
        :return:  total upwelling transmittance due to gas aborption
        '''
        pass


class AerosolCalculator():
    '''
    interface of scattering calculation
    '''
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'cal_reflectance') and
                callable(subclass.cal_reflectance) and
                hasattr(subclass, 'cal_trans_up') and
                callable(subclass.cal_trans_up) and
                hasattr(subclass,'cal_trans_down') and
                callable(subclass.cal_trans_down) and
                hasattr(subclass, 'cal_spherical_albedo') and
                callable(subclass.cal_spherical_albedo) or
                NotImplemented)

    @abc.abstractmethod
    def cal_reflectance(self,iband,solz,senz,phi,aerosol_type,taua):
        '''
        calculating aerosol reflectance for a given aerosol_type and AOT and band
        :param iband:band index , int
        :param solz: solar zenith
        :param senz: viewing zenith
        :param phi: relative azimuth
        :param aerosol_type: aerosol type index
        :param taua: aerosol optical thickness
        :return:
        '''
        pass

    def cal_reflectance_candidates(self,iband,solz,senz,phi):
        pass

    @abc.abstractmethod
    def cal_trans_up(self,iband,senz,aerosol_type,taua):
        '''
        calculating upwelling transmittance due to aerosol scattering
        :param iband:
        :param senz:
        :param aerosol_type:
        :param taua:
        :return:
        '''
        pass

    @abc.abstractmethod
    def cal_trans_down(self,iband,solz,aerosol_type,taua):
        '''
        calculating downwelling transmittance due to aerosol scattering
        :param iband:
        :param solz:
        :param aerosol_type:
        :param taua:
        :return:
        '''
        pass

    @abc.abstractmethod
    def cal_spherical_albedo(self,iband,aerosol_type,taua):
        '''
        calculating spherical albedo due to aerosol scattering,it is dependant to aerosol type and AOT
        :param iband:
        :param aerosol_type:
        :param taua:
        :return:
        '''
        pass

    def get_aerosol_type(self,index):
        pass

    def get_aerosol_types(self):
        pass

class GlintCalcultor():
    '''
    glint interface
    '''
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'cal_reflectance') and
                callable(subclass.cal_reflectance) or
                NotImplemented)

    @abc.abstractmethod
    def cal_reflectance(self,iband,solz,senz,phi,**kwargs):
        '''
        calculate glint reflectrance rhog
        :param iband:
        :param solz:
        :param senz:
        :param phi:
        :param kwargs: wind speed,Trans_s etc.
        :return:  rhog
        '''
        pass

    @abc.abstractmethod
    def set_obs_geo(self,solz,senz,phi):
        pass

    @abc.abstractmethod
    def set_ancillary(self,**kwargs):
        '''
        set ancillary like wind speed,etc
        :param kwargs:
        :return:
        '''
        pass

    @abc.abstractmethod
    def get_reflectance(self,Trans_s):
        '''
        get reflectance at iband
        :param Trans_s: Direct transmittance at the direction of sun
        :return:
        '''
        pass

    @abc.abstractmethod
    def get_nlg(self):
        pass



