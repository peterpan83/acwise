import numpy as np
import os
from tqdm import tqdm
import glob

from ..atm_correction.interface import AtmCorrectionInterface
from ..level1 import Level1_Base
from ..level2 import Level2
from ..decorators import processing_info

import logging

_logger = logging.getLogger("Path AC")

@processing_info
class PathacAtmCorrection(AtmCorrectionInterface):
    def __init__(self,level_1:Level1_Base,level_2:Level2,**kwargs):
        self._gas_cal = None
        self._glint_cal = None
        self._path_alg = None
        self._adj_alg = None
        self._level_1 = level_1
        self._level_2 = level_2
        self.__out_dir = self._level_2.outputer.output_dir
        self.tile_sile = 2.0 if 'TILE_SIZE' not in kwargs else float(kwargs['TILE_SIZE'])
        self._water_vapor = 3.0 if 'WATER_VAPOR' not in kwargs else float(kwargs['WATER_VAPOR'])
        _logger.info("WATER VAPOR:{:.2f}".format(self._water_vapor))
        self._ozone = 0.3  if 'OZONE' not in kwargs else float(kwargs['OZONE'])
        _logger.info("OZONE:{:.2f}".format(self._ozone))
        self._windspeed = 5.0 if 'WIND_SPEED' not in kwargs else float(kwargs['WIND_SPEED'])
        _logger.info("WIND SPEED:{:.2f}".format(self._windspeed))

    def setLevel_1(self,level_1:Level1_Base):
        self._level_1 = level_1

    def setLevel_2(self,level_2:Level2):
        self._level_2 = level_2
        self.__out_dir = self._level_2.outputer.output_dir

    def setGasCalculator(self,gas_cal,**kwargs):
        self._gas_cal = gas_cal(self._level_1.sensor,**kwargs)

    def setPath(self,gas_cal,**kwargs):
        ## it is fucking wired,the passing parameter can not be named path_alg
        # self._gas_cal = path_alg(self._level_1.sensor,**kwargs)
        self._path_alg = gas_cal(self._level_1.sensor, **kwargs)

    def setGlintCalculator(self,glint_cal,**kwargs):
        self._glint_cal =glint_cal(self._level_1.sensor,**kwargs)
        self.__nlg = np.full((self._level_1.nrows, self._level_1.ncols), np.nan, np.float)

    # def setPathAlgorithm(self,path_alg,**kwargs):
    #     _logger.debug("{},{}".format(path_alg,kwargs))
    #     self._path_alg = path_alg(self._level_1.sensor,**kwargs)
    # def setPathAlgorithm(self,path_alg,**kwargs):
    #     self._path_alg = path_alg(self._level_1.sensor,**kwargs)
    #     _logger.debug('{}'.format(self._path_alg))


    def __push_nlg(self,tile,values):
        self.__nlg[tile[0]:tile[1],tile[2]:tile[3]] = values

    def __save_nlg(self):
        fpath = os.path.join(self.__out_dir,"nLg.png")
        _logger.info("Save NLg to {}".format(fpath))
        import matplotlib.pyplot as plt
        try:
            plt.figure(figsize=(12,12))
            plt.imshow(self.__nlg)
            plt.title("nLg ($sr^{-1}$)")
            plt.colorbar(shrink=0.8)
            plt.savefig(fpath)
        except Exception as e:
            _logger.error("error when save nLg")
            raise e


    def Run(self):
        '''
        run atmospheric correction
        :return:
        '''
        # rhow_t = np.full_like(self._level_1.viewing_zenith,np.nan,np.float)


        def get_trans_no_extinction(*args,**kwargs):
            return 1.

        def get_reflectance_no_exist(*args,**kwargs):
            return 0.

        def do_nothing(*args,**kwargs):
            pass


        get_gas_trans_up = get_trans_no_extinction  if self._gas_cal is None else  self._gas_cal.get_trans_up
        get_gas_trans_down = get_trans_no_extinction if self._gas_cal is None else self._gas_cal.get_trans_down
        set_gas_obs_geo = do_nothing if self._gas_cal is None else self._gas_cal.set_obs_geo
        set_gas_ozone_watervapor = do_nothing if self._gas_cal is None else self._gas_cal.set_ozone_watervapor


        set_glint_obs_geo = do_nothing if self._glint_cal is None else self._glint_cal.set_obs_geo
        set_glint_ancillary = do_nothing if self._glint_cal is None else self._glint_cal.set_ancillary
        # get_glint_reflectance = get_reflectance_no_exist if self._glint_cal is None else self._glint_cal.get_reflectance
        get_glint_nlg = get_reflectance_no_exist if self._glint_cal is None else self._glint_cal.get_nlg
        push_nlg = do_nothing if self._glint_cal is None else self.__push_nlg
        save_nlg = do_nothing if self._glint_cal is None else self.__save_nlg

        path_retrival_push = do_nothing if self._path_alg is None else self._path_alg.push_rhot
        path_retrive = do_nothing if self._path_alg is None else self._path_alg.retrieve
        path_get_results = self._path_alg.get_results
        path_clear = do_nothing if self._path_alg is None else self._path_alg.clear
        set_path_obs_geo = do_nothing if self._path_alg is None else self._path_alg.set_obs_geo


        _logger.info("Everything get ready, Start atmospheric correction, this may take hours, be patient!")


        itile = 0

        for tile in tqdm(self._level_1.gen_tiles(size=self.tile_sile),desc='Processing Tile'):
            _logger.info("{},{}".format(tile.sline, tile.spixl))
            # if tile[0]!=667 and tile[2]!=2001:
            #     continue
            solz, viewz, phi = self._level_1.getObsGeo(tile)

            if np.all(np.isnan(viewz)):
                for i, wave in tqdm(enumerate(self._level_1.bandwaves), desc='No valid pixels'):
                    self._level_2.update_tile(tile=tile, data=np.full_like(viewz,np.nan,float), iBandIndex=i)
                continue

            ### set up gas
            set_gas_obs_geo(solz, viewz)
            set_gas_ozone_watervapor(np.full_like(solz, self._ozone, dtype=float),np.full_like(viewz, self._water_vapor, dtype=float))

            ### set up glint
            set_glint_obs_geo(solz, viewz, phi)
            dic_anc = {'windspeed':np.full_like(solz,self._windspeed,np.float)}
            set_glint_ancillary(**dic_anc)
            nlg = get_glint_nlg()
            push_nlg(tile,nlg)
            cos_solar = np.cos(np.deg2rad(solz))

            ### set up aerosol
            set_path_obs_geo(solz, viewz, phi)
            albedo_r_s = []
            try:
                for i, wave in tqdm(enumerate(self._level_1.bandwaves), desc='Processing Band'):
                    trans_g_up = get_gas_trans_up(i)
                    # trans_g_down = get_gas_trans_down(i)

                    rhot = self._level_1.getRhotBand(i,tile=tile)

                    # could not be divied by the downwelling transmittance due to gas absorption
                    # rhoaw = rhot/trans_g_up/trans_g_down-rhor-trans_r_up*rhog

                    rhot_gc = rhot/trans_g_up
                    rhot_gc.tofile(os.path.join(self.__out_dir, "rhot_gc_{}_{}".format(itile, i)))
                    path_retrival_push(i,rhot_gc)

                    del trans_g_up

                path_retrive()
                path_clear()

                for i, wave in tqdm(enumerate(self._level_1.bandwaves),desc='Correcting path reflectance and sky reflectance reflected by air-water surface'):
                    rhot_gc = np.fromfile(os.path.join(self.__out_dir, "rhot_gc_{}_{}".format(itile, i))).reshape(solz.shape)
                    rhop_t,albeo_,trans_up,trans_down = path_get_results(i)
                    rhog = nlg*np.pi*trans_up/cos_solar
                    rhow_m = (rhot_gc -rhop_t-rhog)/(trans_up*trans_down)
                    rhow = rhow_m/(1+rhow_m*(albeo_))
                    Rrs = rhow/np.pi/(trans_down)
                    self._level_2.update_tile(tile=tile,data=Rrs,iBandIndex=i)
            except Exception as e:
                _logger.error(e)
            finally:
                tempfiles = glob.glob(os.path.join(self.__out_dir, "rhot_gc_{}*".format(itile)))
                if len(tempfiles)>0:
                    for tf in tqdm(tempfiles,desc="removing temp files"):
                        os.remove(tf)

            itile+=1
        save_nlg()
        self._level_2.finish_update()
