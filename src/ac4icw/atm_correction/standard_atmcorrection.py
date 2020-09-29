import numpy as np
import os
from tqdm import tqdm
import glob

from ..atm_correction.interface import AtmCorrectionInterface,RayleighCalculator
from ..level1 import Level1_Base
from ..level2 import Level2
from ..decorators import processing_info

import logging


_logger = logging.getLogger("Standard AC")

@processing_info
class StandardAtmCorrection(AtmCorrectionInterface):
    '''
    Normal AC, step by step:
    GAS Correction->Rayleigh Coreection->Glint->Adjacency effect->Aerosol

    the basic theory of atmosphereic radiative transfer refers to 6SV User Guide (v1.1):

    Equations of (5),(19)
    '''
    def __init__(self,level_1:Level1_Base,level_2:Level2,**kwargs):
        self._gas_cal = None
        self._rayleigh_cal =None
        self._aero_alg = None
        self._glint_cal = None
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



    def setGasCalculator(self,gas_cal,**kwargs):
        self._gas_cal = gas_cal(self._level_1.sensor,**kwargs)

    def setRayleighCalculator(self,rayleigh_cal,**kwargs):
        if not issubclass(rayleigh_cal,RayleighCalculator):
            raise Exception()
        self._rayleigh_cal = rayleigh_cal(self._level_1.sensor,**kwargs)

    def setAeroAlgorithm(self,aero_alg,**kwargs):
        self._aero_alg = aero_alg(self._level_1.sensor,**kwargs)

    def setGlintCalculator(self,glint_cal,**kwargs):
        self._glint_cal =glint_cal(self._level_1.sensor,**kwargs)
        self.__nlg = np.full((self._level_1.nrows, self._level_1.ncols), np.nan, np.float)

    def setAdjAlgorithm(self,adj_alg,**kwargs):
        self._adj_alg = adj_alg(self._level_1.sensor,**kwargs)

    def setLevel_1(self,level_1:Level1_Base):
        self._level_1 = level_1

    def setLevel_2(self,level_2:Level2):
        self._level_2 = level_2
        self.__out_dir = self._level_2.outputer.output_dir

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

        def aerosol_results(*args,**kwargs):
            return (0., 0., 1.0, 1.0)


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

        aerosol_retrival_push = do_nothing if self._aero_alg is None else self._aero_alg.push_rhoaw
        aerosol_retrive = do_nothing if self._aero_alg is None else self._aero_alg.retrieve
        aerosol_get_results = aerosol_results if self._aero_alg is None else self._aero_alg.get_results
        aerosol_clear = do_nothing if self._aero_alg is None else self._aero_alg.clear
        set_aero_obs_geo = do_nothing if self._aero_alg is None else self._aero_alg.set_obs_geo

        set_rayleigh_obs_geo = self._rayleigh_cal.set_obs_geo
        rayleigh_get_reflectance = self._rayleigh_cal.get_reflectance
        rayleigh_get_trans_up = self._rayleigh_cal.get_trans_up
        rayleigh_get_trans_down = self._rayleigh_cal.get_trans_down
        rayleigh_get_spherical_albedo = self._rayleigh_cal.cal_spherical_albedo

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


            ### set up rayleigh
            set_rayleigh_obs_geo(solz, viewz, phi)

            ### set up aerosol
            set_aero_obs_geo(solz, viewz, phi)
            albedo_r_s = []
            try:
                for i, wave in tqdm(enumerate(self._level_1.bandwaves), desc='Processing Band'):
                    trans_g_up = get_gas_trans_up(i)
                    trans_g_down = get_gas_trans_down(i)

                    rhor = rayleigh_get_reflectance(i)
                    trans_r_up,trans_r_down =  rayleigh_get_trans_up(i),rayleigh_get_trans_down(i)
                    trans_r_up.tofile(os.path.join(self.__out_dir,"trans_r_up_{}_{}".format(itile,i)))
                    trans_r_down.tofile(os.path.join(self.__out_dir, "trans_r_down_{}_{}".format(itile, i)))


                    albeo_r = rayleigh_get_spherical_albedo(i)
                    albedo_r_s.append(albeo_r)

                    # rhog = get_glint_reflectance(trans_r_down)
                    rhog = nlg*np.pi*trans_r_down/cos_solar

                    rhot = self._level_1.getRhotBand(i,tile=tile)

                    rhoaw = rhot/trans_g_up/trans_g_down-rhor-trans_r_up*rhog
                    rhoaw.tofile(os.path.join(self.__out_dir, "rhoaw_{}_{}".format(itile, i)))
                    aerosol_retrival_push(i,rhoaw)

                    del rhor,trans_g_up,trans_g_down,rhog

                aerosol_retrive()
                aerosol_clear()

                for i, wave in tqdm(enumerate(self._level_1.bandwaves),desc='Correcting aerosol'):
                    rhoaw = np.fromfile(os.path.join(self.__out_dir, "rhoaw_{}_{}".format(itile, i))).reshape(solz.shape)
                    trans_r_up = np.fromfile(os.path.join(self.__out_dir,"trans_r_up_{}_{}".format(itile,i))).reshape(solz.shape)
                    trans_r_down = np.fromfile(os.path.join(self.__out_dir,"trans_r_down_{}_{}".format(itile,i))).reshape(solz.shape)
                    rhoa,albeo_a,trana_up,trana_down = aerosol_get_results(i)
                    rhow_m = (rhoaw -rhoa)/(trana_up*trans_r_up*trana_down*trans_r_down)
                    rhow = rhow_m/(1+rhow_m*(albeo_a+albedo_r_s[i]))
                    Rrs = rhow/np.pi/(trana_down*trans_r_down)
                    self._level_2.update_tile(tile=tile,data=Rrs,iBandIndex=i)
            except Exception as e:
                _logger.error(e)
            finally:
                tempfiles = glob.glob(os.path.join(self.__out_dir, "rhoaw_{}*".format(itile)))+glob.glob(os.path.join(self.__out_dir, "trans_r_up_{}*".format(itile)))+glob.glob(os.path.join(self.__out_dir, "trans_r_down_{}*".format(itile)))
                if len(tempfiles)>0:
                    for tf in tqdm(tempfiles,desc="removing temp files"):
                        os.remove(tf)

            itile+=1
        save_nlg()
        self._level_2.finish_update()




        # for tile in tqdm(self._level_1.gen_tiles(size=self.tile_sile),desc='Processing Tile'):
        #     _logger.info("{},{}".format(tile.sline,tile.spixl))
        #     # if tile.sline!=1334 or tile.spixl!=1334:
        #     #     continue
        #     rhoaw_s, trans_r_up_s,trans_r_down_s,albeo_r_s = [] ,[],[],[]
        #     solz, viewz,phi = self._level_1.getObsGeo(tile)
        #
        #     if np.all(np.isnan(viewz)):
        #         for i, wave in tqdm(enumerate(self._level_1.bandwaves), desc='No valid pixels'):
        #             self._level_2.update_tile(tile=tile, data=np.full_like(viewz,np.nan,float), iBandIndex=i)
        #         continue
        #
        #     self._gas_cal.set_obs_geo()
        #
        #     self._aero_alg.setObsGeometry(solz,viewz,phi)
        #
        #     for i,wave in tqdm(enumerate(self._level_1.bandwaves),desc='Processing Band'):
        #         rhot = self._level_1.getRhotBand(i,tile=tile)
        #
        #         trans_g_up = self._gas_cal.cal_trans_up(i,viewz,water_vapor=np.full_like(viewz,3,dtype=float),ozone=np.full_like(viewz,0.3,dtype=float))
        #         trans_g_down = self._gas_cal.cal_trans_down(i, solz, water_vapor=np.full_like(solz,3,dtype=float),ozone=np.full_like(solz,0.3,dtype=float))
        #         rhot = rhot/(trans_g_up*trans_g_down)
        #
        #         if self._glint_cal!=None:
        #             rhog = self._glint_cal.calculate(iband=i,solz=solz,senz=viewz,phi=phi,windspeed=2.0)
        #             rhot = rhot- rhog
        #         rhor = self._rayleigh_cal.cal_reflectance(i,solz,viewz,phi)
        #         trans_r_down =  self._rayleigh_cal.cal_trans_down(i,solz)
        #         trans_r_up =  self._rayleigh_cal.cal_trans_down(i,viewz)
        #         albeo_r = self._rayleigh_cal.cal_spherical_albedo(iBandIndex=i)
        #
        #         rhoaw = rhot-rhor
        #         self._aero_alg.push_rhoaw(iBandIndex=i,rhoaw=rhoaw)
        #         rhoaw_s.append(rhoaw)
        #         trans_r_up_s.append(trans_r_up)
        #         trans_r_down_s.append(trans_r_down)
        #         albeo_r_s.append(albeo_r)
        #
        #         del rhot,rhor,trans_r_down,trans_r_up,albeo_r,trans_g_up,trans_g_down
        #
        #     # retrieve aerosol type and AOT
        #     self._aero_alg.retrieve()
        #
        #     self._aero_alg.clear()
        #     # for i,wave in tqdm(enumerate(self._level_1.bandwaves),desc='Processing Band(aerosol correction)'):
        #     for i, wave in tqdm(enumerate(self._level_1.bandwaves),desc='Correcting aerosol'):
        #         rhoa,albeo_a,trana_up,trana_down = self._aero_alg.get_results(iBandIndex=i)
        #         rhow_m = (rhoaw_s[i] -rhoa)/(trana_up*trans_r_up_s[i]*trana_down*trans_r_down_s[i])
        #         rhow = rhow_m/(1+rhow_m*(albeo_a+albeo_r_s[i]))
        #         Rrs = rhow/np.pi/(trana_down*trans_r_down_s[i])
        #         self._level_2.update_tile(tile=tile,data=Rrs,iBandIndex=i)

                # self._level_2.update_tile(tile=tile, data=rhoaw_s[i], iBandIndex=i)

        # self._level_2.finish_update()







