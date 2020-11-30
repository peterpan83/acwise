# -*- coding: utf-8 -*-
"""
This is a skeleton file that can serve as a starting point for a Python
console script. To run this script uncomment the following lines in the
[options.entry_points] section in setup.cfg:

    console_scripts =
         fibonacci = ac4icw.skeleton:run

Then run `python setup.py install` which will install the command `fibonacci`
inside your current environment.
Besides console scripts, the header (i.e. until _logger...) of this file can
also be used as template for Python modules.

Note: This skeleton file can be safely removed if not needed!
"""

import argparse
import sys,os
import numpy as np
import logging
import configparser
from plumbum import cli
import plumbum.colors as colors
from rasterio.plot import show
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# plt.style.use('presentation.mplstyle')

from ac4icw import __version__

__author__ = "panyq"
__copyright__ = "panyq"
__license__ = "mit"

_logger = logging.getLogger(__name__)

from ac4icw.main import build_level1
from ac4icw.helper import covert_config_to_dic
from ac4icw.helper import createInstance
from ac4icw.helper import getClass
from ac4icw.tile import Tile

if os.path.exists('presentation.mplstyle'):
    plt.style.use('presentation.mplstyle')
    
def plot_image(img,title,watermask=None):

    plt.figure(figsize=(8,8))
    _max = img.max()*0.1
    if watermask is not None:
        _max = img[watermask].max()*1.5

    img[img==0] = np.nan
    plt.imshow(img,vmax=_max)
    plt.colorbar()
    plt.title(title)
    plt.show()

def plot_image_(img,title):

    plt.figure(figsize=(8,8))
    plt.imshow(img)
    plt.colorbar()
    plt.title(title)
    plt.show()

def plot_spectrum(xaxis,yaxis,names,title,datatype):
    _,axes = plt.subplots(ncols=len(datatype),nrows=1,figsize=(5*len(datatype),4))
     
    # plt.figure(figsize=(10,10))
    for y_v,name in zip(yaxis,names):
        if name.find('trans')>-1:
            axes[1].plot(xaxis,y_v,label=name)
        else:
            axes[0].plot(xaxis,y_v,label=name)
    axes[0].set_title(title)
    axes[1].set_title(title+'\n transmittance due to gas absorption')
    axes[0].set_xlabel("Wavelength (nm)")
    axes[1].set_xlabel("Wavelength (nm)")
    axes[0].set_ylabel("reflectance")
    axes[1].set_ylabel("transmittance")
    axes[0].legend(ncol=2)
    axes[1].legend()
    plt.show()


class ACPlot(cli.Application):
    PROGNAME = colors.green
    VERSION = colors.blue | "0.1"
    COLOR_GROUPS = {"Meta-switches" : colors.bold & colors.yellow}
    verbose = cli.Flag(["v", "verbose"], help = "set loglevel to INFO")
    very_verbose = cli.Flag(["vv","veryverbose"],help="set loglevel to DEBUG")

    def main(self, *args):
        print(self.verbose,self.very_verbose)

        if self.verbose:
            setup_logging(logging.INFO)
        if self.very_verbose:
            setup_logging(logging.DEBUG)
            
        if not self.nested_command:           # will be ``None`` if no sub-command follows
            print("No command given")
            return 1   # error exit code


@ACPlot.subcommand("image")
class Image(cli.Application):
    PROGNAME = colors.green
    VERSION = colors.blue | "1.0"
    COLOR_GROUPS = {"Meta-switches" : colors.bold & colors.yellow}
    IMAGE_NAMES_DIC = {'t':'$\\rho_t$','r':'$\\rho_r$','m':'$\\rho_{rc}$','a':'$\\rho_a$','w':'$\\rho_w$','k':'$\\rho_{gc}$','g':'trans'}
    SLINE,ELINE,SPIXL,EPIXL = 0,-1,0,-1
    _bandindex = 0

    @cli.switch(["-c"],str,mandatory=True,help="config.ini file that runs the AC program")
    def config_file(self,config_f):
        self._config_f = config_f
        if not os.path.exists(config_f):
            print("{} does not exsit!".format(config_f))
            sys.exit(-1)
        config = configparser.ConfigParser()
        config.read(config_f)
        config_dic = covert_config_to_dic(config)
        self._config_dic = config_dic

    @cli.switch(['-N'],str,mandatory=True,help="name of the image you want to plot,"
                                 "like tramw for :rhot,rhor,rhoa,rhorc,rhow")
    def spectrum_names(self,name):
        self._name = str.lower(name)

    @cli.switch(['-I'],int,help="band index,starts from 0")
    def band_index(self,index):
        self._bandindex = index

    def main(self, *args):
        if self._name not in self.IMAGE_NAMES_DIC:
            _logger.error("not recognized image name:{}".format(self._name))
            sys.exit(-1)
        config_dic = self._config_dic
        level_1 = build_level1(config_dic)
        level_1.cal_water_mask()
        level_1.cal_viewing_geo()
        level_1.cal_phi()

        if self.ELINE == -1:
            self.ELINE = level_1.nrows
        if self.EPIXL == -1:
            self.EPIXL = level_1.ncols

        tile = Tile(self.SLINE,self.ELINE,self.SPIXL,self.EPIXL)
        water_mask = level_1.get_water_mask(tile)

        # solz, senz, phi = level_1.getObsGeo(tile)
        title = level_1.image_name

        waves = level_1.bandwaves
        data_dir = config_dic['DATA']['DATA_DIR']
        basic_parameter_dic = {'groud_altitude': level_1.ground_altitude, 'sensor_altitude': level_1.sensor_altitude,
                               'month': level_1.acq_time.month, 'latitude': level_1.center_latitude,
                               'data_dir': data_dir}

        image = None
        if self._name == 't':
            title = title +"\n {}({})".format(self.IMAGE_NAMES_DIC[self._name],waves[self._bandindex])
            rhot = level_1.getRhotBand(bandindex=self._bandindex, tile=tile)
            _logger.info(str(rhot.shape))
            image = rhot

        plot_image(image,title=title,watermask=water_mask)

@ACPlot.subcommand("spectrum")
class Spectrum(cli.Application):
    PROGNAME = colors.green
    VERSION = colors.blue | "1.0"
    COLOR_GROUPS = {"Meta-switches" : colors.bold & colors.yellow}
    _aero_type_index = 2
    _taua550 = 0.005

    SPECTRUM_NAMES_DIC = {'t':'$\\rho_t$','r':'$\\rho_r$','m':'$\\rho_{rc}$','a':'$\\rho_a$',
    'w':'$\\rho_w$','k':'$\\rho_{gc}$','g':'trans','f':'F0 ($$)','b':'$\\rho_b$'}

    @cli.switch(["-c"],str,mandatory=True,help="config.ini file that runs the AC program")
    def config_file(self,config_f):
        self._config_f = config_f
        if not os.path.exists(config_f):
            print("{} does not exsit!".format(config_f))
            sys.exit(-1)
        config = configparser.ConfigParser()
        config.read(config_f)
        config_dic = covert_config_to_dic(config)
        self._config_dic = config_dic

    @cli.switch(['-N'],list,mandatory=True,help="names of the spectrum you want to plot,like tramw for :rhot,rhor,rhoa,rhorc,rhow")
    def spectrum_names(self,names):
        self._names = names

    @cli.switch(["-y","--row"],int,mandatory=True,help="the y position of the pixel")
    def set_irow(self,y):
        self._y = y

    @cli.switch(["-x","--column"],int,mandatory=True,help="the x position of the pixel")
    def set_icol(self,x):
        self._x = x

    @cli.switch(["--iaero"],int,help="aerosol type index")
    def set_aerosoltype(self,index):
        self._aero_type_index = index

    @cli.switch(["--taua"],float,help="AOT at 550nm")
    def set_taua(self,value):
        self._taua550 = value

    @cli.switch(["--saveto"],str,help="specific a path to save the data as csv")
    def set_csv(self,value):
        self._csvf = value


    def main(self, *args):

        # x:2505 y:1886
        config_dic = self._config_dic
        level_1 = build_level1(config_dic)
        level_1.cal_water_mask()
        level_1.cal_viewing_geo()
        level_1.cal_phi()

        # level_1.setWaterMask()
        ylabel = 'Reflectance'
        tile = Tile(self._y,self._y+1,self._x,self._x+1)
        solz, senz, phi = level_1.getObsGeo(tile)
        title = level_1.image_name
        title = title+"\nx:{},y:{},$\\theta_s$:{:.3f},$\\theta_v$:{:.3f},$\\phi$:{:.3f}".format(self._x,self._y,solz.flatten()[0], senz.flatten()[0], phi.flatten()[0])

        _logger.debug("solar zenith:{},veiwing zenith:{},phi:{}".format(solz,senz,phi))
        waves = level_1.bandwaves

        y_values,names = [],[]
        data_dir = config_dic['DATA']['DATA_DIR']
        basic_parameter_dic = {'groud_altitude': level_1.ground_altitude, 'sensor_altitude': level_1.sensor_altitude,
                               'month': level_1.acq_time.month, 'latitude': level_1.center_latitude,
                               'data_dir': data_dir}

        rhot = level_1.getRhotSpectrum(iline=self._y, isample=self._x)

        procedure = str.lower(config_dic['AC']['PROCEDURE'])

        #-----------------------------gas-------------------------------#
        gas_cal_name = config_dic[str.upper(procedure)]['GAS_CALCULATOR']
        gas_cal_cal_class = getClass('ac4icw.atm_correction.gas', 'Gas{}'.format(gas_cal_name))
        gas_cal = gas_cal_cal_class(level_1.sensor, **basic_parameter_dic)
        trans_g_up_all, trans_g_down_all = [], []
        for i in tqdm(range(len(waves)), desc='gas absorption calculating'):
            trans_g_up = gas_cal.cal_trans_up(i, senz, water_vapor=np.full_like(senz, 3, dtype=float),
                                          ozone=np.full_like(senz, 0.3, dtype=float))
            trans_g_down = gas_cal.cal_trans_down(i, solz, water_vapor=np.full_like(solz, 3, dtype=float),
                                              ozone=np.full_like(solz, 0.3, dtype=float))
            trans_g_up_all.append(trans_g_up)
            trans_g_down_all.append(trans_g_down)
        trans_g_up_all, trans_g_down_all = np.asarray(trans_g_up_all).flatten(),np.asarray(trans_g_down_all).flatten()

        #--------------------------------------rayleigh----------------------------------#
        ray_cal_name = config_dic[str.upper(procedure)]['RAY_CALCULATOR']
        rayleigh_cal_class = getClass('ac4icw.atm_correction.rayleigh', 'Rayleigh{}'.format(ray_cal_name))
        ray_cal = rayleigh_cal_class(level_1.sensor, **basic_parameter_dic)
        rhor_all, trans_r_down_all, trans_r_up_all, albeo_r_all = [], [], [], []
        for i in tqdm(range(len(waves)), desc='rayleigh calculating'):
            rhor = ray_cal.cal_reflectance(i, solz, senz, phi)
            trans_r_down = ray_cal.cal_trans_down(i, solz)
            trans_r_up = ray_cal.cal_trans_down(i, senz)
            albeo_r = ray_cal.cal_spherical_albedo(iBandIndex=i)
            rhor_all.append(rhor)
            trans_r_down_all.append(trans_r_down)
            trans_r_up_all.append(trans_r_up)
            albeo_r_all.append(albeo_r)
        rhor_all, trans_r_down_all, trans_r_up_all, albeo_r_all = np.asarray(rhor_all).flatten(), np.asarray(
            trans_r_down_all).flatten(), np.asarray(trans_r_up_all).flatten(), np.asarray(albeo_r_all).flatten()


        #---------------------------------aerosol-------------------------------------#
        aero_cal_name = config_dic[str.upper(procedure)]['AERO_CALCULATOR']
        aerosol_cal_class = getClass('ac4icw.atm_correction.aerosol', 'Aerosol{}'.format(aero_cal_name))
        aero_cal = aerosol_cal_class(level_1.sensor, **basic_parameter_dic)
        rhoa_all, trans_a_up_all, trans_a_down_all, albedo_a_all = [], [], [], []
        for i in tqdm(range(len(waves)), desc='aerosol calculating'):
            rhoa = aero_cal.cal_reflectance(i, solz, senz, phi, self._aero_type_index, self._taua550)
            _logger.debug("{}".format(rhoa.shape))
            trans_a_down = aero_cal.cal_trans_down(i, solz, self._aero_type_index, self._taua550)
            trans_a_up = aero_cal.cal_trans_down(i, senz, self._aero_type_index, self._taua550)
            albeo_a = aero_cal.cal_spherical_albedo(i, self._aero_type_index, self._taua550)
            rhoa_all.append(rhoa)
            trans_a_up_all.append(trans_a_up)
            trans_a_down_all.append(trans_a_down)
            albedo_a_all.append(albeo_a)

        rhoa_all, trans_a_down_all, trans_a_up_all, albedo_a_all = np.asarray(rhoa_all).flatten(),\
                                                                   np.asarray(trans_a_down_all).flatten(), np.asarray(trans_a_up_all).flatten(), np.asarray(albedo_a_all).flatten()
        data_type = ['reflectance']
        if 'r' in self._names:
            y_values.append(rhor_all)
            names.append(self.SPECTRUM_NAMES_DIC['r'])
        if 'a' in self._names:
            y_values.append(rhoa_all)
            names.append(self.SPECTRUM_NAMES_DIC['a'])
            # title+='\naerosol:{},$\\tau_a(550)$:{:.3f}'.format(aero_cal.get_aerosol_type(self._aero_type_index),self._taua550)
        if 't' in self._names:
            y_values.append(rhot)
            names.append(self.SPECTRUM_NAMES_DIC['t'])
        if 'k' in self._names:
            #rhogc = rhot / trans_g_down_all/ trans_g_up_all
            print(str(trans_g_up_all))
            rhogc = rhot / trans_g_up_all
            y_values.append(rhogc)
            names.append(self.SPECTRUM_NAMES_DIC['k'])
        if 'g' in self._names:
            # ylabel = 'Transmittance'
            data_type.append('Transmittance')
            # title = title+'\n transmittance due to gas absorption'
            y_values.append(trans_g_down_all)
            names.append('downwelling '+self.SPECTRUM_NAMES_DIC['g'])
            y_values.append(trans_g_up_all)
            names.append('upwelling '+self.SPECTRUM_NAMES_DIC['g'])
        if 'w' in self._names or 'b' in self._names:
            rhogc = rhot / trans_g_up_all
            rhorc = rhogc - rhor_all
            if 'm' in self._names:
                # rhorc[rhorc<0] = np.nan
                y_values.append(rhorc)
                names.append(self.SPECTRUM_NAMES_DIC['m'])

            rhow_m = (rhorc - rhoa_all) / (
                        trans_a_up_all * trans_r_up_all * trans_a_down_all * trans_r_down_all)
            rhow = rhow_m / (1 + rhow_m * (albedo_a_all + albeo_r_all))
            # rhow[rhow<0] = np.nan
            y_values.append(rhow)
            if 'w' in self._names:
                names.append(self.SPECTRUM_NAMES_DIC['w'])
            else:
                names.append(self.SPECTRUM_NAMES_DIC['b'])

            if self._csvf is not None:
                # if not os.path.(self._csvf):
                #     _logger.warn("{} invalid".format(self._csvf))
                # else:
                pd.DataFrame(data=y_values,columns=waves,index=names).to_csv(self._csvf)


        
        plot_spectrum(waves,y_values,names,title=title,datatype=data_type)





@ACPlot.subcommand('obgeo')
class ObGeo(cli.Application):
    PROGNAME = colors.green
    VERSION = colors.blue | "1.0"
    COLOR_GROUPS = {"Meta-switches" : colors.bold & colors.yellow}

    @cli.switch(["-c"],str)
    def config_file(self,config_f):
        self._config_f = config_f
        if not os.path.exists(config_f):
            print("{} does not exsit!".format(config_f))
            sys.exit(-1)
        config = configparser.ConfigParser()
        config.read(config_f)
        config_dic = covert_config_to_dic(config)
        self._config_dic = config_dic


    def main(self, *args):
        level_1 = build_level1(self._config_dic)
        # level_1.cal_solar_geo()
        level_1.cal_water_mask()
        level_1.cal_viewing_geo()
        level_1.cal_phi()

        solz, senz, phi = level_1.getObsGeo()
        solar_aimuth = level_1.solar_azimuth
        viewing_azimuth = level_1.viewing_azimuth

        plot_image_(solz,'solar zenith angle')
        plot_image_(senz,'viewing zenith angle')

        plot_image_(solar_aimuth,'solar azimuth')
        plot_image_(viewing_azimuth,'viewing azimuth')
        plot_image_(phi, 'relative azimuth angle')
def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(prog='ac4icw',
                                     description="Atmospheric Correction for Inland and Coastal Waters (v2020.0918.01)\n Author:Yanqun Pan (panyq213@163.com))")
    parser.add_argument('command', metavar='N', type=str, nargs='+',
                        help='plot command, like obgeo,etc.')

    parser.add_argument(
        "--version",
        action="version",
        version="ac4icw {ver}".format(ver=__version__))

    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO)
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG)

    parser.add_argument(
        '-c', '--confg', type=str, help='config file')


    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")

def plot_obgeo(level_1):
    level_1.cal_solar_geo()
    level_1.cal_viewing_geo()
    level_1.cal_phi()

    solz,senz,phi = level_1.getObsGeo()
    show(solz,transform=level_1.affine,title='solar zenith')
    show(senz,transform=level_1.affine,title='viewing znith')
    show(phi,transform=level_1.affine,title='relative azimuth')

def main(args):
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    args = parse_args(args)
    setup_logging(args.loglevel)
    print(args.command)

    config_f = args.confg
    if config_f is None or not os.path.exists(config_f):
        print('config file does not exist:{}'.format(config_f))
        sys.exit()
    config = configparser.ConfigParser()
    config.read(config_f)

    config_dict = covert_config_to_dic(config)
    level_1 = build_level1(config_dict)
    plot_obgeo(level_1)


def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    # run()
    ACPlot.run()
