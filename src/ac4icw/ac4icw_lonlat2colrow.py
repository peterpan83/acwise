import argparse
import sys,os
import logging
import configparser
import numpy as np

from ac4icw.exception_ac import ErrorRayleighMissing
from ac4icw import __version__

from ac4icw.main import build_level1

def get_dict(config:configparser.ConfigParser):
    '''
    convert ConfigParser to dict
    :param config:
    :return:
    '''
    sections_dict = {}
    sections = config.sections()
    for section in sections:
        options = config.options(section)
        temp_dict = {}
        for option in options:
            value = None if str.upper(config.get(section,option)) == 'NONE' else config.get(section,option)
            temp_dict[str.upper(option)] = value

        sections_dict[str.upper(section)] = temp_dict

    return sections_dict

def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(prog='ac4icw',
                                     description="Atmospheric Correction for Inland and Coastal Waters (v2020.0918.01)\n Author:Yanqun Pan (panyq213@163.com))")
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


def build_l1(config_f):
    config = configparser.ConfigParser()
    config.read(config_f)
    config_dict = get_dict(config)
    level_1 = build_level1(config_dict)
    return level_1

def find_colrow(l1,location):
    print(longitude.min(),longitude.max(),latitude.min(), latitude.max())
    longitude, latitude = l1.longitude, l1.latitude
    lon,lat = location
    if lon< longitude.min() or lon>longitude.max():
        print("lon:{} not in this scene longitude:[{},{}]".format(lon,longitude.min(),longitude.max()))
        return
    if lat< latitude.min() or lat>  latitude.max():
        print("lat:{} not in this scene latitude:[{},{}]".format(lat, latitude.min(), latitude.max()))

    distance =  np.power(longitude-lon,2)+np.power(latitude-lat,2)
    min_index = distance.argmin()
    row,col = np.unravel_index(min_index,distance.shape)
    return row, col

def find_obgeo(l1,row,col):
    szen = l1.solar_zenith[row,col]
    s_azimuth = l1.solar_azimuth[row,col]
    vzen = l1.viewing_zenith[row,col]
    v_azimuth = l1.viewing_azimuth[row,col]
    print('szen,s_azimuth,vzen,v_azimuth:{:.4f},{:.4f},{:.4f},{:.4f}'.format(szen,s_azimuth,vzen,v_azimuth))



