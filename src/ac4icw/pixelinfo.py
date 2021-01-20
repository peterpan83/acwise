import configparser

from ac4icw.tile import Tile
from ac4icw.main import build_level1
import matplotlib.pyplot as plt


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

config = configparser.ConfigParser()
config.read("d:/Work/Programe/ac4icw/config.ini")

config_dict = get_dict(config)


level_1 = build_level1(config_dict)
level_1.cal_water_mask()
level_1.cal_viewing_geo()
level_1.cal_phi()
_y,_x = 1077,795
rhot = level_1.getRhotSpectrum(iline=_y, isample=_x)

Lt = level_1.read_spectrum(iline=_y, isample=_x)



plt.plot(level_1.bandwaves,rhot,label="$\\rho_t$")
plt.show()

plt.plot(level_1.bandwaves,Lt,label="$L_t$")
plt.show()

print(level_1.F0['values'])

tile = Tile(_y,_y+1,_x,_x+1)
solz, senz, phi = level_1.getObsGeo(tile)
title = "\nx:{},y:{},$\\theta_s$:{:.3f},$\\theta_v$:{:.3f},$\\phi$:{:.3f}".format(_x, _y,solz.flatten()[0],
                                                                                          senz.flatten()[0],
                                                                                          phi.flatten()[0])

print(title)


