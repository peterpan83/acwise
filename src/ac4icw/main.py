import sys,os
from .level2 import Level2
from .output import RasterIOWriter
from .helper import createInstance
from .helper import getClass

import logging

sys.path.append(...)

_logger = logging.getLogger("Main")

def build_level1(config_dic:dict):
    sensor = str.upper(config_dic['SENSOR']['NAME'])

    data_dir = config_dic['DATA']['DATA_DIR']
    if data_dir is None:
        _logger.error('Please specify DATA_DIR')
        sys.exit(-1)
    elif not (os.path.isdir(data_dir) and os.path.exists(data_dir)):
        _logger.debug(data_dir)
        _logger.debug(os.path.exists(data_dir))
        _logger.error('Illegal or no exist DATA_DIR')
        sys.exit(-1)

    # _logger.info('Sensor:{}'.format(sensor))
    if sensor not in config_dic:
        # _logger.error('config of the {} section not found')
        pass
    # ---------------------------------Initialize Level 1------------------------#
    dic_sensor = config_dic[sensor]
    dic_sensor.update(config_dic['DATA'])
    level_1 = createInstance('ac4icw.sensor.level1_{}'.format(str.lower(sensor)), 'L1_{}'.format(sensor), **dic_sensor)
    # level_1.extract_f0()
    return level_1


def build_ac(config_dic:dict):
    '''
    build atmospheric correction instance from the config dictionary
    :param config_dic:
    :return: instance of AtmCorrectionInterface
    '''
    sensor = str.upper(config_dic['SENSOR']['NAME'])

    data_dir = config_dic['DATA']['DATA_DIR']
    if data_dir is None:
        _logger.error('Please specify DATA_DIR')
        sys.exit(-1)
    elif not (os.path.isdir(data_dir) and os.path.exists(data_dir)):
        _logger.debug(data_dir)
        _logger.debug(os.path.exists(data_dir))
        _logger.error('Illegal or no exist DATA_DIR')
        sys.exit(-1)

    # _logger.info('Sensor:{}'.format(sensor))
    if sensor not in config_dic:
        # _logger.error('config of the {} section not found')
        pass
#---------------------------------Initialize Level 1------------------------#
    dic_sensor =  config_dic[sensor]
    dic_sensor.update(config_dic['DATA'])
    level_1 = createInstance('ac4icw.sensor.level1_{}'.format(str.lower(sensor)),'L1_{}'.format(sensor),**dic_sensor)
    level_1.cal_water_mask()
    level_1.cal_viewing_geo()
    level_1.cal_phi()
    # level_1.extract_f0()
    # level_1.setWaterMask()

#--------------------------------Initialize Level 2-----------------------------#
    level_2 = Level2.FromLevel1(level_1)
    writer = RasterIOWriter.FromLevel2(level_2,**config_dic['OUTPUT'])
    # writer = PlotShow()
    level_2.setOutputer(writer)

#---------------------------------build AC Chain-----------------------------------#
    procedure = str.lower(config_dic['AC']['PROCEDURE'])
    # tile_size = float(config_dic['AC']['TILE_SIZE']) if 'TILE_SIZE' in config_dic['AC'] else 2.0
    atm_c = createInstance('ac4icw.atm_correction.{}_atmcorrection'.format(procedure),'{}AtmCorrection'.format(procedure.capitalize()),level_1,level_2,**config_dic['AC'])

    basic_parameter_dic = {'groud_altitude': level_1.ground_altitude, 'sensor_altitude': level_1.sensor_altitude,
                           'month': level_1.acq_time.month,'latitude': level_1.center_latitude,'data_dir':data_dir}


    gas_cal = config_dic[str.upper(procedure)]['GAS_CALCULATOR']
    if gas_cal is None:
        _logger.warning("No gas absorption correction")
    else:
        gas_cal_cal_class = getClass('ac4icw.atm_correction.gas', 'Gas{}'.format(gas_cal))
        atm_c.setGasCalculator(gas_cal_cal_class, **basic_parameter_dic)



    ray_cal = config_dic[str.upper(procedure)]['RAY_CALCULATOR']
    if ray_cal is None:
        _logger.warning('No rayleigh correction')
    else:
        rayleigh_cal_class = getClass('ac4icw.atm_correction.rayleigh', 'Rayleigh{}'.format(ray_cal))
        atm_c.setRayleighCalculator(rayleigh_cal_class, **basic_parameter_dic)



    adj_alg = config_dic[str.upper(procedure)]['ADJ_ALGRITHM']
    if adj_alg is None:
        _logger.warning("No adjacency effection correction!")
    else:
        _logger.warning("Adjacency effect correction not implemented yet!")
        pass


    glint_cal = config_dic[str.upper(procedure)]['GLINT_CALCULATOR']
    if glint_cal is None:
        _logger.warning("No glint correction!")
    else:
        # _logger.warning("Glint correction not implemented yet!")
        glint_cal_class = getClass('ac4icw.atm_correction.glint', 'Glint{}'.format(glint_cal))
        atm_c.setGlintCalculator(glint_cal_class,**basic_parameter_dic)


    aero_alg, aero_cal = config_dic[str.upper(procedure)]['AERO_ALGRITHM'],config_dic[str.upper(procedure)]['AERO_CALCULATOR']
    if ~(aero_alg is None or aero_cal is None):
        aerosol_cal_class = getClass('ac4icw.atm_correction.aerosol', 'Aerosol{}'.format(aero_cal))
        aerosol_retrival_calss = getClass('ac4icw.atm_correction.aero_retrival', 'Aerosol{}'.format(aero_alg))
        basic_parameter_dic.update({'aero_cal': aerosol_cal_class})
        basic_parameter_dic.update(config_dic[aero_alg])
        atm_c.setAeroAlgorithm(aerosol_retrival_calss, **basic_parameter_dic)


    return atm_c






    




