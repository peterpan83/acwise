# -*- coding: utf-8 -*-
import argparse
import sys,os
import logging
import configparser

from ac4icw.exception_ac import ErrorRayleighMissing
from ac4icw import __version__

__author__ = "panyq"
__copyright__ = "panyq"
__license__ = "mit"

_logger = logging.getLogger("AC COMMAND LINE")

from ac4icw.main import build_ac


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


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")


def main(args):
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    args = parse_args(args)
    setup_logging(args.loglevel)
    _logger.info("Starting AC...")

    config_f = args.confg
    if config_f is None or not os.path.exists(config_f):
        print('config file does not exist:{}'.format(config_f))
        sys.exit()
    config = configparser.ConfigParser()
    config.read(config_f)

    config_dict = get_dict(config)
    try:
        atc = build_ac(config_dict)
    except ErrorRayleighMissing as e:

        _logger.error("can not build AC process chain, no Rayleigh calculator")
        sys.exit(-1)
    # except Exception as e:
    #     _logger.error("{}".format(e))
    #     _logger.error("can not build AC process chain,exit")
    #     sys.exit(-1)
    else:
        atc.Run()



def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
