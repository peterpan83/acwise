import os
from configparser import ConfigParser

class Config():
    def __init__(self):
        self.ROOTPATH = os.path.split(os.path.split(os.path.realpath(__file__))[0])[0]

        confg = ConfigParser()
        confg.read(os.path.join(self.ROOTPATH,'confg.ini'))
        self.DATADIR = os.path.join(self.ROOTPATH, confg['ROOT']['DATADIR'])
