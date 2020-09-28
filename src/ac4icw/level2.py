import numpy as np

from .level1 import Level1_Base,Tile
from .output import OutputInterface

import logging

_logger = logging.getLogger("Level 2")
class Level2(object):

    @classmethod
    def FromLevel1(cls,level1:Level1_Base,**kwargs):

        return cls(level1.bandwaves,level1.nbands,level1.sensor,
                   level1.proj,level1.proj_str, level1.affine,level1.ncols,level1.nrows,
                   level1.F0,level1.file_dir,level1.file_name,level1.image_name)

    def __init__(self,bandwaves,nbands,sensor,csr,crs_str,transform,width,height,F0,file_dir,file_name,image_name):
        _logger.info("Initializing Level-2...")
        self.bandwaves = bandwaves
        self.nbands = nbands
        self.sensor = sensor
        self.F0 = F0
        self.src_width =self.dst_width= width
        self.src_height =self.dst_height =  height

        ## the original coordinate system reference and transform
        self.src_csr = csr
        self.src_csr_str = crs_str
        self.src_transform = transform

        ## the modified coordinate system reference and transform after reprojection
        self.dst_csr = self.src_csr
        self.dst_csr_str = self.src_csr_str
        self.dst_transform = self.src_transform
        
        self.file_dir = file_dir
        self.file_name = file_name
        self.image_name = image_name

    def setCRS(self,proj,proj4str):
        self.dst_csr = proj
        self.dst_csr_str = proj4str

    def __reproject(self):
        pass

    def __resample(self):
        pass

    def setOutputer(self,output:OutputInterface):
        self.outputer = output

    def finish_update(self):
        self.outputer.close()

    def update_tile(self,tile:Tile,data:np.ndarray,iBandIndex):
        # print("update {},{}....".format(iBandIndex,tile))
        window,indexes, data= tile.to_window(),iBandIndex, data
        self.outputer.output(window,indexes, data)

