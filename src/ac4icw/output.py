import abc
import os
import rasterio
import numpy as np
from rasterio.windows import Window
import logging

_logger = logging.getLogger("Output")

class OutputInterface(metaclass=abc.ABCMeta):
    '''
    interface of writer
    '''

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'output') and
                callable(subclass.output) and
                hasattr(subclass, 'close') and
               callable(subclass.close)
                or
                NotImplemented)

    @abc.abstractmethod
    def output(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def close(self):
        pass


class RasterIOWriter(OutputInterface):

    @classmethod
    def FromLevel2(cls,level2,**kwargs):
        output_dir = level2.file_dir if 'DIR' not in kwargs else kwargs['DIR']
        _logger.info(kwargs)
        if output_dir is None:
            output_dir = level2.file_dir

        file_name = level2.image_name+'_l2' if 'NAME' not in kwargs else kwargs['NAME'].replace('$NAME$',level2.image_name)
        output_dir = os.path.join(output_dir, file_name)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        _logger.info("Output:{},{}".format(output_dir,file_name))
        writer =  cls(output_dir,file_name,
                   level2.dst_width,level2.dst_height,level2.dst_csr_str,level2.dst_transform,nbands=level2.nbands,bandwaves=level2.bandwaves)
        return writer

    def __init__(self,output_dir, file_name, width, height, crs_str, transform, **kwargs):
        _logger.info('Initializing image writer...')
        self.output_dir =  output_dir
        fpath = os.path.join(output_dir,file_name)
        nbands = kwargs['nbands'] if 'nbands' in kwargs else 1
        bandwaves = kwargs['bandwaves'] if 'bandwaves' in kwargs else None
        self.__dtype = kwargs['dtype'] if 'dtype' in kwargs else rasterio.uint16
        driver = 'GTiff'
        if driver.lower() == 'gtiff':
            fpath = fpath+'.tif'

        meta_dic = {'driver':driver,
                    'width':width,
                    'height':height,
                    'count':nbands,
                    'dtype':self.__dtype,
                    'crs':crs_str,
                    'transform':transform,
                    'compress':'lzw',
                    'tiled':True,
                    'blockxsize':512,
                    'blockysize':512,
                    'BIGTIFF':'YES'
        }

        self.__dst = rasterio.open(fpath, 'w', **meta_dic)
        self.__dst.nodata = 0
        self.__dst.update_tags(scale=1e-5,offset=0)
        self.__dst.update_tags(ns='software',software_name='ac4icw',version='0.1',author='Yanqun Pan,panyq213@163.com')
        if bandwaves is not None:
            self.__dst.update_tags(ns='bandwaves',wavelengths=','.join([str(round(w,2)) for w in bandwaves]),unit='nm')
        _logger.info("Level2 file:{}".format(fpath))

    def output(self, *args, **kwargs):
        # import matplotlib.pyplot as plt
        try:

            window, indexes, data = args[0], args[1], args[2]*1e5
            data[np.isnan(data)] = 0
            data[data<0] = 0
            data = data.astype(self.__dtype)
            self.__dst.write(data, window=window, indexes=indexes+1)

        except Exception as e:
            print(e)
            self.__dst.close()

    def close(self):
        if not self.__dst.closed:
            self.__dst.close()


class PlotShow(OutputInterface):

    def __init__(self):
        self.__dtype = rasterio.uint16

    def output(self, *args, **kwargs):
        import matplotlib.pyplot as plt
        window, indexes, data = args[0], args[1], args[2] * 1e5

        if window.row_off>0:
            plt.imshow(data.astype(self.__dtype))
            plt.title(str(window)+'_'+str(indexes))
            plt.show()

    def close(self):
        i= 1

