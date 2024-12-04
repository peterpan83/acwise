import os
import numpy as np
try:
    import gdal
except:
    from osgeo import gdal

import spectral.io.envi as envi
from spectral.io.envi import FileNotAnEnviHeader
import pendulum
from tqdm import tqdm
import scipy.interpolate as intp

from ..level1 import Level1_Base, Tile
from ..aerial.flightline import FlightLine
from ..exception_ac import ExceptionEnviDimensionNotMatch
from ..exception_ac import ErrorL1WISE
from ..helper import getAffineAndProjFromMapInfo
# from ac4icw.helper import transform_xy
from ..helper import transform_rowcol
from ..helper import FillNA2D
from ..helper import FWMH2RSR
from ..helper import rm_illegal_sym_envi_hdr

from ..decorators import processing_info
from ..F0 import loadF0,calF0Ext

import logging

_logger = logging.getLogger("Level1G WISE")
class L1_WISE(Level1_Base):
    '''
    Level1 georeferenced image,
    this size (samples,lines), resolution may be not as same as the Level 1 image
    '''
    @processing_info
    def __init__(self, *args,**kwargs):
        '''
        the format of L1G image is ENVI format, the file type is PCI with subfix of '.pix', which can not be read corredtly by spectral.
         so, the header and data file are read seperately by different library, rasterio and gdal
        :param hdrf:  header file
        :param dataf: binary data file,
        '''
        # print('{0}Initializing LevelG wise image{0}'.format(''.join(['-']*15)))

        # IMAGE_NAME =
        # L1G_DIR =
        # L1A_GLU_DIR =
        # L1A_DIR =
        # NAV_LOG_DIR =

        if 'IMAGE_NAME' not in kwargs or 'L1G_DIR' not in kwargs:
            raise ErrorL1WISE("WISE L1G_DIR and IMAGE_NAME are required to initialize Level1 image!")
        l1g_dir,image_name = kwargs['L1G_DIR'],kwargs['IMAGE_NAME']

        l1g_hdrf, l1g_dataf = os.path.join(l1g_dir, image_name + '-L1G.pix.hdr'),os.path.join(l1g_dir, image_name + '-L1G.pix')
        if not (os.path.exists(l1g_hdrf) and os.path.exists(l1g_dataf)):
            raise ErrorL1WISE("WISE L1G Header:{} or {} doesn't exist! ".format(l1g_hdrf,l1g_dataf))

        if 'L1A_GLU_DIR' not in kwargs :
            raise ErrorL1WISE('L1A GLU DIR is missing!')
        l1a_glu_dir = kwargs['L1A_GLU_DIR']
        # l1a_glu_hdrf,l1a_glu_dataf =kwargs['L1A_GLU_HDR'],kwargs['L1A_GLU_DATA']
        l1a_glu_hdrf,l1a_glu_dataf = os.path.join(l1a_glu_dir, image_name + '-L1A.glu.hdr'),os.path.join(l1a_glu_dir,image_name+'-L1A.glu')
        if not (os.path.exists(l1a_glu_hdrf) and os.path.exists(l1a_glu_dataf)):
            raise ErrorL1WISE("WISE L1A_GLU Header:{} or {} doesn't exist! ".format(l1a_glu_hdrf,l1a_glu_dataf))



        if 'NAV_LOG_DIR' not in kwargs:
            raise ErrorL1WISE("Navigation sum log file is missing!")
        nav_log_dir = kwargs['NAV_LOG_DIR']
        nav_logf = os.path.join(nav_log_dir, image_name + '-Navcor_sum.log')
        if not os.path.exists(nav_logf):
            raise ErrorL1WISE("WISE NAV_LOG_LOG {}  doesn't exist! ".format(nav_logf))


        if 'DATA_DIR' not in kwargs:
            raise ErrorL1WISE("DATA_DIR is missing!")
        data_dir = kwargs['DATA_DIR']
        if not (os.path.exists(data_dir) and os.path.isdir(data_dir)):
            raise ErrorL1WISE("DATA_DIR:{} is not a directory or does not exist!".format(data_dir))
        # self.data_dir = kwargs['DATA_DIR']

        fl = FlightLine.FromWISEFile(nav_sum_log=nav_logf, L1A_Header=l1a_glu_hdrf)

        self.flightline = fl
        self.glu_file = l1a_glu_dataf

        self.header_f, self.data_f = l1g_hdrf, l1g_dataf
        try:
            self.header = envi.open(l1g_hdrf,image=l1g_dataf)
        except FileNotAnEnviHeader as e:
            _logger.warning(e)
            new_lines = rm_illegal_sym_envi_hdr(l1g_hdrf)
            new_l1g_hdrf = l1g_hdrf.replace('.hdr','.copy.hdr')
            with open(new_l1g_hdrf,'w') as f:
                f.writelines(new_lines)
            self.header =  envi.open(new_l1g_hdrf,image=l1g_dataf)
            self.header_f = new_l1g_hdrf

        file_dir, file_name = os.path.split(self.header_f)[0], \
                                        os.path.splitext(os.path.split(self.header_f)[1])[0]

        mapinfo = self.header.metadata['map info']
        nrows,ncols,nbands = self.header.nrows,self.header.ncols,self.header.nbands
        waves = np.array([float(w) for w in self.header.metadata['wavelength']])

        d = gdal.Open(self.data_f)
        self.data = d
        if d.RasterCount !=nbands or d.RasterXSize != ncols or d.RasterYSize!=nrows:
            raise ExceptionEnviDimensionNotMatch(header_dimension=( self.ncols,self.nrows,self.nbands),data_dimension=(d.RasterXSize,d.RasterYSize,d.RasterCount))

        # self.data = d

        ## obtaine the affine transform, projection and proj4string. +proj=utm +zone=19 +ellps=WGS84 +datum=WGS84 +units=m +no_defs
        affine,proj,proj4string = getAffineAndProjFromMapInfo(mapinfo)

        # acq_time,nrows,ncols,affine,proj):
        acq_time = pendulum.parse(self.header.metadata['acquisition time'],)

        super().__init__(acq_time,nrows,ncols,nbands,affine,proj,proj4string,file_dir,file_name,image_name,'WISE',data_dir,bandwaves=waves)
        self.FWHM = np.full(nbands,5.05)

        rgb_bands = tuple([int(item)-1 for item in self.header.metadata['default bands']])
        super().setRGBBands(rgb=rgb_bands)

        # super().setValidMask(self.read_band(bandindex=rgb_bands[0]) > 0)

        super().setSensorAltitude(int(round(fl.height / 1000.0)))
        super().setGroundAltitude(0)

        f0_file = kwargs['F0'] if 'F0' in kwargs else None
        self.extract_f0(f0_file)
        _logger.info("Calculating solar zenith...")
        super().cal_solar_geo()
        # _logger.info('Setting water mask...')
        # super().setWaterMask()

    def extract_f0(self,specific_file=None):
        '''
        implement the abstract method of getF0
        :return:
        '''
        if specific_file is None:
            _logger.info("Calculating {} equvilent F0...".format(self.sensor))
            _waves,_f0,unit = loadF0(data_dir=self.data_dir)
            _f0_c = calF0Ext(_f0,self.acq_time_local)
            resolution, wrange = 0.01, (_waves[0],_waves[-1]+0.0001)

            _f0_c_fine = intp.interp1d(_waves,_f0_c)(np.arange(wrange[0],wrange[1],resolution))

            rsr, x = FWMH2RSR(self.FWHM, self.bandwaves, wrange=wrange, resolution=resolution)
            f0_wise = np.zeros_like(self.bandwaves,dtype=np.float)
            for i,item in enumerate(rsr):
                f0_wise[i] =np.sum(_f0_c_fine*item)/np.sum(item)
            self.F0['values'] = f0_wise
            self.F0['unit'] = unit
        else:
            _logger.info("Loading F0 of {} from {}".format(self.sensor,specific_file))
            _waves, _f0, unit = loadF0(data_dir=self.data_dir,sensor=str.upper(self.sensor),name='F0_6SV')
            self.F0['values'] = _f0
            self.F0['unit'] = unit
            _logger.debug("loading F0 finished....")


    # def calSolarGeometry(self):
    #     '''
    #     calculate solar zenith and azimuth
    #     :return:  self.solar_zenith, self.solar_azimuth
    #     '''
    #     self.__cal_solar_geometry()

    @processing_info
    def cal_viewing_geo(self):
        '''
        extract viewing zenith angle
        the orginal data from the flight line is not georeferenced, and the nrows and ncols are not the same as the georeferenced ones
        so, we need to transfer the original viewing geometry to the georefernce grid using the georeference LUT
        :return:
        '''
        if self.flightline is None:
            raise Exception(message='no flight line found')

        glu_data = None
        if self.glu_file is not None:
            glu_data = gdal.Open(self.glu_file)
            nchannels,nsamples_glu,nlines_glu = glu_data.RasterCount, glu_data.RasterXSize, glu_data.RasterYSize
            if nchannels !=3:
                raise Exception(message='the glu file does not have three channels')
            if nsamples_glu!=self.flightline.samples or nlines_glu!=self.flightline.lines:
                raise Exception(message='samples or lines of flightline and glu do not match')


        # data_glu =  glu_data.ReadAsArray()
        _logger.info('reading georeference look up tables')
        band_x, band_y = glu_data.GetRasterBand(1), glu_data.GetRasterBand(2)
        x_glu, y_glu =  band_x.ReadAsArray(), band_y.ReadAsArray()
            
        v_zenith_fl, v_azimuth_fl = self.flightline.calSZA_AZA()
        # v_zenith_fl, v_azimuth_fl = v_zenith_fl.flatten(), v_azimuth_fl.flatten()

        ## initialize viewing zenith and azimuth with default values
        v_zenith_level1, v_azimuth_level1 = np.full((self.nrows,self.ncols),np.nan), np.full((self.nrows,self.ncols),np.nan)

        # self._XY()
        # print('looking for correlated corrdinates')
        # for row in tqdm(range(self.nrows),desc='Processing GLU'):
        #     y =  self.Y[row]
        #     temp_y =  np.abs(y_glu-y)
        #
        #     for col in tqdm(range(self.ncols),desc='Line {}'.format(row)):
        #         x = self.X[col]
        #         min_index = np.argmin(np.abs(x_glu-x)+temp_y)
        #         v_zenith_level1[row,col] = v_zenith_fl[min_index]
        #         v_azimuth_level1[row,col] =  v_azimuth_fl[min_index]
        #
        # del v_zenith_fl, v_azimuth_fl,x_glu,y_glu
        # self.viewing_zenith, self.viewing_azimuth =  v_zenith_level1, v_azimuth_level1
        for row in tqdm(range(self.flightline.lines),desc='Processing GLU'):
            xs, ys = x_glu[row],y_glu[row]
            # print(xs,ys)
            rows_c,cols_c = transform_rowcol(self.affine,xs=xs,ys=ys,precision=5)
            mask = (rows_c<self.nrows) & (cols_c<self.ncols)
            # print(np.max(rows_c),np.max(cols_c))
            rows_c = rows_c[mask]
            cols_c = cols_c[mask]

            v_zenith_level1[rows_c,cols_c] = v_zenith_fl[row][mask]
            v_azimuth_level1[rows_c,cols_c] = v_azimuth_fl[row][mask]

        _logger.info('Filling NA')
        self.viewing_zenith =  FillNA2D(v_zenith_level1)
        self.viewing_azimuth = FillNA2D(v_azimuth_level1)
        
        self.viewing_zenith[~self.get_valid_mask()] = np.nan
        self.viewing_azimuth[~self.get_valid_mask()] = np.nan



    def read_band(self,bandindex,tile:Tile=None):
        '''
        read DN for a given band
        :param bandindex: bandindex starts with 0
        :param extent: if geo=False: (sline,eline,spixl,epixl) else (north,south,west,east)
        :param geo: if the extent is longitude,latitude or line,pixl
        :return: re
        '''
        band_temp = self.data.GetRasterBand(bandindex+1)
        if tile:
            Lt = band_temp.ReadAsArray(xoff=tile[2],yoff=tile[0],
                                           win_xsize=tile.xsize,win_ysize=tile.ysize)
        else:
            Lt = band_temp.ReadAsArray()
        return Lt*1e-6

    def read_spectrum(self,iline,isample):
        '''
        read DN spectrum for a given pixel
        :param iline:  iline th line, y
        :param isample: ismaple th sample, x
        :return: 1D DN spectrum
        '''
        Lt_spectrum = self.data.ReadAsArray(xoff=isample,yoff=iline,xsize=1,ysize=1)*1e-6
        return Lt_spectrum[:,0,0]

    def read_tile(self,tile:Tile):
        # convert to  mW/sr^1/cm^2/nm
        Lt =self.data.ReadAsArray(xoff=tile.spixl,yoff=tile.sline,xsize=tile.xsize,ysize=tile.ysize)*1e-6
        return Lt


    




