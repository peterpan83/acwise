import numpy as np
import pycrs
import pyproj
import pendulum
from abc import abstractmethod

from .helper import transform_xy
from .helper import calSolarHourAngel
from .helper import calDeclination
from .helper import calSolarZenithAzimuth
from .helper import findLocalTimeZone
from .tile import Tile

from .decorators import processing_info
from .decorators import ProcessingInfo
import logging.config

_logger = logging.getLogger("Level 1")

# logging.basicConfig(level=logging.INFO)

class Level1_Base():

    def __init__(self,acq_time,nrows,ncols,nbands,affine,proj,proj_str,file_dir,file_name,image_name,
                 sensor,data_dir,timezone_local_name=None,bandwaves=None):
        '''
        base classs of level 1 image that is georeferenced.
        :param acq_time:  acquired time , instance of datetime
        :param nrows:  number of rows
        :param ncols:  number of clolumns
        :param nbands:  number of bands
        :param affine:  affine ,instance of rasterio.transform.Affine
        :param proj:  projection, instance of  pyproj.Proj
        :param timezone_name: name of the local timezone where the image located
        :param data: GDAL Data
        '''
        self.sensor = sensor
        self.file_name,self.file_dir,self.image_name = file_name,file_dir,image_name
        self.data_dir= data_dir
        self.F0 = {'values':None,'unit':None}

        self.acq_time = acq_time #the time when the image was acquired
        acq_time_zone = self.acq_time.timezone_name
        self.acq_time_zone = acq_time_zone

        self.acq_time_local = None

        # time_zone_offset
        self.nrows = nrows
        self.ncols = ncols
        self.nbands = nbands
        self.affine = affine
        self.x_resolution,self.y_resolution = abs(self.affine.a),abs(self.affine.e)
        self.proj = proj
        self.proj_str =   proj_str

        self.longitude, self.latitude = None,None
        ## longitude and latitude of the center pixel
        self.center_longitude, self.center_latitude = None,None

        self.solar_zenith = None
        self.solar_azimuth = None
        self.viewing_zenith = None
        self.viewing_azimuth = None
        self.phi = None

        self.sensor_altitude = None
        self.ground_altitude = 0

        ## original corrinates obtained by affine transformation, 1-D ndarray
        self.X, self.Y = None,None
        self.timezone_local = None
        if timezone_local_name is not None:
            self.timezone_local =pendulum.timezone(timezone_local_name)

        self.central_lon_localimezone = None

        self.__valid_mask = None
        self.__water_mask = None
        
        self.bandwaves = bandwaves
        _logger.info("Number of bands:{}".format(self.nbands))
        _logger.info("Central wavelengths are:{}".format(','.join([str(w) for w in self.bandwaves])))
        self._acq_localtime()

    def setSensorAltitude(self,value):
        self.sensor_altitude =  value

    def setGroundAltitude(self,value):
        self.ground_altitude = value

    # def _getF0Ext(self):
    #     '''
    #     :param F0: extraterritrial irradiance at level1 bandwaves
    #     :return:  F0 for the specific day
    #     '''
    #     # return  calF0Ext(F0,self.acq_time_local)
    #     _,_,F0_,unit = getF0(self.bandwaves)
    #     self.F0['values']=calF0Ext(F0_,self.acq_time_local)
    #     self.F0['unit'] = unit
    def setRGBBands(self,rgb:tuple):
        '''
        set RGB band index for visualization
        :param rgb: RGB band index, tuple
        :return: None
        '''
        self.__red_band_index,self.__green_band_index,self.__blue_band_index = rgb
    def getRGBBands(self):
        return (self.__red_band_index,self.__green_band_index,self.__blue_band_index)

    def cal_valid_mask(self):
        if self.__valid_mask is None:
            iband = 0 if self.__red_band_index is None else self.__red_band_index
            Lt = self.read_band(iband)
            self.__valid_mask = (Lt > 0)

    def set_valid_mask(self,mask):
        self.__valid_mask = mask


    def get_valid_mask(self,tile:Tile=None):
        if self.__valid_mask is None:
            self.cal_valid_mask()
        if tile is None:
            return self.__valid_mask
        else:
            return self.__valid_mask[tile.sline:tile.eline,tile.spixl:tile.epixl]

    @ProcessingInfo(message="calculating water mask")
    def cal_water_mask(self):
        if self.__water_mask is not None:
            return
        wavelength = self.bandwaves[self.bandwaves > 800.0][0]
        iband = int(np.where(self.bandwaves == wavelength)[0][0])
        _logger.info("Water mask band:{},wavelength:{}".format(iband, wavelength))

        Lt = self.read_band(iband)
        if self.__valid_mask is None: self.__valid_mask = (Lt > 0)
        sz = self.solar_zenith
        rhot = np.pi * Lt / (self.F0['values'][iband] * np.cos(np.deg2rad(sz)))
        self.__water_mask = (rhot < 0.01)&self.__valid_mask

    def set_water_mask(self,mask):
        self.__water_mask = mask

    def get_water_mask(self,tile:Tile=None):
        if self.__water_mask is None:
            self.cal_water_mask()
        if tile is None:
            return self.__water_mask
        else:
            return self.__water_mask[tile.sline:tile.eline,tile.spixl:tile.epixl]

    def show_rgb(self,tile:Tile=None):
        '''
        show rgb image
        :param tile:
        :return:
        '''
        import matplotlib.pyplot as plt
        from skimage import exposure
        def adjust_gamma(img):
            corrected = exposure.adjust_gamma(img, 0.3)
            return corrected
        Rdata_a = self.getRhotBand(bandindex=self.__red_band_index,tile=tile)
        Gdata_a = self.getRhotBand(bandindex=self.__green_band_index,tile=tile)
        Bdata_a = self.getRhotBand(bandindex=self.__blue_band_index,tile=tile)

        y_ticks,x_ticks = list(range(0,self.nrows,1000)),list(range(0,self.ncols,1000))
        cor_x, cor_y = transform_xy(self.affine, rows=y_ticks, cols=x_ticks)
        
        RGBdata = np.zeros((Rdata_a.shape[0],Rdata_a.shape[1],3),dtype=np.float)
        RGBdata[:,:,0] = Rdata_a
        RGBdata[:, :, 1] = Gdata_a
        RGBdata[:, :, 2] = Bdata_a
        dst = adjust_gamma(RGBdata)
        dst[~self.__valid_mask] = 1.0
        # dst = RGBdata
        plt.imshow(dst)
        plt.xticks(ticks=x_ticks,labels=cor_x)
        plt.yticks(ticks=y_ticks, labels=cor_y,rotation=90)
        plt.show()


    def _acq_localtime(self):
        '''
        calculate local time of the acquired time
        and central longitude of the local time zone
        :return:
        '''
        tz = self.__localtimezone()
        # print()
        _logger.info("local time zone; {}".format(tz))
        self.acq_time_local = tz.convert(self.acq_time)
        # print("acquired UTC time:{}, and local time：{}".format(self.acq_time,self.acq_time_local))
        _logger.info("acquired UTC time:{}, and local time：{}".format(self.acq_time,self.acq_time_local))
        offset_hours = self.acq_time_local.offset_hours
        self.central_lon_localimezone = offset_hours*15
        # print("central longitude of {}:{}".format(tz,self.central_lon_localimezone))
        _logger.info("central longitude of {}:{}".format(tz,self.central_lon_localimezone))


    def __localtimezone(self):
        '''
        find the local timezone from the central longitude and latitude
        :return: timezone, instance of pendulum.timezone
        '''
        if self.timezone_local:
            return self.timezone_local
        if self.center_longitude is None or self.center_latitude is None:
            self._lonlat()

        return findLocalTimeZone(self.center_longitude,self.center_latitude)


    def _lonlat(self):
        '''
        transform the corrdinates to WGS84, EPSG:4326
        extract longitude and latitude
        :return: longitude (ndarry), latitude (ndarry)
        '''
        if self.longitude is not None and self.latitude is not None:
            return
        if self.X is None or self.Y is None:
            self._XY()

        _logger.info('calculating longitude and latitude each pixel')
        from_proj, to_proj = self.proj, pyproj.Proj(pycrs.parse.from_epsg_code(4326).to_proj4())

        xv, yv = np.meshgrid(self.X, self.Y)
        lon_v, lat_v = pyproj.transform(self.proj, to_proj, xv, yv)
        self.longitude, self.latitude =   lon_v, lat_v
        self.center_longitude,self.center_latitude =  lon_v[self.nrows//2,self.ncols//2],lat_v[self.nrows//2,self.ncols//2]

    def _XY(self):
        '''
        obtain the original corrdinates
        :return: X, Y
        '''
        if self.X is not None or self.Y is not None:
            return
        cor_x, cor_y = transform_xy(self.affine, rows=list(range(self.nrows)), cols=list(range(self.ncols)))
        self.X, self.Y = cor_x, cor_y
        # return xv, yv

    def cal_phi(self):
        '''
        calculate relative azimuth
        :return:
        '''
        self.phi = np.abs(self.viewing_azimuth-self.solar_azimuth)


    def gen_tiles(self,size=1.0):
        '''
        generate tiles
        :param size: the side length of a square tile (unit=km), the default is 1 km
        :return:
        '''
        size = size*1000 # transform km to m
        x_pixels,y_pixels = \
            int(round(size/self.x_resolution)) if int(round(size/self.x_resolution))>1 else 1,\
            int(round(size/self.y_resolution))  if int(round(size/self.y_resolution))>1 else 1

        y_t,x_t = list(range(0,self.nrows,y_pixels)),list(range(0,self.ncols,x_pixels))
        if self.nrows not in y_t:
            y_t.append(self.nrows)
        if self.ncols not in x_t:
            x_t.append(self.ncols)
        _logger.info('Tile size:{} m,Tiles:{}'.format(size,len(y_t[:-1])*len(x_t[:-1])))
        for sline,eline in zip(y_t[:-1],y_t[1:]):
            for spixl,epixl in zip(x_t[:-1],x_t[1:]):
                yield Tile(sline,eline,spixl,epixl)


    def cal_solar_geo(self):
        '''
        calculate solar zenith angle and azimuth angle
        :return:
        '''
        self._lonlat()
        # self._acq_localtime()
        hourAngle = calSolarHourAngel(self.longitude,self.central_lon_localimezone,self.acq_time_local)


        ## year, month and day
        year, month, day = self.acq_time.year, self.acq_time.month, self.acq_time.day
        
        declination = calDeclination(year,month,day)
        self.solar_zenith, self.solar_azimuth = calSolarZenithAzimuth(self.latitude,hourAngle,declination)




    @abstractmethod
    def cal_viewing_geo(self):
        pass

    @abstractmethod
    def extract_f0(self,specific_file=None):
        pass

    @abstractmethod
    def read_band(self, bandindex, tile:Tile=None):
        pass
    @abstractmethod
    def read_spectrum(self, iline, isample):
        pass

    @abstractmethod
    def read_tile(self,tile:Tile):
        pass

    def getObsGeo(self,tile:Tile=None):
        solz = self.solar_zenith.copy()
        invalid_mask = ~self.__valid_mask
        solz[invalid_mask] = np.nan
        if not tile:
            return solz,self.viewing_zenith,self.phi

        return solz[tile.sline:tile.eline, tile.spixl:tile.epixl], self.viewing_zenith[tile.sline:tile.eline,tile.spixl:tile.epixl],self.phi[tile.sline:tile.eline,tile.spixl:tile.epixl]

    def getRhotBand(self, bandindex, tile:Tile=None):
        '''
        TOA reflectance for a given band
        :param bandindex:
        :param extent: if geo=False: (sline,eline,spixl,epixl) else (north,south,west,east)
        :param geo:
        :return: TOA reflectance for specific bands
        '''
        # unit: uW/cm-2 sr-1 / nm * 1000
        Lt = self.read_band(bandindex,tile)
        # unit (f0): mW/cm^2/um
        # assert Lt.shape == (tile.ysize,tile.xsize)
        if tile:
            sz = self.solar_zenith[tile[0]:tile[1], tile[2]:tile[3]]
        else:
            sz = self.solar_zenith
        # assert sz.shape==(tile.ysize,tile.xsize)
        return np.pi * Lt / (self.F0['values'][bandindex] * np.cos(np.deg2rad(sz)))

    def getRhotSpectrum(self, iline, isample):
        '''
        TOA reflectance spectrun for a given pixel
        :param iline:
        :param isample:
        :return:
        '''
        Lt_spectrum = self.read_spectrum(iline,isample)
        sz = self.solar_zenith[iline,isample]
        return np.pi* Lt_spectrum/(self.F0['values']*np.cos(np.deg2rad(sz)))

    def getRhotTile(self,tile:Tile)->np.ndarray:
        Lt = self.read_tile(tile)
        assert Lt.shape == (self.nbands, tile.ysize, tile.xsize)
        sz = self.solar_zenith[tile[0]:tile[1],tile[2]:tile[3]]
        cosz = np.cos(np.deg2rad(sz))
        return np.pi * Lt / (self.F0['values'][:,np.newaxis,np.newaxis] * cosz[np.newaxis,:,:])



if __name__ == '__main__':
    tile = Tile(10,50,2,20)
    print(tile,tile[2])

