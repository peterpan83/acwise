#!/usr/bin/env python
# encoding: utf-8
from __future__ import print_function, division, absolute_import

import os
import numpy as np
import pendulum
from pandas import Series
import spectral.io.envi as envi
from spectral.io.envi import FileNotAnEnviHeader

from ..exception_ac import ErrorFlightLine
from ..helper import rm_illegal_sym_envi_hdr
import logging

_logger = logging.getLogger("root")
class FlightLine():

    @classmethod
    def FromWISEFile(cls,nav_sum_log,L1A_Header):
        '''
        initialize WISE FlightLing based on navigation log file, L1A header or L1A-GLU header
        :param nav_sum_log:  subfixed with '-Navcor_sum.log'
        :L1A_Header: l1a header subfixed with '-L1A.pix.hdr, -L1A.glu.hdr'
        '''

        _logger.info("Initializing Flight Line from WISE navigation sum log file and L1A header:")
        _logger.info("{},{}".format(nav_sum_log,L1A_Header))
        if not os.path.exists(L1A_Header):
            _logger.error("{} doesn't exist,initilation of flight line failed".format(L1A_Header))
            raise ErrorFlightLine()

        def read():
            try:
                header = envi.read_envi_header(L1A_Header)
            except FileNotAnEnviHeader as e:
                _logger.warning(e)
                new_lines = rm_illegal_sym_envi_hdr(L1A_Header)
                new_hdr = L1A_Header.replace('.hdr','.copt.hdr')
                with open(new_hdr, 'w') as f:
                    f.writelines(new_lines)
                header = envi.read_envi_header(new_hdr)

            ncols = int(header['samples'])
            nrows = int(header['lines'])
            resolution_x,resolution_y = float(header['pixel size'][0]),float(header['pixel size'][1])

            with open(nav_sum_log, 'r') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if line.startswith('Average Value'):
                    line_strip = line.replace('Average Value', '').replace('+ ground pixel size', '').replace('+',
                                                                                                          '').strip()
                    _logger.debug(line_strip.split(' '))
                    values = [float(item) for item in line_strip.split(' ') if item.strip() != '']+[ncols,nrows,resolution_x,resolution_y]
                    return Series(data=values,
                                 index=['Roll', 'Pitch', 'Heading', 'Distance', 'Height', 'Easting', 'Norhting','Samples','Lines','ResX','ResY'])

        s = read()
        return cls(s['Height'],s['Heading'],int(s['Samples']),int(s['Lines']),s['ResX'])


    def __init__(self,height,heading,samples,lines,resolution_x,**kwargs):
        '''
        :param height:  fly height (m)
        :param heading: attitude of flight refering to NORTH (degree)
        :param samples: number of pixels in each scanning line (int)
        :param lines:  number of scanning lines
        :param resolution_x:  resolution of the x direction (m)
        :param kwargs:   unit of distance is meter, unit of angle is degree
        '''
        self.height = height
        self.heading = heading
        self.samples = samples
        self.lines = lines
        self.resolution_x = resolution_x

        self.roll = 0.0 if 'roll' not in kwargs else kwargs['roll']
        self.pitch = 0.0 if 'pitch' not in kwargs else kwargs['pitch']
        self.center_x = self.samples/2 if 'center_x' not in kwargs else kwargs['center_x']
        self.surface_altitude = 0.0 if 'surface_altitude' not in kwargs else kwargs['surface_altitude']

        self.sample_center = [self.resolution_x*s +self.resolution_x/2 for s in range(samples)]


    def __calNadirX(self):
        '''
        calculate the Nadir position in each scanning line
        :return:  nadir point (pixel), nadir point (meter)
        '''
        distance_nadir2center = self.height*np.tan(np.deg2rad(self.roll))
        nadir = self.center_x * self.resolution_x - distance_nadir2center
        nadir_x = int(nadir/self.resolution_x)
        return nadir_x, nadir


    def calSZA_AZA(self):
        '''
        calculate sensor zenith angle and azimuth angle
        :return: (SZA,AZA)
        '''
        # print('calculating veiwing zenith and azimuth angle of flight line.....')
        # start_time = pendulum.now()

        nadir_x, nadir = self.__calNadirX()
        sza = np.rad2deg(np.arctan(np.abs(nadir - np.asarray(self.sample_center))/self.height))

        # aza_ = 90+self.heading if  (90+self.heading)<360 else self.heading-270
        # aza = np.full_like(sza,aza_)
        # aza[nadir_x:] = 180+90+self.heading if (180+90+self.heading)<360 else self.heading-90

        aza_ = 180+90+self.heading if (180+90+self.heading)<360 else self.heading-90
        aza = np.full_like(sza,aza_)
        aza[nadir_x:] = 90+self.heading if  (90+self.heading)<360 else self.heading-270


        # print("took {} seconds".format((pendulum.now()-start_time).seconds))
        return np.tile(sza,(self.lines,1)), np.tile(aza,(self.lines,1))

