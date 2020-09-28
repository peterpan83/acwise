import time
import functools

dic_method = {
    'cal_solar_geo':'calculating solar zenith and azimuth',
    'cal_viewing_geo':'calculating veiwing zenith and azimuth',
    'L1_WISE.__init__':'Initializing LevelG wise image',
    'extract_f0':'Extracting extraterritaril irradiance for current sensor',
    'RayleighSixS.cal_reflectance':'Calculating rayleigh reflectance and transmittance',
    'GasSixS.cal_trans_up':'Calculating gas absorption transmittance',
    'StandardAtmCorrection.Run':'Standard atmospheric correction processing'
}

def processing_info(func,message=None):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # print(func.__qualname__)
        if message is not None:
            print('{0}{1}{0}'.format(''.join(['-'] * 15), message))
        elif func.__qualname__ in dic_method:
            print('{0}{1}{0}'.format(''.join(['-'] * 15), dic_method[func.__qualname__]))
        elif func.__name__ in dic_method:
            print('{0}{1}{0}'.format(''.join(['-'] * 15), dic_method[func.__name__]))
        else:
            print('{0}{1}{0}'.format(''.join(['-'] * 15), func.__qualname__))
        start_now = time.perf_counter()
        value = func(*args, **kwargs)
        print("It took {:.4f} seconds\n".format((time.perf_counter() - start_now)))
        return value
    return wrapper


class ProcessingInfo(object):
    def __init__(self, message=None):
        """
        If there are decorator arguments, the function
        to be decorated is not passed to the constructor!
        """
        self.message = message


    def __call__(self, func):
        """
        If there are decorator arguments, __call__() is only called
        once, as part of the decoration process! You can only give
        it a single argument, which is the function object.
        """
        def wrapped_f(*args,**kwargs):
            if self.message is not None:
                print('{0}{1}{0}'.format(''.join(['-'] * 15), self.message))
            elif func.__qualname__ in dic_method:
                print('{0}{1}{0}'.format(''.join(['-'] * 15), dic_method[func.__qualname__]))
            elif func.__name__ in dic_method:
                print('{0}{1}{0}'.format(''.join(['-'] * 15), dic_method[func.__name__]))
            else:
                print('{0}{1}{0}'.format(''.join(['-'] * 15), func.__qualname__))
            start_now = time.perf_counter()
            value = func(*args,**kwargs)
            print("It took {:.4f} seconds\n".format((time.perf_counter() - start_now)))
            return value
        return wrapped_f
