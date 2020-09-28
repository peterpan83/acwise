
class ExceptionEnviImage(Exception):
    '''
    base  exception for reading ENVI format imagery
    '''
    def __init__(self,message="Envi image error"):
        self.message = message
        super().__init__(self.message)


class ExceptionEnviDimensionNotMatch(ExceptionEnviImage):
    '''
    exception rasied for errors when nrows, ncols and nbands read in header do not match withe info in the binary data
    '''
    def __init__(self,header_dimension,data_dimension, message="ENVI dimension doesn't match "):
        self.message = message
        self.message ="{}: header {},data {}".format(self.message,str(header_dimension),str(data_dimension))
        super().__init__(self.message)

class ErrorFlightLine(Exception):
    def __init__(self,message="Initialize FlightLine from WISE files failed,L1A or L1A-GLU header is needed"):
        self.message = message
        super().__init__(self.message)

class ErrorL1WISE(Exception):
    def __init__(self,message='L1WISE initialize failed'):
        self.message = message
        super().__init__(self.message)
        



