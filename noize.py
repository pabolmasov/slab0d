import matplotlib
from matplotlib import rc
from numpy import *
from mslab import ifplot
if ifplot:
    from pylab import *
from scipy.integrate import *
from scipy.interpolate import interp1d
from scipy.signal import *

def flickgen(tmax, dt, nslope = 2, ifplot = False):
    '''
    produces a correlated time series of a given length and PDS slope
    '''
    nx = int(floor(tmax/dt))
    t = arange(nx)*dt
    fwhite = rand(nx)
    freqfilter = abs(fftfreq(nx))**(-nslope/2.) * exp(1j * rand(nx)*2.*pi) 
    freqfilter[isnan(freqfilter)] = 0.
    freqfilter[isinf(freqfilter)] = 0.
    f = ifft(fft(fwhite)*freqfilter)

    return t, real(f)

def brown(tmax, dt, tbreak):
    '''
    brownian noize with a break at tbreak (GM/c^3 units)
    '''
    nx = int(floor(tmax/dt))
    t = arange(nx)*dt
    fwhite = rand(nx)
    freqfilter = exp(1j * rand(nx)*2.*pi) / sqrt(abs(fftfreq(nx, d=dt)*tbreak)**(2)+1.)
    freqfilter[isnan(freqfilter)] = 0.
    freqfilter[isinf(freqfilter)] = 0.
    f = ifft(fft(fwhite)*freqfilter)
    return t, real(f)
   
