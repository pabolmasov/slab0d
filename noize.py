from mslab import ifplot 
if ifplot:
    import matplotlib
    from matplotlib import rc
import numpy.random as random
from numpy import *
from numpy.fft import fft, ifft, fftfreq
from numpy.random import rand, seed
# from numpy import random
from scipy.integrate import *
from scipy.interpolate import interp1d
from scipy.signal import *

'''
routines making noise: flickering or Brownian
'''

def flickgen(tmax, dt, nslope = 2, ifplot = False, rseed = None):
    '''
    produces a correlated time series of a given length and PDS slope
    '''
    if rseed is not None:
        seed(seed=rseed)
    nx = int(floor(tmax/dt))
    t = arange(nx)*dt
    fwhite = rand(nx)
    freqfilter = abs(fftfreq(nx))**(-nslope/2.) * exp(1j * rand(nx)*2.*pi) 
    freqfilter[isnan(freqfilter)] = 0.
    freqfilter[isinf(freqfilter)] = 0.
    f = ifft(fft(fwhite)*freqfilter)

    return t, real(f)

def brown(tmax, dt, tbreak, rseed = None):
    '''
    brownian noize with a break at tbreak (GM/c^3 units)
    '''
    if rseed is not None:
        seed(seed=rseed)
    nx = int(floor(tmax/dt))
    t = arange(nx)*dt
    fwhite = rand(nx)
    freqfilter = exp(1j * rand(nx)*2.*pi) / sqrt(abs(fftfreq(nx, d=dt)*tbreak)**(2)+1.)
    freqfilter[isnan(freqfilter)] = 0.
    freqfilter[isinf(freqfilter)] = 0.
    f = ifft(fft(fwhite)*freqfilter)
    return t, real(f)
   
def randomsin(mdot, sinefreq, samp, rseed = None):
    '''
    '''
    if rseed is not None:
        seed(seed=rseed)
    sinedphi = rand()
    s = lambda x: mdot * (1. + samp * sin(sinefreq * x + sinedphi))
    return s
