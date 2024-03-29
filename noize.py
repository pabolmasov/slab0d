from mslab import ifplot 
if ifplot:
    import plots
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
    freq = fftfreq(nx)
    df = (1.-freq.min()/freq.max())/double(size(freq))
    freqfilter = exp(1j * rand(nx)*2.*pi - (nslope/4.) * (log(freq.real**2+freq.imag**2+0.01*df**2)- 2.*log(freq.max())))
    # abs(freq/freq.max() + df * 0.01j)**(-nslope/2.) * exp(1j * rand(nx)*2.*pi) 
    #    freqfilter[isnan(freqfilter)] = 0.
    #    freqfilter[isinf(freqfilter)] = 0.
    # 
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
    s = lambda x: mdot * (1. + samp * sin(sinefreq * double(x) + sinedphi))
    return s

def noizetest():

    nflick =2.
    # t, x = flickgen(1., 1.e-5, nslope = nflick)

    t=arange(10000)*0.01
    xfun  = randomsin(1., 2.*pi*1., 0.05)
    x=xfun(t)
    
    xf = fft(x)
    freq = fftfreq(size(x))

    pds = abs(xf)**2

    w= (freq >0.)
    
    plots.xydy(freq[w], xf[w], xf[w]*0., outfile = 'noizetest', ylog = True, xlog = True)
