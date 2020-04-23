from numpy import *
from scipy.integrate import *
from scipy.interpolate import interp1d
from scipy.signal import *

# for normal distribution fit:
# import matplotlib.mlab as mlab
# from scipy.stats import norm
# for curve fitting:
from scipy.optimize import curve_fit
import os
import multiprocessing
from multiprocessing import Pool
# from mpi4py import MPI

# GM = c = kappa = 1

# dl/dt = mdot * j - alpha * geff * M * R * (omega > omegaNS)
# dm/dt = mdot - alpha * geff * M^2/L * R * (omega > omegaNS)

mNS = 1.5 # NS mass in Solar units
r = 6./(mNS/1.5) # NS radius in GM/c**2 units
alpha = 1e-4
tdepl = 1e4 # depletion time in GM/c^3 units
j = 0.9*sqrt(r)
pspin = 0.003 # spin period, s
tscale = 4.92594e-06 * mNS # time scale, s
mscale = 6.41417e10 * mNS # mass scale, g
omegaNS = 2.*pi/pspin *tscale

print("q = "+str(r**2/alpha/tdepl/j))

# noise parameters:
regimes = ['const', 'sine', 'flick', 'brown']
regime = 'flick'

nflick = 2.
tbreak = 1.
# accretion rate and amplitude:
mdot = 1. * 4.*pi # mean mass accretion rate, GM/kappa c units
dmdot =  0.5 # relative variation dispersion

sinefreq = 2.*pi * 10. * tscale # frequency of the sinusoudal variation
samp = 0.5 # amplitude of the sine

# time grid
maxtimescale = (tdepl+1./alpha)
mintimescale = 1./(1./tdepl+alpha)
dtdyn = r**1.5
dtout = minimum(0.3*dtdyn,3e-2*mintimescale) # this gives 10^5 data points
tmax = 30.*maxtimescale
nt = int(ceil(tmax/dtout))
print(str(nt)+" points in time")
tar = dtout * arange(nt)

# outputs:
ifplot = True # if we are plotting against the computer (disabled for now)
ifasc = True # if we are writing ASCII output
ifzarr = True
hname = 'slabout' # output HDF5 file name
if ifplot:
    import plots

import noize
import hdfoutput as hdf

def onestep(m, l, mdot):
    '''
    single step of boundary layer evolution
    input: mass, luminosity, mass accr. rate
    output: dm/dt, dl/dt, luminosity release, cos(theta) for the flow
    '''
    omega = l /m /r**2
    geff = 1./r**2 - omega**2*r
    dl = mdot * j - l/tdepl - alpha * geff * m * r * sign(omega - omegaNS)
    dm = mdot - m/tdepl
    # alpha * geff * m**2/l * (omega > omegaNS)
    #    a = alpha * m * r / omega * (omega-omegaNS)**2
    lout = alpha * r * geff * (omega - omegaNS) *m # + 1./tdepl * (omega**2-omegaNS**2) * r**2 /2.) * m 
    a = lout / geff
    cth = a/(4.*pi*r**2)
    
    # (geff*a+l/tdepl/2.)*(omega-omegaNS)
    if geff < 0:
        print("geff = "+str(geff))
        print("l="+str(l))
        print("m="+str(m))
        ii=input('m')
    #    print(omegaNS)
    return dm, dl, lout, cth

def singlerun(krepeat):
    '''
    evolution of a layer spun up by a variable mass accretion rate. 
    If nflick is set, it is the power-law index of the noize spectrum
    if tbreak is set, it is the characteristic time scale for the spectral break
    if neither is set, mass accretion rate is assumed constant
    runs an iteration with the number krepeat
    '''
    print("simulation No"+str(krepeat))
    # initial conditions:
    tstore = 0.;
    m =  0.1*mdot*tdepl # starting mass, 10% from equilibrium
    l = omegaNS*r**2*m   # starting angular momentum
    t = 0. ; dt = 0.1 ; ctr = 0
    # setting the input mdot variability spectrum:
    if regime == 'flick':
        tint, mint = noize.flickgen(tmax, dtdyn, nslope = nflick, rseed = krepeat)
        mmean = mint.mean() ; mrms = mint.std()
    if regime == 'brown':
        tint, mint = noize.brown(tmax, dtdyn, tbreak, rseed = krepeat)  # nslope = nflick)
        mmean = mint.mean() ; mrms = mint.std()
    if regime == 'const':
        mconst = True
        meq = mdot * tdepl
        q = r**2/alpha/tdepl/j
        oeq = j/r**2 * (q - sqrt(q**2-4.*q+4.*r/j**2))/2.
    if (regime == 'flick') | (regime == 'brown'):
        mint = mdot * exp( (mint-mmean)/mrms * dmdot)
        mconst = False
        mfun = interp1d(tint, mint, bounds_error = False, fill_value=(mint[0], mint[-1]))
    if regime == 'sine':
        mfun = noize.randomsin(mdot, sinefreq, samp, rseed = krepeat)
        mconst = False
        
    # ASCII output:
    if ifasc:
        # ASCII output
        #        print("writing file number "+str(krepeat)+" of "+str(nrepeat)+"\n")
        fout = open('slabout'+hdf.entryname(krepeat)+'.dat', 'w')
        fout.write('# parameters:')
        fout.write('# mdot = '+str(mdot)+'\n')
        fout.write('# std(log(mdot)) = '+str(dmdot)+'\n')
        if regime == 'flick':
            fout.write('# flickering with p = '+str(nflick)+'\n')
        if regime == 'brown':
            fout.write('# brownian with tbreak = '+str(tbreak)+'\n')
        if regime == 'sine':
            fout.write('# sine wave with omega = '+str(sinefreq)+' and amplitude '+str(samp)+'\n')
        fout.write('# t  mdot m lout orot\n')
    mdotar = zeros(nt) ; loutar = zeros(nt) ; mar = zeros(nt) ; orotar = zeros(nt)
    while((t<tmax) & (ctr<nt)):           
        # halfstep:
        dt = .25/(10.*mdot/m + 10.*mdot*j/l + 10./tdepl+1./dtout+1.*alpha)
        if mconst:
            mdotcurrent = mdot
            mdotcurrent1 = mdot
        else:
            mdotcurrent = mfun(t)
            mdotcurrent1 = mfun(t+dt/2.)
        dm, dl, lout, cth = onestep(m, l, mdotcurrent)
        if isinf(dm+dl):
            print("dm = "+str(dm))
            print("dl = "+str(dl))
            ii=input("ddl")
        m1 = m+dm*dt/2. ; l1 = l + dl*dt/2. ; t1 = t+dt/2.
        dm, dl, lout, cth = onestep(m1, l1, mdotcurrent1)
        m += dm*dt ; l += dl*dt ; t +=dt

        if(t>=tstore):
            orot = l / m /r**2/tscale / 2. /pi
            if ifasc:
                fout.write(str(t*tscale)+" "+str(mdotcurrent1/4./pi)+" "+str(m)+" "+str(lout/ (4.*pi))+" "+str(orot)+"\n")
                fout.flush()
            tstore += dtout
            mdotar[ctr] = mdotcurrent1/ (4.*pi) ; loutar[ctr] = lout/ (4.*pi) ; mar[ctr] = m1 ; orotar[ctr] = orot
            ctr+=1
    if ifasc:
        fout.close()
    if hfile is not None:
        hdf.dump(hfile, krepeat, ["mdot", "L", "M", "omega"], [mdotar, loutar, mar, orotar])
    if ifplot and regime == 'const':
        plots.mconsttests(tar, mar*mscale, orotar, meq, oeq)
    return True

def slab_evolution(nrepeat = 1, nproc = None, somega = None):
    global hfile 
    global sinefreq

    if somega is not None:
        sinefreq = 2. * pi * somega  * tscale # adjusting the sine frequency                                  

    hfile = hdf.init(hname, tar * tscale, mdot = mdot, alpha = alpha, tdepl = tdepl,
                     nsims = nrepeat, nflick = nflick, tbreak = tbreak, regime = regime)
        
    krepeat = linspace(0, nrepeat, num=nrepeat, endpoint=False, dtype=int)
    print(krepeat)
    if nproc is not None:
        pool = multiprocessing.Pool(nproc)
        pool.map(singlerun, krepeat)
    else:
        print('sequential mapping\n')
        # [singlerun(x, nflick=nflick, tbreak=tbreak, hfile=hfile) for x in krepeat]
        map(singlerun, krepeat)
        if not ifzarr:
            hfile.close()

