from numpy import *
from scipy.integrate import *
from scipy.interpolate import interp1d
from scipy.signal import *
from numpy.random import rand, seed

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
alpha = 1.e-7
tdepl = 1.e8 # r**1.5/alpha/2. # depletion time in GM/c^3 units
j = .9999*sqrt(r)
pspin = 0.003 # spin period, s
tscale = 4.92594e-06 * mNS # time scale, s
mscale = 6.41417e10 * mNS # mass scale, g
omegaNS = 2.*pi/pspin * tscale
dtdyn = r**1.5

atd = alpha * tdepl / r**1.5 
print("q = "+str(atd))
omegaplus = 1./2./atd + sqrt((1./2./atd-1.)**2 + (1.-j/sqrt(r))/atd)
omegaminus = 1./2./atd - sqrt((1./2./atd-1.)**2 + (1.-j/sqrt(r))/atd)
print("Omega +/- = "+str(omegaplus)+", "+str(omegaminus)+"\Omega_K \n")
# ii = input("O")

# noise parameters:
regimes = ['const', 'sine', 'flick', 'brown']
regime = 'flick'

nflick = 1.3
tbreak = tdepl
# accretion rate and amplitude:
mdot = 1. * 4.*pi # mean mass accretion rate, GM/kappa c units
dmdot =  0.5 # relative variation dispersion
# time grid
maxtimescale = (tdepl+1./alpha)
mintimescale = 1./(1./tdepl+alpha)
dtout = 3e-2*mintimescale # this gives 10^5 data points
tmax = 100.*maxtimescale
nt = int(ceil(tmax/dtout))
print(str(nt)+" points in time")
print("alpha = "+str(alpha))
print("tdepl = "+str(tdepl*tscale)+"s")
tar = dtout * arange(nt)

sinefreq = 2.*pi * 0.1 / mintimescale  # frequency of the sinusoudal variation
samp = 0.05 # amplitude of the sine
print("somega = "+str(sinefreq/tscale))

# outputs:
ifplot = True # if we are plotting against the computer (disabled for now)
ifasc = False # if we are writing ASCII output
ifzarr = True
hname = 'slabout' # output HDF5 file name
if ifplot:
    import plots

import noize
import hdfoutput as hdf
from timing import spec_parallel

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
    lout = alpha * r * geff * (omega - omegaNS) * m / 2.  + 1./tdepl * (omega**2-omegaNS**2) * r**2 /2. * m + mdot * ((j/r)**2 - (omega*r)**2 ) / 2.
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

def singlerun(krepeat, ooutput = False, epicyclic = False):
    '''
    evolution of a layer spun up by a variable mass accretion rate. 
    If nflick is set, it is the power-law index of the noize spectrum
    if tbreak is set, it is the characteristic time scale for the spectral break
    if neither is set, mass accretion rate is assumed constant
    runs an iteration with the number krepeat
    '''
    print("simulation No"+str(krepeat))
    print("alpha * tdepl = "+str(alpha * tdepl))
    # initial conditions:
    tstore = 0.;
    meq = mdot * tdepl
    q = r**2/alpha/tdepl/j
    oeq = maximum(j/r**2 * (q - sqrt(q**2-4.*q+4.*r/j**2))/2., omegaNS)
    print("oeq = "+str(oeq))
    m =  meq # starting mass, 100% equilibrium
    l = oeq*r**2*m   # starting angular momentum, = equilibrium
    t = 0. ; dt = 0.1 ; ctr = 0
    dt_est = dtout*0.25
    # setting the input mdot variability spectrum:
    if regime == 'flick':
        tint, mint = noize.flickgen(tmax, dt_est, nslope = nflick, rseed = krepeat)
        mmean = mint.mean() ; mrms = mint.std()
    if regime == 'brown':
        tint, mint = noize.brown(tmax, dt_est, tbreak, rseed = krepeat)  # nslope = nflick)
        mmean = mint.mean() ; mrms = mint.std()
    if regime == 'const':
        mconst = True
    if (regime == 'flick') | (regime == 'brown'):
        mint = mdot * exp( (mint-mmean)/mrms * dmdot)
        mconst = False
        mfun = interp1d(tint, mint, bounds_error = False, fill_value='extrapolate', kind='nearest')
                        # (mint[0], mint[-1]), kind='nearest')
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
    mdotar = zeros(nt, dtype = double) ; loutar = zeros(nt) ; mar = zeros(nt) ; orotar = zeros(nt)
    while((t<tmax) & (ctr<nt)):           
        # halfstep:
        dt = .5/(10.*mdot/m + 10.*mdot*j/l + 10./tdepl+1./dtout+1.*alpha)
        if mconst:
            if t <= dt:            
                mdotcurrent = mdot
                mdotcurrent1 = mdot
        else:
            if t <= dt:
                mdotcurrent = mfun(t)
            mdotcurrent1 = mfun(t+dt/2.)
        dm, dl, lout, cth = onestep(m, l, mdotcurrent)
        if isinf(dm+dl):
            print("dm = "+str(dm))
            print("dl = "+str(dl))
            ii=input("ddl")
        m1 = m+dm*dt/2. ; l1 = l + dl*dt/2. ; t1 = t+dt/2.
        dm, dl, lout1, cth = onestep(m1, l1, mdotcurrent1)
        m += dm*dt ; l += dl*dt ; t +=dt

        mdotcurrent = mfun(t) 
        if(t>=tstore):
            # t is a bit offset!
            mdotreal = mfun(tstore) # (mdotcurrent-mdotcurrent1) * (tstore - t) * 2. / dt + mdotcurrent
            mreal = (tstore - t) * dm + m
            lreal = (tstore - t) * dl + l
            orot = lreal / mreal /r**2/tscale / 2. /pi
            if epicyclic:
                orot *= cth *2.
            loutreal = (lout-lout1) * ((t-dt) - tstore) * 2. / dt + lout
            if ifasc:
                fout.write(str(tstore*tscale)+" "+str(mdotreal/4./pi)+" "+str(mreal)+" "+str(loutreal/ (4.*pi))+" "+str(orot)+"\n")
                fout.flush()
            mdotar[ctr] = mdotreal/(4.*pi) ; loutar[ctr] = loutreal/ (4.*pi) ; mar[ctr] = mreal ; orotar[ctr] = orot
            ctr+=1
            tstore += dtout
    if ifasc:
        fout.close()
    if hfile is not None:
        hdf.dump(hfile, krepeat, ["mdot", "L", "M", "omega"], [mdotar, loutar, mar, orotar])
    if ifplot and regime == 'const':
        plots.mconsttests(tar, mar*mscale, orotar, meq, oeq)
    if ooutput:
        return orotar.mean(), orotar.std(), oeq
    else:
        return tar, mdotar, loutar, mar, orotar

##############################################################################

def slab_evolution(nrepeat = 1, nproc = 1, somega = None):
    # simulates the evolution "nrepeat" times on "nproc" cores
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
        outlist = pool.map(singlerun, krepeat)
        pool.close()
        print(shape(outlist))
        spec_parallel(hname, nproc = nproc, rawdata = outlist, binning = 1000)
    else:
        print('sequential mapping\n')
        # [singlerun(x, nflick=nflick, tbreak=tbreak, hfile=hfile) for x in krepeat]
        map(singlerun, krepeat)
        if not ifzarr:
            hfile.close()
            
def tvar(nrepeat = 30, epicyclic = False):
    '''
    special regime with variable tdepl
    '''
    global tdepl
    global hfile
    global hname
    global tmax

    seed()

    hname = 'tvar'

    tmax = maximum(tmax, 1000.)
    
    td1 = dtdyn / alpha * 0.4
    td2 = 2.5 * td1
    nd = nrepeat
    tdar = arange(nd) / double(nd) * (td2 - td1) + td1

    omar = zeros(nd) ; ostar = zeros(nd) ; oeq = zeros(nd)
    
    hfile = hdf.init(hname, tar * tscale, mdot = mdot, alpha = alpha, tdepl = td1,
                     nsims = nrepeat, nflick = nflick, tbreak = tbreak, regime = regime)
    fout = open('tvar.dat', 'w')
    for k in arange(nd):
        tdepl = tdar[k]
        omean, ostd, ooeq = singlerun(k, ooutput = True, epicyclic = epicyclic)
        omar[k] = omean * (2.*pi*tscale) ; ostar[k] = ostd * (2.*pi*tscale) ; oeq[k] = ooeq
        print(str(alpha*tdepl)+" "+str(omean)+" "+str(ostd)+"\n")
        fout.write(str(alpha*tdepl/dtdyn)+" "+str(omar[k]*r**1.5)+" "+str(ostar[k]*r**1.5)+" "+str(oeq[k] * r**1.5)+"\n")
    fout.close()
    
    plots.xydy(alpha*tdar/dtdyn, omar*r**1.5, ostar*r**1.5,
               xlog = (td2/td1) > 5.,
               addlines = [oeq * r**1.5, oeq*0.+omegaNS*r**1.5],
               xl = r'$q$', yl = r'$\Omega/\Omega_{\rm K}$', outfile = 'tvar')
    # \alpha \Omega_{\rm K} t_{\rm depl} 
    
    
