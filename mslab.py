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

import noize
import hdfoutput as hdf

# GM = c = kappa = 1

# dl/dt = mdot * j - alpha * geff * M * R * (omega > omegaNS)
# dm/dt = mdot - alpha * geff * M^2/L * R * (omega > omegaNS)

mNS = 1.5 # NS mass in Solar units
r = 6./(mNS/1.5) # NS radius in GM/c**2 units
alpha = 1e-5
tdepl = 1e3 # depletion time in GM/c^3 units
j = 0.9*sqrt(r)
pspin = 0.003 # spin period, s
tscale = 4.92594e-06 * mNS
mscale = 6.41417e10 * mNS
omegaNS = 2.*pi/pspin *tscale
ifplot = False
ifasc = False
if ifplot:
    import plots

def onestep(m, l, mdot):
    '''
    single step of boundary layer evolution
    input: mass, luminosity, mass accr. rate
    output: dm/dt, dl/dt, luminosity release, cos(theta) for the flow
    '''
    omega = l /m /r**2
    geff = maximum(1./r**2 - omega**2*r, 0.)
    dl = mdot * j - l / tdepl - alpha * geff * m * r * sign(omega - omegaNS)
    dm = mdot - m / tdepl
    # alpha * geff * m**2/l * (omega > omegaNS)
    #    a = alpha * m * r / omega * (omega-omegaNS)**2
    lout = geff*alpha*m * r * abs(omega - omegaNS) + m/tdepl * (omega**2-omegaNS**2) * r**2 /2.
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

def slab_evolution(nflick = None, tbreak = None, nrepeat = 1, hname = 'slabout'):
    '''
    evolution of a layer spun up by a variable mass accretion rate. 
    If nflick is set, it is the power-law index of the noize spectrum
    if tbreak is set, it is the characteristic time scale for the spectral break
    if neither is set, mass accretion rate is assumed constant
    the run is repeated "nrepeat" times, and the resulting PDS averaged
    '''
    # initial conditions:
    mdot = 1. * 4.*pi # mean mass accretion rate, GM/kappa c units
    dmdot =  0.5 # relative variation dispersion

    maxtimescale = (tdepl+1./alpha)
    mintimescale = 1./(1./tdepl+alpha)
    t = 0. ; dt = 0.1 ; tmax = 10.*maxtimescale
    tstore = 0.; dtout = minimum(1.,1e-2*mintimescale) # this gives 10^5 data points
    dtdyn = r**1.5
    #    tbreak = 100. * dtdyn 

    m = mdot * tdepl * 0.1 # starting mass, 10% from equilibrium
    l = m*omegaNS*r**2   # starting angular momentum

    for krepeat in arange(nrepeat):
        # single run
        tlist = [] ; mlist = [] ; llist = [] ; loutlist = [] ; clist = []
        # time, mass, angular momentum, total luminosity, thickness of the layer
        t = 0. ; tstore = 0.
        # input noize
        # the highest frequency present in the noise is the local dynamical scale, anyway; inside the dynamical time scale, it is safe to interpolate
        if (nflick is not None):
            tint, mint = noize.flickgen(tmax, dtdyn, nslope = nflick)
            mmean = mint.mean() ; mrms = mint.std()
        else:
            if tbreak is not None:
                tint, mint = noize.brown(tmax, dtdyn, tbreak)  # nslope = nflick)
                mmean = mint.mean() ; mrms = mint.std()
        if(nflick is not None) or (tbreak is not None):
            mint = mdot * exp( (mint-mmean)/mrms * dmdot)
            mconst = False
            mfun = interp1d(tint, mint, bounds_error = False, fill_value=(mint[0], mint[-1]))
        else:
            mconst = True
            meq = mdot * tdepl
            q = r**2/alpha/tdepl/j
            oeq = j/r**2 * (q - sqrt(q**2-4.*q+4.*r/j**2))/2.
        if ifasc:
            # ASCII output
            print("writing file number "+str(krepeat)+" of "+str(nrepeat)+"\n")
            fout = open('slabout'+hdf.entryname(krepeat)+'.dat', 'w')
            fout.write('# parameters:')
            fout.write('# mdot = '+str(mdot)+'\n')
            fout.write('# std(log(mdot)) = '+str(dmdot)+'\n')
            if (nflick is not None):
                fout.write('# flickering with p = '+str(nflick)+'\n')
            if (tbreak is not None):
                fout.write('# brownian with tbreak = '+str(tbreak)+'\n')
            fout.write('# t  mdot m lout orot\n')
        while(t<tmax):           
            # halfstep:
            dt = 1./(100.*(mdot/m)+100.*(mdot*j/l)+1./tdepl+1./dtout)
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
                    fout.write(str(t*tscale)+" "+str(mdotcurrent1/4./pi)+" "+str(m*mscale)+" "+str(lout/ (4.*pi))+" "+str(orot)+"\n")
                    fout.flush()
                tlist.append(t)
                mlist.append(m)
                llist.append(l)
                loutlist.append(lout)
                clist.append(cth)
                #   print(str(t)+" "+str(m)+" "+str(l)+"\n")
                #                print("dt = "+str(dt))
                tstore += dtout
        if ifasc:
            fout.close()
        tar = array(tlist, dtype = double) * tscale
        mar = array(mlist, dtype = double) * mscale
        orot = array(llist, dtype = double) / array(mlist, dtype = double) / r**2/tscale/2./pi
        loutar = array(loutlist, dtype = double) / (4.*pi)
        if mconst:
            mdotar = zeros(size(tar)) + mdot
            ldisc = zeros(size(tar)) + mdot / r / 2.
        else:
            mdotar = mfun(tar/tscale)
            ldisc = mfun(tar/tscale)/r/2.
        ldisc /= 8.*pi # half of potential energy
        cthar = array(clist, dtype = double)
        # HDF output
        if krepeat <1:
            hfile = hdf.init(hname, tar, mdot = mdot, alpha = alpha, tdepl = tdepl,
                             nsims =nrepeat, nflick = nflick, tbreak = tbreak )
            #        print(size(["mdot", "L", "M", "omega"]))
            # ii=input("time")
        hdf.dump(hfile, krepeat, ["mdot", "L", "M", "omega"],
                 [mdotar, loutar, mar, orot])
        if ifplot:
            if mconst:
                plots.mconsttests(tar, mar, orot, meq, oeq)
            else:
                print("plotting\n")
                w=(tar > (0.9*tmax * tscale))
                print(w.sum())
                plots.generalcurve(tar[w], mdotar[w], mar[w], orot[w], cthar[w], loutar[w], ldisc[w])
    hfile.close()

