import numpy
import numpy.fft
from numpy import *
from numpy.fft import *
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from functools import partial

# from timing import oldscipy
oldscipy = False

if oldscipy:
    from scipy.optimize import fsolve
else:
    from scipy.optimize import root_scalar

import hdfoutput as hdf
#import plots as plots
from mslab import j, r, ifzarr, tscale, ifplot, tdepl, alpha, omegaNS, regime, mscale

import multiprocessing
from multiprocessing import Pool

q = alpha * tdepl / r**1.5 

ons = omegaNS * r**1.5

if ifplot:
    import matplotlib
    from pylab import *
    from matplotlib import interactive, use
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    #Uncomment the following if you want to use LaTeX in figures
    rc('font',**{'family':'serif','serif':['Times']})
    rc('mathtext',fontset='cm')
    rc('mathtext',rm='stix')
    rc('text', usetex=True)
    # #add amsmath to the preamble
    matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amssymb,amsmath}"] 
    ioff()
    use('Agg')
    from plots import xydy

####################
def lsine(x, A, B, Omega, x0):
    return A+B*sin(Omega * (x-x0))

################################################################
def readrange(infile, entries):
    print(entries)
    t, datalist = hdf.read(infile, entries[0])
    L, M, mdot, omega = datalist

    x = mdot
    
    xsum = copy(x) ; xsumsq = x
    
    for k in arange(size(entries)-1)+1:
        t, datalist = hdf.read(infile, entries[k])
        L, M, mdot, omega = datalist
        x = mdot    
        xsum += x ; xsumsq += x**2

    return t, xsum, xsumsq

def curvestat(infile, nproc = 1, nentries = 1):

    timestart = time.time()
    
    pool = multiprocessing.Pool(processes = nproc)

    nperpro = ((nentries-1) // nproc) + 1 # number of times we require each core (maximal)

    xsumtot = 0. ; xsumsqtot = 0.

    entries_raw = range(nentries) 
    entries = []
    
    for k in arange(nproc):
        k1 = k*nperpro
        k2 = minimum(k1+nperpro, nentries)
        entries.append(entries_raw[k1:k2])
    print("entries: "+str(entries))
    print("check n(entries) = "+str(size(asarray(entries).flatten()))+" = "+str(nentries))

    readrange_partial = partial(readrange, infile)

    res = pool.map(readrange_partial, entries)

    l = squeeze(asarray(list(res)))
    print(shape(l))
    t = l[0,0,:] ; xsum = l[:,1,:].sum(axis=0) ; xsumsq = l[:,2,:].sum(axis=0)
    print((l[0,1,:]-l[1,1,:]).max())
    xmean = xsum / double(nentries)
    xstd = sqrt(xsumsq / double(nentries) - xmean**2)

    timeend = time.time()
    print("reading and calculations took "+str(timeend-timestart)+"s")
    
    xydy(t, xmean, xstd, outfile = 'curvestat')


#######################################################################

def viewcurve(infile, nentry, trange = None, ascout = False, stored = False,
              taver = None, ttest = False, cb = True, q=q, xstretch = 1.,
              gapping = None, xrange = None):
    '''
    viewing and plotting a simulated light curve 
    taver is the size of the window for time-average
    gapping is gapping period in seconds: data are shown only for half of the period
    '''
    if stored:
        lines = loadtxt(infile+hdf.entryname(nentry)+'.dat')
        t = lines[:,0] ; mdot = lines[:,1] ; L   = lines[:,3] ; M = lines[:,2] ; omega = lines[:,4]
    else:
        t, datalist = hdf.read(infile, nentry)
        L, M, mdot, omega = datalist
        
    omega *= r**1.5*tscale * 2.*pi # omega to physical frequency

    if trange is not None:
        w = (t>trange[0]) & (t<trange[1])
        t=t[w] ; L=L[w] ; M=M[w] ; mdot=mdot[w] ; omega=omega[w]
    niter = shape(mdot)[0] ; nt = size(t)

    if ttest:
        dt = t[1:]-t[:-1]
        print("t interval RMS "+str(dt.std()))
        xydy(t[1:], dt, dt*0., outfile = 'ttest')
        return 0.
        
    if taver is not None:
        dtmean = (t.max()-t.min())/double(nt)
        if dtmean >= taver:
            print("taver doing nothing")
        else:
            tnew = arange(t.min(), t.max(), taver)
            tc = (tnew[1:]+tnew[:-1])/2.
            ntnew = size(tnew)-1
            #            print(str(ntnew)+" points")
            Lnew = zeros(ntnew) ; Mnew = zeros(ntnew) ; Onew = zeros(ntnew)
            mdnew = zeros(ntnew)
            dLnew = zeros(ntnew) ; dMnew = zeros(ntnew) ; dOnew = zeros(ntnew)
            dmdnew = zeros(ntnew) ; omegaminus_new = zeros(ntnew)
            for k in arange(ntnew):
                w = (t>tnew[k]) * (t<tnew[k+1])
                print(str(w.sum())+" points")
                Lnew[k] = L[w].mean() ; dLnew[k] = L[w].std()
                Mnew[k] = M[w].mean() ; dMnew[k] = M[w].std()
                Onew[k] = omega[w].mean() ; dOnew[k] = omega[w].std()
                mdnew[k] = mdot[w].mean() ; dmdnew[k] = mdot[w].std()
                atd = alpha * M/1. / r**1.5
                omegaminus_new[k] = (1./2./atd - sqrt((1./2./atd-1.)**2 + (1.-j/sqrt(r))/atd))[w].mean()
                omegaminus_new[k] = (1./atd - 1.)[w].mean()
        if gapping is not None:
            wshow = (tc%gapping)/gapping < 0.5
            tc = tc[wshow]
            Mnew = Mnew[wshow] ; dMnew = dMnew[wshow]
            Lnew = Lnew[wshow] ; dLnew = dLnew[wshow]
            Onew = Onew[wshow] ; dOnew = dOnew[wshow]
            mdnew = mdnew[wshow] ; dmdnew = dmdnew[wshow]
            
    atd = alpha * Mnew/mdnew * r**1.5 
    omegaplus = 1./2./atd + sqrt((1./2./atd-1.)**2 + (1.-j/sqrt(r))/atd)
    omegaminus = 1./2./atd - sqrt((1./2./atd-1.)**2 + (1.-j/sqrt(r))/atd)
    
    if ifplot:
        Ldisc = mdot/r/2.
        oepi = omega * L # epicyclic frequency
        clf()
        # fig, ax = subplots(2,1)
        #   subplot(2,1,0)
        cmap = cm.viridis
        norm = Normalize(vmin=t.min(), vmax=t.max())
        fig = figure()      
        if taver is not None:
            sc1 = scatter(Lnew, Onew, c = tc, s=5.)
            
            errorbar(Lnew, Onew, xerr = dLnew, yerr = dOnew, fmt = 'none',
                     ecolor = cmap(norm(tc)))
            mdevmin = 1e-4 ; mdevmax = 1. ; ndev =10
            mdev = (mdevmax/mdevmin)**(arange(ndev)/double(ndev-1))*mdevmin
            l0 = 0.1 ; noo = 10000 
            otmp = arange(noo)/double(noo-1)*(1.-ons)+ons
            ttmp = arange(noo)*0.01
            if gapping is not None:
                for k in arange(size(mdev)):
                #                plot( l0 * ((1.-otmp**2)*( mdev[k] * q /(1.-otmp)**2 + mdev[k] * q * (otmp-ons)) + mdev[k] * (otmp**2-ons**2)), otmp, '-k')
                    plot( l0 *mdev[k] / (1.-otmp), otmp, '-k')
                #  plot(l0*mdev[k]*(otmp**2-ons**2+2.*q*(1.-otmp**2)*(2.*otmp+1.-ons)), otmp, 'r-')
            print(q)
        else:
            sc1 = scatter(L, omega, c = t, s=1.)
            #   cc = scatter(-L, 2.*pi*omega, c = t, s=1.)
        if cb:
            cbar1 = fig.colorbar(sc1)
            cbar1.ax.tick_params(labelsize=12, length=3, width=1., which='major', direction ='in')
            cbar1.set_label(r'$t$, s', fontsize=14)
        ylabel(r'$\Omega/\Omega_{\rm K}$', fontsize = 20)
        xlabel(r'$L/L_{\rm Edd}$', fontsize = 20)
        ylim(omega.min(), omega.max())
        if xrange is None:
            xrange = [L.min(), L.max()]
        xlim(xrange[0], xrange[1])
        #     xscale('log')
#        if taver is not None:
#            xlim(Lnew.min(), Lnew.max())
#            ylim(2.*pi*Onew.min(), 2.*pi*Onew.max())
        tick_params(labelsize=14, length=6, width=1., which='major', direction ='in')
        tick_params(labelsize=14, length=3, width=1., which='minor', direction ='in')
        if cb:
            fig.set_size_inches(5*xstretch, 6)
        else:
            fig.set_size_inches(4*xstretch, 6)
        fig.tight_layout()
        savefig(infile+"_O.pdf")
        savefig(infile+"_O.png")
        savefig(infile+"_O.eps")

        wpos = (L > 0.)
        clf()
        fig = figure()
        plot(t[wpos], L[wpos], 'k,')
        if taver:
            plot(tc, Lnew, 'k-')
            l1 = 0.5*mdnew/r*(1.-Onew**2)
            l2 = 0.5*alpha*Mnew*(1.-Onew**2)*(Onew-ons)/r**2.5/(4.*pi)
            l3 = 0.5*alpha*Mnew/q*(Onew**2-ons**2)/r**2.5/(4.*pi)
            #            plot(tc, l1+l2+l3, 'gray')
            plot(tc, l1, 'g--')
            plot(tc, l2, 'r-.')
            plot(tc, l3, 'b:')
        #        plot(t, Ldisc * j**2 / r, 'r:')
        xlabel(r'$t$, s', fontsize = 20) ; ylabel(r'$L/L_{\rm Edd}$', fontsize = 20)
        #        ylim(minimum(L,Ldisc)[wpos].min(), maximum(L, Ldisc)[wpos].max())
        #        yscale('log')
        tmax = 10.*tdepl*tscale
        xlim(0.,tmax)
        if taver:
            ylim(0., Lnew[tc < tmax].max())
        tick_params(labelsize=14, length=6, width=1., which='major', direction='in')
        tick_params(labelsize=14, length=3, width=1., which='minor', direction='in')
        fig.set_size_inches(4*xstretch, 4)
        fig.tight_layout()
        savefig(infile+"_lBL.eps")
        savefig(infile+"_lBL.png")
        savefig(infile+"_lBL.pdf")

        if taver is not None:
            clf()
            plot(tc, Onew/(Onew).max(), 'k.')
            plot(tc, Mnew/Mnew.max(), 'r.')
            plot(tc, Lnew/Lnew.max(), 'b.')
            savefig(infile+"_MJ.png")
            clf()
            fig = figure()
            plot(tc, Onew, 'k-')
            ominus = mdnew/alpha/Mnew*r**1.5*4.*pi-1.
            plot(tc, ominus, 'b:')
            plot(tc, tc*0.+j/r**.5, 'b:')
            #        plot(t[wpos], omegaminus[wpos], 'r--')
            plot(tc, tc*0.+ons, 'g-.')
            xlabel(r'$t$, s', fontsize = 20) ; ylabel(r'$\Omega/\Omega_{\rm K}$', fontsize = 20)
            xlim(0., tmax) ; ylim(0., ominus[tc < tmax].max())
            tick_params(labelsize=14, length=6, width=1., which='major', direction='in')
            tick_params(labelsize=14, length=3, width=1., which='minor', direction='in')
            fig.set_size_inches(4*xstretch, 4)
            fig.tight_layout()
            savefig(infile+"_Opm.eps")
            savefig(infile+"_Opm.png")
            savefig(infile+"_Opm.pdf")
            #      print(Lnew/(Mnew/mscale))
            clf()
            sc1 = scatter(Mnew*mscale/1.e20, Onew, c = tc, s=5.)
            errorbar(Mnew*mscale/1.e20, Onew, xerr =dMnew/1.e9, yerr = dOnew, fmt = 'none',
                     ecolor = cmap(norm(tc)))
            for k in arange(size(mdev)):
                plot(mdev[k]/(1.-otmp) * M.mean()*mscale/1e20, otmp, 'k-')
            ylabel(r'$\Omega/\Omega_{\rm K}$', fontsize = 20)
            xlabel(r'$M$, $10^{20}$g', fontsize = 20)
            tick_params(labelsize=14, length=6, width=1., which='major', direction ='in')
            tick_params(labelsize=14, length=3, width=1., which='minor', direction ='in')
            ylim(omega.min(), omega.max())
            
            xlim(round(Mnew.min()*mscale/1.e20,1), round(Mnew.max()*mscale/1.e20,1))
            fig.set_size_inches(4*xstretch, 6)
            fig.tight_layout()
            savefig(infile+"_OM.pdf")
            savefig(infile+"_OM.png")
            savefig(infile+"_OM.eps")
            clf()
            sc1 = scatter(Mnew, mdnew, c = tc, s=5.)
            errorbar(Mnew, mdnew, xerr =dMnew, yerr = dmdnew, fmt = 'none',
                     ecolor = cmap(norm(tc)))
            # for k in arange(size(mdev)):
            #     plot(.5*q*mdev[k]/(1.-otmp)**2, otmp, 'k-')
            ylabel(r'$\dot m$', fontsize = 20)
            xlabel(r'$M$, relative units', fontsize = 20)
            tick_params(labelsize=14, length=6, width=1., which='major', direction ='in')
            tick_params(labelsize=14, length=3, width=1., which='minor', direction ='in')
            ylim(mdot.min(), mdot.max())
            xlim(M.min(), M.max())
            fig.tight_layout()
            savefig(infile+"_MMass.pdf")
            savefig(infile+"_MMass.png")

        close("all")
    if regime == 'sine':
        cfit, ccov = curve_fit(lsine, t[:2000], mdot[:2000], sigma = mdot[:2000],
                               p0 = [1., 0.05, 0.09, -5.8], ftol = 1e-15, xtol = 1e-15)
        print(cfit)
        mdot -= lsine(t, cfit[0], cfit[1], cfit[2], cfit[3])
        clf()
        plot(t, mdot, 'k-')
        xlabel(r'$t$, s', fontsize = 20) ; ylabel(r'$\dot m$', fontsize = 20)
        savefig(infile+"_msin.eps")
        savefig(infile+"_msin.png")
        savefig(infile+"_msin.pdf")
        
    if ascout:
        fout = open(infile+hdf.entryname(nentry)+'.dat', 'w')
        #        fout.write('# parameters:')
        # fout.write('# mdot = '+str(mdot)+'\n')
        # fout.write('# std(log(mdot)) = '+str(dmdot)+'\n')
        fout.write('#  t, mdot, M, L, omega')
        for k in arange(nt):
            fout.write(str(t[k])+" "+str(mdot[k])+" "+str(M[k])+" "+str(L[k])+" "+str(omega[k])+"\n")
            fout.flush()
        fout.close()
   
