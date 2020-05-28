import numpy
import numpy.fft
from numpy import *
from numpy.fft import *
from scipy.interpolate import interp1d
from functools import partial

oldscipy = False
if oldscipy:
    from scipy.optimize import fsolve
else:
    from scipy.optimize import root_scalar

import hdfoutput as hdf
import plots as plots
from mslab import j, r, ifzarr, tscale, ifplot, tdepl, alpha, omegaNS

import multiprocessing
from multiprocessing import Pool


if ifplot:
    import matplotlib
    from pylab import *
    from matplotlib import interactive, use
    #Uncomment the following if you want to use LaTeX in figures
    rc('font',**{'family':'serif','serif':['Times']})
    rc('mathtext',fontset='cm')
    rc('mathtext',rm='stix')
    rc('text', usetex=True)
    # #add amsmath to the preamble
    matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amssymb,amsmath}"] 
    ioff()
    use('Agg')
    import plots

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
    t = l[0,0,:] ; xsum = l[:,1,:].sum(axis=0) ; xsumsq = l[:,2,:].sum(axis=0)

    xmean = xsum / double(nentries)
    xstd = sqrt(xsumsq / double(nentries) - xmean**2)

    timeend = time.time()
    print("reading and calculations took "+str(timeend-timestart)+"s")
    
    plots.xydy(t, xmean, xstd, outfile = 'curvestat')


#######################################################################

def viewcurve(infile, nentry, trange = None, ascout = False, stored = False):
    if stored:
        lines = loadtxt(infile+hdf.entryname(nentry)+'.dat')
        t = lines[:,0] ; mdot = lines[:,1] ; L   = lines[:,3] ; M = lines[:,2] ; omega = lines[:,4]
    else:
        t, datalist = hdf.read(infile, nentry)
        L, M, mdot, omega = datalist
        
    if trange is not None:
        w = (t>trange[0]) & (t<trange[1])
        t=t[w] ; L=L[w] ; M=M[w] ; mdot=mdot[w] ; omega=omega[w]
    niter = shape(mdot)[0] ; nt = size(t)
    if ifplot:
        Ldisc = mdot/r/2.
        oepi = 2.*omega * r**1.5*tscale * L # epicyclic frequency
        clf()
        # fig, ax = subplots(2,1)
        #   subplot(2,1,0)
        fig = figure()
        sc1 = scatter(L+Ldisc, 2.*pi*omega * r**1.5*tscale, c=t, s=1.)
        cbar1 = colorbar(sc1)
        cbar1.ax.tick_params(labelsize=14, length=3, width=1., which='major')
        cbar1.set_label(r'$t$, s', fontsize=18)
        ylabel(r'$\Omega/\Omega_{\rm K}$', fontsize = 20)
        xlabel(r'$L/L_{\rm Edd}$', fontsize = 20)
        ylim(2.*pi*omega.min() * r**1.5*tscale, 2.*pi*omega.max() * r**1.5*tscale)
        tick_params(labelsize=14, length=6, width=1., which='major')
        tick_params(labelsize=14, length=3, width=1., which='minor')
        fig.set_size_inches(5, 6)
        fig.tight_layout()
        savefig(infile+"_O.pdf")
        savefig(infile+"_O.png")

        wpos = (L > 0.)
        clf()
        fig = figure()
        plot(t[wpos], L[wpos], 'k-')
        plot(t, Ldisc * j**2 / r, 'r:')
        xlabel(r'$t$, s', fontsize = 20) ; ylabel(r'$L/L_{\rm Edd}$', fontsize = 20)
        ylim(minimum(L,Ldisc)[wpos].min(), maximum(L, Ldisc)[wpos].max())
        #        yscale('log')
        fig.set_size_inches(10, 4)
        tick_params(labelsize=14, length=6, width=1., which='major')
        tick_params(labelsize=14, length=3, width=1., which='minor')
        fig.tight_layout()
        savefig(infile+"_lBL.eps")
        savefig(infile+"_lBL.png")
        savefig(infile+"_lBL.pdf")
        atd = alpha * M/mdot * r**1.5 
        print("q = "+str(r**2/alpha/tdepl/j))
        omegaplus = 1./2./atd + sqrt((1./2./atd-1.)**2 + (1.-j/sqrt(r))/atd)
        omegaminus = 1./2./atd - sqrt((1./2./atd-1.)**2 + (1.-j/sqrt(r))/atd)
        
        clf()
        plot(t[wpos], omega[wpos] * r**1.5*tscale, 'k-')
        plot(t[wpos], omegaplus[wpos], 'b:')
        #        plot(t[wpos], omegaminus[wpos], 'r--')
        plot(t, t*0.+omegaNS, 'g-.')
        xlabel(r'$t$, s', fontsize = 20) ; ylabel(r'$\Omega/\Omega_{\rm K}$', fontsize = 20)
        tick_params(labelsize=14, length=6, width=1., which='major')
        tick_params(labelsize=14, length=3, width=1., which='minor')
        savefig(infile+"_Opm.eps")
        savefig(infile+"_Opm.png")
        savefig(infile+"_Opm.pdf")
        
        close("all")
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
   
