from numpy import *
from numpy.fft import *
import matplotlib
from pylab import *
from scipy.interpolate import interp1d

import hdfoutput as hdf

#Uncomment the following if you want to use LaTeX in figures
rc('font',**{'family':'serif','serif':['Times']})
rc('mathtext',fontset='cm')
rc('mathtext',rm='stix')
rc('text', usetex=True)
# #add amsmath to the preamble
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amssymb,amsmath}"] 
ioff()

def viewcurve(infile):
    t, mdot = hdf.vread(infile, valname = "mdot")
    t, lBL = hdf.vread(infile, valname = "L")
    niter = shape(mdot)[0] ; nt = size(t)
    clf()
    for k in arange(niter):
        plot(t, mdot[k][:])
    xlabel('$t$, s') ; ylabel('$\dot{M}c^2/L_{\rm Edd}$')
    yscale('log') ; xlim(0,1)
    savefig(infile+"_mdot.png")
    clf()
    for k in arange(niter):
        plot(t, lBL[k][:])
    xlabel('$t$, s') ; ylabel('$\dot{L}/L_{\rm Edd}$')
    savefig(infile+"_lBL.png")
    close("all")
    plot(log(mdot[0][:]), log(mdot[1][:]), '.k')
    savefig("sample_corr.png")

def spec(infile = 'slabout', trange = [0.1,1e5]):
    '''
    makes spectra and cross-spectra out of the blslab output
    '''
    # infile has the form t -- mdot -- m -- lBL -- orot
    
    #    lines = np.loadtxt(infile+".dat")
    #   t = lines[:,0] ; mdot = lines[:,1] ; m = lines[:,2] ; lBL = lines[:,3] ; orot = lines[:,4]

    t, mdot = hdf.vread(infile, valname = "mdot")
    t, lBL = hdf.vread(infile, valname = "L")
    t, omega = hdf.vread(infile, valname = "omega")
    niter = shape(mdot)[0] ; nt = size(t)
    #   print("nt = "+str(nt))
    #   print("nt = "+str(size(lBL[1])))
    #   print("size(mdot) = " + str(shape(mdot)))
    #   ii=input("t")
    mdotar = zeros([niter, nt])
    lBLar = zeros([niter, nt])
    orot = zeros([niter, nt])
    for k in arange(niter):
        #        print(k)
        mdotar[k,:] = (mdot[k])[:]
        lBLar[k,:] = (lBL[k])[:]
        orot[k,:] = (omega[k])[:]
    mdot = copy(mdotar) ; lBL = copy(lBLar) 
    mdot_demean = copy(mdot) ; lBL_demean = copy(lBL) ; orot_demean = copy(orot)
    # subtracting mean values
    for k in arange(niter):
        mdot_demean[k,:] = mdot[k,:] - mdot[k,:].mean()
        lBL_demean[k,:] = lBL[k,:] - lBL[k,:].mean()
        orot_demean[k,:] = orot[k,:] - orot[k,:].mean()
        
    nt = np.size(t) ;    tspan = t.max() - t.min() 
    dt = tspan / np.double(nt)
    #frequencies:
    freq1 =1./tspan/2. ; freq2=freq1*np.double(nt)/2.
    freq = np.fft.fftfreq(nt, dt)
    
    # Fourier images: 
    mdot_f=2.*fft(mdot_demean)/mdot.sum()  # last axis is the FFT by default
    lBL_f=2.*fft(lBL_demean)/lBL.sum()
    orot_f = 2.*fft(orot_demean)/orot.sum()

    # PDS and cross-spectra:
    mdot_pds = abs(mdot_f)**2
    lBL_pds = abs(lBL_f)**2
    orot_pds = abs(orot_f)**2
    mmdot_cross = mdot_f * conj(lBL_f)

    mdot_pds_av = mdot_pds.mean(axis = 0) ; mdot_pds_std = mdot_pds.std(axis = 0)
    lBL_pds_av = lBL_pds.mean(axis = 0) ; lBL_pds_std = lBL_pds.std(axis = 0)
    mmdot_cross_angle_av = angle(mmdot_cross).mean(axis = 0)
    mmdot_cross_angle_std = angle(mmdot_cross).std(axis = 0)
    coherence = mmdot_cross/sqrt(abs(mdot_pds * lBL_pds))
    coherence_av = abs((coherence).mean(axis=0))
    coherence_std = coherence.std(axis=0)

    print(shape(mdotar))
    print(shape(mdot))
    print(shape(mdot_f))
    print(shape(mdot_pds_av))
    
    # ASCII output:
    fout = open(infile+'_sp.dat', 'w')
    fout.write("# f  mdot dmdot  lBL dlBL  coherence dcoherence phaselag dphaselag\n")
    for k in arange(size(freq)):
        fout.write(str(freq[k])+" "
                   +str(mdot_pds_av[k])+" "+str(mdot_pds_std[k])+" "
                   +str(lBL_pds_av[k])+" "+str(lBL_pds_std[k])+" "
                   +str(coherence_av[k])+" "+str(coherence_std[k])+" "
                   +str(mmdot_cross_angle_av[k])+" "+str(mmdot_cross_angle_std[k])+" "
                   +"\n")
    fout.close()

    w= (freq > 0.)
    
    # graphic output:
    clf()
    # plot(freq, mdot_pds, 'k,')
    # plot(freq, lBL_pds, 'r,')
    errorbar(freq[w], mdot_pds_av[w], yerr = mdot_pds_std[w]/sqrt(niter-1.), fmt = 'ks')
    errorbar(freq[w], lBL_pds_av[w], yerr = lBL_pds_std[w]/sqrt(niter-1.), fmt = 'rd')    
    # plot(freq[freq>0.], 1e-3/((freq[freq>0.]*1000.*4.92594e-06*1.5)**2+1.), 'g-')
    xlim([1./tspan/2., freq.max()])
    xscale('log') ; yscale('log')
    xlabel(r'$f$, Hz') ; ylabel(r'$PDS$')
    savefig('pdss.png')
    savefig('pdss.eps')
    clf()
    fig, ax = subplots(2,1)
    ax[0].errorbar(freq[w], mmdot_cross_angle_av[w],
                   yerr =  mmdot_cross_angle_std[w]/sqrt(niter-1.), fmt = 'k.')
    ax[0].plot(freq[w], freq[w]*0., 'r-')
    ax[0].plot(freq[w], freq[w]*0.+pi/2., 'r-')
    ax[0].plot(freq[w], freq[w]*0.+pi, 'r-')
    ax[0].set_xscale('log')  ; ax[0].set_ylabel(r'$\Delta \varphi$', fontsize=18)
    ax[1].errorbar(freq[w], coherence_av[w], yerr = coherence_std[w]/sqrt(niter-1.), fmt = 'k.')
    ax[1].set_xscale('log')
    ax[1].set_xlabel(r'$f$, Hz', fontsize=18) ; ax[1].set_ylabel(r'coherence', fontsize=18)
    ax[0].tick_params(labelsize=14, length=6, width=1., which='major')
    ax[0].tick_params(labelsize=14, length=3, width=1., which='minor')
    ax[1].tick_params(labelsize=14, length=6, width=1., which='major')
    ax[1].tick_params(labelsize=14, length=3, width=1., which='minor')
    fig.set_size_inches(5, 6)
    fig.tight_layout()
    savefig('coherence.png')
    savefig('coherence.eps')
    close('all')

