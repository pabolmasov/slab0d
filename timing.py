from numpy import *
from numpy.fft import *
import matplotlib
from pylab import *
from scipy.interpolate import interp1d

import hdfoutput as hdf
import plots as plots

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

def spec_sequential(infile = 'slabout', trange = [0.1, 1e10]):
    '''
    makes spectra and cross-spectra out of the blslab output
    reads the entries one by one, thus avoiding memory issues
    '''
    keys = hdf.keyshow(infile+'.hdf5')
    nsims = size(keys)-1 # one key points to globals

    for k in arange(nsims):
        t, datalist = hdf.read(infile+'.hdf5', 0, entry = keys[k])
        L, M, mdot, orot = datalist
        if k == 0:
            nt = size(t) ;  tspan = t.max() - t.min() 
            dt = tspan / np.double(nt)
            #frequencies:
            freq1 =1./tspan/2. ; freq2=freq1*np.double(nt)/2.
            freq = np.fft.fftfreq(nt, dt)
            #            mdot = zeros([nsims, nt], dtype = double)
            #            L = zeros([nsims, nt], dtype = double)
            #    orot = zeros([nsims, nt], dtype = double)
            mdot_PDS_av = zeros(nt, dtype = double)
            mdot_PDS_std = zeros(nt, dtype = double)
            orot_PDS_av = zeros(nt, dtype = double)
            orot_PDS_std = zeros(nt, dtype = double)
            cross_av = zeros(nt, dtype = complex)
            dcross_im = zeros(nt, dtype = complex)
            dcross_re = zeros(nt, dtype = complex)
        #        mdot_av += mdot ; orot_av += orot
        #        mdot_std += mdot**2 ; orot_std += orot**2
        # Fourier images
        mdot_f=2.*fft(mdot-mdot.mean())/mdot.sum()  # last axis is the FFT by default
        #        lBL_f=2.*fft(lBL_demean)/lBL.sum()
        orot_f = 2.*fft(orot-orot.mean())/orot.sum()
        cross = mdot_f * conj(orot_f)
        mdot_PDS = abs(mdot_f)**2 ; orot_PDS = abs(orot_f)**2
        mdot_PDS_av += mdot_PDS ; mdot_PDS_std += mdot_PDS**2
        orot_PDS_av += orot_PDS ; orot_PDS_std += orot_PDS**2
        cross = mdot_f * conj(orot_f)
        cross_av += cross
        dcross_im += cross.imag**2
        dcross_re += cross.real**2

    # mean values:
    mdot_PDS_av /= double(nsims) ; orot_PDS_av /= double(nsims)
    cross_av /= double(nsims)
    # RMS:
    mdot_PDS_std = sqrt(mdot_PDS_std / double(nsims) - mdot_PDS_av**2) / sqrt(double(nsims-1))
    orot_PDS_std = sqrt(orot_PDS_std / double(nsims) - orot_PDS_av**2) / sqrt(double(nsims-1))
    dcross_im = sqrt(dcross_im / double(nsims) - cross_av.imag**2) / sqrt(double(nsims-1))
    dcross_re = sqrt(dcross_re / double(nsims) - cross_av.real**2) / sqrt(double(nsims-1))
    # coherence:
    coherence = abs(cross_av) / sqrt(mdot_PDS_av*orot_PDS_av)
    dcoherence = (sqrt(dcross_im**2 + dcross_re**2)/abs(cross_av) +  (mdot_PDS_std/mdot_PDS_av + orot_PDS_std/orot_PDS_av) * 0.5) * coherence 
    phaselag = angle(cross_av)
    dphaselag = sqrt(dcross_im**2 + dcross_re**2)/abs(cross_av) # estimate for the angular size of the spot
    
    w = freq > 0.
    
    clf()
    # plot(freq, mdot_pds, 'k,')
    # plot(freq, lBL_pds, 'r,')
    errorbar(freq[w], mdot_PDS_av[w], yerr = mdot_PDS_std[w], fmt = 'ks')
    errorbar(freq[w], orot_PDS_av[w], yerr = orot_PDS_std[w], fmt = 'rd')   
    # plot(freq[freq>0.], 1e-3/((freq[freq>0.]*1000.*4.92594e-06*1.5)**2+1.), 'g-')
    xlim([1./tspan/2., freq.max()])
    xscale('log') ; yscale('log')
    xlabel(r'$f$, Hz') ; ylabel(r'$PDS$')
    savefig('pdss.png')
    savefig('pdss.eps')
    clf()
    fig, ax = subplots(2,1)
    ax[0].errorbar(freq[w], phaselag[w],
                   yerr =  dphaselag[w], fmt = 'k.')
    ax[0].plot(freq[w], freq[w]*0., 'r-')
    ax[0].plot(freq[w], freq[w]*0.+pi/2., 'r-')
    ax[0].plot(freq[w], freq[w]*0.+pi, 'r-')
    ax[0].set_xscale('log')  ; ax[0].set_ylabel(r'$\Delta \varphi$', fontsize=18)
    ax[1].errorbar(freq[w], coherence[w], yerr = dcoherence[w], fmt = 'k.')
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
        
def spec_readall(infile = 'slabout', trange = [0.1,1e5]):
    '''
    makes spectra and cross-spectra out of the blslab output
    Note! it reads the entire HDF5 tables for multiple simulations at once. 
    If the number of simulations times the number of timesteps Nsim X Nt \gtrsim 10^6, 
    the memory is overloaded. For a large number of time series, use spec_sequential
    '''
    # infile has the form t -- mdot -- m -- lBL -- orot
    
    #    lines = np.loadtxt(infile+".dat")
    #   t = lines[:,0] ; mdot = lines[:,1] ; m = lines[:,2] ; lBL = lines[:,3] ; orot = lines[:,4]

    # mdot is global!
    t, mdot = hdf.vread(infile, valname = "mdot")
    t, lBL = hdf.vread(infile, valname = "L")
    t, orot = hdf.vread(infile, valname = "omega")
    niter = shape(mdot)[0] ; nt = size(t)
    # print("nt = "+str(nt))
    # print("lBLshape = "+str(shape(lBL)))
    #  print("size(mdot) = " + str(shape(mdot)))
    #   ii=input("t")
    '''
    mdotar = zeros([niter, nt])
    lBLar = zeros([niter, nt])
    orot = zeros([niter, nt])
    for k in arange(niter):
        #        print(k)
        mdotar[k,:] = (mdot[k])[:]
        lBLar[k,:] = (lBL[k])[:]
        orot[k,:] = (omega[k])[:]
    mdot = copy(mdotar) ; lBL = copy(lBLar) 
    '''
    mdot_demean = copy(mdot) ; lBL_demean = copy(lBL) ; orot_demean = copy(orot)
    # subtracting mean values
    for k in arange(niter):
        mdot_demean[k,:] = mdot[k,:] - mdot[k,:].mean()
        lBL_demean[k,:] = lBL[k,:] - lBL[k,:].mean()
        orot_demean[k,:] = orot[k,:] - orot[k,:].mean()
        
    nt = np.size(t) ;  tspan = t.max() - t.min() 
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
    mmdot_cross = mdot_f * conj(orot_f)

    mdot_pds_av = mdot_pds.mean(axis = 0) ; mdot_pds_std = mdot_pds.std(axis = 0)
    lBL_pds_av = lBL_pds.mean(axis = 0) ; lBL_pds_std = lBL_pds.std(axis = 0)
    orot_pds_av = orot_pds.mean(axis = 0) ; orot_pds_std = orot_pds.std(axis = 0)
    mmdot_cross_angle_av = angle(mmdot_cross).mean(axis = 0)
    mmdot_cross_angle_std = angle(mmdot_cross).std(axis = 0)
    coherence = mmdot_cross/sqrt(abs(mdot_pds * lBL_pds))
    coherence_av = abs((coherence).mean(axis=0))
    coherence_std = coherence.std(axis=0)

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
    # binning
    

    '''    
    # graphic output:
    if ifplot:
        plots.pds(binfreq, mdot_pdsbin, mdot_dpdsbin, lBL_pdsbin, lBL_dpdsbin, npoints)
        #        plots.phaselag(binfreq, phaselag_bin, dphaselag_bin, npoints)
        plots.coherence(binfreq, mmdot_crossbin, dmmdot_crossbin,
              mdot_pdsbin, mdot_dpdsbin, lBL_pdsbin, lBL_dpdsbin,
              npoints)
    '''
    
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

