from numpy import *
from numpy.fft import *
import matplotlib
from pylab import *
from matplotlib import interactive, use
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

import hdfoutput as hdf
import plots as plots
from mslab import r

#Uncomment the following if you want to use LaTeX in figures
rc('font',**{'family':'serif','serif':['Times']})
rc('mathtext',fontset='cm')
rc('mathtext',rm='stix')
rc('text', usetex=True)
# #add amsmath to the preamble
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amssymb,amsmath}"] 
ioff()
use('Agg')

def Cfun(x, f1, f2, rhs):
    return (f1-x)*log(abs((f2-x)/(f1-x)))+x - rhs

def Bfun(x, rhs):
    '''
    (exp(x)-1)/x = y solve for x
    '''
    return (exp(x)-1.)/x-rhs

def logABC(x, frange, rhs):
    '''
    makes a modified logarithmic mesh f = A exp(Bx)+C, where x is uniformly spaced between 0 and 1
    '''
    # finding C:
    #   brhs = (frange[1]-frange[0])/rhs
    sol = root_scalar(Bfun, args = (1./rhs), rtol = 1e-5, x0 = log(frange[1]/frange[0]), x1 = -log(rhs))
    print(sol)
    print("basic B = "+str(log(frange[1]/frange[0])))
    print("basic A = "+str(frange[0]))
    bcoef = sol.root
    acoef = rhs/bcoef * (frange[1]-frange[0])
    ccoef = frange[0] - acoef
    print("acoef = "+str(acoef))
    print("bcoef = "+str(bcoef))
    print("ccoef = "+str(ccoef))
    
    return acoef * exp( bcoef * x) + ccoef

def viewcurve(infile, nentry):
    trange = [1., 10.]
    t, datalist = hdf.read(infile, nentry)
    w = (t>trange[0]) & (t<trange[1])
    L, M, mdot, omega = datalist
    niter = shape(mdot)[0] ; nt = size(t)
    clf()
    scatter(mdot, omega * r**1.5, c=t, s=1.)
    xlabel(r'$\dot{m}$') ; ylabel(r'$\Omega/\Omega_{\rm K}$')
    xlim(mdot[w].min(), mdot[w].max()) ; ylim(omega[w].min() * r**1.5, omega[w].max() * r**1.5)
    savefig(infile+"_O.png")
    Ldisc = mdot/r/8./pi
    clf()
    fig = figure()
    plot(t, L, 'k-')
    plot(t, Ldisc, 'r:')
    xlabel(r'$t$, s') ; ylabel(r'$\dot{L}/L_{\rm Edd}$') ; xlim(trange[0],trange[1]) ; ylim(minimum(L,Ldisc)[w].min(), maximum(L, Ldisc)[w].max())
    fig.set_size_inches(10, 3)
    savefig(infile+"_lBL.png")
    close("all")

def spec_sequential(infile = 'slabout', trange = [0.1, 1e10], binning = 100, ifplot = True, logbinning = False, simfilter = None, cotest = False):
    '''
    makes spectra and cross-spectra out of the blslab output
    reads the entries one by one, thus avoiding memory issues
    binning, if set, should be the number of frequency bins 
    simfilter = [N1, N2]  sets the number range of the files used in the simulation
    cotest (boolean) is used to test the correct work of the covariance analysis
    '''
    keys = hdf.keyshow(infile+'.hdf5')
    nsims = size(keys)-1 # one key points to globals

    if simfilter is not None:
        keys = keys[simfilter[0]:simfilter[1]]
        nsims  = size(keys)-1
    
    for k in arange(nsims):
        t, datalist = hdf.read(infile+'.hdf5', 0, entry = keys[k])
        L, M, mdot, orot = datalist
        if k == 0:
            nt = size(t) ;  tspan = t.max() - t.min() 
            dt = tspan / np.double(nt)
            print("dt = "+str(dt)+"\n")
            #frequencies:
            freq = np.fft.rfftfreq(nt, d=dt)
            print("no of freqs = "+str(size(freq)))
            print("nt = "+str(nt))
            nf = size(freq)
            mdot_PDS_av = zeros(nf, dtype = double)
            mdot_PDS_std = zeros(nf, dtype = double)
            orot_PDS_av = zeros(nf, dtype = double)
            orot_PDS_std = zeros(nf, dtype = double)
            l_PDS_av = zeros(nf, dtype = double)
            l_PDS_std = zeros(nf, dtype = double)
            o_cross_av = zeros(nf, dtype = complex)
            o_dcross_im = zeros(nf, dtype = double)
            o_dcross_re = zeros(nf, dtype = double)
            l_cross_av = zeros(nf, dtype = complex)
            l_dcross_im = zeros(nf, dtype = double)
            l_dcross_re = zeros(nf, dtype = double)
        # Fourier images
        mdot_f=2.*rfft(mdot-mdot.mean())/mdot.sum()  # last axis is the FFT by default
        #        lBL_f=2.*fft(lBL_demean)/lBL.sum()
        if cotest:
            mdotfun = interp1d(t, mdot, bounds_error=False, fill_value = mdot.mean())
            tlag = 1e-2
            orot = 1e-3 * mdotfun(t+tlag)
        orot_f = 2.*rfft(orot-orot.mean())/orot.sum()
        l_f = 2.*rfft(L-L.mean())/L.sum()
        o_cross = mdot_f * conj(orot_f)
        mdot_PDS = abs(mdot_f)**2 ; orot_PDS = abs(orot_f)**2 ; l_PDS = abs(l_f)**2
        mdot_PDS_av += mdot_PDS ; mdot_PDS_std += mdot_PDS**2
        orot_PDS_av += orot_PDS ; orot_PDS_std += orot_PDS**2
        l_PDS_av += l_PDS ; l_PDS_std += l_PDS**2
        o_cross = copy(mdot_f * conj(orot_f))
        l_cross = copy(mdot_f * conj(l_f))
        o_cross_av += o_cross ;     l_cross_av += l_cross
        o_dcross_im += o_cross.imag**2 ;   o_dcross_re += o_cross.real**2 # dispersions of imaginary and real components
        l_dcross_im += l_cross.imag**2 ;   l_dcross_re += l_cross.real**2

    # mean values:
    mdot_PDS_av /= (nsims) ; orot_PDS_av /= (nsims) ; l_PDS_av /= (nsims)
    o_cross_av /= (nsims) ; l_cross_av /= (nsims)
    # RMS:
    mdot_PDS_std = sqrt(mdot_PDS_std / (nsims) - mdot_PDS_av**2) / sqrt(double(nsims-1))
    orot_PDS_std = sqrt(orot_PDS_std / (nsims) - orot_PDS_av**2) / sqrt(double(nsims-1))
    l_PDS_std = sqrt(l_PDS_std / (nsims) - l_PDS_av**2) / sqrt(double(nsims-1))

    o_coherence = abs(o_cross)/sqrt(orot_PDS_av*mdot_PDS_av)
    o_phaselag = angle(o_cross)
    o_dcross_im = sqrt(o_dcross_im / (nsims) - o_cross_av.imag**2) / sqrt(double(nsims-1))
    o_dcross_re = sqrt(o_dcross_re / (nsims) - o_cross_av.real**2) / sqrt(double(nsims-1))
    o_dcoherence = 0.5 * ((o_dcross_im * abs(o_cross_av.imag) + o_dcross_re * abs(o_cross_av.real))/abs(o_cross_av)**2
                          + orot_PDS_std / orot_PDS_av + mdot_PDS_std / mdot_PDS_std / orot_PDS_av) * o_coherence
    o_dphaselag = (o_dcross_im * abs(o_cross_av.real) + o_dcross_re * abs(o_cross_av.imag))/abs(o_cross_av)
    l_coherence = abs(l_cross)/sqrt(l_PDS_av*mdot_PDS_av)
    l_phaselag = angle(l_cross)
    l_dcross_im = sqrt(l_dcross_im / (nsims) - l_cross_av.imag**2) / sqrt(double(nsims-1))
    l_dcross_re = sqrt(l_dcross_re / (nsims) - l_cross_av.real**2) / sqrt(double(nsims-1))
    l_dcoherence = 0.5 * ((l_dcross_im * abs(l_cross_av.imag) + l_dcross_re * abs(l_cross_av.real))/abs(l_cross_av)**2
                          + l_PDS_std / l_PDS_av + mdot_PDS_std / mdot_PDS_std / orot_PDS_av)* l_coherence
    
    w = freq > 0.
    
    if cotest:   
        clf()
        fig, ax = subplots(2,1)
        ax[0].errorbar(freq[w], phaselag[w],
                       yerr =  dphaselag[w], fmt = 'k.')
        ax[0].plot(freq[w], ((pi-2.*pi*tlag*freq[w]) % (2.*pi))-pi,
                   'c-', linewidth = 3)
        ax[0].plot(freq[w], freq[w]*0., 'r-')
        ax[0].plot(freq[w], freq[w]*0.+pi/2., 'r-')
        ax[0].plot(freq[w], freq[w]*0.+pi, 'r-')
        ax[0].set_xscale('log')  ; ax[0].set_ylabel(r'$\Delta \varphi$', fontsize=18)
        ax[1].errorbar(freq[w], coherence[w], yerr = dcoherence[w], fmt = 'k.')
        ax[1].set_xscale('log') ; ax[0].set_ylim(-pi,pi) ; ax[1].set_ylim(-pi,pi)
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

    # binning:
    if binning is not None:
        nbins = binning
        npoints = zeros(nbins, dtype = integer)
        mdot_PDS_avbin = zeros(nbins) ; orot_PDS_avbin = zeros(nbins) ; l_PDS_avbin = zeros(nbins)
        mdot_dPDS_bin = zeros(nbins) ; orot_dPDS_bin = zeros(nbins) ; l_dPDS_bin = zeros(nbins)
        # --variation within the bin
        mdot_dPDS_ensemble = zeros(nbins) ; orot_dPDS_ensemble = zeros(nbins) ; l_dPDS_ensemble = zeros(nbins)
        # --mean uncertainty
        o_cross_avbin = zeros(nbins, dtype = complex) ; o_coherence_avbin = zeros(nbins)
        o_dcross_im_ensemble = zeros(nbins) ; o_dcross_re_ensemble = zeros(nbins)
        o_dcross_im_bin = zeros(nbins) ; o_dcross_re_bin = zeros(nbins)
        #        o_dcoherence_bin = zeros(nbins) # variation within the bin
        #        o_dcoherence_ensemble = zeros(nbins) # mean uncertainty
        #  o_phaselag_bin = zeros(nbins)
        # o_dphaselag_avbin = zeros(nbins) # variation within the bin
        # o_phaselag_stdbin = zeros(nbins) # mean uncertainty
        l_cross_avbin = zeros(nbins, dtype = complex) ; o_coherence_avbin = zeros(nbins)
        l_dcross_im_ensemble = zeros(nbins) ; l_dcross_re_ensemble = zeros(nbins)
        l_dcross_im_bin = zeros(nbins) ; l_dcross_re_bin = zeros(nbins)
        # l_dcoherence_avbin = zeros(nbins) # variation within the bin
        # l_coherence_stdbin = zeros(nbins) # mean uncertainty
        # l_phaselag_avbin = zeros(nbins)
        # l_dphaselag_avbin = zeros(nbins) # variation within the bin
        # l_phaselag_stdbin = zeros(nbins) # mean uncertainty
       
        print(str(nbins)+" bins\n")
        freq1 =1./tspan/2. ; freq2=freq1*np.double(nt)/2.
        x = arange(nbins+1)/np.double(nbins)
        if(logbinning):
            df = (freq2-freq1) / double(nf) ; kfactor = 5 # k restricts the number of points per unit freqbin
            binfreq = logABC(x, [freq1, freq2], kfactor * double(nbins)/ double(nf))
            # (freq2-freq1) * exp(kfactor * double(nbins)/double(nf) * (x-1.))+freq1
            print(binfreq)
            ii =input("BF")
            # (freq2/freq1)**((np.arange(nbins+1)/np.double(nbins)))*freq1
            #  binfreq[0] = 0.
        else:
            binfreq=(freq2-freq1)*x+freq1
        for kb in arange(nbins):
            freqrange=(freq>=binfreq[kb])&(freq<binfreq[kb+1])
            npoints[kb] = freqrange.sum()
            #    print(npoints[kb])
            mdot_PDS_avbin[kb]=mdot_PDS_av[freqrange].mean() 
            mdot_dPDS_ensemble[kb]=mdot_PDS_std[freqrange].mean() 
            mdot_dPDS_bin[kb]=mdot_PDS_av[freqrange].std() / sqrt(double(npoints[kb]-1))
            orot_PDS_avbin[kb]=orot_PDS_av[freqrange].mean() 
            orot_dPDS_ensemble[kb]=orot_PDS_std[freqrange].mean() 
            orot_dPDS_bin[kb]=orot_PDS_av[freqrange].std() / sqrt(double(npoints[kb]-1))
            l_PDS_avbin[kb]=l_PDS_av[freqrange].mean() # intra-bin variations
            l_dPDS_ensemble[kb]=l_PDS_std[freqrange].mean() # ensemble variations
            l_dPDS_bin[kb]=l_PDS_av[freqrange].std() / sqrt(double(npoints[kb]-1))
            o_cross_avbin[kb]=o_cross_av[freqrange].mean()
            o_dcross_im_ensemble[kb] = o_dcross_im[freqrange].mean()
            o_dcross_im_bin[kb] = o_cross_av.imag[freqrange].std() / sqrt(double(npoints[kb]-1))
            o_dcross_re_ensemble[kb] = o_dcross_re[freqrange].mean()
            o_dcross_re_bin[kb] = o_cross_av.real[freqrange].std() / sqrt(double(npoints[kb]-1))
            l_cross_avbin[kb]=l_cross_av[freqrange].mean()
            l_dcross_im_ensemble[kb] = l_dcross_im[freqrange].mean()
            l_dcross_im_bin[kb] = l_cross_av.imag[freqrange].std() / sqrt(double(npoints[kb]-1))
            l_dcross_re_ensemble[kb] = l_dcross_re[freqrange].mean()
            l_dcross_re_bin[kb] = l_cross_av.real[freqrange].std() / sqrt(double(npoints[kb]-1))

        o_phaselag_avbin = angle(o_cross_avbin)      
        o_coherence_avbin = abs(o_cross_avbin)/sqrt(mdot_PDS_avbin * orot_PDS_avbin)
        o_dphaselag_ensemble = ((sin(o_phaselag_avbin) * o_dcross_re_ensemble/abs(o_cross_avbin))**2 +
                                (cos(o_phaselag_avbin) * o_dcross_im_ensemble/abs(o_cross_avbin))**2)
        o_dphaselag_bin = ((sin(o_phaselag_avbin) * o_dcross_re_bin/abs(o_cross_avbin))**2 +
                           (cos(o_phaselag_avbin) * o_dcross_im_bin/abs(o_cross_avbin))**2)
        o_dcoherence_ensemble = o_coherence_avbin * ( (o_cross_avbin.real * o_dcross_re_ensemble +
                                                       o_cross_avbin.imag * o_dcross_im_ensemble) / abs(o_cross_avbin)**2+
                                                      0.5 * orot_dPDS_ensemble / orot_PDS_avbin +
                                                      0.5 * mdot_dPDS_ensemble / mdot_PDS_avbin) 
        o_dcoherence_bin = o_coherence_avbin * ( (o_cross_avbin.real * o_dcross_re_bin +
                                                  o_cross_avbin.imag * o_dcross_im_bin) / abs(o_cross_avbin)**2+
                                                 0.5 * orot_dPDS_bin / orot_PDS_avbin +
                                                 0.5 * mdot_dPDS_bin / mdot_PDS_avbin) 
        l_phaselag_avbin = angle(l_cross_avbin)
        l_coherence_avbin = abs(l_cross_avbin)/sqrt(mdot_PDS_avbin * l_PDS_avbin)
        l_dphaselag_ensemble = ((sin(l_phaselag_avbin) * l_dcross_re_ensemble/abs(l_cross_avbin))**2 +
                                (cos(l_phaselag_avbin) * l_dcross_im_ensemble/abs(l_cross_avbin))**2)
        l_dphaselag_bin = ((sin(l_phaselag_avbin) * l_dcross_re_bin/abs(l_cross_avbin))**2 +
                           (cos(l_phaselag_avbin) * l_dcross_im_bin/abs(l_cross_avbin))**2)       
        l_dcoherence_ensemble = l_coherence_avbin * ( (l_cross_avbin.real * l_dcross_re_ensemble +
                                                       l_cross_avbin.imag * l_dcross_im_ensemble) / abs(l_cross_avbin)**2+
                                                      0.5 * l_dPDS_ensemble / l_PDS_avbin +
                                                      0.5 * mdot_dPDS_ensemble / mdot_PDS_avbin) 
        l_dcoherence_bin = l_coherence_avbin * ( (l_cross_avbin.real * l_dcross_re_bin +
                                                  l_cross_avbin.imag * l_dcross_im_bin) / abs(l_cross_avbin)**2+
                                                 0.5 * l_dPDS_bin / l_PDS_avbin +
                                                 0.5 * mdot_dPDS_bin / mdot_PDS_avbin) 
       
        # ASCII output:
        fout = open(infile+'_osp.dat', 'w')
        fout.write("# f1  f2  mdot dmdot d1mdot Omega dOmega d1Omega coherence dcoherence d1coherence  phaselag dphaselag d1phaselag npoints\n")
        for k in arange(nbins):
            fout.write(str(binfreq[k])+" "+str(binfreq[k+1])+" "
                       +str(mdot_PDS_avbin[k])+" "+str(mdot_dPDS_ensemble[k])+" "+str(mdot_dPDS_bin[k])+" "
                       +str(orot_PDS_avbin[k])+" "+str(orot_dPDS_ensemble[k])+" "+str(orot_dPDS_bin[k])+" "
                       +str(o_coherence_avbin[k])+" "+str(o_dcoherence_ensemble[k])+" "+str(o_dcoherence_bin[k])+" "
                       +str(o_phaselag_avbin[k])+" "+str(o_dphaselag_ensemble[k])+" "+str(o_dphaselag_bin[k])+" "
                       +str(npoints[k])+"\n")
            fout.flush()
        fout.close()
        fout = open(infile+'_lsp.dat', 'w')
        fout.write("# f1  f2  mdot dmdot d1mdot LBL dLBL d1LBL coherence dcoherence d1coherence  phaselag dphaselag d1phaselag npoints\n")
        for k in arange(nbins):
            fout.write(str(binfreq[k])+" "+str(binfreq[k+1])+" "
                       +str(mdot_PDS_avbin[k])+" "+str(mdot_dPDS_ensemble[k])+" "+str(mdot_dPDS_bin[k])+" "
                       +str(l_PDS_avbin[k])+" "+str(l_dPDS_ensemble[k])+" "+str(l_dPDS_bin[k])+" "
                       +str(l_coherence_avbin[k])+" "+str(l_dcoherence_ensemble[k])+" "+str(l_dcoherence_bin[k])+" "
                       +str(l_phaselag_avbin[k])+" "+str(l_dphaselag_ensemble[k])+" "+str(l_dphaselag_bin[k])+" "
                       +str(npoints[k])+"\n")
            fout.flush()
        fout.close()
        if ifplot:
            plots.pds(binfreq, mdot_PDS_avbin, mdot_dPDS_ensemble, mdot_dPDS_bin,
                              orot_PDS_avbin, orot_dPDS_ensemble, orot_dPDS_bin, npoints, outfile = 'o_pdss')
            plots.pds(binfreq, mdot_PDS_avbin, mdot_dPDS_ensemble, mdot_dPDS_bin,
                              l_PDS_avbin, l_dPDS_ensemble, l_dPDS_bin, npoints, outfile = 'l_pdss')

            plots.coherence(binfreq, o_coherence_avbin, o_dcoherence_ensemble, o_dcoherence_bin,
                            o_phaselag_avbin, o_dphaselag_ensemble, o_dphaselag_bin, npoints, outfile = 'o_cobin')
            plots.coherence(binfreq, l_coherence_avbin, l_dcoherence_ensemble, l_dcoherence_bin,
                            l_phaselag_avbin, l_dphaselag_ensemble, l_dphaselag_bin, npoints, outfile = 'l_cobin')

    
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

