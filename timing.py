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
    if oldscipy:
        bcoef = fsolve(Bfun, args = (1./rhs), xtol = 1e-5, x0 = log(frange[1]/frange[0]))
        print(bcoef)
        # print(coefinfo)
    else:
        sol = root_scalar(Bfun, args = (1./rhs), rtol = 1e-5, x0 = log(frange[1]/frange[0]), x1 = -log(rhs))
        print(sol)
        bcoef = sol.root
    print("basic B = "+str(log(frange[1]/frange[0])))
    print("basic A = "+str(frange[0]))
    acoef = rhs/bcoef * (frange[1]-frange[0])
    ccoef = frange[0] - acoef
    print("acoef = "+str(acoef))
    print("bcoef = "+str(bcoef))
    print("ccoef = "+str(ccoef))
    
    return acoef * exp( bcoef * x) + ccoef
     
def spec_onevar(varno, infile = 'slabout', trange = [0.0, 1e10], binning = 100, logbinning = False, simfilter = None, cotest = False):
    '''
    makes spectra and cross-spectra for a given parameter of the output
    varno is the number of the output variable: 0 is luminosity, 1 is mass, 2 is mdot, 3 is orot
    '''
    keys = hdf.keyshow(infile)
    nsims = size(keys)-1 # one key points to globals

    if simfilter is not None:
        keys = keys[simfilter[0]:simfilter[1]]
        nsims  = size(keys)-1
    for k in arange(nsims):
        t, datalist = hdf.read(infile, 0, entry = keys[k])
        v = datalist[varno] ; mdot = datalist[2]
        if trange is not None:
            w = (t>trange[0]) * (t<trange[1])
            t=t[w] ; v=v[w] ; mdot=mdot[w]
        if k == 0:
            nt = size(t) ;  tspan = t.max() - t.min() 
            dt = tspan / double(nt)
            print("dt = "+str(dt)+"\n")
            #frequencies:
            freq = rfftfreq(nt, d=dt)
            print("no of freqs = "+str(size(freq)))
            print("nt = "+str(nt))
            nf = size(freq)
            mdot_PDS_av = zeros(nf, dtype = double)
            mdot_PDS_std = zeros(nf, dtype = double)
            v_PDS_av = zeros(nf, dtype = double)
            v_PDS_std = zeros(nf, dtype = double)
            v_cross_av = zeros(nf, dtype = complex)
            v_dcross_im = zeros(nf, dtype = double)
            v_dcross_re = zeros(nf, dtype = double)
        # Fourier images
        mdot_f=2.*rfft(mdot-mdot.mean())/mdot.sum()  # last axis is the FFT by default
        v_f = 2.*rfft(v-v.mean())/v.sum()
        mdot_PDS = abs(mdot_f)**2 ; v_PDS = abs(v_f)**2
        mdot_PDS_av += mdot_PDS ; mdot_PDS_std += mdot_PDS**2
        v_PDS_av += v_PDS ; v_PDS_std += v_PDS**2
        v_cross = copy(mdot_f * conj(v_f))
        v_cross_av += v_cross 
        v_dcross_im += v_cross.imag**2 ;   v_dcross_re += v_cross.real**2 # dispersions of imaginary and real components
    # mean values:
    mdot_PDS_av /= (nsims) ; v_PDS_av /= (nsims) ;    v_cross_av /= (nsims)
    # RMS:
    mdot_PDS_std = sqrt(mdot_PDS_std / double(nsims) - mdot_PDS_av**2) / sqrt(double(nsims-1))
    v_PDS_std = sqrt(v_PDS_std / double(nsims) - v_PDS_av**2) / sqrt(double(nsims-1))
    v_coherence = abs(v_cross)/sqrt(v_PDS_av*mdot_PDS_av)
    v_phaselag = angle(v_cross)
    v_dcross_im = sqrt(v_dcross_im / double(nsims) - v_cross_av.imag**2) / sqrt(double(nsims-1))
    v_dcross_re = sqrt(v_dcross_re / double(nsims) - v_cross_av.real**2) / sqrt(double(nsims-1))
    v_dcoherence = 0.5 * ((v_dcross_im * abs(v_cross_av.imag) + v_dcross_re * abs(v_cross_av.real))*2./abs(v_cross_av)**2
                          + v_PDS_std / v_PDS_av + mdot_PDS_std / mdot_PDS_av) * v_coherence
    v_dphaselag = (v_dcross_im * abs(v_cross_av.real) + v_dcross_re * abs(v_cross_av.imag))/abs(v_cross_av)**2

    if ifplot:
        w = (freq > 0.)
        clf()
        fig, ax = subplots(2,1)
        ax[0].errorbar(freq[w], v_phaselag[w],
                       yerr =  v_dphaselag[w], fmt = 'k.')
        ax[0].plot(freq[w], arctan(2.*pi*freq[w]*tdepl*tscale), 'c-', linewidth = 3)
        ax[0].plot(freq[w], freq[w]*0., 'r-')
        ax[0].plot(freq[w], freq[w]*0.+pi/2., 'r-')
        ax[0].plot(freq[w], freq[w]*0.+pi, 'r-')
        ax[0].set_xscale('log')  ; ax[0].set_ylabel(r'$\Delta \varphi$', fontsize=18)
        ax[1].errorbar(freq[w], v_coherence[w], yerr = v_dcoherence[w], fmt = 'k.')
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

############################################################################   
# Fourier namespace (does not deserve the name "class")
class fourier:
    def define(self, nt, dt):
        self.freq = rfftfreq(nt, d=dt)
        self.nf = size(self.freq)
    def FT(self, lc):
        self.mean = lc.mean()
        self.tilde = 2.*rfft(lc-self.mean)/lc.sum() # Miyamoto norm
        self.pds = double(abs(self.tilde)**2)
    def crossT(self, lcref_fft):
        self.cross = conj(self.tilde) * lcref_fft

class binobject:
    def __init__(self):
        print('binobject')
        
    def interpolmake(self, f1, f2, iniar, diniar):
        # makes a binned array out of initial "iniar". Calculates two sets of uncertainties, one
        # averages diniar (ensemble errors) and one intra-bin
        nf1 = size(f1) ; nf2 = size(f2)-1
        ifcomplex = iscomplex(iniar[0])
        self.av = zeros(nf2, dtype = type(iniar[0])) # type could be double or complex
        self.densemble = zeros(nf2, dtype = type(iniar[0])) # type could be double or complex
        self.dbin = zeros(nf2, dtype = type(iniar[0])) # type could be double or complex
        self.npoints = zeros(nf2, dtype = integer)
        for k in arange(nf2):
            w = (f1 > f2[k]) & (f1 < f2[k+1])
            self.npoints[k] = w.sum()
            self.av[k] = iniar[w].mean()
            self.densemble[k] = diniar[w].mean() # mean ensemble error
            if ifcomplex:
                self.dbin[k] = iniar[w].real.std() + iniar[w].imag.std() * 1j
            else:
                self.dbin[k] = iniar[w].std()
                
    def comake(self, pds1, pds2):
        # makes coherence and phase lags out of a cross-spectrum
        #        print(sqrt(double(pds1.av)))
        self.c = abs(self.av) / sqrt(pds1.av * pds2.av)
        self.dc_ensemble = (self.densemble.real * self.av.real + self.densemble.imag * self.av.imag) / sqrt(pds1.av*pds2.av) + abs(self.av) / 2. * (pds1.densemble / pds1.av + pds2.densemble / pds2.av)
        self.dc_bin = (self.dbin.real * self.av.real + self.dbin.imag * self.av.imag) / sqrt(pds1.av*pds2.av) + abs(self.av) / 2. * (pds1.dbin / pds1.av + pds2.dbin / pds2.av)
        self.phlag = angle(self.av)
        self.dphlag_ensemble = sqrt((self.densemble.real/self.av.real)**2 +
                                   (self.densemble.imag/self.av.imag)**2)/(1.+tan(self.phlag)**2)
        self.dphlag_bin = sqrt((self.dbin.real/self.av.real)**2 +
                                   (self.dbin.imag/self.av.imag)**2)/(1.+tan(self.phlag)**2)

def asc_pdsout(freq, pdsobjects, outfile):
    '''
    outputs 
    '''
    fout = open(outfile+'.dat', 'w')
    nf = size(freq)-1 ; nobjects = size(pdsobjects)
    fout.write('#  f1 f2 pds1 d1pds1 d2pds1 pds2 d1pds2 d2pds2 ... npoints \n')
    for k in arange(nf):
        s = str(freq[k])+' '+str(freq[k+1])
        for ko in arange(nobjects):
            s+=' '+str(pdsobjects[ko].av[k])+' '+str(pdsobjects[ko].densemble[k])+' '+str(pdsobjects[ko].dbin[k])
        s+=' '+str(pdsobjects[0].npoints[k])+'\n'
        print(s)
        fout.write(s)
    fout.close()

def asc_coherence(freq, cobjects, outfile):
    fout = open(outfile+'.dat', 'w')
    nf = size(freq)-1 ; nobjects = size(cobjects)
    fout.write('#  f1 f2 co1 d1co1 d2co1 phlag1 d1phlag1 d2phlag1 co2 ... ')
    for k in arange(nf):
        s = str(freq[k])+' '+str(freq[k+1])
        for ko in arange(nobjects):
            s+=' '+str(cobjects[ko].c[k])+' '+str(cobjects[ko].dc_ensemble[k])+' '+str(cobjects[ko].dc_bin[k])
            s+=' '+str(cobjects[ko].phlag[k])+' '+str(cobjects[ko].dphlag_ensemble[k])+' '+str(cobjects[ko].dphlag_bin[k])
        s+=' '+str(cobjects[0].npoints[k])+'\n'
        print(s)
        fout.write(s)
    fout.close()
        
def pdsmerge(fourierlist):
    '''
    makes a mean and std arrays for PDS out of a list of "Fourier" objects
    '''
    nl = size(fourierlist)
    nf = size(fourierlist[0].pds)
    pdssum = zeros(nf, dtype = double) ; pdssqsum = zeros(nf, dtype = double)
    
    for k in arange(nl):
        pdssum += fourierlist[k].pds
        pdssqsum += fourierlist[k].pds**2
    pdsmean = pdssum/double(nl)
    pdsstd = sqrt(pdssqsum/double(nl) - pdsmean**2)
    return pdsmean, pdsstd

def crossmerge(fourierlist):
    '''
    calculates the mean cross-spectrum (complex) and its uncertainties (stored as a complex value)
    '''
    nl = size(fourierlist)
    nf = size(fourierlist[0].pds)
    crosssum = zeros(nf, dtype = complex) ; crosssqsum = zeros(nf, dtype = complex)
    for k in arange(nl):
        crosssum += fourierlist[k].cross
        crosssqsum += fourierlist[k].cross.real**2 + fourierlist[k].cross.imag**2 * 1j
    crossmean = crosssum / double(nl)
    crossstd = sqrt(crosssqsum.real/ double(nl) - crossmean.real**2) + \
               sqrt(crosssqsum.imag/ double(nl) - crossmean.imag**2) * 1j
    return crossmean, crossstd
    
def spec_retrieve(infile, entries):
    '''
    reads several entries from the zarr file, converts them to PDS and calculates mean and dispersions of the PDS, co-spectra and their uncertainties
    infile is a string without extension (should be ".zarr")
    entries is a list or array of numbers
    trange may be of the shape [t1, t2] and sets the time limits for the series
    '''
    mdotsps = [] ; msps = [] ; lsps = [] ; osps = []
    mdotsp = fourier() ; msp = fourier() ; lsp = fourier() ; osp = fourier()
    for k in arange(size(entries)):
        t, datalist = hdf.read(infile, 0, entry = entries[k])
        if k == 0:
                nt = size(t) ; dt = (t.max()-t.min())/double(nt)
        mdotsp.define(nt, dt)
        msp.define(nt, dt)    ;    lsp.define(nt, dt)   ;  osp.define(nt, dt) 
        L, M, mdot, omega = datalist
        mdotsp.FT(mdot)
        mdotsps.append(mdotsp)
        msp.FT(M)
        msp.crossT(mdotsp.tilde)
        msps.append(msp)
        lsp.FT(L) ; lsp.crossT(mdotsp.tilde)
        lsps.append(lsp)
        osp.FT(omega) ; osp.crossT(mdotsp.tilde)
        osps.append(osp)
    return mdotsps, msps, osps, lsps

def spec_parallel(infile, nproc = 2, trange = None, simlimit = None, binning = 100):
    '''
    reads from a zarr file light curves in parallel, calculates and merges PDS and cross-spectra
    '''
    keys = hdf.keyshow(infile)
    nsims = size(keys)-1 # one key points to globals
    if simlimit is not None:
        nsims = minimum(nsims, simlimit)
    entrieslist = keys[:nsims]        
    print(entrieslist)
    nperproc = nsims//nproc
    # entries chunked to fit the number of cores
    entrieslist = [entrieslist[i:(i+nperproc)] for i in range(0,nsims,nperproc)]
    #    print(entrieslist)
    #    ii = input('L')
    t1 = time.time()
    pool = multiprocessing.Pool(processes = nproc)
    spec_retrieve_partial = partial(spec_retrieve, infile)
    res = pool.map(spec_retrieve_partial, entrieslist)
    t2 = time.time()
    # first dimension: processor
    # second dimension: variable
    l = squeeze(asarray(list(res)))
    print(shape(l))
    #    freq = l[0,0].freq
    mdotsps = l[:,0,:].flatten() ; msps = l[:,1,:].flatten() ; osps = l[:,2,:].flatten()
    lsps = l[:,3,:].flatten()
    print(shape(mdotsps))
    freq = mdotsps[0].freq
    nf = size(freq)
    mdot_pds, dmdot_pds = pdsmerge(mdotsps)
    m_pds, dm_pds = pdsmerge(msps)
    o_pds, do_pds = pdsmerge(osps)
    l_pds, dl_pds = pdsmerge(lsps)
    m_c, dm_c = crossmerge(msps)
    o_c, do_c = crossmerge(osps)
    l_c, dl_c = crossmerge(lsps)
    t3 = time.time()
    print("parallel reading took "+str(t2-t1)+"s")
    print("merging took "+str(t3-t2)+"s")
    
    clf()
    errorbar(freq, mdot_pds, yerr = dmdot_pds)
    errorbar(freq, m_pds, yerr = dm_pds)
    xscale('log') ; yscale('log') ; xlim(freq[freq>0.].min(), freq.max())
    savefig('pdstest.png')
    t4 = time.time()
    
    # binning:
    nbins = binning
    npoints = zeros(nbins, dtype = integer)
    mdot_pds_bin = binobject()  ; m_pds_bin = binobject() ; l_pds_bin = binobject()  ;  o_pds_bin = binobject()
    mdot_cross_bin = binobject()  ; m_cross_bin = binobject() ; l_cross_bin = binobject()  ;  o_cross_bin = binobject()
    freq1 = freq[freq>0.].min() ; freq2 = freq.max()
    #  freqbin = (freq2/freq1)**(arange(nbins+1)/double(nbins))*freq1
    kfactor = 5
    x = arange(nbins+1)/double(nbins)
    freqbin = logABC(x, [freq1, freq2], kfactor * double(nbins)/ double(nf))
    freqbin[0] = 0.
    mdot_pds_bin.interpolmake(freq, freqbin, mdot_pds, dmdot_pds)
    m_pds_bin.interpolmake(freq, freqbin, m_pds, dm_pds)
    l_pds_bin.interpolmake(freq, freqbin, l_pds, dl_pds)
    o_pds_bin.interpolmake(freq, freqbin, o_pds, do_pds)
    print(m_pds_bin.av)
    m_cross_bin.interpolmake(freq, freqbin, m_c, dm_c)
    l_cross_bin.interpolmake(freq, freqbin, l_c, dl_c)
    o_cross_bin.interpolmake(freq, freqbin, o_c, do_c)
    m_cross_bin.comake(mdot_pds_bin, m_pds_bin) # calculate coherence and phase lags
    l_cross_bin.comake(mdot_pds_bin, l_pds_bin)
    o_cross_bin.comake(mdot_pds_bin, o_pds_bin)
    t5 = time.time()
    asc_pdsout(freqbin, [mdot_pds_bin, m_pds_bin, l_pds_bin, o_pds_bin], infile+'_pds')
    asc_coherence(freqbin, [m_cross_bin, l_cross_bin, o_cross_bin], infile+'_cross')
    plots.object_pds(freqbin, [mdot_pds_bin, m_pds_bin, l_pds_bin, o_pds_bin], infile+'_pds')
    plots.object_coherence(freqbin, [m_cross_bin], infile+'_mcoherence')
    plots.object_coherence(freqbin, [l_cross_bin], infile+'_lcoherence')
    plots.object_coherence(freqbin, [o_cross_bin], infile+'_ocoherence')
    t6 = time.time()
    print("binning "+str(t5-t4)+"s")
    print("outputs "+str(t6-t5)+"s")    

def object_pds_stored(infile, nvar = 0):
    lines = np.loadtxt(infile+".dat")
    f1 = lines[:,0] ; f2 = lines[:,1] ; npoints=lines[:, -1]
    nf = size(f1)
    freq = zeros(nf+1)
    freq[:-1] = f1[:] ; freq[-1] = f2[-1]
    q = lines[:, nvar*3+2] ; dq_ensemble = lines[:, nvar*3+3] ; dq_bin = lines[:, nvar*3+4]
    qobj = binobject()
    qobj.av = q ; qobj.densemble = dq_ensemble ; qobj.dbin = dq_bin ; qobj.npoints = npoints
    plots.object_coherence(freq, [qobj], infile)
   
def object_coherence_stored(infile, nvar = 0):
    lines = np.loadtxt(infile+".dat")
    f1 = lines[:,0] ; f2 = lines[:,1] ; npoints=lines[:, -1]
    nf = size(f1)
    freq = zeros(nf+1)
    freq[:-1] = f1[:] ; freq[-1] = f2[-1]
    c = lines[:, nvar*6+2] ; dc_ensemble = lines[:, nvar*6+3] ; dc_bin = lines[:, nvar*3+4]
    phlag  = lines[:, nvar*6+5] ; dphlag_ensemble = lines[:, nvar*6+6] ; dphlag_bin = lines[:, nvar*6+7]
    qobj = binobject()
    qobj.c = c ; qobj.dc_ensemble = dc_ensemble ; qobj.dc_bin = dc_bin ; qobj.npoints = npoints
    qobj.phlag = phlag ; qobj.dphlag_ensemble = dphlag_ensemble ; qobj.dphlag_bin = dphlag_bin
    plots.object_coherence(freq, [qobj], infile)
    
    
################################################################################################



def spec_sequential(infile = 'slabout', trange = [0.1, 1e10],
                    binning = 100, logbinning = False, simfilter = None, cotest = False):
    '''
    makes spectra and cross-spectra out of the blslab output
    reads the entries one by one, thus avoiding memory issues
    binning, if set, should be the number of frequency bins 
    simfilter = [N1, N2]  sets the number range of the files used in the simulation
    cotest (boolean) is used to test the correct work of the covariance analysis
    '''
    keys = hdf.keyshow(infile)
    nsims = size(keys)-1 # one key points to globals

    if simfilter is not None:
        keys = keys[simfilter[0]:simfilter[1]]
        nsims  = size(keys)-1
    
    for k in arange(nsims):
        t, datalist = hdf.read(infile, 0, entry = keys[k])
        L, M, mdot, orot = datalist
        if trange is not None:
            w = (t>trange[0]) * (t<trange[1])
            t=t[w] ; L=L[w] ; M=M[w] ; mdot=mdot[w] ; orot=orot[w]
        if k == 0:
            nt = size(t) ;  tspan = t.max() - t.min() 
            dt = tspan / double(nt)
            print("dt = "+str(dt)+"\n")
            #frequencies:
            freq = rfftfreq(nt, d=dt)
            print("no of freqs = "+str(size(freq)))
            print("nt = "+str(nt))
            nf = size(freq)
            mdot_PDS_av = zeros(nf, dtype = double)
            mdot_PDS_std = zeros(nf, dtype = double)
            orot_PDS_av = zeros(nf, dtype = double)
            orot_PDS_std = zeros(nf, dtype = double)
            l_PDS_av = zeros(nf, dtype = double)
            l_PDS_std = zeros(nf, dtype = double)
            mass_PDS_av = zeros(nf, dtype = double)
            mass_PDS_std = zeros(nf, dtype = double)
            o_cross_av = zeros(nf, dtype = complex)
            o_dcross_im = zeros(nf, dtype = double)
            o_dcross_re = zeros(nf, dtype = double)
            l_cross_av = zeros(nf, dtype = complex)
            l_dcross_im = zeros(nf, dtype = double)
            l_dcross_re = zeros(nf, dtype = double)
            mass_cross_av = zeros(nf, dtype = complex)
            mass_dcross_im = zeros(nf, dtype = double)
            mass_dcross_re = zeros(nf, dtype = double)
        # Fourier images
        mdot_f=2.*rfft(mdot-mdot.mean())/mdot.sum()  # last axis is the FFT by default
        #        lBL_f=2.*fft(lBL_demean)/lBL.sum()
        if cotest:
            mdotfun = interp1d(t, mdot, bounds_error=False, fill_value = mdot.mean())
            tlag = 1e-2
            orot = 1e-3 * mdotfun(t+tlag)
        orot_f = 2.*rfft(orot-orot.mean())/orot.sum()
        l_f = 2.*rfft(L-L.mean())/L.sum()
        mass_f = 2.*rfft(M-M.mean())/M.sum()

        mdot_PDS = abs(mdot_f)**2 ; orot_PDS = abs(orot_f)**2 ; l_PDS = abs(l_f)**2 ; mass_PDS = abs(mass_f)**2
        mdot_PDS_av += mdot_PDS ; mdot_PDS_std += mdot_PDS**2
        orot_PDS_av += orot_PDS ; orot_PDS_std += orot_PDS**2
        mass_PDS_av += mass_PDS ; mass_PDS_std += mass_PDS**2
        l_PDS_av += l_PDS ; l_PDS_std += l_PDS**2
        o_cross = copy(mdot_f * conj(orot_f))
        l_cross = copy(mdot_f * conj(l_f))
        mass_cross = copy(mdot_f * conj(mass_f))
        o_cross_av += o_cross ;     l_cross_av += l_cross ;     mass_cross_av += mass_cross
        o_dcross_im += o_cross.imag**2 ;   o_dcross_re += o_cross.real**2 # dispersions of imaginary and real components
        l_dcross_im += l_cross.imag**2 ;   l_dcross_re += l_cross.real**2
        mass_dcross_im += l_cross.imag**2 ;   mass_dcross_re += l_cross.real**2

    # mean values:
    mdot_PDS_av /= (nsims) ; orot_PDS_av /= (nsims) ; l_PDS_av /= (nsims) ; mass_PDS_av /= (nsims)
    o_cross_av /= (nsims) ; l_cross_av /= (nsims) ; mass_cross_av /= (nsims)
    # RMS:
    mdot_PDS_std = sqrt(mdot_PDS_std / double(nsims) - mdot_PDS_av**2) / sqrt(double(nsims-1))
    orot_PDS_std = sqrt(orot_PDS_std / double(nsims) - orot_PDS_av**2) / sqrt(double(nsims-1))
    l_PDS_std = sqrt(l_PDS_std / double(nsims) - l_PDS_av**2) / sqrt(double(nsims-1))
    mass_PDS_std = sqrt(mass_PDS_std / double(nsims) - mass_PDS_av**2) / sqrt(double(nsims-1))

    o_coherence = abs(o_cross)/sqrt(orot_PDS_av*mdot_PDS_av)
    o_phaselag = angle(o_cross)
    o_dcross_im = sqrt(o_dcross_im / double(nsims) - o_cross_av.imag**2) / sqrt(double(nsims-1))
    o_dcross_re = sqrt(o_dcross_re / double(nsims) - o_cross_av.real**2) / sqrt(double(nsims-1))
    o_dcoherence = 0.5 * ((o_dcross_im * abs(o_cross_av.imag) + o_dcross_re * abs(o_cross_av.real))*2./abs(o_cross_av)**2
                          + orot_PDS_std / orot_PDS_av + mdot_PDS_std / mdot_PDS_av) * o_coherence
    o_dphaselag = (o_dcross_im * abs(o_cross_av.real) + o_dcross_re * abs(o_cross_av.imag))/abs(o_cross_av)**2

    l_coherence = abs(l_cross)/sqrt(l_PDS_av*mdot_PDS_av)
    l_phaselag = angle(l_cross)
    l_dcross_im = sqrt(l_dcross_im / double(nsims) - l_cross_av.imag**2) / sqrt(double(nsims-1))
    l_dcross_re = sqrt(l_dcross_re / double(nsims) - l_cross_av.real**2) / sqrt(double(nsims-1))
    l_dcoherence = 0.5 * ((l_dcross_im * abs(l_cross_av.imag) + l_dcross_re * abs(l_cross_av.real))*2./abs(l_cross_av)**2
                          + l_PDS_std / l_PDS_av + mdot_PDS_std / mdot_PDS_av)* l_coherence
    l_dphaselag = (l_dcross_im * abs(l_cross_av.real) + l_dcross_re * abs(l_cross_av.imag))/abs(l_cross_av)**2
    
    mass_coherence = abs(mass_cross)/sqrt(mass_PDS_av*mdot_PDS_av)
    mass_phaselag = angle(mass_cross)
    mass_dcross_im = sqrt(mass_dcross_im / double(nsims) - mass_cross_av.imag**2) / sqrt(double(nsims-1))
    mass_dcross_re = sqrt(mass_dcross_re / double(nsims) - mass_cross_av.real**2) / sqrt(double(nsims-1))
    mass_dcoherence = 0.5 * ((mass_dcross_im * abs(mass_cross_av.imag) + mass_dcross_re * abs(mass_cross_av.real))*2./abs(mass_cross_av)**2
                          + mass_PDS_std / mass_PDS_av + mdot_PDS_std / mdot_PDS_av)* mass_coherence
    mass_dphaselag = (mass_dcross_im * abs(mass_cross_av.real) + mass_dcross_re * abs(mass_cross_av.imag))/abs(mass_cross_av)**2

    w = freq > 0.
    
    if cotest and ifplot:   
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
        mdot_PDS_avbin = zeros(nbins) ; orot_PDS_avbin = zeros(nbins) ; l_PDS_avbin = zeros(nbins) ; mass_PDS_avbin = zeros(nbins)
        mdot_dPDS_bin = zeros(nbins) ; orot_dPDS_bin = zeros(nbins) ; l_dPDS_bin = zeros(nbins) ; mass_dPDS_bin = zeros(nbins)
        # --variation within the bin
        mdot_dPDS_ensemble = zeros(nbins) ; orot_dPDS_ensemble = zeros(nbins) ; l_dPDS_ensemble = zeros(nbins) ; mass_dPDS_ensemble = zeros(nbins)
        # --mean uncertainty
        o_cross_avbin = zeros(nbins, dtype = complex) ; o_coherence_avbin = zeros(nbins)
        o_dcross_im_ensemble = zeros(nbins) ; o_dcross_re_ensemble = zeros(nbins)
        o_dcross_im_bin = zeros(nbins) ; o_dcross_re_bin = zeros(nbins)
        l_cross_avbin = zeros(nbins, dtype = complex) ; l_coherence_avbin = zeros(nbins)
        l_dcross_im_ensemble = zeros(nbins) ; l_dcross_re_ensemble = zeros(nbins)
        l_dcross_im_bin = zeros(nbins) ; l_dcross_re_bin = zeros(nbins)
        mass_cross_avbin = zeros(nbins, dtype = complex) ; mass_coherence_avbin = zeros(nbins)
        mass_dcross_im_ensemble = zeros(nbins) ; mass_dcross_re_ensemble = zeros(nbins)
        mass_dcross_im_bin = zeros(nbins) ; mass_dcross_re_bin = zeros(nbins)
       
        print(str(nbins)+" bins\n")
        freq1 =1./tspan/2. ; freq2=freq1*double(nt)/2.
        x = arange(nbins+1)/double(nbins)
        if(logbinning):
            df = (freq2-freq1) / double(nf) ; kfactor = 5 # k restricts the number of points per unit freqbin
            binfreq = logABC(x, [freq1, freq2], kfactor * double(nbins)/ double(nf))
            # (freq2-freq1) * exp(kfactor * double(nbins)/double(nf) * (x-1.))+freq1
            print(binfreq)
            #            ii =input("BF")
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
            mass_PDS_avbin[kb]=mass_PDS_av[freqrange].mean() # intra-bin variations
            mass_dPDS_ensemble[kb]=mass_PDS_std[freqrange].mean() # ensemble variations
            mass_dPDS_bin[kb]=mass_PDS_av[freqrange].std() / sqrt(double(npoints[kb]-1))
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
            mass_cross_avbin[kb]=mass_cross_av[freqrange].mean()
            mass_dcross_im_ensemble[kb] = mass_dcross_im[freqrange].mean()
            mass_dcross_im_bin[kb] = mass_cross_av.imag[freqrange].std() / sqrt(double(npoints[kb]-1))
            mass_dcross_re_ensemble[kb] = mass_dcross_re[freqrange].mean()
            mass_dcross_re_bin[kb] = mass_cross_av.real[freqrange].std() / sqrt(double(npoints[kb]-1))

        o_phaselag_avbin = angle(o_cross_avbin)      
        o_coherence_avbin = abs(o_cross_avbin)/sqrt(mdot_PDS_avbin * orot_PDS_avbin)
        o_dphaselag_ensemble = sqrt((sin(o_phaselag_avbin) * o_dcross_re_ensemble/abs(o_cross_avbin))**2 +
                                    (cos(o_phaselag_avbin) * o_dcross_im_ensemble/abs(o_cross_avbin))**2)
        o_dphaselag_bin = sqrt((sin(o_phaselag_avbin) * o_dcross_re_bin/abs(o_cross_avbin))**2 +
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
        l_dphaselag_ensemble = sqrt((sin(l_phaselag_avbin) * l_dcross_re_ensemble/abs(l_cross_avbin))**2 +
                                    (cos(l_phaselag_avbin) * l_dcross_im_ensemble/abs(l_cross_avbin))**2)
        l_dphaselag_bin = sqrt((sin(l_phaselag_avbin) * l_dcross_re_bin/abs(l_cross_avbin))**2 +
                               (cos(l_phaselag_avbin) * l_dcross_im_bin/abs(l_cross_avbin))**2)       
        l_dcoherence_ensemble = l_coherence_avbin * ( (l_cross_avbin.real * l_dcross_re_ensemble +
                                                       l_cross_avbin.imag * l_dcross_im_ensemble) / abs(l_cross_avbin)**2+
                                                      0.5 * l_dPDS_ensemble / l_PDS_avbin +
                                                      0.5 * mdot_dPDS_ensemble / mdot_PDS_avbin) 
        l_dcoherence_bin = l_coherence_avbin * ( (l_cross_avbin.real * l_dcross_re_bin +
                                                  l_cross_avbin.imag * l_dcross_im_bin) / abs(l_cross_avbin)**2+
                                                 0.5 * l_dPDS_bin / l_PDS_avbin +
                                                 0.5 * mdot_dPDS_bin / mdot_PDS_avbin) 
        mass_phaselag_avbin = angle(mass_cross_avbin)
        mass_coherence_avbin = abs(mass_cross_avbin)/sqrt(mdot_PDS_avbin * mass_PDS_avbin)
        mass_dphaselag_ensemble = sqrt((sin(mass_phaselag_avbin) * mass_dcross_re_ensemble/abs(mass_cross_avbin))**2 +
                                    (cos(mass_phaselag_avbin) * mass_dcross_im_ensemble/abs(mass_cross_avbin))**2)
        mass_dphaselag_bin = sqrt((sin(mass_phaselag_avbin) * mass_dcross_re_bin/abs(mass_cross_avbin))**2 +
                               (cos(mass_phaselag_avbin) * mass_dcross_im_bin/abs(mass_cross_avbin))**2)       
        mass_dcoherence_ensemble = mass_coherence_avbin * ( (mass_cross_avbin.real * mass_dcross_re_ensemble +
                                                       mass_cross_avbin.imag * mass_dcross_im_ensemble) / abs(mass_cross_avbin)**2+
                                                      0.5 * mass_dPDS_ensemble / mass_PDS_avbin +
                                                      0.5 * mdot_dPDS_ensemble / mdot_PDS_avbin) 
        mass_dcoherence_bin = l_coherence_avbin * ( (mass_cross_avbin.real * mass_dcross_re_bin +
                                                  mass_cross_avbin.imag * mass_dcross_im_bin) / abs(mass_cross_avbin)**2+
                                                 0.5 * mass_dPDS_bin / mass_PDS_avbin +
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
        fout = open(infile+'_msp.dat', 'w')
        fout.write("# f1  f2  mdot dmdot d1mdot M dM d1M coherence dcoherence d1coherence  phaselag dphaselag d1phaselag npoints\n")
        for k in arange(nbins):
            fout.write(str(binfreq[k])+" "+str(binfreq[k+1])+" "
                       +str(mdot_PDS_avbin[k])+" "+str(mdot_dPDS_ensemble[k])+" "+str(mdot_dPDS_bin[k])+" "
                       +str(mass_PDS_avbin[k])+" "+str(mass_dPDS_ensemble[k])+" "+str(mass_dPDS_bin[k])+" "
                       +str(mass_coherence_avbin[k])+" "+str(mass_dcoherence_ensemble[k])+" "+str(mass_dcoherence_bin[k])+" "
                       +str(mass_phaselag_avbin[k])+" "+str(mass_dphaselag_ensemble[k])+" "+str(mass_dphaselag_bin[k])+" "
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
            plots.coherence(binfreq, mass_coherence_avbin, mass_dcoherence_ensemble, mass_dcoherence_bin,
                            mass_phaselag_avbin, mass_dphaselag_ensemble, mass_dphaselag_bin, npoints, outfile = 'm_cobin')

            
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
        
    nt = size(t) ;  tspan = t.max() - t.min() 
    dt = tspan / double(nt)
    #frequencies:
    freq1 =1./tspan/2. ; freq2=freq1*double(nt)/2.
    freq = fftfreq(nt, dt)
    
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

