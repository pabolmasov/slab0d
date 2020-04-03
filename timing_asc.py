from numpy import *
from numpy.fft import *
import matplotlib
from pylab import *
from scipy.interpolate import interp1d

import os
import glob

import plots


def cartesian_product(x,y):
    '''
    produces a Cartesian (direct) product of two linear arrays
    '''
    nx = size(x) ; ny = size(y)
    z = zeros([nx, ny], dtype = complex)
    for ky in arange(ny):
        z[:,ky] = x[:] * y[ky]
    return z

def spec(infile = 'slabout', nbins = 100, trange = [0.1,1e6],
         mocking = False, logbinning = True, ifplot = True):
    '''
    makes spectra and cross-spectra out of the blslab output
    '''
    # infile has the form t -- mdot -- m -- lBL -- orot
    
    lines = np.loadtxt(infile+".dat")
    t = lines[:,0] ; mdot = lines[:,1] ; m = lines[:,2] ; lBL = lines[:,3] ; orot = lines[:,4]

    if(trange is not None):
        w=np.where((t<trange[1])&(t>trange[0]))
        t=t[w] ; mdot=mdot[w] ; m=m[w] ; lBL=lBL[w] ; orot=orot[w]

    # mocking time shift:
    if mocking:
        mockmfun = interp1d(t, mdot, bounds_error =False, fill_value = 0.)
        deltat = 1.
        mockm = mockmfun(t-deltat)
        lBL = mockm

    nt = np.size(t) ;    tspan = t.max() - t.min() 
    dt = tspan / np.double(nt)
    #frequencies:
    freq1 =1./tspan/2. ; freq2=freq1*np.double(nt)/2.
    freq = np.fft.fftfreq(nt, dt)
    
    # Fourier images:
    mdot_f=2.*fft(mdot-mdot.mean())/mdot.sum() 
    m_f=2.*fft(m-m.mean())/m.sum()
    lBL_f=2.*fft(lBL-lBL.mean())/lBL.sum()
    orot_f = 2.*fft(orot-orot.mean())/orot.sum()

    # PDS and cross-spectra:
    mdot_pds = abs(mdot_f)**2
    m_pds = abs(m_f)**2
    lBL_pds = abs(lBL_f)**2
    orot_pds = abs(orot_f)**2
    mmdot_cross = mdot_f * conj(lBL_f)
    phaselag = angle(mmdot_cross)
    # mmdot_cc = ifft(mmdot_cross) # cross-correlation
            
    # binning:
    if(logbinning):
        binfreq=(freq2/freq1)**((np.arange(nbins+1)/np.double(nbins)))*freq1
    else:
        binfreq=(freq2-freq1)*((np.arange(nbins+1)/np.double(nbins)))+freq1
    
    mdot_pdsbin = zeros(nbins)
    lBL_pdsbin = zeros(nbins)
    mdot_dpdsbin = zeros(nbins)
    lBL_dpdsbin = zeros(nbins)
    mmdot_crossbin = zeros(nbins, dtype = complex)
    dmmdot_crossbin = zeros(nbins)
    phaselag_bin = zeros(nbins)
    dphaselag_bin = zeros(nbins)

    npoints = zeros(nbins)
    
    for kb in arange(nbins):
        freqrange=(freq>=binfreq[kb])&(freq<binfreq[kb+1])
        npoints[kb] = freqrange.sum()
        mdot_pdsbin[kb]=mdot_pds[freqrange].mean() 
        lBL_pdsbin[kb]=lBL_pds[freqrange].mean() 
        mdot_dpdsbin[kb]=mdot_pds[freqrange].std()
        lBL_dpdsbin[kb]=lBL_pds[freqrange].std()
        mmdot_crossbin[kb] = mmdot_cross[freqrange].mean()
        dmmdot_crossbin[kb] = mmdot_cross[freqrange].std()
        phaselag_bin[kb] = phaselag[freqrange].mean()
        dphaselag_bin[kb] = phaselag[freqrange].std()
        # /np.sqrt(np.double(freqrange.sum())-1.)
    
#    w = npoints>=2.

    # ASCII output:
    fout = open(infile+'_sp.dat', 'w')
    fout.write("# f1  f2  mdot dmdot  lBL dlBL  Re(cross) Im(cross) dcross npoints\n")
    for k in arange(nbins):
        fout.write(str(binfreq[k])+" "+str(binfreq[k+1])+" "
                   +str(mdot_pdsbin[k])+" "+str(mdot_dpdsbin[k])+" "
                   +str(lBL_pdsbin[k])+" "+str(lBL_dpdsbin[k])+" "
                   +str(real(mmdot_crossbin[k]))+" "+str(imag(mmdot_crossbin[k]))+" "
                   +str(dmmdot_crossbin[k])+" "+str(npoints[k])+"\n")
    fout.close()
    # graphic output:
    if ifplot:
        plots.pds(binfreq, mdot_pdsbin, mdot_dpdsbin, lBL_pdsbin, lBL_dpdsbin, npoints)
        #        plots.phaselag(binfreq, phaselag_bin, dphaselag_bin, npoints)
        plots.coherence(binfreq, mmdot_crossbin, dmmdot_crossbin,
              mdot_pdsbin, mdot_dpdsbin, lBL_pdsbin, lBL_dpdsbin,
              npoints)

def bispec(prefix = 'slabout', nbins = 100, logbinning = False):
    '''
    makes a bispectrum from a collection of light curves
    '''

    filelist = np.sort(glob.glob( os.path.join("%s[0-9][0-9][0-9][0-9][0-9][0-9].dat"%prefix) ) )
    nf = size(filelist)
    nf = 3
    
    for kf in arange(nf):
        infile = (filelist[kf])[:-4]
        print("reading "+infile)
        lines = np.loadtxt(infile+".dat")
        t = lines[:,0] ; mdot = lines[:,1] ; m = lines[:,2] ; lBL = lines[:,3] ; orot = lines[:,4]
        if (kf == 0):
            nt = np.size(t) ;    tspan = t.max() - t.min() 
            dt = tspan / np.double(nt)
            #frequencies:
            freq1 =1./tspan/2. ; freq2=freq1*np.double(nt)/2.
            freq = np.fft.fftfreq(nt, dt)
            # binning:
            if(logbinning):
                binfreq=(freq2/freq1)**((np.arange(nbins+1)/np.double(nbins)))*freq1
            else:
                binfreq=(freq2-freq1)*((np.arange(nbins+1)/np.double(nbins)))+freq1
            bsp_l = zeros([nbins, nbins], dtype=complex)
            bsp_m = zeros([nbins, nbins], dtype=complex)
            bspnorm_l = zeros([nbins, nbins]) ;  bspnorm_m = zeros([nbins, nbins])
            bspnorm1_l = zeros([nbins, nbins]) ;  bspnorm1_m = zeros([nbins, nbins])
        # Fourier images:
        m_f=2.*fft(mdot-mdot.mean())/mdot.sum() # mdot
        l_f=2.*fft(lBL-lBL.mean())/lBL.sum() # LBL
        for f_k in arange(nbins):
            freqrange_k=(freq>=binfreq[f_k])&(freq<binfreq[f_k+1])
            if freqrange_k.sum() <= 0:
                continue
            freqk = freq[freqrange_k]
            for f_l in arange(nbins):
                freqrange_l=(freq>=binfreq[f_l])&(freq<binfreq[f_l+1])
                if freqrange_l.sum() <= 0:
                    continue
                freql = freq[freqrange_l]
                freq2k, freq2l = meshgrid(freqk, freql)
                n2l, n2k = meshgrid(where(freqrange_l), where(freqrange_k)) # indices of the corresponding elements
                nsum = n2k + n2l # position of the frequency sum
                insideflag =  nsum < nt # whether we are inside the integration domain
                nsum = minimum(nsum, nt-1) # within the integration domain
                tmp_bsp_l = cartesian_product(l_f[freqrange_k], l_f[freqrange_l])
                #                tmp_bsp_l *= conj(l_f[n2k + n2l]) * insideflag
                tmp_bsp_m = cartesian_product(m_f[freqrange_k], m_f[freqrange_l])
                #                tmp_bsp_m *= conj(m_f[n2k + n2l]) * insideflag
                bsp_l[f_k, f_l] += ( tmp_bsp_l * conj(l_f[nsum]) * insideflag).sum()
                bsp_m[f_k, f_l] += ( tmp_bsp_m * conj(m_f[nsum]) * insideflag).sum()
                bspnorm_l[f_k, f_l] += (abs(tmp_bsp_l)**2).sum()
                bspnorm_m[f_k, f_l] += (abs(tmp_bsp_m)**2).sum()
                bspnorm1_l[f_k, f_l] += (abs(l_f[n2k+n2l] * insideflag)**2).sum() 
                bspnorm1_m[f_k, f_l] += (abs(m_f[n2k+n2l] * insideflag)**2).sum() 

    bicoherence_l = abs(bsp_l)**2/bspnorm_l / bspnorm1_l # see Kim & Powers (1979), or Arur & Maccarone (2020)
    bicoherence_m = abs(bsp_m)**2/bspnorm_m / bspnorm1_m
    biphase_l = angle(bsp_l)+pi ; biphase_m = angle(bsp_m)+pi
    w0 = (bspnorm_m * bspnorm_l * bspnorm1_m * bspnorm1_l) <= 0.
    bicoherence_l[w0] = 0. ;    bicoherence_m[w0] = 0.
    biphase_l[w0] = 0. ;    biphase_m[w0] = 0.
    # ASCII output
    fout = open(prefix+'_bisp.dat', 'w')
    fout.write('#  freq1start freq1finish   freq2start freq2finish bicoherence_l bicoherence_m biphase_l biphase_m \n')
    for f_k in arange(nbins):
        for f_l in arange(nbins):
            fout.write(str(binfreq[f_k])+" "+str(binfreq[f_k+1])+" "+str(binfreq[f_l])+" "+str(binfreq[f_l+1])+" "+\
                       str(bicoherence_l[f_k, f_l])+" "+str(bicoherence_m[f_k, f_l])+" "+\
                       str(biphase_l[f_k, f_l])+" "+str(biphase_m[f_k, f_l])+"\n")
            fout.flush()
    fout.close()
    freqc = (binfreq[1:]+binfreq[:-1])/2.
    plots.biplot(binfreq, bicoherence_l, biphase_l, outname = 'biplot_l')
    plots.biplot(binfreq, bicoherence_m, biphase_m, outname = 'biplot_m')         
        
def multispec(prefix = 'slabout', recalc = False, nbins = 100):
    '''
    collects (or recalculates) all the PDS files matching a certain template, sums the PDSs and makes a joint plot
    '''
    filelist = np.sort(glob.glob( os.path.join("%s[0-9][0-9][0-9][0-9][0-9][0-9].dat"%prefix) ) )
    nf = size(filelist)
    
    for kf in arange(nf):
        if recalc:
            print("infile = "+(filelist[kf])[:-4])
            spec(infile = (filelist[kf])[:-4], nbins = nbins, logbinning = True, ifplot = False)
        # reading the input data
        lines = loadtxt((filelist[kf])[:-4]+'_sp.dat')
        #        print(shape(lines))
        #        ii =input('lines')
        freq1 = lines[:,0] ; freq2 = lines[:,1]
        mdot = lines[:,2] ; dmdot = lines[:,3]
        l = lines[:,4] ; dl = lines[:,5]
        cross = lines[:,6] + 1j * lines[:,7]
        dcross = lines[:,8] ; npoints = lines[:,9]
        print("file "+(filelist[kf])[:-4]+"_sp.dat successfully read \n")
        if kf == 0:
            nx = size(freq1)
            mean_mdot = mdot ; disp_mdot = mdot**2
            mean_l = l ; disp_l = l**2
            mean_cross = cross ; disp_cross = abs(cross)**2
            mean_dmdot = dmdot ; disp_dmdot = dmdot**2
            mean_dl = dl ; disp_dl = dl**2
            mean_dcross = dcross ; disp_dcross = dcross**2 
        else:
            if nx != size(freq1):
                print("array lengths do not match")
                print("nx = "+str(nx)+" or "+str(size(freq1))+"?")
                return 1
            mean_mdot += mdot ; disp_mdot += mdot**2
            mean_l += l ; disp_l += l**2
            mean_cross += cross ; disp_cross += abs(cross)**2
            mean_dmdot += dmdot ; disp_dmdot += dmdot**2
            mean_dl += dl ; disp_dl += dl**2
            mean_dcross += dcross ; disp_dcross += dcross**2
            
    mean_mdot /= double(nf)  ; mean_l /= double(nf) ; mean_cross /= double(nf) + 0j
    mean_dmdot /= double(nf)  ; mean_dl /= double(nf) ; mean_dcross /= double(nf) 
    disp_mdot = disp_mdot/double(nf) - mean_mdot**2
    disp_l = disp_l/double(nf) - mean_l**2
    disp_dmdot = disp_dmdot/double(nf) - mean_dmdot**2
    disp_dl = disp_dl/double(nf) - mean_dl**2
    disp_cross = disp_cross/double(nf) - abs(mean_cross)**2
    disp_dcross = disp_dcross/double(nf) - mean_dcross**2

    dispfactor = sqrt(double(npoints*nf)-1.)
    plots.pds_doubled(freq1, freq2, mean_mdot, mean_dmdot / dispfactor,
                      sqrt(disp_mdot)/dispfactor,
                      mean_l, mean_dl/dispfactor, sqrt(disp_l)/dispfactor, npoints)
    plots.coherence_doubled(freq1, freq2, mean_cross, mean_dcross/dispfactor,
                            sqrt(disp_cross)/dispfactor,
                            mean_mdot, mean_l)
   
def dists(infile = 'slabout', trange = [0.1,1e4]):
    # infile has the form t -- mdot -- m -- lBL -- orot
    
    lines = np.loadtxt(infile+".dat")
    t = lines[:,0] ; mdot = lines[:,1] ; m = lines[:,2] ; lBL = lines[:,3] ; orot = lines[:,4]
    if trange is not None:
        w = (t<trange[1]) & (t> trange[0])
        t = t[w]  ; mdot = mdot[w] ; m=m[w] ; lBL = lBL[w] ; orot = orot[w]
    
    nmdot, binsmdot = histogram(mdot, bins=30, density = False)
    nlBL, binslBL = histogram(lBL, bins=30, density = False)

    binsmdot_c = (binsmdot[1:]+binsmdot[:-1])/2.
    binsmdot_s = (binsmdot[1:]-binsmdot[:-1])
    binslBL_c = (binslBL[1:]+binslBL[:-1])/2.
    binslBL_s = (binslBL[1:]-binslBL[:-1])
   
    dnmdot = sqrt(nmdot) ; dnlBL = sqrt(nlBL)
    fmdot = nmdot/(binsmdot[1:]-binsmdot[:-1]) ;     flBL = nlBL/(binslBL[1:]-binslBL[:-1])
    dfmdot = dnmdot/(binsmdot[1:]-binsmdot[:-1]) ;     dflBL = dnlBL/(binslBL[1:]-binslBL[:-1])

    r=5.0

    clf()
    fig = figure()
    errorbar(binslBL_c, flBL, xerr = binslBL_s/2., yerr = dflBL/sqrt(1.), fmt = 'k.')
    errorbar(binsmdot_c/r, fmdot, xerr = binsmdot_s/2./r, yerr = dfmdot/sqrt(1.), fmt = 'g.')
    xlabel(r'$L/L_{\rm Edd}$', fontsize=18) ; ylabel(r'dN/dL', fontsize=18)
    xscale('log') ;    yscale('log')
    xlim(binslBL_c.min(), (binsmdot_c/r).max())
    tick_params(labelsize=14, length=6, width=1., which='major')
    tick_params(labelsize=14, length=3, width=1., which='minor')
    fig.set_size_inches(5, 6)
    fig.tight_layout()
    savefig('dists.png')
    savefig('dists.eps')
    close('all')
    
 
