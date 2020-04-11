from numpy import *
import numpy.ma as ma
import matplotlib
from pylab import *
from matplotlib.colors import BoundaryNorm
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

from mslab import mscale, tscale, omegaNS, r
from mslab import alpha, tdepl

#################################################################################
# for mslab:
def mconsttests(tar, mar, orot, meq, oeq):
    # mass and momentum test
    clf()
    fig, ax = subplots(2,1)
    ax[0].plot(tar, mar*0. + meq*mscale/1e17 , 'r-')
    ax[0].plot(tar, mar/1e17, 'k-')
    #            ax[0].set_xlabel(r'$t$, s', fontsize=18)
    ax[0].set_ylabel(r'$M$, $10^{17}$g', fontsize=18)
    ax[1].plot(tar, orot*0. + oeq/tscale/2./pi, 'r-')
    ax[1].plot(tar, orot, 'k-')
    ax[1].plot(tar, orot*0. + omegaNS/tscale/2./pi, 'b:')
    ax[1].plot(tar, orot*0. + r**(-1.5)/tscale/2./pi, 'b:')
    ax[1].set_xlabel(r'$t$, s', fontsize=18)
    ax[1].set_ylabel(r'$f$, Hz', fontsize=18)
    ax[0].set_xscale('log')
    ax[1].set_xscale('log')
    #            ax[0].set_xlim(0., efftimescale * 50.)
    #            ax[1].set_xlim(0., efftimescale * 50.)
    ax[0].tick_params(labelsize=14, length=6, width=1., which='major')
    ax[0].tick_params(labelsize=14, length=3, width=1., which='minor')
    ax[1].tick_params(labelsize=14, length=6, width=1., which='major')
    ax[1].tick_params(labelsize=14, length=3, width=1., which='minor')
    fig.set_size_inches(5, 6)
    fig.tight_layout()
    savefig('motest.png')
    savefig('motest.eps')
    close()

def generalcurve(tar, mdot, mar, orot, cthar, loutar, ldisc):
    clf()
    plot(tar, orot, 'k-')
    #    plot(tar, orot*2.*cthar, 'k:')
    plot(tar, orot*0. + omegaNS/tscale/2./pi, 'b:')
    plot(tar, orot*0. + r**(-1.5)/tscale/2./pi, 'b:')
    xlabel(r'$t$, s') ; ylabel(r'$f$, Hz')
    savefig('oplot.png')
    savefig('oplot.eps')
    fig = figure()
    clf()
    plot(tar, loutar+ldisc, 'k-')
    plot(tar, ldisc, 'g-')
    plot(tar, loutar, 'b-')
    plot(tar, orot*0. + mdot/4./pi/r, 'r-')
    plot(tar, orot*0. + mdot/8./pi/r, 'r:')
    xlabel(r'$t$, s', fontsize=18) ; ylabel(r'$L/L_{\rm Edd}$', fontsize=18)
    tick_params(labelsize=14, length=6, width=1., which='major')
    tick_params(labelsize=14, length=4, width=1., which='minor')
    ylim([(loutar+ldisc).min()*0.5, (loutar+ldisc).max()*2.])
    yscale('log')
    fig.set_size_inches(8, 4)
    fig.tight_layout()
    savefig('lplot.png')
    savefig('lplot.eps')
    clf()
    plot(ldisc, loutar, 'k.')
    plot(orot*0. + mdot/8./pi/r, loutar, 'r-')
    plot(ldisc, orot*0. + mdot/8./pi/r, 'r-')
    xlabel(r'$L_{\rm disc}/L_{\rm Edd}$') ; ylabel(r'$L_{\rm BL}/L_{\rm Edd}$')
    ylim([(loutar+ldisc).min()*0.5, (loutar+ldisc).max()*2.])
    #        yscale('log') ; xscale('log')
    savefig('ls.png')
    clf()
    fig = figure()
    plot(loutar+ldisc, orot, 'k,')
    plot(loutar+ldisc, orot*2.*cthar, 'g,')
    plot([(loutar+ldisc).min(), (loutar+ldisc).max()], [ omegaNS/tscale/2./pi,  omegaNS/tscale/2./pi], 'r-')
    plot([(loutar+ldisc).min(), (loutar+ldisc).max()], [ r**(-1.5)/tscale/2./pi, r**(-1.5)/tscale/2./pi], 'r-')
    plot(orot*0. + mdot/4./pi/r, orot*2.*cthar, 'r-')
    xlabel(r'$L/L_{\rm Edd}$', fontsize=18) ; ylabel(r'$f$, Hz', fontsize=18)
    xscale('log') # ; yscale('log')
    ylim([(orot*cthar).min()*0.9, (orot).max()*1.1])
    tick_params(labelsize=14, length=6, width=1., which='major')
    tick_params(labelsize=14, length=4, width=1., which='minor')
    fig.set_size_inches(5, 6)
    fig.tight_layout()
    savefig('lfreq.png')
    savefig('lfreq.eps')
    close('all')

####################################################################################
# for timing:

def pds(binfreq, mdot_pdsbin, mdot_dpdsbin, lBL_pdsbin, lBL_dpdsbin, npoints, outfile = 'pdss'):

    binfreqc = (binfreq[1:]+binfreq[:-1])/2.
    binfreqs = (binfreq[1:]-binfreq[:-1])/2.    
    w = npoints>=2.
    
    clf()
    # plot(freq, mdot_pds, 'k,')
    # plot(freq, lBL_pds, 'r,')
    errorbar(binfreqc[w], mdot_pdsbin[w], xerr = binfreqs[w], yerr = mdot_dpdsbin[w]/sqrt(npoints[w]-1.), fmt = 'ks')
    errorbar(binfreqc[w], lBL_pdsbin[w], xerr = binfreqs[w], yerr = lBL_dpdsbin[w]/sqrt(npoints[w]-1.), fmt = 'rd')
    # plot(freq[freq>0.], 1e-3/((freq[freq>0.]*1000.*4.92594e-06*1.5)**2+1.), 'g-')
    xlim([binfreqc.min()/2., binfreq.max()])
    xscale('log') ; yscale('log')
    xlabel(r'$f$, Hz') ; ylabel(r'$PDS$')
    savefig(outfile + '.png')
    savefig(outfile + '.eps')
    close()

def pds_doubled(freq1, freq2, mdot_pdsbin, mdot_dpdsbin1,  mdot_dpdsbin2, lBL_pdsbin, lBL_dpdsbin1, lBL_dpdsbin2, npoints):
    # for a double error source, variability within the frequency bin and ensemble variations
    freqc = (freq1+freq2)/2.
    freqs = (freq2-freq1)/2.    
    clf()
    errorbar(freqc, mdot_pdsbin, xerr = freqs, yerr = mdot_dpdsbin1, fmt = 'gs', linewidth = 3.)
    errorbar(freqc, mdot_pdsbin, xerr = freqs, yerr = mdot_dpdsbin2, fmt = 'ks')
    errorbar(freqc, lBL_pdsbin, xerr = freqs, yerr = lBL_dpdsbin1, fmt = 'gd', linewidth = 3.)
    errorbar(freqc, lBL_pdsbin, xerr = freqs, yerr = lBL_dpdsbin2, fmt = 'kd')
    xlim([freqc.min()/2., freq2.max()])
    xscale('log') ; yscale('log')
    xlabel(r'$f$, Hz') ; ylabel(r'$PDS$')
    savefig('pdss2.png')
    savefig('pdss2.eps')
    close()
    
def phaselag(binfreq, phaselag_bin, dphaselag_bin, mmdot_crossbin, npoints):
    binfreqc = (binfreq[1:]+binfreq[:-1])/2.
    binfreqs = (binfreq[1:]-binfreq[:-1])/2.    
    w = npoints>=2.
    clf()
    errorbar(binfreqc[w], phaselag_bin[w], xerr = binfreqs[w], yerr = dphaselag_bin[w]/sqrt(npoints[w]-1.), fmt = 'k.')
    errorbar(binfreqc[w], angle(mmdot_crossbin[w]), xerr = binfreqs[w], yerr = dmmdot_crossbin[w]/abs(mmdot_crossbin[w])/sqrt(npoints[w]-1.), fmt = 'g.')
    plot(freq[freq>0.], freq[freq>0.]*0., 'r-')
    plot(freq[freq>0.], freq[freq>0.]*0.+pi/2., 'r-')
    xscale('log')
    xlabel(r'$f$, Hz') ; ylabel(r'$\Delta \varphi$')
    savefig('phaselag.png')
    savefig('phaselag.eps')
    close()

def coherence(binfreq, coherence, dcoherence, phaselag, dphaselag,
              mdot_PDS, mdot_dPDS, orot_PDS, orot_dPDS,
              npoints, outfile = 'cobin'):
    # (binfreq, mmdot_crossbin, dmmdot_crossbin,
    #              mdot_pdsbin, mdot_dpdsbin, lBL_pdsbin, lBL_dpdsbin,
    #              npoints, outfile = 'coherence'):
    #!!! do we need PDS uncertainties?
    freq = binfreq
    binfreqc = (binfreq[1:]+binfreq[:-1])/2.
    binfreqs = (binfreq[1:]-binfreq[:-1])/2.    
    w = npoints>=2.
    clf()
    fig, ax = subplots(2,1)
    # ax[0].errorbar(binfreqc[w], abs(mmdot_crossbin[w]), xerr = binfreqs[w], yerr = dmmdot_crossbin[w]/sqrt(npoints[w]-1.), fmt = 'k.')
    # ax[0].set_xlabel(r'$f$, Hz') ;
    ax[0].plot([r**(-1.5)/tscale,r**(-1.5)/tscale], [-pi,pi], 'g')
    ax[0].plot([r**(-1.5)/tscale*alpha,r**(-1.5)/tscale*alpha], [-pi,pi], 'g--')
    ax[0].plot([1./tscale/tdepl,1./tscale/tdepl], [-pi,pi], 'g:')
    ax[0].plot(freq[freq>0.], freq[freq>0.]*0., 'r-')
    ax[0].plot(freq[freq>0.], freq[freq>0.]*0.+pi/2., 'r-')
    ax[0].plot(freq[freq>0.], freq[freq>0.]*0.+pi, 'r-')
    ax[0].plot(freq[freq>0.], freq[freq>0.]*0.-pi/2., 'r-')
    ax[0].plot(freq[freq>0.], freq[freq>0.]*0.-pi, 'r-')
    ax[0].errorbar(binfreqc[w], phaselag, xerr = binfreqs[w],
                   yerr = dphaselag, fmt = 'k.')
    ax[0].set_xscale('log')  ; ax[0].set_ylabel(r'$\Delta \varphi$', fontsize=18) ; ax[0].set_ylim(-pi,pi)
    ax[1].plot([r**(-1.5)/tscale,r**(-1.5)/tscale], [0.,1.], 'g')
    ax[1].plot([r**(-1.5)/tscale*alpha,r**(-1.5)/tscale*alpha], [0.,1.], 'g--')
    ax[1].plot([1./tscale/tdepl,1./tscale/tdepl], [0.,1.], 'g:')
    ax[1].errorbar(binfreqc[w], coherence,
                   xerr = binfreqs[w], yerr = dcoherence, fmt = 'k.')
    ax[1].set_xscale('log')
    ax[1].set_xlabel(r'$f$, Hz', fontsize=18) ; ax[1].set_ylabel(r'coherence', fontsize=18)
    ax[0].tick_params(labelsize=14, length=6, width=1., which='major')
    ax[0].tick_params(labelsize=14, length=3, width=1., which='minor')
    ax[1].tick_params(labelsize=14, length=6, width=1., which='major')
    ax[1].tick_params(labelsize=14, length=3, width=1., which='minor')
    fig.set_size_inches(5, 6)
    fig.tight_layout()
    savefig(outfile+'.png')
    savefig(outfile+'.eps')
    close()

def coherence_doubled(freq1, freq2, cross, dcross1, dcross2, mdot_pds, lBL_pds):
    # for a double error source, variability within the frequency bin and ensemble variations
    freqc = (freq1+freq2)/2.
    freqs = (freq2-freq1)/2.    

    clf()
    fig, ax = subplots(2,1)
    # ax[0].errorbar(binfreqc[w], abs(mmdot_crossbin[w]), xerr = binfreqs[w], yerr = dmmdot_crossbin[w]/sqrt(npoints[w]-1.), fmt = 'k.')
    # ax[0].set_xlabel(r'$f$, Hz') ;
    ax[0].errorbar(freqc, angle(cross), xerr = freqs,
                   yerr = dcross1/abs(cross), fmt = 'g.', linewidth = 3.)
    ax[0].errorbar(freqc, angle(cross), xerr = freqs,
                   yerr = dcross2/abs(cross), fmt = 'k.')
    ax[0].plot(freqc, freqc*0., 'r-')
    ax[0].plot(freqc, freqc*0.+pi/2., 'r-')
    ax[0].plot(freqc, freqc*0.+pi, 'r-')
    ax[0].set_xscale('log')  ; ax[0].set_ylabel(r'$\Delta \varphi$', fontsize=18)
    ax[0].set_ylim(0., pi)
    ax[1].errorbar(freqc, abs(cross)/sqrt(mdot_pds * lBL_pds),
                   xerr = freqs, yerr = dcross1 / sqrt(mdot_pds * lBL_pds), fmt = 'g.', linewidth = 3.)
    ax[1].errorbar(freqc, abs(cross)/sqrt(mdot_pds * lBL_pds),
                   xerr = freqs, yerr = dcross2 / sqrt(mdot_pds * lBL_pds), fmt = 'k.')
    ax[1].set_xscale('log')
    ax[1].set_xlabel(r'$f$, Hz', fontsize=18) ; ax[1].set_ylabel(r'coherence', fontsize=18)
    ax[0].tick_params(labelsize=14, length=6, width=1., which='major')
    ax[0].tick_params(labelsize=14, length=3, width=1., which='minor')
    ax[1].tick_params(labelsize=14, length=6, width=1., which='major')
    ax[1].tick_params(labelsize=14, length=3, width=1., which='minor')
    fig.set_size_inches(5, 6)
    fig.tight_layout()
    savefig('coherence2.png')
    savefig('coherence2.eps')
    close()

def biplot(f, bc, bphi, outname = 'biplot'):

    f2, g2 = meshgrid(f, f)

    lbc = log10(bc)
    #    lbc = bc
    bcmin =  lbc[bc>0.].min() ; bcmax = lbc.max() ; nbc = 30
    bclevs = (bcmax-bcmin)*(arange(nbc)/double(nbc))+bcmin
    cmap = plt.get_cmap('hot')
    bcnorm = BoundaryNorm(bclevs, ncolors=cmap.N, clip=True)

    nphi = 30
    philevs = 2.*pi*arange(nphi)/double(nphi)
    phinorm = BoundaryNorm(philevs, ncolors=cmap.N, clip=True)

    fmin=(f2[:-1,:-1])[bc >= 0.].min()    
    fmax=(f2[:-1,:-1])[bc >= 0.].max()    

    lbc_masked = ma.masked_array(lbc, mask = (bc<=0.))
    bphi_masked = ma.masked_array(bphi, mask = (bc<=0.))
    
    clf()
    fig, ax = subplots(1,2)
    c0 = ax[0].pcolormesh(f2, g2, lbc_masked, norm = bcnorm, cmap = cmap)
    colorbar(c0, ax=ax[0])
    ax[0].set_xlabel(r'$f_1$, Hz', fontsize=18)
    ax[0].set_ylabel(r'$f_2$, Hz', fontsize=18)
    c1 = ax[1].pcolormesh(f2, g2, bphi_masked, norm = phinorm, cmap = cmap)
    colorbar(c1, ax=ax[1])
    ax[1].set_xlabel(r'$f_1$, Hz', fontsize=18)
    ax[1].set_ylabel(r'$f_2$, Hz', fontsize=18)
    ax[0].set_xscale('log') ; ax[0].set_yscale('log')
    ax[1].set_xscale('log') ; ax[1].set_yscale('log')
    ax[0].set_xlim(fmin, fmax)  ;  ax[1].set_xlim(fmin, fmax)
    ax[0].set_ylim(fmin, fmax)  ;  ax[1].set_ylim(fmin, fmax)
    ax[0].tick_params(labelsize=14, length=6, width=1., which='major')
    ax[0].tick_params(labelsize=14, length=3, width=1., which='minor')
    ax[1].tick_params(labelsize=14, length=6, width=1., which='major')
    ax[1].tick_params(labelsize=14, length=3, width=1., which='minor')
    ax[0].set_title('bicoherence') ; ax[1].set_title('biphase')
    fig.set_size_inches(10, 4)
    fig.tight_layout()
    savefig(outname+'.png')
    close()
