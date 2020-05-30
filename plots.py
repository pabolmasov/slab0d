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

from mslab import mscale, tscale, omegaNS, r, pspin
from mslab import alpha, tdepl

linestyles = ['-', '--', ':', '-.', '--.']

colorsequence = ['k', 'r', 'g', 'b', 'm']

###########################################################################
def xydyfile(infile):
    lines = loadtxt(infile+'.dat')
    x = lines[:,0] ; y = lines[:,1] ; dy = lines[:,2] ; z = lines[:,3]
    xydy(x, y, dy, outfile = infile+'_xydy', addlines=[z])

def xydy(x, y, dy, xl = None, yl = None, outfile = 'xydy', xlog = False, ylog = False, addlines = None):
    '''
    general plotter for quantity y as a function of x. dy is shown as errorbars for y. 
    '''
    clf()
    fig = figure()
    if addlines is not None:
        # addlines should be a list
        nlines = shape(addlines)[0]
        for k in arange(nlines):
            plot(x, addlines[k], 'r', linestyle = linestyles[k])
    plot(x, y, 'ko')
    eb = errorbar(x, y, yerr = dy, fmt = 'none')
    eb[-1][0].set_linestyle('-')
    if xl is not None:
        xlabel(xl, fontsize = 18)
    if yl is not None:
        ylabel(yl, fontsize = 18)
    if xlog:
        xscale('log')
    if ylog:
        yscale('log')
    tick_params(labelsize=14, length=6, width=1., which='major')
    tick_params(labelsize=14, length=3, width=1., which='minor')
    fig.set_size_inches(10, 4)
    fig.tight_layout()
    savefig(outfile+'.png')
    savefig(outfile+'.pdf')
    close()
        
#################################################################################
# for mslab:
def mconsttests(tar, mar, orot, meq, oeq):
    # mass and momentum test
    print("Meq = "+str(meq))
    clf()
    fig, ax = subplots(2,1)
    ax[0].plot(tar, mar*0. + meq*mscale/1e17 , 'r-')
    ax[0].plot(tar, mar/1e17, 'k-')
    #            ax[0].set_xlabel(r'$t$, s', fontsize=18)
    ax[0].set_ylabel(r'$M$, $10^{17}$g', fontsize=18)
    if oeq>0.:
        ax[1].plot(tar, orot*0. + oeq/tscale/2./pi, 'r-')
    ax[1].plot(tar, orot, 'k-')
    ax[1].plot(tar, orot*0. + 1./pspin, 'b:')
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

def pds(binfreq, mdot_pdsbin, mdot_dpdsbin, mdot_dpdsbin1, lBL_pdsbin, lBL_dpdsbin, lBL_dpdsbin1, npoints, outfile = 'pdss'):

    binfreqc = (binfreq[1:]+binfreq[:-1])/2.
    binfreqs = (binfreq[1:]-binfreq[:-1])/2.    
    w = npoints>=2.
    
    clf()
    fig=figure()
    # plot(freq, mdot_pds, 'k,')
    # plot(freq, lBL_pds, 'r,')
    eb1 = errorbar(binfreqc[w]+binfreqs[1]*0.2, (binfreqc*mdot_pdsbin)[w], yerr = mdot_dpdsbin1[w], fmt = 'none', color='k')
    eb1[-1][0].set_linestyle(':')
    errorbar(binfreqc[w], (binfreqc*mdot_pdsbin)[w], xerr = binfreqs[w], yerr = (binfreqc*mdot_dpdsbin)[w], fmt = 'ks')
    eb2 = errorbar(binfreqc[w]+binfreqs[1]*0.2, (binfreqc*lBL_pdsbin)[w], yerr = (binfreqc*lBL_dpdsbin1)[w], fmt = 'none', color='r')
    eb2[-1][0].set_linestyle(':')
    errorbar(binfreqc[w], (binfreqc*lBL_pdsbin)[w], xerr = binfreqs[w], yerr = (binfreqc*lBL_dpdsbin)[w], fmt = 'rd')
    # plot(freq[freq>0.], 1e-3/((freq[freq>0.]*1000.*4.92594e-06*1.5)**2+1.), 'g-')
    xlim([binfreqc.min()/2., binfreq.max()])
    xscale('log') ; yscale('log')
    xlabel(r'$f$, Hz') ; ylabel(r'$f \, PDS$')
    tick_params(labelsize=14, length=6, width=1., which='major')
    tick_params(labelsize=14, length=4, width=1., which='minor')
    fig.set_size_inches(5, 6)
    fig.tight_layout()
    savefig(outfile + '.png')
    savefig(outfile + '.eps')
    close()

def multipds_stored(prefix):
    lines1 = loadtxt(prefix+'_osp.dat', comments='#')
    #     f1, f2, mdot, dmdot, d1mdot, y, dy, dy1, c, dc, dc1, p, dp, dp1, np
    f1_1 = lines1[:,0] ; f1_2 = lines1[:,1]
    mpds = lines1[:,2] ; dmpds = lines1[:,3]+lines1[:,4]
    opds = lines1[:,5] ; dopds = lines1[:,6]+lines1[:,7]
    lines2 = loadtxt(prefix+'_lsp.dat', comments='#')
    f2_1 = lines2[:,0] ; f2_2 = lines2[:,1]
    lpds = lines2[:,5] ; dlpds = lines2[:,6]+lines2[:,7]
    multipds([f1_1, f2_1, f1_1], [f1_2, f2_2, f1_2], [mpds, lpds, opds], [dmpds, dlpds, dopds], prefix+'_pds3')
    
def multipds(freq1, freq2, pds, dpds, outfile):
    nsp, rest = shape(freq1)
    formats = ['ko', 'rs', 'gd', 'bx']
    clf()
    fig=figure()
    for k in arange(nsp):
        freqc = (freq1[k]+freq2[k])/2.
        freqs = (freq2[k]-freq1[k])/2.    
        errorbar(freqc, pds[k] * freqc, xerr = freqs, yerr = dpds[k], fmt = formats[k], linewidth = 1.)
    xscale('log') ; yscale('log')
    xlabel(r'$f$, Hz', fontsize=20) ; ylabel(r'$f \, PDS$', fontsize=20)
    tick_params(labelsize=14, length=6, width=1., which='major')
    tick_params(labelsize=14, length=4, width=1., which='minor')
    fig.set_size_inches(5, 6)
    fig.tight_layout()
    savefig(outfile+'.png')
    savefig(outfile+'.eps')
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

def plot_stored(ascfile):
    lines = loadtxt(ascfile+'.dat', comments='#')
    #     f1, f2, mdot, dmdot, d1mdot, y, dy, dy1, c, dc, dc1, p, dp, dp1, np
    f1 = lines[:,0] ; f2 = lines[:,1]
    binfreq = concatenate([f1, f2[-1:]])
    mdot = lines[:,2] ; dmdot = lines[:,3] ; d1mdot = lines[:,4]
    y = lines[:,5] ; dy = lines[:,6]  ; d1y = lines[:,7]
    c = lines[:,8] ; dc = lines[:,9] ; dc1 = lines[:,10]
    p = lines[:,11] ; dp = lines[:,12] ; dp1 = lines[:,13]
    np = lines[:,14]
    coherence(binfreq, c, dc, dc1, p, dp, dp1, np, outfile = ascfile+'_c')
    pds(binfreq, mdot, dmdot, d1mdot, y, dy, d1y, np, outfile = ascfile+'_pds')

def coherence(binfreq, coherence, dcoherence, dcoherence1,
              phaselag, dphaselag, dphaselag1,
              npoints, outfile = 'cobin'):
    freq = binfreq
    binfreqc = (binfreq[1:]+binfreq[:-1])/2.
    binfreqs = (binfreq[1:]-binfreq[:-1])/2.    
    w = npoints>=2.
    clf()
    fig, ax = subplots(2,1)
    # ax[0].errorbar(binfreqc[w], abs(mmdot_crossbin[w]), xerr = binfreqs[w], yerr = dmmdot_crossbin[w]/sqrt(npoints[w]-1.), fmt = 'k.')
    # ax[0].set_xlabel(r'$f$, Hz') ;
    ax[0].plot([r**(-1.5)/tscale,r**(-1.5)/tscale], [-pi,pi], 'g')
    ax[0].plot([r**(-1.5)*alpha/tscale,r**(-1.5)*alpha/tscale], [-pi,pi], 'g--')
    ax[0].plot([1./tscale/tdepl,1./tscale/tdepl], [-pi,pi], 'g-.')
    ax[0].plot(freq[freq>0.], freq[freq>0.]*0., 'r-')
    ax[0].plot(freq[freq>0.], freq[freq>0.]*0.+pi/2., 'r-')
    ax[0].plot(freq[freq>0.], freq[freq>0.]*0.+pi, 'r-')
    ax[0].plot(freq[freq>0.], freq[freq>0.]*0.-pi/2., 'r-')
    ax[0].plot(freq[freq>0.], freq[freq>0.]*0.-pi, 'r-')
    ax[0].errorbar(binfreqc[w], phaselag[w], xerr = binfreqs[w],
                   yerr = (dphaselag)[w], fmt = 'k.')
    e1 = ax[0].errorbar(binfreqc[w]+binfreqs[w]*0.2, phaselag[w],
                        yerr = (dphaselag1)[w], fmt = 'none')
    e1[-1][0].set_linestyle(':')
    ax[0].set_xscale('log')  ; ax[0].set_ylabel(r'$\Delta \varphi$', fontsize=18) ; ax[0].set_ylim(-pi,pi)
    ax[1].plot([r**(-1.5)/tscale,r**(-1.5)/tscale], [0.,1.], 'g')
    ax[1].plot([r**(-1.5)*alpha/tscale,r**(-1.5)*alpha/tscale], [0.,1.], 'g--')
    ax[1].plot([1./tscale/tdepl,1./tscale/tdepl], [0.,1.], 'g-.')
    ax[1].errorbar(binfreqc[w], coherence,
                   xerr = binfreqs[w], yerr = dcoherence[w], fmt = 'k.')
    e1 = ax[1].errorbar(binfreqc[w]+binfreqs[w]*0.2, coherence[w],
                        yerr = (dcoherence1)[w], fmt = 'none')
    e1[-1][0].set_linestyle(':')
    ax[1].set_xscale('log') # ;   ax[1].set_yscale('log')
    ax[1].set_xlabel(r'$f$, Hz', fontsize=18) ; ax[1].set_ylabel(r'coherence', fontsize=18)
    ax[0].tick_params(labelsize=14, length=6, width=1., which='major')
    ax[0].tick_params(labelsize=14, length=3, width=1., which='minor')
    ax[1].tick_params(labelsize=14, length=6, width=1., which='major')
    ax[1].tick_params(labelsize=14, length=3, width=1., which='minor')
    fig.set_size_inches(5, 6)
    fig.tight_layout()
    savefig(outfile+'.png')
    savefig(outfile+'.eps')
    savefig(outfile+'.pdf')
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

####################################################3
# plotting from binobject
from timing import binobject

def object_pds(freq, objlist, outfile):
    nf = size(freq)-1 ;  no = size(objlist)
    freqc = (freq[1:]+freq[:-1])/2.
    freqs = (freq[1:]-freq[:-1])/2.

    ebs = zeros(no, dtype = matplotlib.container.ErrorbarContainer)
    
    w = (objlist[0].npoints > 0)
    clf()
    fig=figure()
    # plot(freq, mdot_pds, 'k,')
    # plot(freq, lBL_pds, 'r,')
    for ko in arange(no):
        ebs[ko] = errorbar(freqc[w]+freqs[1]*0.2, (freqc*objlist[ko].av)[w], yerr = (freqc*objlist[ko].dbin/sqrt(double(objlist[ko].npoints-1)))[w], fmt = 'none', color=colorsequence[ko])
        ebs[ko][-1][0].set_linestyle(':')
        errorbar(freqc[w], (freqc*objlist[ko].av)[w], xerr = freqs[w], yerr = (freqc*objlist[ko].densemble/sqrt(double(objlist[ko].npoints-1)))[w], fmt = colorsequence[ko]+'s')
    xlim([freqc.min()/2., freq.max()])
    xscale('log') ; yscale('log')
    xlabel(r'$f$, Hz') ; ylabel(r'$f \, PDS$')
    tick_params(labelsize=14, length=6, width=1., which='major')
    tick_params(labelsize=14, length=4, width=1., which='minor')
    fig.set_size_inches(5, 6)
    fig.tight_layout()
    savefig(outfile + '.png')
    savefig(outfile + '.pdf')
    close()
    
def object_coherence(freq, objlist, outfile):
    nf = size(freq)-1 ;  no = size(objlist)
    freqc = (freq[1:]+freq[:-1])/2.
    freqs = (freq[1:]-freq[:-1])/2.

    ebs_c = zeros(no, dtype = matplotlib.container.ErrorbarContainer)
    ebs_p = zeros(no, dtype = matplotlib.container.ErrorbarContainer)
    
    w = (objlist[0].npoints > 0)
    clf()
    fig, ax = subplots(2,1)
    # plot(freq, mdot_pds, 'k,')
    # plot(freq, lBL_pds, 'r,')
    for ko in arange(no):
        ebs_c[ko] = ax[1].errorbar(freqc[w]+freqs[1]*0.2, objlist[ko].c[w], yerr = (freqc*objlist[ko].dc_bin/sqrt(double(objlist[ko].npoints-1)))[w], fmt = 'none', color=colorsequence[ko])
        ebs_c[ko][-1][0].set_linestyle(':')
        ax[1].errorbar(freqc[w], objlist[ko].c[w], xerr = freqs[w], yerr = (objlist[ko].dc_ensemble/sqrt(double(objlist[ko].npoints-1)))[w], fmt = colorsequence[ko]+'.')
        ebs_p[ko] = ax[0].errorbar(freqc[w]+freqs[1]*0.2, objlist[ko].phlag[w], yerr = (objlist[ko].dphlag_bin/sqrt(double(objlist[ko].npoints-1)))[w], fmt = 'none', color=colorsequence[ko])
        ebs_p[ko][-1][0].set_linestyle(':')
        ax[0].errorbar(freqc[w], objlist[ko].phlag[w], xerr = freqs[w], yerr = (objlist[ko].dphlag_ensemble/sqrt(double(objlist[ko].npoints-1)))[w], fmt = colorsequence[ko]+'.')
    ax[0].plot([r**(-1.5)/tscale,r**(-1.5)/tscale], [-pi,pi], 'g')
    ax[0].plot([r**(-1.5)*alpha/tscale,r**(-1.5)*alpha/tscale], [-pi,pi], 'g--')
    ax[0].plot([1./tscale/tdepl,1./tscale/tdepl], [-pi,pi], 'g-.')
    ax[0].plot(freq[freq>0.], freq[freq>0.]*0., 'r-')
    ax[0].plot(freq[freq>0.], freq[freq>0.]*0.+pi/2., 'r-')
    ax[0].plot(freq[freq>0.], freq[freq>0.]*0.+pi, 'r-')
    ax[0].plot(freq[freq>0.], freq[freq>0.]*0.-pi/2., 'r-')
    ax[0].plot(freq[freq>0.], freq[freq>0.]*0.-pi, 'r-')
    ax[1].plot([r**(-1.5)/tscale,r**(-1.5)/tscale], [0.,1.], 'g')
    ax[1].plot([r**(-1.5)*alpha/tscale,r**(-1.5)*alpha/tscale], [0.,1.], 'g--')
    ax[1].plot([1./tscale/tdepl,1./tscale/tdepl], [0.,1.], 'g-.')
    ax[0].set_xlim([freqc.min()/2., freq.max()])
    ax[1].set_xlim([freqc.min()/2., freq.max()])
    ax[0].set_xscale('log')
    ax[1].set_xscale('log') # ;   ax[1].set_yscale('log')
    ax[1].set_xlabel(r'$f$, Hz', fontsize=18) ; ax[1].set_ylabel(r'coherence', fontsize=18)
    ax[0].set_ylabel(r'$\Delta \varphi$', fontsize=18) ; ax[0].set_ylim(-pi,pi)
    ax[0].tick_params(labelsize=14, length=6, width=1., which='major')
    ax[0].tick_params(labelsize=14, length=3, width=1., which='minor')
    ax[1].tick_params(labelsize=14, length=6, width=1., which='major')
    ax[1].tick_params(labelsize=14, length=3, width=1., which='minor')
    fig.set_size_inches(5, 6)
    fig.tight_layout()
    savefig(outfile+'.png')
    savefig(outfile+'.eps')
    savefig(outfile+'.pdf')
    close()
 
