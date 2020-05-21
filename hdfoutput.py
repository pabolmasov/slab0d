import h5py
import zarr
from numpy import *

from mslab import ifzarr

'''
inputs and outputs to HDF5 and zarr 
'''

def entryname(n, ndig = 6):
    entry = str(n).rjust(ndig, '0') # allows for 6 positions (hundreds of thousand of entries)
    return entry

def init(hname, t, mdot = None, alpha = None, tdepl = None,
         nsims = None, nflick = None, tbreak = None, regime = None): 
    '''
    creating the file and writing the time grid
    '''
    if ifzarr:
        hfile = zarr.open(hname+".zarr", "w")
    else:
        hfile = h5py.File(hname+".hdf5", "w")
    glo = hfile.create_group("globals")
    if regime is not None:
        glo.attrs['regime']  = regime
    if mdot is not None:
        glo.attrs['mdotmean']      = mdot
    if alpha is not None:
        glo.attrs['alpha']      = alpha
    if tdepl is not None:
        glo.attrs['tdepl']      = tdepl
    glo.attrs['nsims']      = nsims
    if nflick is not None:
        glo.attrs['nflick']      = nflick
    if tbreak is not None:
        glo.attrs['tbreak']      = tbreak

    glo.create_dataset("time", data = t)

    if not ifzarr:
        hfile.flush()
        
    return hfile # returns file stream reference
    
def dump(hfile, nout, valnames, valarray):
    '''
    writing one snapshot
    '''
    entry = entryname(nout)
    grp = hfile.create_group("entry"+entry)
    #    grp.attrs["N"] = nout
    for k in arange(size(valnames)):
        grp.create_dataset(valnames[k], data=valarray[k][:])
        print("writing "+valnames[k]+" to entry"+entry)
    if not ifzarr:
        hfile.flush()

#########################
def keyshow(filename):
    '''
    showing the list of keys (entries) in a given data file
    '''
    if ifzarr:
        f = zarr.open(filename+'.zarr','r')
    else:
        f = h5py.File(filename+'.hdf5','r', libver='latest')
    keys = list(f.keys())
    #    print(list(f.keys()))
    if not ifzarr:
        f.close()
    return keys

def read(hname, nentry, entry = None):
    '''
    read a single entry from an HDF5
    '''
    print("hname = "+hname)
    if ifzarr:
        hfile = zarr.open(hname+".zarr", "r")
    else:
        hfile = h5py.File(hname+".hdf5", "r")
    glo=hfile["globals"]
    if entry is None:
        entry = "entry" + entryname(nentry)
    time=glo["time"][:]
    data=hfile[entry]

    vals = data.keys()
    print(vals)
    datalist = [] # list of the arrays read from the file
    for theval in vals:
        #        print(theval)
        datalist.append(data[theval][:])
    if not ifzarr:
        hfile.close()
    return time, datalist
    
def vread(hname, valname = "mdot"):
    '''
    read particular variable from all the entries from an HDF5
    global array t is read simultaneously
    '''
    if ifzarr:
        hfile = zarr.open(hname+".zarr", "r")
    else:
        hfile = h5py.File(hname+".hdf5", "r")
    glo=hfile["globals"]
    time=glo["time"][:]
    keys = hfile.keys()
    #    print(keys)
    nsim = glo.attrs['nsims'] ;  nt = size(time)
    datarray = zeros([nsim, nt], dtype=double)
    k=0
    
    for entry in keys:
        if(entry == "globals"):
            glo=hfile[entry]
            t=glo["time"][:]
        else:
            data=hfile[entry]
            print(data.keys())
            #            datalist.append(data[valname][:])
            datarray[k,:] = data[valname][:]
            k+=1
    if not ifzarr:
        hfile.close()
    print("size(t) = "+str(size(t)))
    print("shape(datarray) = "+str(shape(datarray)))
    return t, datarray
