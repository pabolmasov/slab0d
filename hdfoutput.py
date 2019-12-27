import h5py
from numpy import *

def entryname(n, ndig = 6):
    entry = str(n).rjust(ndig, '0') # allows for 6 positions (hundreds of thousand of entries)
    return entry

def init(hname, t, mdot = None, alpha = None, tdepl = None,
         nsims = None, nflick = None, tbreak = None): 
    '''
    creating the file and writing the time grid
    '''
    hfile = h5py.File(hname+".hdf5", "w")
    glo = hfile.create_group("globals")
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
    
    hfile.flush()
    return hfile # returns file stream reference
    
def dump(hfile, nout, valnames, valarray):
    '''
    writing one snapshot
    '''
    entry = entryname(nout)
    grp = hfile.create_group("entry"+entry)
    grp.attrs["N"] = nout
    for k in arange(size(valnames)):
        grp.create_dataset(valnames[k], data=valarray[k][:])
        print("writing "+valnames[k]+" to entry"+entry)
    hfile.flush()

#########################
def keyshow(filename):
    '''
    showing the list of keys (entries) in a given data file
    '''
    f = h5py.File(filename,'r', libver='latest')
    keys = list(f.keys())
    #    print(list(f.keys()))
    f.close()
    return keys

def read(hname, nentry):
    '''
    read a single entry from an HDF5
    '''
    hfile = h5py.File(hname, "r")
    glo=hfile["globals"]
    entry = entryname(nentry)
    time=glo["time"][:]
    data=hfile["entry"+entry]

    vals = data.keys()
    print(vals)
    datalist = [] # list of the arrays read from the file
    for theval in vals:
        datalist.append(data[theval][:])
        #    hfile.close()
    return time, datalist

def vread(hname, valname = "mdot"):
    '''
    read particular variable from all the entries from an HDF5
    '''
    hfile = h5py.File(hname+".hdf5", "r")
    #    glo=hfile["globals"]
    #    time=glo["time"][:]
    keys = hfile.keys()
    #    print(keys)
    datalist = []
    
    for entry in keys:
        if(entry == "globals"):
            glo=hfile[entry]
            t=glo["time"][:]
        else:
            data=hfile[entry]
            print(data.keys())
            datalist.append(data[valname][:])
    hfile.close()
    print("size(t) = "+str(size(t)))
    print("size(datalist[0]) = "+str(size(datalist[0])))
    return t, datalist
