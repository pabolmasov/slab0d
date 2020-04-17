from mslab import *

sfreq1 = dtdyn ; sfreq2 = tmax ;  nsines = 10
sines =	(sfreq2/sfreq1)**(arange(nsines)/double(nsines-1))*sfreq1

for k in arange(nsines):
    slab_evolution(nrepeat = 1, somega = sines[k])
    os.system('mv slabout.zarr slabout_sine'+str(k)+'.zarr')
    
