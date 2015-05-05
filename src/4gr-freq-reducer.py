import glob
import sys
import gc
import numpy as np
import pickle
from joblib import Parallel, delayed
import set_up
print '[4gr freq reducer]'


repl = pickle.load(open(set_up.tmp_path + '4gr_replacer', 'rb'))
goodset = set(repl.keys())

in_dir = set_up.feats_folder_path + '4gr/'



def worker(i,fname):
    print '[{}] {}'.format(i, fname)
    (ptr, vals) = pickle.load(open(fname))
    new_ptr = []
    new_vals = []
    for ind, val in zip(ptr, vals):
        if ind in goodset:
            new_ptr.append(repl[ind])
            new_vals.append(val)
    del ptr, vals
    pickle.dump((new_ptr, new_vals), open(set_up.feats_folder_path + '4gr/' + fname.split('/')[-1] + '.freq', 'wb'), protocol=2)
    gc.collect()
    

files = glob.glob(in_dir + '*.bytes')

#print files
Parallel(n_jobs=15)(delayed(worker)(i,f) for i,f in enumerate(files))

print 'Done!'
   