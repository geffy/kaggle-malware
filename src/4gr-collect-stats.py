import cPickle as pickle
import numpy as np
import hickle
import glob
import sys
import gc
import set_up
print '[4gr freq stat collector]'

in_dir = set_up.feats_folder_path + '4gr/'
all_tokens = np.zeros(257**4)
files = glob.glob(in_dir + '*.bytes')[::5]
for i, fname in enumerate(files):
    print '[{}] {}'.format(i, fname)
    ptr, vals = pickle.load(open(fname))
    #for key in ptr:
    #    all_tokens[key] += 1
    all_tokens[ptr] = all_tokens[ptr] + 1
    del ptr, vals
    if i%100==0:
        gc.collect()
    if (i%4000==0) and (i>0):
        print 'pickled_state: {}'.format(i)
        hickle.dump(all_tokens, open(set_up.tmp_path + '4gr_stats_2', 'w')) 
        break