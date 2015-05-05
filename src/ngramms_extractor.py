import glob
from joblib import Parallel, delayed
import set_up
import pickle 
import os
import sys
print '[n-gramm extractor]'

if sys.argv[1]=='train':
    in_dir = set_up.train_folder_path + '*.bytes'
elif sys.argv[1]=='test':
    in_dir = set_up.test_folder_path + '*.bytes'  
else:
    print 'Unknown option'
    sys.exit()
    
ng_order = int(sys.argv[2])
out_dir = set_up.feats_folder_path + '{}gr/'.format(ng_order)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

files = glob.glob(in_dir)[::-1]

def get_dict():
    d = {format(key, '02X'): key for key in range(256)}
    d['??'] = 256
    return d

def indexer4gr(tokens):
    return tokens[0]*16974593 + tokens[1]*66049 + tokens[2]*257 + tokens[3]


def count_4f(all_elem_codes,order):
    counts_4g = {}
    if order == 4:
        indexer = indexer4gr
    elif order == 10:
        print "Order10 not prepared"
    else:
        print 'WFT?'
    # collect counts    
    for i in range(len(all_elem_codes)-order+1):
        index = indexer(all_elem_codes[i:i+order])
        counts_4g[index] = counts_4g.get(index,0)+1
    # dump it!
    ptr = []
    vals = []
    for key in counts_4g:
        ptr.append(key)
        vals.append(counts_4g[key])
    return (ptr, vals)

def extract_4g (filename,order):
        convert_dict = get_dict()
        with open(filename,'r') as f:
            text = f.read()
        lines = text.split('\r\n')
        all_elems_codes = []
        for l in lines:
            elems = l.split(' ')
            all_elems_codes.extend([convert_dict[x] for x in elems[1:]])
            
        with open(out_dir + filename.split('/')[-1],'w') as f_dump:
            pickle.dump(count_4f(all_elems_codes,order),f_dump)   
    
#print files
Parallel(n_jobs=-1)(delayed(extract_4g)(fi, ng_order) for fi in files)

#pickle.dump(four_gr,open('../data/feats/%s/four_gr/4g' % what,'w'))
#pickle.dump([x.split('/')[-1] for x in files],open('../data/feats/%s/four_gr/names' % what,'w'))