# coding: utf-8

import pandas as pd
import numpy as np
import cPickle
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
import set_up
feats_path = '../data/feats/'


import logging
reload(logging)
logging.basicConfig(format = u'[%(asctime)s]  %(message)s', level = logging.INFO)
logging.info('[Base feature packer]')



def calc_entropy(X):
    ent = np.zeros(len(X))
    for i in range(len(X)):
        x = X[i]*1.0 / (sum(X[i]) + 0.00001)
        ent[i] = -np.sum(x*np.log(x+0.00000001))
    return ent




trainLabels = pd.read_csv(set_up.train_labels_path)
sampleSubmission = pd.read_csv(set_up.test_sample_path)



# sections features
logging.info('Collect sections...')
from sklearn.feature_extraction import DictVectorizer
section_whitelist = set(['.bss', '.data', '.edata', '.idata', '.rdata', '.reloc',
                       '.rsrc', '.text', '.tls', 'bss', 'code', 'data', 'header'])

y_tr = np.zeros(len(trainLabels))
lines_tr = []

#for train
for i,row in trainLabels.iterrows():
    y_tr[i] = row['Class']
    
    # lines
    feat = cPickle.load(open('{}sections_hist/{}'.format(feats_path, row['Id']), 'r'))
    del feat['sum']
    whitened = {}
    for x in feat:
        if x in section_whitelist:
            whitened[x] = feat[x]
    lines_tr.append(whitened)

        
#for test
lines_te = []
for i,row in sampleSubmission.iterrows():
    # lines
    feat = cPickle.load(open('{}sections_hist/{}'.format(feats_path, row['Id']), 'r'))
    del feat['sum']
    whitened = {}
    for x in feat:
        if x in section_whitelist:
            whitened[x] = feat[x]
    lines_te.append(whitened)
    
# convert to matrix
dv = DictVectorizer(sparse=False)
dv.fit(lines_tr)
X_lines_tr = dv.transform(lines_tr)
X_lines_te = dv.transform(lines_te)


E_lines_tr = calc_entropy(X_lines_tr)
E_lines_te = calc_entropy(X_lines_te)




# filesize
logging.info('Collect file sizes')
import os
# train
X_sizes_tr = np.zeros([len(trainLabels), 2])
for i,row in trainLabels.iterrows():
    fname = row['Id']
    X_sizes_tr[i, 0] = os.path.getsize('{}{}.bytes'.format(set_up.train_folder_path, fname))
    X_sizes_tr[i, 1] = os.path.getsize('{}{}.asm'.format(set_up.train_folder_path, fname))
size_ratio_tr = (X_sizes_tr[:, 0] *1.0 / X_sizes_tr[:, 1])[:, np.newaxis]

#test
X_sizes_te = np.zeros([len(sampleSubmission), 2])
for i,row in sampleSubmission.iterrows():
    fname = row['Id']
    X_sizes_te[i, 0] = os.path.getsize('{}{}.bytes'.format(set_up.test_folder_path, fname))
    X_sizes_te[i, 1] = os.path.getsize('{}{}.asm'.format(set_up.test_folder_path, fname))
size_ratio_te = (X_sizes_te[:, 0] *1.0 / X_sizes_te[:, 1])[:, np.newaxis]



# spectral asm
logging.info('Collect spectral asm...')
def read_file(filename):
    fin = open(filename, 'r')
    data = []
    for line in fin:
        data.append(line.strip())
    return data

fnames = read_file('{}spectral_asm/fnames'.format(feats_path))
asm_dict = {}
specter = ['jmp', 'mov', 'retf', 'push', 'pop', 'xor', 'retn', 'nop', 'sub', 'inc', 'dec', 'add',
                'imul', 'xchg', 'or', 'shr', 'cmp', 'call', 'shl', 'ror', 'rol', 'jnb']
for op in specter:
    values = read_file('{}spectral_asm/{}'.format(feats_path, op))
    for fname, val in zip(fnames, values):
        asm_dict[fname] = asm_dict.get(fname, {})
        asm_dict[fname][op] = val
# train
X_asm_tr = np.zeros((len(trainLabels), 22))
for i, fname in enumerate(trainLabels.Id.values):
    for j,op in enumerate(specter):
        X_asm_tr[i,j] = asm_dict[fname][op]
        
E_asm_tr = calc_entropy(X_asm_tr)

#test
X_asm_te = np.zeros((len(sampleSubmission), 22))
for i, fname in enumerate(sampleSubmission.Id.values):
    for j,op in enumerate(specter):
        X_asm_te[i,j] = asm_dict[fname][op]
        
E_asm_te = calc_entropy(X_asm_te)



# line counts
logging.info('Collect line counts...')
fnames = read_file('{}spectral_asm/fnames'.format(feats_path))
line_counts = read_file('{}spectral_asm/line_count'.format(feats_path))

line_dict = {}
for i, fname in enumerate(fnames):
    line_dict[fname] = line_counts[i]

# train
X_lcounts_tr = np.zeros((len(trainLabels), 1))
for i, fname in enumerate(trainLabels.Id.values):
    X_lcounts_tr[i, 0] = line_dict[fname]
    
E_lcounts_tr = calc_entropy(X_lcounts_tr)

# test
X_lcounts_te = np.zeros((len(sampleSubmission), 1))
for i, fname in enumerate(sampleSubmission.Id.values):
    X_lcounts_te[i, 0] = line_dict[fname]
    
E_lcounts_te = calc_entropy(X_lcounts_te)



# import calls
logging.info('Collect calls...')
def get_call_list(fname):
    calls =[]
    lines = read_file(fname)
    for line in lines:
        calls.append(line.split('__stdcall')[1].split('(')[0].split('_')[0].strip())
    return calls

# train
call_txt_tr = []
for i,row in trainLabels.iterrows():
    call_txt_tr.append(' '.join(get_call_list('{}stdcall_grepper/'.format(feats_path) + row['Id'])))

# train
call_txt_te = []
for i,row in sampleSubmission.iterrows():
    call_txt_te.append(' '.join(get_call_list('{}stdcall_grepper/'.format(feats_path) + row['Id'])))


logging.info('-> vectorizing...')
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
vect = TfidfVectorizer(max_features=10000)
vect.fit(call_txt_tr + call_txt_te)
X_call_tr = vect.transform(call_txt_tr)
X_call_te = vect.transform(call_txt_te)


logging.info('-> apply NMF...')
from sklearn.decomposition import TruncatedSVD, NMF
from scipy import sparse

nmf = NMF(n_components=10, sparseness='data')
nmf.fit(sparse.vstack([X_call_tr, X_call_te]))
X_calls_nmf_tr = nmf.transform(X_call_tr)
X_calls_nmf_te = nmf.transform(X_call_te)



# funcs


logging.info('Collect FUNCs...')

def get_func_list(fname):
    procs =[]
    lines = read_file(fname)
    for line in lines:
        line2 = line.split('FUNCTION')[1]
        if 'PRESS' in line2:
            procs.append(line2.split('PRESS')[0].strip().replace('.', ''))
    return procs

func_txt_tr = []
func_txt_te = []

for i,row in trainLabels.iterrows():
    func_txt_tr.append(' '.join(get_func_list('{}func_grepper/'.format(feats_path) + row['Id'])))
        
for i,row in sampleSubmission.iterrows():
    func_txt_te.append(' '.join(get_func_list('{}func_grepper/'.format(feats_path) + row['Id'])))

logging.info('-> vectorizing...')
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
vect_func = TfidfVectorizer(max_features=10000)
vect_func.fit(func_txt_tr + func_txt_te)

X_func_tr = vect_func.transform(func_txt_tr)
X_func_te = vect_func.transform(func_txt_te)

logging.info('-> apply NMF...')
nmf_func = NMF(n_components=10, sparseness='data')
nmf_func.fit(sparse.vstack([X_func_tr, X_func_te]))
X_func_nmf_tr = nmf_func.transform(X_func_tr)
X_func_nmf_te = nmf_func.transform(X_func_te)



# building
logging.info('Build all together...')
X_train_tr = np.hstack((
                          X_lines_tr, 
                          size_ratio_tr, 
                          X_asm_tr,
                          X_sizes_tr, #!
                          E_lines_tr[:, np.newaxis], #!
                          X_calls_nmf_tr,
                          X_func_nmf_tr
                          ))

X_train_te = np.hstack((
                          X_lines_te, 
                          size_ratio_te, 
                          X_asm_te,
                          X_sizes_te, #!
                          E_lines_te[:, np.newaxis], #!
                          X_calls_nmf_te,
                          X_func_nmf_te
                          ))

# dump
joblib.dump((X_train_tr, X_train_te), '{}X_basepack'.format(feats_path))
logging.info('Done!')
