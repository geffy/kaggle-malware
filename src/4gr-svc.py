import pandas as pd
import numpy as np
import hickle, cPickle
import set_up
import scipy.sparse as sp
from sklearn.externals import joblib
import os

import logging
reload(logging)
logging.basicConfig(format = u'[%(asctime)s]  %(message)s', level = logging.INFO)
logging.info('[Script for 1st stage feature selection by SVC]')

# assembling train
logging.info('[train] Load data...')
trainLabels = pd.read_csv(set_up.train_labels_path)

data = []
indices = []
ptrs = [0]
cur_bound = 0
for i, row in trainLabels.iterrows():
    fname = set_up.feats_folder_path + '4gr/' + '{}.bytes.freq'.format(row['Id'])
    (inds, vals) = cPickle.load(open(fname, 'rb'))
    assert len(vals) == len(inds)
    data.extend(vals)
    indices.extend(inds)
    cur_bound += len(vals)
    ptrs.append(cur_bound)
    if i%1000==0:
        print '[{}] {}'.format(i, row['Id'])        
hickle.dump((np.array(data), np.array(indices), np.array(ptrs)), set_up.tmp_path + '4gr_train_raw.hi')

logging.info('[train] Build csr...')
X = sp.csr_matrix((np.array(data), np.array(indices), np.array(ptrs)), dtype=int)

logging.info('[train] dump to {}...'.format(set_up.feats_folder_path + '4gr/4gr_train_csr.joblib'))
joblib.dump(X, set_up.feats_folder_path + '4gr/4gr_train_csr.joblib')
del X, data, indices, ptrs


# assembling test
logging.info('[test] Load data...')
sampleSubmission = pd.read_csv(set_up.test_sample_path) #ToDo

data = []
indices = []
ptrs = [0]
cur_bound = 0
for i, row in sampleSubmission.iterrows():
    fname = set_up.feats_folder_path + '4gr/' + '{}.bytes.freq'.format(row['Id'])
    (inds, vals) = cPickle.load(open(fname, 'rb'))
    assert len(vals) == len(inds)
    data.extend(vals)
    indices.extend(inds)
    cur_bound += len(vals)
    ptrs.append(cur_bound)
    if i%1000==0:
        print '[{}] {}'.format(i, row['Id'])

hickle.dump((np.array(data), np.array(indices), np.array(ptrs)), set_up.tmp_path + '4gr_test_raw.hi')

logging.info('[test] Build csr...')
X = sp.csr_matrix((np.array(data), np.array(indices), np.array(ptrs)), dtype=int)

logging.info('[test] dump to {}...'.format(set_up.feats_folder_path + '4gr/4gr_test_csr.joblib'))
joblib.dump(X, set_up.feats_folder_path + '4gr/4gr_test_csr.joblib')

logging.info('Cleaning memory...')
del X, data, indices, ptrs


# Feature selection
logging.info('Load train for feature selection...')

import scipy.sparse as sp
from sklearn.externals import joblib
from sklearn.svm import LinearSVC

trainLabels = pd.read_csv(set_up.train_labels_path)
X_csr_tr = joblib.load(set_up.feats_folder_path + '4gr/4gr_train_csr.joblib')

logging.info('Fit lsvc...')
model = LinearSVC(penalty='l1',max_iter=20, dual=False, verbose=1)
model.fit(X_csr_tr, trainLabels.Class.values)

logging.info('Dump fitted model..')
joblib.dump(model, set_up.tmp_path + '4gr_csr_model.joblib')


# reduce train
logging.info('Reduce train...')
X_mini = model.transform(X_csr_tr)

logging.info('Dump train...')
joblib.dump(X_mini, set_up.feats_folder_path + '4gr/4gr_train_dim10k.joblib')
del X_csr_tr


# reduce test
logging.info('Read test...')
X_csr_te = joblib.load(set_up.feats_folder_path + '4gr/4gr_test_csr.joblib')

logging.info('Reduce test...')
X_mini = model.transform(X_csr_te)

logging.info('Dump test...')
joblib.dump(X_mini, set_up.feats_folder_path + '4gr/4gr_test_dim10k.joblib')

logging.info('Done!')