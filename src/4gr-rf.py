import pandas as pd
import numpy as np
import hickle, cPickle
import set_up
import scipy.sparse as sp
from sklearn.externals import joblib

import logging
reload(logging)
logging.basicConfig(format = u'[%(asctime)s]  %(message)s', level = logging.INFO)
logging.info('[Script for 2nd stage feature selection by RF]')

logging.info('Load data...')
trainLabels = pd.read_csv(set_up.train_labels_path)
X_tr = joblib.load(set_up.feats_folder_path + '4gr/4gr_train_dim10k.joblib').todense()
X_te = joblib.load(set_up.feats_folder_path + '4gr/4gr_test_dim10k.joblib').todense()

logging.info('Fitting RF...')
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X_tr, trainLabels.Class.values, random_state=42, test_size=0.3)
rf = RandomForestClassifier(n_jobs=-1, n_estimators=1000)
rf.fit(X_train, y_train)
logging.info('RF acc: {}'.format(np.mean(rf.predict(X_test) == y_test)))

fmask = (rf.feature_importances_> 0.0014)
logging.info('Total feats: {}, selected: {}'.format(len(fmask), sum(fmask)))

logging.info('Dumping...')
data = (X_tr[:, fmask], X_te[:, fmask])
cPickle.dump(data, open(set_up.feats_folder_path + '4gr_pack_dim100.pickled', 'wb'), protocol=2)

logging.info('Done!')