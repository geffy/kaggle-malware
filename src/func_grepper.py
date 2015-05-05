import sys
import cPickle
import glob
import subprocess
import os
import set_up
print '[Grepper for FUNCTION]'

if sys.argv[1]=='train':
    in_dir = set_up.train_folder_path
elif sys.argv[1]=='test':
    in_dir = set_up.test_folder_path   
else:
    print 'Unknown option'
    sys.exit()

out_dir = set_up.feats_folder_path + 'func_grepper/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


def worker(fname):
    # preparation
    subprocess.call('grep "FUNCTION" {}{}.asm > {}{}'.format(in_dir, fname, out_dir, fname), shell=True)


raw_filenames = glob.glob(in_dir + '*.asm')
fnames = map(lambda x: x.split('/')[-1].split('.')[0], raw_filenames)

from multiprocessing import Process
def wrapper(fname_list):
    for fname in fname_list:
        worker(fname)

nJobs = 15
workers = []
for workerId in range(nJobs):
    p = Process(target=wrapper, args=[[param for i, param in enumerate(fnames) if i % nJobs == workerId]])
    workers.append(p)
    p.start()
for p in workers:
    p.join()
print 'Done!'