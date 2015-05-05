import sys
import cPickle
import glob
import set_up
import os

nJobs = set_up.nJobs
print '[Section extraction script]'

if sys.argv[1]=='train':
    in_dir = set_up.train_folder_path
elif sys.argv[1]=='test':
    in_dir = set_up.test_folder_path   
else:
    print 'Unknown option'
    sys.exit()

out_dir = set_up.feats_folder_path + 'sections_hist/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

    
def worker(fname):
        # preparation
        stat = {}
        fin = open(in_dir + fname + '.asm', 'r')
        for line in fin:
                line_type = line.split(':')[0].lower()
                stat[line_type] = stat.get(line_type, 0) + 1
                stat['sum'] = stat.get('sum', 0) + 1
        cPickle.dump(stat, open(out_dir + fname, 'w'))

        
raw_filenames = glob.glob(in_dir + '*.asm')
fnames = map(lambda x: x.split('/')[-1].split('.')[0], raw_filenames)


from multiprocessing import Process
def wrapper(fname_list):
        for fname in fname_list:
                worker(fname)

                
workers = []
for workerId in range(nJobs):
        p = Process(target=wrapper, args=[[param for i, param in enumerate(fnames) if i % nJobs == workerId]])
        workers.append(p)
        p.start()
for p in workers:
        p.join()
print 'Done!'
