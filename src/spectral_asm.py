import sys
import cPickle
import glob
import subprocess
import os
import set_up
print 'ASM specter extractor'

if sys.argv[1]=='train':
    in_dir = set_up.train_folder_path
elif sys.argv[1]=='test':
    in_dir = set_up.test_folder_path   
else:
    print 'Unknown option'
    sys.exit()

out_dir = set_up.feats_folder_path + 'spectral_asm/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


def worker(fname):
        # preparation
        subprocess.call('echo "{}" >> {}{}'.format(fname, out_dir, 'fnames'), shell=True)
        subprocess.call('cat {}{}.asm | wc -l >> {}{}'.format(in_dir, fname, out_dir, 'line_count'), shell=True)

        specter = ['jmp', 'mov', 'retf', 'push', 'pop', 'xor', 'retn', 'nop', 'sub', 'inc', 'dec', 'add',
                'imul', 'xchg', 'or', 'shr', 'cmp', 'call', 'shl', 'ror', 'or', 'rol', 'jnb']
        for op in specter:
                subprocess.call('grep "\s{}\s" {}{}.asm | wc -l >> {}{}'.format(op, in_dir, fname, out_dir, op), shell=True)



raw_filenames = glob.glob(in_dir + '*.asm')
fnames = map(lambda x: x.split('/')[-1].split('.')[0], raw_filenames)
for i, fname in enumerate(fnames):
        worker(fname)
        if i%200==0:
                print i
print 'Done!'