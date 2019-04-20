#!/usr/bin/env python

import os
from glob import glob
import fitsio

from . import files

def collate_trials(run):

    outfile=files.get_collated_url(run)
    outdir=files.get_collated_dir(run)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    dir=files.get_output_dir(run)
    pattern=os.path.join(dir, '*.fits')
    flist=glob(pattern)

    nf=len(flist)
    print(nf,"files found")
    if len(flist)==0:
        return

    flist.sort()

    print('will write to',outfile)
    with fitsio.FITS(outfile,mode="rw",clobber=True) as output:

        first=True
        for i,f in enumerate(flist):
            print('%d/%d %s' % (i+1, nf, f))

            try:
                t=fitsio.read(f)
            except IOError as err:
                print("caught IOError: %s" % str(err))
                continue

            if first:
                output.write(t)
                first=False
            else:
                output[-1].append(t)

def get_tmp_file(fname):
    tmpdir=os.environ['TMPDIR']
    bname=os.path.basename(fname)
    return os.path.join(tmpdir, bname) 

