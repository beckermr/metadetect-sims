import os

#
# config files
#

def get_config_dir():
    """
    this is used by the batch maker; in this case you need to
    put them in a standard place.

    the basic executable dbsim-run can take the paths
    """
    assert 'DBSIM_CONFIG_DIR' in os.environ,'DBSIM_CONFIG_DIR not set'

    return os.environ['DBSIM_CONFIG_DIR']

def get_config_file(identifier):
    """
    get a config file based on identifier, in the "usual" place
    spcified by the BDSIM_CONFIG_DIR environment variable
    """
    return os.path.join(
        get_config_dir(),
        identifier+'.yaml',
    )

def read_config_file(identifier):
    """
    get a config file based on identifier, in the "usual" place
    specified  by the BDSIM_CONFIG_DIR environment variable
    """
    f=get_config_file(identifier)
    print("reading:",f)
    return read_yaml(f)

def get_dbsim_dir():
    """
    all outputs go here
    """
    assert 'DBSIM_DIR' in os.environ,'DBSIM_DIR not set'

    d=os.environ['DBSIM_DIR']
    return d

def get_run_dir(run):
    """
    base dir for a run
    """
    dir=get_dbsim_dir()
    return os.path.join(dir,run)

#
# condor
#


def get_condor_dir(run):
    """
    location of condor submit files
    """
    dir=get_run_dir(run)
    dir=os.path.join(dir, 'condor')
    return dir

def get_condor_job_url(run, filenum):
    """
    location of condor submit files
    """
    d=get_condor_dir(run)

    fname='%(run)s-%(filenum)02d.condor' % {
        'run':run,
        'filenum':filenum,
    }
    return os.path.join(d,fname)

def get_condor_master_url(run):
    """
    location of condor master script
    """
    d=get_condor_dir(run)
    return os.path.join(d,'%s.sh' % run)


#
# wq
#

def get_wq_dir(run):
    """
    location of wq submit scripts
    """
    dir=get_run_dir(run)
    dir=os.path.join(dir, 'wq')
    return dir

def get_wq_job_url(run, filenum):
    """
    path to wq submit script
    """
    d=get_wq_dir(run)

    fname='%(run)s-%(filenum)06d.yaml' % {
        'run':run,
        'filenum':filenum,
    }

    return os.path.join(d,fname)

def get_wq_master_url(run):
    """
    location of master script
    """
    d=get_wq_dir(run)
    return os.path.join(d,'%s.sh' % run)

#
# lsf
#

def get_lsf_dir(run):
    """
    location of lsf submit scripts
    """
    dir=get_run_dir(run)
    dir=os.path.join(dir, 'lsf')
    return dir

def get_lsf_job_url(run, filenum, missing=False):
    """
    path to lsf submit script
    """
    d=get_lsf_dir(run)

    fname='%(run)s-%(filenum)06d.lsf' % {
        'run':run,
        'filenum':filenum,
    }
    if missing:
        fname=fname.replace('.lsf','-missing.lsf')

    return os.path.join(d,fname)


def get_lsf_master_url(run):
    """
    path to master script
    """
    d=get_lsf_dir(run)
    return os.path.join(d,'%s.sh' % run)

#
# slr
#


def get_slr_dir(run):
    """
    location of slr submit scripts
    """
    dir=get_run_dir(run)
    dir=os.path.join(dir, 'slr')
    return dir

def get_slr_job_url(run, filenum):
    """
    path to slr submit script
    """
    d=get_slr_dir(run)
    fname='%(run)s-%(filenum)06d.slr' % {
        'run':run,
        'filenum':filenum,
    }

    return os.path.join(d,fname)

def get_slr_minions_command_list(run):
    d=get_slr_dir(run)
    fname = '%s.dat' % run
    return os.path.join(d,fname)


def get_slr_minions_job_url(run):
    d=get_slr_dir(run)
    fname = '%s.slr' % run
    return os.path.join(d,fname)


def get_output_dir(run):
    """
    output directory
    """
    dir=get_run_dir(run)
    return os.path.join(dir, 'outputs')

def get_collated_dir(run):
    """
    output directory
    """
    dir=get_run_dir(run)
    return os.path.join(dir, 'collated')


def get_output_url(run, filenum):
    """
    path to output file
    """
    dir=get_output_dir(run)

    fname='%(run)s-%(filenum)06d.fits' % {
        'run':run,
        'filenum':filenum,
    }
    return os.path.join(dir, fname)

def get_collated_url(run):
    """
    path to collated file
    """
    dir=get_collated_dir(run)

    fname='%s.fits' % run
    return os.path.join(dir, fname)


def get_means_dir(run):
    """
    location of aggregate statistics
    """
    dir=get_run_dir(run)
    dir=os.path.join(dir, 'fit-m-c')
    return dir

def get_means_url(run, extra=None):
    """
    path to file with aggregate statistics
    """
    dir=get_means_dir(run)
    if extra is not None:
        extra = '-'+extra
    else:
        extra=''

    f='%s-means%s.fits' % (run,extra)
    return os.path.join(dir, f)

def get_sums_url(run, extra=None):
    """
    path to file with sums statistics
    """
    dir=get_means_dir(run)
    if extra is not None:
        extra = '-'+extra
    else:
        extra=''

    f='%s-sums%s.fits' % (run,extra)
    return os.path.join(dir, f)


def read_output(run, filenum, get_meta=False):
    """
    Read an output file
    """
    import fitsio

    fname=get_output_url(run, filenum)

    print("reading:",fname)
    with fitsio.FITS(fname) as fits: 
        model_fits=fits['model_fits'][:]
        meta=fits['meta'][:]

    if get_meta:
        return model_fits, meta
    else:
        return model_fits

def read_collated(run, **kw):
    """
    Read an output file
    """
    import fitsio

    fname=get_collated_url(run)
    return fitsio.read(fname, **kw)

def read_yaml(fname):
    """
    read a yaml file
    """
    import yaml

    with open(fname) as fobj:
        data=yaml.load(fobj)

    return data


