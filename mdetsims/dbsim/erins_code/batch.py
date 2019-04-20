import os
import numpy as np
from . import files
from . import util

_slr_template="""#!/bin/bash -l
#SBATCH -p  shared  
#SBATCH -n  1        # number of logical cores   
#SBATCH -t  02:00:00     
#SBATCH -J  %(job_name)s
#SBATCH -o  %(job_name)s.log
#SBATCH -e  %(job_name)s.err
#SBATCH -L  SCRATCH
#SBATCH -C  haswell
#SBATCH -A  des

dbsim-run                  \\
        --seed=%(seed)s   \\
        %(run)s           \\
        %(ntrials_per)d      \\
        %(output)s
"""

_slr_template_shifter="""#!/bin/bash -l
#SBATCH --partition=%(queue)s
#SBATCH --nodes=%(nodes)d
#SBATCH --time=%(walltime)s
#SBATCH --job-name=%(job_name)s
#SBATCH --output=%(job_name)s.out
#SBATCH --license=SCRATCH
#SBATCH --constraint=%(constraint)s
#SBATCH --account=des
#SBATCH --image=%(docker_image)s
#SBATCH --volume="/global/cscratch1/sd/esheldon/lensing/shapesim:/data/lensing/shapesim"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=erin.sheldon@gmail.com

# -n is ncores, should equal 32*number of nodes for haswell
srun -n %(ncores)d shifter dbsim-run-mpi %(run)s
"""


_slr_template_minions="""#!/bin/bash -l
#SBATCH --partition=%(queue)s
#SBATCH --nodes=%(nodes)d
#SBATCH --time=%(walltime)s
#SBATCH --job-name=%(job_name)s
#SBATCH --output=%(job_name)s.log
#SBATCH --error=%(job_name)s.err
#SBATCH --license=SCRATCH
#SBATCH --constraint=%(machine)s
#SBATCH --account=des

# -n is ntasks, should equal 32*number of nodes
# -c is number of cpus per task, should be 1
srun -n %(ncores)d -c 1 minions < "%(command_list)s"
"""



_lsf_template="""#!/bin/bash
#BSUB -J %(job_name)s
#BSUB -n 1
#BSUB -oo ./%(job_name)s.oe
#BSUB -W 36:00
#BSUB -R "linux64 && rhel60 && scratch > 2"

echo "working on host: $(hostname)"
uptime

command=%(master_script)s
ntrials=%(ntrials_per)s
output="%(output)s"
logfile="%(logfile)s"
seed=%(seed)s

${command} ${ntrials} ${seed} ${output} ${logfile}
"""



_lsf_master_template="""#!/bin/bash
function runsim {
    echo "host: $(hostname)"
    echo "will write to file: $output"

    dbsim-run ${sim_config} ${fit_config} ${ntrials} ${seed} ${output}
    status=$?

    echo "time: $SECONDS"

    if [[ $status != "0" ]]; then
        echo "error running sim: $status"
    fi

    return $status
}

export OMP_NUM_THREADS=1

ntrials=$1
seed=$2
output=$3
logfile=$4

sim_config=%(sim_config)s
fit_config=%(fit_config)s

tmpdir="/scratch/esheldon/${LSB_JOBID}"
mkdir -p ${tmpdir}
echo "cd $tmpdir"
cd $tmpdir

tmplog=$(basename $logfile)

runsim &> ${tmplog}
status=$?

echo "moving log file ${tmplog} -> ${logfile}" >> ${tmplog}

# errors go to the jobs stderr
mv -fv "${tmplog}" "${logfile}" 1>&2
status2=$?

if [[ $status2 != "0" ]]; then
    # this error message will go to main error file
    echo "error ${status2} moving log to: ${logfile}" 1>&2

    status=$status2
fi

cd $HOME

echo "removing temporary directory"
rm -rfv ${tmpdir}

exit $status
"""



#
# condor templates
#

_condor_master_template="""#!/bin/bash
function runsim {
    echo "host: $(hostname)"
    echo "will write to file: $output"

    dbsim-run ${sim_config} ${fit_config} ${ntrials} ${seed} ${output}
    status=$?

    echo "time: $SECONDS"

    if [[ $status != "0" ]]; then
        echo "error running sim: $status"
    fi

    return $status
}

export OMP_NUM_THREADS=1

ntrials=$1
seed=$2
output=$3
logfile=$4

sim_config=%(sim_config)s
fit_config=%(fit_config)s

if [[ -n $_CONDOR_SCRATCH_DIR ]]; then
    tmpdir=$_CONDOR_SCRATCH_DIR
    export TMPDIR=$tmpdir
else
    tmpdir=$TMPDIR
    mkdir -p $tmpdir
fi

pushd $tmpdir

tmplog=$(basename $logfile)

runsim &> ${tmplog}
status=$?

echo "moving log file ${tmplog} -> ${logfile}" >> ${tmplog}

mv -fv "${tmplog}" "${logfile}" 1>&2

popd
"""

_condor_submit_head="""
Universe        = vanilla

Notification    = Never

# Run this exe with these args
Executable      = %(master_script)s

Image_Size       =  1000000

GetEnv = True

kill_sig        = SIGINT

#requirements = (cpu_experiment == "star") || (cpu_experiment == "phenix")
#requirements = (cpu_experiment == "phenix")

+Experiment     = "astro"

"""

_condor_job_template="""
+job_name = "%(job_name)s"
Arguments = %(ntrials_per)s %(seed)d %(output)s %(logfile)s
Queue
"""







_wq_template="""
command: |

    source activate nsim3

    command=%(master_script)s
    ntrials=%(ntrials_per)s
    output="%(output)s"
    logfile="%(logfile)s"
    seed=%(seed)d
    ${command} ${ntrials} ${seed} ${output} ${logfile}
    
job_name: "%(job_name)s"
"""


_wq_master_template="""#!/bin/bash
function runsim {
    echo "host: $(hostname)"
    echo "will write to file: $output"

    dbsim-run ${sim_config} ${fit_config} ${ntrials} ${seed} ${output}
    status=$?

    echo "time: $SECONDS"

    if [[ $status != "0" ]]; then
        echo "error running sim: $status"
    fi

    return $status
}

export OMP_NUM_THREADS=1

ntrials=$1
seed=$2
output=$3
logfile=$4

sim_config=%(sim_config)s
fit_config=%(fit_config)s

tmpdir=$TMPDIR
mkdir -p $tmpdir
cd $tmpdir

tmplog=$(basename $logfile)

runsim &> ${tmplog}
status=$?

echo "moving log file ${tmplog} -> ${logfile}" >> ${tmplog}

# errors go to the jobs stderr
mv -fv "${tmplog}" "${logfile}" 1>&2
status2=$?

if [[ $status2 != "0" ]]; then
    # this error message will go to main error file
    echo "error ${status2} moving log to: ${logfile}" 1>&2

    status=$status2
fi

exit $status
"""


def get_command():
    dir=files.get_nsim_dir()
    cmd = os.path.join(dir,'bin','dbsim-run')
    return cmd

class MakerBase(dict):
    def __init__(self, conf, missing=False):
        self.update(conf)
        self.missing=missing

        if 'seed' in conf:
            seed=conf['seed']
        else:
            seed=util.convert_run_to_seed(conf['run'])

        self.rng = np.random.RandomState(seed)

        self['sim_config'] = files.get_config_file(self['sim'])
        self['fit_config'] = files.get_config_file(self['fit'])

        assert os.path.exists(self['sim_config'])
        assert os.path.exists(self['fit_config'])

    def go(self):
        self.make_some_dirs()
        self.write_master()

        self.write_batch_files()

    def write_master(self):
        master_url=self.get_master_url()

        print("writing master:", master_url)
        with open(master_url,'w') as fobj:
            text = self.get_master_text()
            fobj.write(text % self)
        print()

        os.system('chmod 755 %s' % master_url)

        self['master_script'] = master_url


    def write_batch_files(self):
        overall_name = '-'.join( (self['run'].split('-'))[1:] )

        njobs=0
        ntrials_tot=0

        filenum=0

        ntrials_per, nsplit, time_hours = util.get_trials_nsplit(self)
        self['ntrials_per'] = ntrials_per

        walltime = int(2.0 + time_hours)
        self['walltime'] = '%02d:00' % walltime

        for isplit in range(nsplit):

            seed = self.rng.randint(0,2**31-1)

            ntrials_tot += ntrials_per
            output = files.get_output_url(self['run'], isplit)

            if self.missing and os.path.exists(output):
                continue

            logfile = output.replace('.fits','.log')
            job_name='%s-%06d' % (overall_name,isplit)

            self['output'] = output
            self['logfile'] = logfile
            self['job_name']=job_name
            self['seed'] = seed
            self.write_script(filenum)

            njobs += 1
            filenum += 1


        print('total jobs: ',njobs)
        print('total trials:',ntrials_tot)


    def write_script(self, filenum):
        job_url=self.get_job_url(filenum)

        with open(job_url,'w') as fobj:
            text = self.get_job_text()
            fobj.write(text)


    def make_some_dirs(self):

        d=self.get_batch_dir()
        print("dir:",d)
        outd = files.get_output_dir(self['run'])


        if not os.path.exists(d):
            os.makedirs(d)

        if not os.path.exists(outd):
            os.makedirs(outd)

        tmpdir=self.get_tmpdir()
        if tmpdir is not None:
            if not os.path.exists(tmpdir):
                os.makedirs(tmpdir)


class WQMaker(MakerBase):
    def get_tmpdir(self):
        tmpdir=os.environ['TMPDIR']
        return tmpdir

    def get_batch_dir(self):
        return files.get_wq_dir(self['run'])

    def get_master_url(self):
        master_url=files.get_wq_master_url(self['run'])
        return master_url

    def get_master_text(self):
        text = _wq_master_template % self
        return text

    def get_job_text(self):
        return _wq_template % self

    def get_job_url(self, filenum):
        job_url=files.get_wq_job_url(self['run'], filenum, missing=self.missing )
        return job_url

class LSFMaker(MakerBase):
    def get_tmpdir(self):
        return None

    def get_batch_dir(self):
        return files.get_lsf_dir(self['run'])

    def get_master_url(self):
        master_url=files.get_lsf_master_url(self['run'])
        return master_url

    def get_master_text(self):
        text = _lsf_master_template % self
        return text

    def get_job_text(self):
        return _lsf_template % self

    def get_job_url(self, filenum):
        job_url=files.get_lsf_job_url(self['run'], filenum, missing=self.missing)
        return job_url


class SLRMaker(MakerBase):
    def write_master(self):
        pass

    def get_tmpdir(self):
        return None

    def get_batch_dir(self):
        return files.get_slr_dir(self['run'])

    def get_master_url(self):
        return None

    def get_master_text(self):
        return None

    def get_job_text(self):
        return _slr_template % self

    def get_job_url(self, filenum):
        job_url=files.get_slr_job_url(
            self['run'],
            filenum,
            missing=self.missing,
        )
        return job_url

def get_walltime(hours):
    hours_int =int(hours)

    minutes = (hours - hours_int)*60
    minutes_int = int(minutes)

    seconds = (minutes - minutes_int)*60
    seconds_int = int(seconds)

    return '%02d:%02d:%02d' % (hours_int, minutes_int, seconds_int)

class SLRMakerShifter(MakerBase):
    def go(self):
        self.make_some_dirs()

        self.set_stats()

        self.write_batch_file()

    def get_ncores_per_node(self):
        if self['constraint']=='haswell':
            ncores=32
        elif self['constraint']=='knl,quad,cache':
            ncores=68
        else:
            raise ValueError("unknown constraint: '%s'" % self['constraint'])

        return ncores

    def set_stats(self):
        """
        for minions, we have a single job file representing
        an MPI job
        """

        # break up jobs evenly
        # note one of the cores is used as the master, so subtract one

        cores_per_node = self.get_ncores_per_node()

        # one job for each ship in the fleet
        njobs = self['nodes']*cores_per_node - 1

        # ncores includes the admiral
        ncores = self['nodes']*cores_per_node

        trials_per_job = util.get_trials_per_job_mpi(
            njobs, self['ntrials'],
        )

        hours = trials_per_job*self['sec_per']/3600.0

        calculated_walltime = get_walltime(hours)
        if 'walltime' not in self:
            self['walltime'] = calculated_walltime

        hours_int =int(hours)

        minutes = (hours - hours_int)*60
        minutes_int = int(minutes)

        seconds = (minutes - minutes_int)*60
        seconds_int = int(seconds)

        #self['walltime'] = '%02d:%02d:%02d' % (hours_int, minutes_int, seconds_int)
        self['ncores']   = ncores
        self['njobs']    = njobs
        self['ntrials_per'] = trials_per_job
        self['job_name'] = self['run']

        print("total cpu hours:",round(hours*ncores))
        print("walltime calculated:",calculated_walltime)
        print("walltime:",self['walltime'])

    def write_batch_file(self):
        """
        the main job file
        """
        fname=self.get_job_url()
        print("writing job submission script:",fname)
        with open(fname,'w') as fobj:
            text = _slr_template_shifter % self
            fobj.write(text)


    def get_tmpdir(self):
        return None

    def get_batch_dir(self):
        return files.get_slr_dir(self['run'])

    def get_master_url(self):
        return None

    def get_master_text(self):
        return None

    def get_job_text(self):
        return _slr_template % self

    def get_job_url(self):
        job_url=files.get_slr_minions_job_url(
            self['run'],
        )
        return job_url


class CondorMaker(MakerBase):
    def _remove_sub_files(self):
        from glob import glob
        d=files.get_condor_dir(self['run'])
        flist=glob(os.path.join(d, '*.condor'))
        for f in flist:
            print("removing:",f)
            os.remove(f)

    def write_batch_files(self):
        """
        currently only one written
        """

        self._remove_sub_files()
        overall_name = '-'.join( (self['run'].split('-'))[1:] )

        njobs=0
        ntrials_tot=0

        filenum=0

        ntrials_per, nsplit, time_hours = util.get_trials_nsplit(self)
        self['ntrials_per'] = ntrials_per


        icondor=0
        fobj=None

        for isplit in range(nsplit):

            seed = self.rng.randint(0,2**31-1)

            ntrials_tot += ntrials_per
            output = files.get_output_url(self['run'], isplit)

            if self.missing and os.path.exists(output):
                continue

            if (njobs % self['jobs_per_condor_sub'])==0:

                if fobj is not None:
                    fobj.close()

                subfile=files.get_condor_job_url(
                    self['run'],
                    icondor,
                )
                icondor+=1

                print("writing sub file",subfile)
                fobj=open(subfile,'w')

                head= _condor_submit_head % self
                fobj.write(head)


            logfile = output.replace('.fits','.log')
            job_name='%s-%05d' % (overall_name,isplit)

            self['output'] = output
            self['logfile'] = logfile
            self['job_name']=job_name
            self['seed'] = seed

            call=_condor_job_template % self
            fobj.write(call)
            fobj.write('\n')

            njobs += 1
            filenum += 1


        print('total jobs: ',njobs)
        print('total trials:',ntrials_tot)



    def get_tmpdir(self):
        return None

    def get_batch_dir(self):
        return files.get_condor_dir(self['run'])

    def get_master_url(self):
        master_url=files.get_condor_master_url(self['run'])
        return master_url

    def get_master_text(self):
        text = _condor_master_template % self
        return text



