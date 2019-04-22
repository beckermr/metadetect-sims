import os
from copy import deepcopy
import time
import numpy as np
import logging
import fitsio
import esutil as eu
import ngmix
from . import simulation
from . import descwl_sim

from . import fitters
from .fitters import MOFFitter, MOFFitterFull, MetacalFitter

from . import util

from . import visualize

logger = logging.getLogger(__name__)

def go(sim_conf,
       fit_conf,
       ntrials,
       seed,
       output_file,
       show=False,
       save=False,
       make_plots=False,
       #max_run_time_hours=1.0
      ):
    """
    run the simulation and fitter
    """
    logger.info('seed: %d' % seed)

    np.random.seed(seed)
    simseed=np.random.randint(0,2**30)
    fitseed=np.random.randint(0,2**30)

    rng = np.random.RandomState(simseed)
    fitrng = np.random.RandomState(fitseed)

    if sim_conf['sim_type']=='descwl':
        pos_sampler=descwl_sim.PositionSampler(sim_conf['positions'], rng)
        cat_sampler=descwl_sim.CatalogSampler(sim_conf, rng)
        sim=descwl_sim.DESWLSim(
            sim_conf,
            cat_sampler,
            pos_sampler,
            rng,
        )
    elif sim_conf['sim_type']=='pair':
        sim=simulation.PairSim(sim_conf, rng)
    else:
        sim=simulation.Sim(sim_conf, rng)

    fitter=get_fitter(sim_conf, fit_conf, fitrng)

    nsim=0
    nfit=0
    tm0_main = time.time()
    tm_sim=0.0
    tm_fit=0.0

    datalist=[]
    for i in range(ntrials):

        logger.debug("trial: %d/%d" % (i+1,ntrials))

        tm0=time.time()
        logger.debug('simulating images')
        sim.make_obs()
        tm_sim += time.time()-tm0

        nsim += 1

        if show:
            sim.show()
            if 'q'==input('hit a key (q to quit): '):
                return
        elif save:
            plt=sim.show(show=False)
            fname='rgb-%06d.png' % i
            logger.info('saving: %s' % fname)
            plt.write_img(1000,1000,fname)

        results, tm = do_fits(
            sim,
            fit_conf,
            fitter,
            show=show,
        )

        if results is not None:
            tm_fit += tm
            datalist.append( results )

    elapsed_time=time.time()-tm0_main
    nkept = len(datalist)

    meta=make_meta(
        ntrials, nsim, nfit,
        nkept, elapsed_time, tm_sim, tm_fit,
    )

    logger.info("kept: %d/%d %.2f" % (nkept,ntrials,float(nkept)/ntrials))
    logger.info("time minutes: %g" % meta['tm_minutes'][0])
    logger.info("time per trial: %g" % meta['tm_per_trial'][0])
    logger.info("time per sim: %g" % meta['tm_per_sim'][0])
    if nfit > 0:
        logger.info("time per fit: %g" % meta['tm_per_fit'][0])

    if nkept == 0:
        logger.info("no results to write")
    else:
        write_output(output_file, datalist, meta, fit_conf)

def do_fits(sim,
            fit_conf,
            fitter,
            show=False):

    tm0 = time.time()
    logger.debug('extracting')
    mbobs_list = sim.get_mbobs_list(
        weight_type=fit_conf['weight_type'],
    )
    if len(mbobs_list)==0:
        results=None
        nobj=0
        tm_fit=0
        logger.debug("no objects detected")
    else:

        # this gets a dict with an entry for each metacal type.  Each entry is
        # a list of MultiBandObsList
        mbobs_lists = get_metacal_with_redetect(sim, mbobs_list, fit_conf)

        results={}
        for type, mbobs_list in mbobs_lists.items():
            results[type] = run_one(fitter, mbobs_list)

    tm = time.time()-tm0
    return results, tm

def run_one(fitter, mbobs_list):
    """
    running on a set of objects
    """

    nobj = len(mbobs_list)
    if nobj > 0:

        res=fitter.go(mbobs_list)
        if res is None:
            logger.debug("failed to fit")

    return res


def redetect(mbobs, sx_config, detect_thresh):
    """
    currently we take all detections, but should probably
    trim to some central region to avoid edges

    since taking all, will want to use big postage stamps
    """
    import sep

    assert len(mbobs)==1
    assert len(mbobs[0])==1
    obs = mbobs[0][0]

    noise = np.sqrt(1.0/obs.weight[0,0])
    cat = sep.extract(
        obs.image,
        detect_thresh,
        err=noise,
        **sx_config
    )
    logger.debug('    redetect found %d' % cat.size)

    mbobs_list=[]
    if cat.size > 0:
        for i in range(cat.size): 
            row=cat['y'][i]
            col=cat['x'][i]

            tobs = obs.copy()
            # makes a copy
            j = tobs.jacobian
            j.set_cen(row=row, col=col)
            tobs.jacobian = j

            tobslist=ngmix.ObsList()
            tobslist.append(tobs)
            tmbobs = ngmix.MultiBandObsList()
            tmbobs.append(tobslist)

            mbobs_list.append( tmbobs )

    return mbobs_list

def get_metacal_with_redetect(sim, mbobs_list, fit_conf):
    """
    output is a dict keyed by the metacal type, each entry
    holding a mbobs_list
    """
    sx_config=deepcopy(sim['sx'])
    sx_config['filter_kernel'] = np.array(sx_config['filter_kernel'])
    detect_thresh = sx_config.pop('detect_thresh')

    metaconf=fit_conf['metacal']
    metacal_pars=metaconf['metacal_pars']
    types=metacal_pars['types']

    mbobs_lists = {}
    for type in types:
        mbobs_lists[type]=[]

    for mbobs in mbobs_list:

        odict = ngmix.metacal.get_all_metacal(
            mbobs,
            **metacal_pars
        )
        for type in types:
            tmbobs = odict[type]
            new_mbobs_list = redetect(tmbobs, sx_config, detect_thresh)

            mbobs_lists[type] += new_mbobs_list

    return mbobs_lists


def make_meta(ntrials,
              nsim,
              nfit,
              nkept,
              elapsed_time,
              tm_sim,
              tm_fit):
    dt=[
        ('ntrials','i8'),
        ('nsim','i8'),
        ('nfit','i8'),
        ('nkept','i8'),
        ('tm','f4'),
        ('tm_minutes','f4'),
        ('tm_per_trial','f4'),
        ('tm_sim','f4'),
        ('tm_fit','f4'),
        ('tm_per_sim','f4'),
        ('tm_per_fit','f4'),
    ]
    meta=np.zeros(1, dtype=dt)
    meta['ntrials'] = ntrials
    meta['nsim'] = nsim
    meta['nfit'] = nfit
    meta['nkept'] = nkept
    meta['tm_sim'] = tm_sim
    meta['tm_fit'] = tm_fit
    meta['tm'] = elapsed_time
    meta['tm_minutes'] = elapsed_time/60
    meta['tm_per_sim'] = tm_sim/nsim
    meta['tm_per_trial'] = elapsed_time/ntrials

    if nfit > 0:
        tm_per_fit=tm_fit/nfit
    else:
        tm_per_fit=-9999


    meta['tm_per_fit'] = tm_per_fit

    return meta


def get_fitter(sim_conf, fit_conf, fitrng):
    """
    get the appropriate fitting class
    """
    if 'bands' in sim_conf:
        nband=len(sim_conf['bands'])
    else:
        nband=sim_conf['nband']
    return fitters.Moments(fit_conf, nband, fitrng)


def profile_sim(seed,sim_conf,fit_conf,ntrials,output_file):
    """
    run the simulation using a profiler
    """
    import cProfile
    import pstats

    cProfile.runctx('go(seed,sim_conf,fit_conf,ntrials,output_file)',
                    globals(),locals(),
                    'profile_stats')
    
    p = pstats.Stats('profile_stats')
    p.sort_stats('time').print_stats()

def write_output(output_file, datalist, meta, fit_conf):

    odir=os.path.dirname(output_file)
    if not os.path.exists(odir):
        try:
            os.makedirs(odir)
        except:
            # probably a race condition
            pass

    logger.info("writing: %s" % output_file)

    types=fit_conf['metacal']['metacal_pars']['types']

    with fitsio.FITS(output_file,'rw',clobber=True) as fits:
        for mtype in types:

            dlist=[]
            for d in datalist:
                if d is not None and mtype in d:
                    # this is a list of results
                    tdata = d[mtype]
                    if tdata is not None:
                        dlist.append(tdata)

            if len(dlist) == 0:
                raise RuntimeError("no results found for type: %s" % mtype)

            data = eu.numpy_util.combine_arrlist(dlist)
            logger.info('    %s' % mtype)
            fits.write(data, extname=mtype)

        fits.write(meta, extname='meta_data')


