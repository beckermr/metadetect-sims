import os
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
    nobj_detected=0
    tm0_main = time.time()
    tm_sim=0.0
    tm_fit=0.0

    fit_conf['meta']=fit_conf.get('meta',None)
    metad=fit_conf['meta']

    datalist=[]
    truth_list=[]
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

        image_id=i
        if metad is not None and metad['dometa']:
            resdict, nobj, tm = do_meta(sim, fit_conf, fitter, image_id, show=show)
            reslist=[resdict]
        else:
            reslist, nobj, tm = do_fits(
                sim,
                fit_conf,
                fitter,
                image_id,
                show=show,
            )

        if reslist is not None:
            nobj_detected += nobj
            tm_fit += tm

            #for r in reslist:
            #    r['image_id'] = image_id
            #truth=sim.get_truth_catalog()
            #truth['image_id'] = image_id

            datalist += reslist
            #truth_list += [truth]

            if nobj > 0:
                nfit += 1

        #time_elapsed_hours=(time.time()-tm0_main)/3600.0
        #if time_elapsed_hours > max_run_time_hours:
        #    logger.info('stopping early to time limit: %g > %g' % (time_elapsed_hours,max_run_time_hours))
        #    break

    elapsed_time=time.time()-tm0_main
    nkept = len(datalist)

    meta=make_meta(
        ntrials, nsim, nfit, nobj_detected,
        nkept, elapsed_time, tm_sim, tm_fit,
    )

    logger.info("kept: %d/%d %.2f" % (nkept,ntrials,float(nkept)/ntrials))
    logger.info("time minutes: %g" % meta['tm_minutes'][0])
    logger.info("time per trial: %g" % meta['tm_per_trial'][0])
    logger.info("time per sim: %g" % meta['tm_per_sim'][0])
    if nfit > 0:
        logger.info("time per fit: %g" % meta['tm_per_fit'][0])
        logger.info("time fit per detected object: %g" % meta['tm_per_obj_detected'][0])

    if nkept == 0:
        logger.info("no results to write")
    else:
        if metad is not None and metad['dometa']:
            write_meta(output_file, datalist, meta, fit_conf)
        else:
            data = eu.numpy_util.combine_arrlist(datalist)
            #truth_data = eu.numpy_util.combine_arrlist(truth_list)
            #truth_
            write_output(output_file, data, meta)


def do_meta(sim, fit_conf, fitter, image_id, show=False):
    """
    currently we only do the full version, making
    metacal images for the full image set and
    sending all to MOF 
    """
    mtype=fit_conf['meta']['type']
    fit_conf['meta']['dosim'] = fit_conf['meta'].get('dosim',False)
    if mtype=='meta-detect':
        tup = do_meta_detect(sim, fit_conf, fitter, image_id, show=show)
    elif mtype in ['meta-mof','meta-max']:
        if fit_conf['fitter']=='mof-full':
            if fit_conf['meta']['dosim']:
                tup = do_meta_mof_full_withsim(sim, fit_conf, fitter, show=show)
            else:
                tup = do_meta_mof_full(sim, fit_conf, fitter, show=show)
        else:
            tup = do_meta_mof(sim, fit_conf, fitter, image_id, show=show)
    else:
        raise ValueError("bad meta type: '%s'" % mtype)
    
    return tup

def do_meta_detect(sim, fit_conf, fitter, image_id, show=False):
    """
    metacal the entire process, including detection.
    This means you lose a lot of detections
    """
    metaconf=fit_conf['metacal']
    metacal_pars=metaconf['metacal_pars']
    if metacal_pars.get('symmetrize_psf',False):
        if 'psf' in metaconf:
            fitters._fit_all_psfs([sim.obs], metaconf['psf'])
        else:
            fitters._fit_all_psfs([sim.obs], fit_conf['mof']['psf'])

    # Note using the simulation rng here not the fitting
    # rng
    odict=ngmix.metacal.get_all_metacal(
        sim.obs,
        rng=sim.rng,
        **metacal_pars
    )

    nobj=0
    tm_fit=0.0
    reslists={}
    for key in odict:
        reslist, tnobj, ttm = do_fits(
            sim,
            fit_conf,
            fitter,
            image_id,
            obs=odict[key],
            show=show,
        )
        nobj = max(nobj, tnobj)
        tm_fit += ttm
        reslists[key] = reslist

    return reslists, nobj, tm_fit

def do_meta_mof(sim, fit_conf, fitter, image_id, show=False):
    """
    metacal the MOF process but not detection

    also can do without mof

    build the catalog based on original images, but then
    run MOF on sheared versions
    """

    # create metacal versions of image
    metacal_pars=fit_conf['metacal']['metacal_pars']
    if metacal_pars.get('symmetrize_psf',False):
        fitters._fit_all_psfs([sim.obs], fit_conf['mof']['psf'])

    odict=ngmix.metacal.get_all_metacal(
        sim.obs,
        **metacal_pars
    )

    # create the catalog based on original images
    # this will just run sx and create seg and
    # cat
    medsifier=sim.get_medsifier()
    if medsifier.cat.size==0:
        return [], 0, 0.0

    nobj=0
    tm_fit=0.0
    reslists={}
    for key in odict:
        reslist, tnobj, ttm = do_fits(
            sim,
            fit_conf,
            fitter,
            image_id,
            cat=medsifier.cat,
            seg=medsifier.seg,
            obs=odict[key],
            show=show,
        )
        nobj = max(nobj, tnobj)
        tm_fit += ttm
        reslists[key] = reslist

    return reslists, nobj, tm_fit
    
def do_meta_mof_full(sim, fit_conf, fitter, show=False):
    """
    metacal the MOF process but not detection, using the
    full MOF not stamp MOF
    """
    import mof

    mofc=fit_conf['mof']

    # create metacal versions of image
    metacal_pars=fit_conf['metacal']['metacal_pars']
    if metacal_pars.get('symmetrize_psf',False):
        fitters._fit_all_psfs([sim.obs], fit_conf['mof']['psf'])

    odict=ngmix.metacal.get_all_metacal(
        sim.obs,
        **metacal_pars
    )

    # create the catalog based on original images
    # this will just run sx and create seg and
    # cat
    medser = sim.get_medsifier()
    if medser.cat.size==0:
        return {}, 0, 0.0


    mm=medser.get_multiband_meds()

    mn=mof.fofs.MEDSNbrs(
        mm.mlist,
        fit_conf['fofs'],
    )

    nbr_data = mn.get_nbrs()

    nf = mof.fofs.NbrsFoF(nbr_data)
    fofs = nf.get_fofs()
    if fofs.size==0:
        return [], 0, 0.0

    cat=medser.cat

    if show:
        sim._plot_fofs(mm, fofs)

    hist,rev=eu.stat.histogram(fofs['fofid'], rev=True)

    nobj=0
    tm_fit=0.0
    reslists={}
    for key in odict:

        mcal_obs = odict[key]

        ttm = time.time()
        reslist=[]
        for i in range(hist.size):
            assert rev[i] != rev[i+1],'all fof groups should be populated'
            w=rev[ rev[i]:rev[i+1] ]

            # assuming number is index+1
            indices=fofs['number'][w]-1
            nobj += indices.size

            subcat = cat[indices]

            # this is an array with all results from objects
            # in the fof
            mof_fitter, data = fitter.go(
                mcal_obs,
                subcat,
                ntry=mofc['ntry'],
                get_fitter=True,
            )

            if show:
                gmix=mof_fitter.get_convolved_gmix()
                tobs = sim.obs[0][0]
                _plot_compare_model(gmix, tobs)

            reslist.append(data)

        reslists[key] = reslist
        tm_fit += time.time() - ttm

    return reslists, nobj, tm_fit
 
def _plot_compare_model(gmix, tobs):
    import biggles
    import images
    model_im = gmix.make_image(tobs.image.shape, jacobian=tobs.jacobian)
    imdiff = model_im - tobs.image
    tab = biggles.Table(2,2,aspect_ratio=1.0)
    tab[0,0] = images.view(tobs.image,show=False,title='im')
    tab[0,1] = images.view(model_im,show=False,title='model')
    tab[1,0] = images.view(imdiff,show=False,title='diff')
    tab.show()
    if input('hit a key (q to quit): ')=='q':
        stop

def do_meta_mof_full_withsim_old(sim, fit_conf, fitter, show=False):
    """
    not finding groups, just fitting everything.  This means
    it won't work on bigger images with lots of empty space
    """
    import mof

    assert fit_conf['fofs']['find_fofs']==False

    tm0 = time.time()
    mofc=fit_conf['mof']

    # create metacal versions of image
    metacal_pars=fit_conf['metacal']['metacal_pars']
    if metacal_pars.get('symmetrize_psf',False):
        fitters._fit_all_psfs([sim.obs], fit_conf['mof']['psf'])

    # create the catalog based on original images
    # this will just run sx and create seg and
    # cat
    medser = sim.get_medsifier()
    cat=medser.cat

    mof_fitter, data = fitter.go(
        sim.obs,
        cat,
        ntry=mofc['ntry'],
        get_fitter=True,
    )

    tobs = sim.obs[0][0]
    if show:
        gmix=mof_fitter.get_convolved_gmix()
        _plot_compare_model(gmix, tobs)

    sim_mbobs_after = ngmix.MultiBandObsList()
    sim_mbobs_before = ngmix.MultiBandObsList()

    for band,obslist in enumerate(sim.obs):
        sim_obslist_after = ngmix.ObsList()
        sim_obslist_before = ngmix.ObsList()
        for obsnum,obs in enumerate(obslist):
            gmix=mof_fitter.get_convolved_gmix(
                band=band,
                obsnum=obsnum,
            )
            sobs_after = ngmix.simobs.simulate_obs(gmix, obs, add_noise=False)
            sobs_before = ngmix.simobs.simulate_obs(gmix, obs, add_noise=True)

            # get another noise field to be used in metacal fixnoise
            # also used for the after obs
            sobs_before2 = ngmix.simobs.simulate_obs(gmix, obs, add_noise=True)

            sobs_before.noise = sobs_before2.noise_image

            # meta data gets passed on by metacal, we can use the total
            # noise image later for the 'after' obs
            sobs_after.meta['total_noise'] = \
                    sobs_before.noise_image + sobs_before2.noise_image
            #import images
            #images.view(sobs_before.image)
            #stop

            sim_obslist_after.append( sobs_after )
            sim_obslist_before.append( sobs_before )

        sim_mbobs_after.append(sim_obslist_after)
        sim_mbobs_before.append(sim_obslist_before)

    odict=ngmix.metacal.get_all_metacal(
        sim.obs,
        **metacal_pars
    )

    after_metacal_pars={}
    after_metacal_pars.update(metacal_pars)
    after_metacal_pars['fixnoise']=False
    odict_after=ngmix.metacal.get_all_metacal(
        sim_mbobs_after,
        **after_metacal_pars
    )

    before_metacal_pars={}
    before_metacal_pars.update(metacal_pars)
    before_metacal_pars['use_noise_image']=True

    odict_before=ngmix.metacal.get_all_metacal(
        sim_mbobs_before,
        **before_metacal_pars
    )

    # now go and add noise after shearing
    for key in odict_after:
        mbobs=odict_after[key]
        for obslist in mbobs:
            for obs in obslist:
                #import images
                #images.view(obs.image)
                #stop

                # because with fixnoise we added extra noise
                obs.weight *= 0.5
                obs.image += obs.meta['total_noise']

    # now run fits on all these images
    allstuff=[
        (None,odict),
        ('after',odict_after),
        ('before',odict_before),
    ]

    reslists={}
    for name,todict in allstuff:
        n=util.Namer(front=name)
        for key,mbobs in todict.items():
            data = fitter.go(
                mbobs,
                cat,
                ntry=mofc['ntry'],
            )

            reslists[n(key)] = [data]

    nobj = cat.size
    tm_fit = time.time()-tm0
    return reslists, nobj, tm_fit
 

def do_meta_mof_full_withsim_old2(sim, fit_conf, fitter, show=False):
    """
    not finding groups, just fitting everything.  This means
    it won't work on bigger images with lots of empty space
    """
    import mof

    assert fit_conf['fofs']['find_fofs']==False

    tm0 = time.time()
    mofc=fit_conf['mof']

    # create metacal versions of image
    metacal_pars=fit_conf['metacal']['metacal_pars']
    if metacal_pars.get('symmetrize_psf',False):
        fitters._fit_all_psfs([sim.obs], fit_conf['mof']['psf'])

    # create the catalog based on original images
    # this will just run sx and create seg and
    # cat
    medser = sim.get_medsifier()
    cat=medser.cat

    mof_fitter, data = fitter.go(
        sim.obs,
        cat,
        ntry=mofc['ntry'],
        get_fitter=True,
    )

    tobs = sim.obs[0][0]
    if show:
        gmix=mof_fitter.get_convolved_gmix()
        _plot_compare_model(gmix, tobs)

    sim_mbobs = ngmix.MultiBandObsList()

    for band,obslist in enumerate(sim.obs):
        sim_obslist = ngmix.ObsList()
        for obsnum,obs in enumerate(obslist):
            gmix=mof_fitter.get_convolved_gmix(
                band=band,
                obsnum=obsnum,
            )

            psf_gmix = obs.psf.gmix
            psf_im = psf_gmix.make_image(
                obs.psf.image.shape,
                jacobian=obs.psf.jacobian,
            )
            pn = psf_im.max()/1000.0
            psf_im += fitter.rng.normal(size=psf_im.shape, scale=pn)
            psf_wt = psf_im*0 + 1.0/pn**2


            sobs = ngmix.simobs.simulate_obs(gmix, obs, add_noise=False)
            sobs.psf.image = psf_im
            sobs.psf.weight = psf_wt

            sobs_noisy = ngmix.simobs.simulate_obs(gmix, obs, add_noise=True, rng=fitter.rng)
            # this one for fixnoise
            sobs_noisy2 = ngmix.simobs.simulate_obs(gmix, obs, add_noise=True, rng=fitter.rng)

            # to be added after shearing
            sobs.meta['noise'] = sobs_noisy.noise_image
            sobs.meta['noise_for_fixnoise'] = sobs_noisy2.noise_image

            sim_obslist.append( sobs )

        sim_mbobs.append(sim_obslist)

    # for the measurement on real data
    odict=ngmix.metacal.get_all_metacal(
        sim.obs,
        rng=fitter.rng,
        **metacal_pars
    )

    # shear the noiseless sim
    sim_metacal_pars={}
    sim_metacal_pars.update(metacal_pars)
    sim_metacal_pars['fixnoise']=False
    sim_metacal_pars['types']=['1p','1m']
    sim_odict=ngmix.metacal.get_all_metacal(
        sim_mbobs,
        **sim_metacal_pars
    )


    # now add the noise; the noise will be the same
    # realization for each
    for key,mbobs in sim_odict.items():
        for obslist in mbobs:
            for obs in obslist:
                #import images
                #images.view(obs.image)
                #stop

                obs.image += obs.meta['noise']
                obs.noise = obs.meta['noise_for_fixnoise']


    sim_metacal_pars2={}
    sim_metacal_pars2.update(metacal_pars)
    sim_metacal_pars2['use_noise_image']=True

    sim_odict_1p=ngmix.metacal.get_all_metacal(
        sim_odict['1p'],
        **sim_metacal_pars2
    )
    sim_odict_1m=ngmix.metacal.get_all_metacal(
        sim_odict['1m'],
        **sim_metacal_pars2
    )




    reslists={}
    reslists.update(
        _process_one_full_mof_metacal(mofc, odict, cat, fitter)
    )
    reslists.update(
        _process_one_full_mof_metacal(mofc, sim_odict_1p, cat, fitter, prefix='sim1p')
    )
    reslists.update(
        _process_one_full_mof_metacal(mofc, sim_odict_1m, cat, fitter, prefix='sim1m')
    )

    nobj = cat.size
    tm_fit = time.time()-tm0

    return reslists, nobj, tm_fit


def do_meta_mof_full_withsim(sim, fit_conf, fitter, show=False):
    """
    not finding groups, just fitting everything.  This means
    it won't work on bigger images with lots of empty space
    """
    import galsim
    import mof

    assert fit_conf['fofs']['find_fofs']==False

    tm0 = time.time()
    mofc=fit_conf['mof']

    # create metacal versions of image
    metacal_pars=fit_conf['metacal']['metacal_pars']
    if metacal_pars.get('symmetrize_psf',False):
        fitters._fit_all_psfs([sim.obs], fit_conf['mof']['psf'])

    # create the catalog based on original images
    # this will just run sx and create seg and
    # cat
    medser = sim.get_medsifier()
    if medser.cat.size==0:
        return {}, 0, 0.0
    cat=medser.cat

    mof_fitter, data = fitter.go(
        sim.obs,
        cat,
        ntry=mofc['ntry'],
        get_fitter=True,
    )

    tobs = sim.obs[0][0]
    if show:
        gmix=mof_fitter.get_convolved_gmix()
        _plot_compare_model(gmix, tobs)

    sim_mbobs_1p = ngmix.MultiBandObsList()
    sim_mbobs_1m = ngmix.MultiBandObsList()

    for band,obslist in enumerate(sim.obs):
        sim_obslist_1p = ngmix.ObsList()
        sim_obslist_1m = ngmix.ObsList()
        band_gmix0=mof_fitter.get_gmix(
            band=band,
        )

        theta=fitter.rng.uniform(low=0.0, high=np.pi)
        #gmix0 = gmix0.get_rotated(theta)
        for obsnum,obs in enumerate(obslist):

            jac=obs.jacobian
            scale=jac.scale

            gmix0 = band_gmix0.copy()

            gmix0.set_flux(gmix0.get_flux()/scale**2)

            # cheating on psf for now
       
            ny,nx = obs.image.shape

            # galsim does everything relative to the canonical center, but
            # for the mof fitter we had the origin a 0,0.  Shift over by
            # the cen
            ccen=(np.array(obs.image.shape)-1.0)/2.0

            gs0 = gmix0.make_galsim_object()

            gs0 = gs0.shift(dx=-ccen[1]*scale, dy=-ccen[1]*scale)

            if show and obsnum==0:
                import images
                #gs = gmix0.make_galsim_object(psf=sim.psf)
                #gs = gs.shift(dx=-ccen[1]*scale, dy=-ccen[1]*scale)
                gs=galsim.Convolve(gs0, sim.psf)
                tim = gs.drawImage(nx=nx, ny=ny, scale=sim['pixel_scale']).array
                images.compare_images(sim.obs[0][0].image, tim)
                if 'q'==input('hit a key (q to quit): '):
                    stop

            gs0_1p = gs0.shear(g1= 0.01, g2=0.0)
            gs0_1m = gs0.shear(g1=-0.01, g2=0.0)

            gs_1p = galsim.Convolve(gs0_1p, sim.psf)
            gs_1m = galsim.Convolve(gs0_1m, sim.psf)

            im_1p = gs_1p.drawImage(nx=nx, ny=ny, scale=sim['pixel_scale']).array
            im_1m = gs_1m.drawImage(nx=nx, ny=ny, scale=sim['pixel_scale']).array

            # adding same noise to both

            noise_image = ngmix.simobs.get_noise_image(obs.weight, rng=fitter.rng)
            im_1p += noise_image
            im_1m += noise_image

            # this one will be used for fixnoise
            noise_image2 = ngmix.simobs.get_noise_image(obs.weight, rng=fitter.rng)

            sobs_1p = ngmix.Observation(
                im_1p,
                weight=obs.weight.copy(),
                jacobian=jac,
                psf=obs.psf.copy(),
                noise=noise_image2.copy(),
            )
            sobs_1m = ngmix.Observation(
                im_1m,
                weight=obs.weight.copy(),
                jacobian=jac,
                psf=obs.psf.copy(),
                noise=noise_image2.copy(),
            )

            sim_obslist_1p.append( sobs_1p )
            sim_obslist_1m.append( sobs_1m )

        sim_mbobs_1p.append( sim_obslist_1p )
        sim_mbobs_1m.append( sim_obslist_1m )

    # for the measurement on real data
    odict=ngmix.metacal.get_all_metacal(
        sim.obs,
        rng=fitter.rng,
        **metacal_pars
    )

    sim_metacal_pars={}
    sim_metacal_pars.update(metacal_pars)
    sim_metacal_pars['use_noise_image']=True

    sim_odict_1p=ngmix.metacal.get_all_metacal(
        sim_mbobs_1p,
        rng=fitter.rng, # not needed
        **sim_metacal_pars
    )
    sim_odict_1m=ngmix.metacal.get_all_metacal(
        sim_mbobs_1m,
        rng=fitter.rng, # not needed
        **sim_metacal_pars
    )

    reslists={}
    reslists.update(
        _process_one_full_mof_metacal(mofc, odict, cat, fitter)
    )
    reslists.update(
        _process_one_full_mof_metacal(mofc, sim_odict_1p, cat, fitter, prefix='sim1p')
    )
    reslists.update(
        _process_one_full_mof_metacal(mofc, sim_odict_1m, cat, fitter, prefix='sim1m')
    )

    nobj = cat.size
    tm_fit = time.time()-tm0

    return reslists, nobj, tm_fit

def _process_one_full_mof_metacal(mofc, odict, cat, fitter, prefix=None):

    n=util.Namer(front=prefix)

    reslist={}
    for key,mbobs in odict.items():
        data = fitter.go(
            mbobs,
            cat,
            ntry=mofc['ntry'],
        )
        reslist[n(key)] = [data]

    return reslist
 



def do_fits(sim,
            fit_conf,
            fitter,
            image_id,
            cat=None,
            seg=None,
            obs=None,
            show=False):

    fof_conf=fit_conf['fofs']
    weight_type=fit_conf['weight_type']

    if fof_conf.get('merge_fofs',False):
        # currently just take the brightest, thinking of the
        # two object case, but we can make this more
        # sophisticated later using the actual fof groups
        mbobs_list = sim.get_mbobs_list(
            weight_type=weight_type,
        )
        if len(mbobs_list) > 0:
            # always do this to center stamp
            mbobs_list = extract_single_obs(sim, mbobs_list)

    else:
        if fof_conf['find_fofs']:
            logger.debug('extracting and finding fofs')


            if fof_conf.get('link_all',False):
                mbobs_list = sim.get_mbobs_list(
                    cat=cat,
                    seg=seg,
                    obs=obs,
                    weight_type=weight_type,
                )
                mbobs_list = [mbobs_list]
            else:
                mbobs_list = sim.get_fofs(
                    fof_conf,
                    cat=cat,
                    seg=seg,
                    obs=obs,
                    weight_type=weight_type,
                    show=show,
                )

            for fof_id,tmbobs_list in enumerate(mbobs_list):
                for mbobs in tmbobs_list:
                    mbobs.meta['image_id'] = image_id
                    mbobs.meta['fof_id'] = fof_id
                    #for obslist in mbobs:
                    #    for obs in obslist:
                    #        obs.meta['image_id'] = image_id
                    #        obs.meta['fof_id'] = fof_id

        else:
            logger.debug('extracting')
            mbobs_list = sim.get_mbobs_list(
                cat=cat,
                seg=seg,
                obs=obs,
                weight_type=weight_type,
            )
            for fof_id,mbobs in enumerate(mbobs_list):
                mbobs.meta['image_id'] = image_id
                mbobs.meta['fof_id'] = fof_id
                #for obslist in mbobs:
                #    for obs in obslist:
                #        obs.meta['image_id'] = image_id
                #        obs.meta['fof_id'] = fof_id

    if len(mbobs_list)==0:
        reslist=None
        nobj=0
        tm_fit=0
        logger.debug("no objects detected")
    else:

        if fof_conf['find_fofs']:
            if show:
                for i,mbl in enumerate(mbobs_list):
                    title='FoF %d/%d' % (i+1,len(mbobs_list))
                    visualize.view_mbobs_list(mbl,title=title,dims=[800,800])
                if 'q'==input('hit a key (q to quit): '):
                    stop
            # mbobs_list is really a list of those
            reslist, nobj, tm_fit = run_fofs(fitter, mbobs_list)
        else:

            if show:
                visualize.view_mbobs_list(mbobs_list,weight=True)
                if 'q'==input('hit a key (q to quit): '):
                    stop

            reslist, nobj, tm_fit = run_one_fof(fitter, mbobs_list)

        logger.debug("    processed %d objects" % nobj)

    return reslist, nobj, tm_fit

def extract_single_obs(sim, mbobs_list):
    assert len(mbobs_list[0])==1,'one band for now'

    box_size = max( [max(m[0][0].image.shape) for m in mbobs_list] )
    half_box_size = box_size//2

    # get the mean position
    flux = np.array( [m[0][0].meta['flux'] for m in mbobs_list] )
    rows = np.array( [m[0][0].meta['orig_row'] for m in mbobs_list] )
    cols = np.array( [m[0][0].meta['orig_col'] for m in mbobs_list] )
    flux_sum=flux.sum()
    if flux_sum == 0.0:
        logger.info('zero flux')
        return []

    mean_row = (rows*flux).sum()/flux.sum()
    mean_col = (cols*flux).sum()/flux.sum()
    print('rows:',rows,'row mean:',mean_row)
    print('cols:',cols,'col mean:',mean_col)

    imax = flux.argmax()
    mbobs_imax=mbobs_list[imax]

    # again assuming single band
    obs=sim.obs[0][0].copy()
    #obs.meta.update(mbobs_list[0][0][0].meta)
    maxrow,maxcol=obs.image.shape


    # now get the adaptive moments center
    j=obs.jacobian
    j.set_cen(row=mean_row, col=mean_col)
    obs.jacobian=j

    shiftmax=3.0 # arcsec
    Tguess=4.0*j.scale**2
    runner=ngmix.bootstrap.EMRunner(obs, Tguess, 1, {'maxiter':1000,'tol':1.0e-4})
    runner.go(ntry=2)
    fitter=runner.fitter
    res=fitter.get_result()
        #try:
        #    fitter=ngmix.admom.run_admom(obs, Tguess, shiftmax=shiftmax)
        #    res=fitter.get_result()
        #except ngmix.GMixRangeError as err:
        #    res={'flags':2}
            
        #if res['flags']==0:
        #    break

    if res['flags'] != 0:
        print("    could not fit for center",res)
        return []

    gm=fitter.get_gmix()
    v,u = gm.get_cen()

    newrow, newcol = j.get_rowcol(v, u)
    print("new cen:",newrow,newcol)

    start_row = int(newrow) - half_box_size + 1
    start_col = int(newcol) - half_box_size + 1
    end_row   = int(newrow) + half_box_size + 1 # plus one for slices
    end_col   = int(newcol) + half_box_size + 1

    if start_row < 0:
        start_row=0
    if start_col < 0:
        start_col=0
    if start_row > maxrow:
        start_row=maxrow
    if start_col > maxcol:
        start_col=maxcol

    cutout_row = newrow - start_row
    cutout_col = newcol - start_col


    im = obs.image[
        start_row:end_row,
        start_col:end_col,
    ]
    wt = obs.weight[
        start_row:end_row,
        start_col:end_col,
    ]

    j=obs.jacobian
    j.set_cen(row=cutout_row, col=cutout_col)

    nobs=ngmix.Observation(
        im,
        weight=wt,
        jacobian=j,
        bmask=np.zeros(im.shape,dtype='i4'),
        meta=mbobs_list[0][0][0].meta,
        psf=obs.psf.copy(),
    )


    mbobs = ngmix.MultiBandObsList()
    obslist=ngmix.ObsList()

    obslist.append(nobs)
    mbobs.append(obslist)


    return [mbobs]

def run_fofs(fitter, fof_mbobs_lists):
    """
    run all fofs that were found
    """
    datalist=[]
    nobj=0
    tm=0.0

    nfofs=len(fof_mbobs_lists)
    logger.debug("processing: %d fofs" % nfofs)
    for i,mbobs_list in enumerate(fof_mbobs_lists):
        logger.debug("    fof: %d/%d has %d members" % (i+1,nfofs,len(mbobs_list)))
        reslist, t_nobj, t_tm = run_one_fof(fitter, mbobs_list)

        nobj += t_nobj
        tm += t_tm
        datalist += reslist

    return datalist, nobj, tm

def run_one_fof(fitter, mbobs_list):
    """
    running on a set of objects
    """

    nobj = len(mbobs_list)

    datalist=[]
    tm0=time.time()
    if nobj > 0:

        res=fitter.go(mbobs_list)
        if res is None:
            logger.debug("failed to fit")
        else:
            datalist.append(res)
    tm = time.time()-tm0

    return datalist, nobj, tm

def make_meta(ntrials,
              nsim,
              nfit,
              nobj_detected,
              nkept,
              elapsed_time,
              tm_sim,
              tm_fit):
    dt=[
        ('ntrials','i8'),
        ('nsim','i8'),
        ('nfit','i8'),
        ('nobj_detected','i8'),
        ('nkept','i8'),
        ('tm','f4'),
        ('tm_minutes','f4'),
        ('tm_per_trial','f4'),
        ('tm_sim','f4'),
        ('tm_fit','f4'),
        ('tm_per_sim','f4'),
        ('tm_per_fit','f4'),
        ('tm_per_obj_detected','f4'),
    ]
    meta=np.zeros(1, dtype=dt)
    meta['ntrials'] = ntrials
    meta['nsim'] = nsim
    meta['nfit'] = nfit
    meta['nobj_detected'] = nobj_detected
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

    if nobj_detected > 0:
        tm_per_obj_detected =tm_fit/nobj_detected
    else:
        tm_per_obj_detected=-9999

    meta['tm_per_fit'] = tm_per_fit
    meta['tm_per_obj_detected'] = tm_per_obj_detected

    return meta


def get_fitter(sim_conf, fit_conf, fitrng):
    """
    get the appropriate fitting class
    """
    if 'bands' in sim_conf:
        nband=len(sim_conf['bands'])
    else:
        nband=sim_conf['nband']

    if fit_conf['fitter'] in ['metacal','metacal-avg','metacal-2comp','metacal-em']:
        if fit_conf['fofs']['find_fofs']:
            mof_fitter = MOFFitter(fit_conf, nband, fitrng)
        else:
            mof_fitter=None

        if fit_conf['fitter'] == 'metacal-avg':
            cls = fitters.MetacalAvgFitter
        elif fit_conf['fitter'] == 'metacal-2comp':
            cls = fitters.Metacal2CompFitter
        elif fit_conf['fitter'] == 'metacal-em':
            cls = fitters.EMMetacalFitter
        else:
            cls = fitters.MetacalFitter

        fitter=cls(
            fit_conf,
            nband,
            fitrng,
            mof_fitter=mof_fitter,
        )

    elif fit_conf['fitter'] in ['metacal-am','metacal-am-avg']:
        if fit_conf['fofs']['find_fofs']:
            mof_fitter = MOFFitter(fit_conf, nband, fitrng)
        else:
            mof_fitter=None

        if fit_conf['fitter']=='metacal-am-avg':
            cls=fitters.AdmomMetacalAvgFitter
        else:
            cls=fitters.AdmomMetacalFitter
        fitter=cls(
            fit_conf,
            nband,
            fitrng,
            mof_fitter=mof_fitter,
        )

    elif fit_conf['fitter'] in ['metacal-mom']:
        if fit_conf['fofs']['find_fofs']:
            mof_fitter = MOFFitter(fit_conf, nband, fitrng)
        else:
            mof_fitter=None

        cls=fitters.MomentMetacalFitter
        fitter=cls(
            fit_conf,
            nband,
            fitrng,
            mof_fitter=mof_fitter,
        )


    elif fit_conf['fitter']=='mof':
        fitter = MOFFitter(fit_conf, nband, fitrng)

    elif fit_conf['fitter']=='mof-full':
        fitter = MOFFitterFull(fit_conf, nband, fitrng)

    elif fit_conf['fitter']=='max':
        fitter = fitters.MaxFitter(fit_conf, nband, fitrng)

    elif fit_conf['fitter']=='mom':
        fitter = fitters.Moments(fit_conf, nband, fitrng)


    else:
        raise ValueError("bad fitter: '%s'" % fit_conf['fitter'])

    return fitter


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


def write_output(output_file, data, meta):
    """
    write an output file, making the directory if needed
    """
    odir=os.path.dirname(output_file)
    if not os.path.exists(odir):
        try:
            os.makedirs(odir)
        except:
            # probably a race condition
            pass

    logger.info("writing: %s" % output_file)
    with fitsio.FITS(output_file,'rw',clobber=True) as fits:
        fits.write(data, extname='model_fits')
        fits.write(meta, extname='meta_data')

def write_meta(output_file, datalist, meta, fit_conf):

    odir=os.path.dirname(output_file)
    if not os.path.exists(odir):
        try:
            os.makedirs(odir)
        except:
            # probably a race condition
            pass

    logger.info("writing: %s" % output_file)

    types=fit_conf['metacal']['metacal_pars']['types']
    if fit_conf['meta']['dosim']:
        types_sim1p = ['sim1p_'+t for t in types]
        types_sim1m = ['sim1m_'+t for t in types]
        types += types_sim1p
        types += types_sim1m

    with fitsio.FITS(output_file,'rw',clobber=True) as fits:
        for mtype in types:

            dlist=[]
            for d in datalist:
                if d is not None:
                    if mtype in d:
                        # this is a list of results
                        if d[mtype] is not None:
                            dlist += d[mtype]

            if len(dlist) == 0:
                raise RuntimeError("no results found for type: %s" % mtype)

            data = eu.numpy_util.combine_arrlist(dlist)
            logger.info('    %s' % mtype)
            fits.write(data, extname=mtype)

        fits.write(meta, extname='meta_data')


