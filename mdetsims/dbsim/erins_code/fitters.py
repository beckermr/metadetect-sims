import logging
import numpy as np
from numpy import array
from pprint import pprint

import esutil as eu
import ngmix
from ngmix.gexceptions import GMixRangeError
from ngmix.observation import Observation
from ngmix.gexceptions import GMixMaxIterEM
from ngmix.gmix import GMixModel
from ngmix.gexceptions import BootPSFFailure, BootGalFailure

from .util import log_pars, TryAgainError, Namer

import mof

logger = logging.getLogger(__name__)

METACAL_TYPES=['noshear','1p','1m','2p','2m']

class FitterBase(dict):
    def __init__(self, run_conf, nband, rng):

        self.nband=nband
        self.rng=rng
        self.update(run_conf)

    def go(self, mbobs_list):
        """
        do measurements.  This is abstract
        """
        raise NotImplementedError("implement go()")

    def _get_prior(self, conf):
        """
        Set all the priors
        """
        import ngmix
        from ngmix.joint_prior import PriorSimpleSep, PriorBDFSep

        if 'priors' not in conf:
            return None

        ppars=conf['priors']
        if ppars.get('prior_from_mof',False):
            return None

        # g
        gp = ppars['g']
        assert gp['type']=="ba"
        g_prior = self._get_prior_generic(gp)

        T_prior = self._get_prior_generic(ppars['T'])
        flux_prior = self._get_prior_generic(ppars['flux'])

        # center
        cp=ppars['cen']
        assert cp['type'] == 'normal2d'
        cen_prior = self._get_prior_generic(cp)

        if conf['model']=='bdf':
            assert 'fracdev' in ppars,"set fracdev prior for bdf model"
            fp = ppars['fracdev']
            assert fp['type'] == 'normal','only normal prior supported for fracdev'

            fracdev_prior = self._get_prior_generic(fp)

            prior = PriorBDFSep(
                cen_prior,
                g_prior,
                T_prior,
                fracdev_prior,
                [flux_prior]*self.nband,
            )

        else:

            prior = PriorSimpleSep(
                cen_prior,
                g_prior,
                T_prior,
                [flux_prior]*self.nband,
            )

        return prior

    def _get_prior_generic(self, ppars):
        ptype=ppars['type']

        if ptype=="flat":
            prior=ngmix.priors.FlatPrior(*ppars['pars'], rng=self.rng)

        elif ptype == 'two-sided-erf':
            prior=ngmix.priors.TwoSidedErf(*ppars['pars'], rng=self.rng)

        elif ptype=='normal':
            prior = ngmix.priors.Normal(
                ppars['mean'],
                ppars['sigma'],
                rng=self.rng,
            )

        elif ptype=='log-normal':
            prior = ngmix.priors.LogNormal(
                ppars['mean'],
                ppars['sigma'],
                rng=self.rng,
            )


        elif ptype=='normal2d':
            prior=ngmix.priors.CenPrior(
                0.0,
                0.0,
                ppars['sigma'],
                ppars['sigma'],
                rng=self.rng,
            )

        elif ptype=='ba':
            prior = ngmix.priors.GPriorBA(ppars['sigma'], rng=self.rng)

        else:
            raise ValueError("bad prior type: '%s'" % ptype)

        return prior

class MOFFitter(FitterBase):
    def __init__(self, *args, **kw):

        super(MOFFitter,self).__init__(*args, **kw)

        self.mof_prior = self._get_prior(self['mof'])


    def go(self, mbobs_list, ntry=2, get_fitter=False):
        """
        run the multi object fitter

        parameters
        ----------
        mbobs_list: list of MultiBandObsList
            One for each object.  If it is a simple
            MultiBandObsList it will be converted
            to a list

        returns
        -------
        data: ndarray
            Array with all output fields
        """
        if not isinstance(mbobs_list,list):
            mbobs_list=[mbobs_list]

        try:
            _fit_all_psfs(mbobs_list, self['mof']['psf'])

            mofc = self['mof']
            guess_from_priors=mofc.get('guess_from_priors',False)
            fitter = mof.MOFStamps(
                mbobs_list,
                mofc['model'],
                prior=self.mof_prior,
            )
            for i in range(ntry):
                guess=mof.moflib.get_stamp_guesses(
                    mbobs_list,
                    mofc['detband'],
                    mofc['model'],
                    self.rng,
                    prior=self.mof_prior,
                    guess_from_priors=guess_from_priors,
                )
                fitter.go(guess)

                res=fitter.get_result()
                if res['flags']==0:
                    break

        except BootPSFFailure as err:
            print(str(err))
            res={'flags':1}

        if res['flags'] != 0:
            fitter=None
            data=None
        else:
            average_fof_shapes = self.get('average_fof_shapes',False)
            if average_fof_shapes:
                logger.debug('averaging fof shapes')
                resavg=fitter.get_result_averaged_shapes()
                data=self._get_output(mbobs_list[0],[resavg], fitter.nband)
            else:
                reslist=fitter.get_result_list()
                data=self._get_output(mbobs_list[0], reslist, fitter.nband)

        if get_fitter:
            return fitter, data
        else:
            return data


    def _get_dtype(self, npars, nband):
        n=Namer(front=self['mof']['model'])
        dt = [
            ('image_id','i4'),
            ('fof_id','i4'), # fof id within image
            ('psf_g','f8',2),
            ('psf_T','f8'),
            (n('nfev'),'i4'),
            (n('s2n'),'f8'),
            (n('pars'),'f8',npars),
            (n('pars_cov'),'f8',(npars,npars)),
            (n('g'),'f8',2),
            (n('g_cov'),'f8',(2,2)),
            (n('T'),'f8'),
            (n('T_err'),'f8'),
            (n('T_ratio'),'f8'),
            (n('flux'),'f8',nband),
            (n('flux_cov'),'f8',(nband,nband)),
            (n('flux_err'),'f8',nband),
        ]

        if self['mof']['model']=='bdf':
            dt += [
                (n('fracdev'),'f8'),
                (n('fracdev_err'),'f8'),
            ]
        return dt

    def _get_output(self, mbobs_example, reslist,nband):

        npars=reslist[0]['pars'].size

        model=self['mof']['model']
        n=Namer(front=model)

        dt=self._get_dtype(npars, nband)
        output=np.zeros(len(reslist), dtype=dt)

        meta=mbobs_example.meta
        output['image_id'] = meta['image_id']
        output['fof_id'] = meta['fof_id']

        for i,res in enumerate(reslist):
            t=output[i] 

            for name,val in res.items():
                if name=='nband':
                    continue

                if 'psf' in name:
                    t[name] = val
                else:
                    nname=n(name)
                    t[nname] = val

        return output

class MOFFitterFull(MOFFitter):
    def __init__(self, *args, **kw):
        """
        we don't use the MOFFitter init
        """
        FitterBase.__init__(self, *args, **kw)

    def go(self, mbobs, cat, ntry=2, get_fitter=False):
        """
        run the multi object fitter

        parameters
        ----------
        mbobs_list: list of MultiBandObsList
            One for each object

        returns
        -------
        data: ndarray
            Array with all output fields
        """

        mofc = self['mof']
        guess_from_priors=mofc.get('guess_from_priors',False)
        nband=len(mbobs)
        jacobian=mbobs[0][0].jacobian

        prior=self._get_prior(
            cat,
            jacobian,
        )

        try:
            _fit_all_psfs([mbobs], mofc['psf'])

            fitter = mof.MOF(
                mbobs,
                mofc['model'],
                cat.size,
                prior=prior,
            )

            for i in range(ntry):
                guess=mof.moflib.get_full_image_guesses(
                    cat,
                    nband,
                    jacobian,
                    mofc['model'],
                    self.rng,
                    prior=prior,
                    guess_from_priors=guess_from_priors,
                )
                fitter.go(guess)

                res=fitter.get_result()
                if res['flags']==0:
                    break

        except BootPSFFailure as err:
            print(str(err))
            res={'flags':1}

        if res['flags'] != 0:
            fitter=None
            data=None
        else:
            average_fof_shapes = self.get('average_fof_shapes',False)
            if average_fof_shapes:
                raise NotImplementedError('make sure works for full mof')
                logger.debug('averaging fof shapes')
                resavg=fitter.get_result_averaged_shapes()
                data=self._get_output([resavg], fitter.nband)
            else:
                reslist=fitter.get_result_list()
                data=self._get_output(reslist, fitter.nband)

        if get_fitter:
            return fitter, data
        else:
            return data

    def _get_prior(self, objects, jacobian):
        """
        Note a single jacobian is being sent.  for multi-band this
        is the same as assuming they are all on the same coordinate system.
        
        assuming all images have the 
        prior for N objects.  The priors are the same for
        structural parameters, the only difference being the
        centers
        """

        conf=self['mof']
        ppars=conf['priors']

        nobj=len(objects)

        cen_priors=[]

        cen_sigma=jacobian.get_scale() # a pixel
        for i in range(nobj):
            row=objects['y'][i]#-1
            col=objects['x'][i]#-1

            v, u = jacobian(row, col)
            p=ngmix.priors.CenPrior(
                v,
                u,
                cen_sigma, cen_sigma,
                rng=self.rng,
            )
            cen_priors.append(p)

        gp = ppars['g']
        assert gp['type']=="ba"
        g_prior = self._get_prior_generic(gp)

        T_prior = self._get_prior_generic(ppars['T'])
        F_prior = self._get_prior_generic(ppars['flux'])

        if conf['model']=='bdf':
            fracdev_prior = ngmix.priors.Normal(0.5, 0.1, rng=self.rng)

            return mof.priors.PriorBDFSepMulti(
                cen_priors,
                g_prior,
                T_prior,
                fracdev_prior,
                [F_prior]*self.nband,
            )
        else:
            return mof.priors.PriorSimpleSepMulti(
                cen_priors,
                g_prior,
                T_prior,
                [F_prior]*self.nband,
            )


class MetacalFitter(FitterBase):
    """
    run metacal on all objects found in the image, using
    the deblended or "corrected" images produced by the
    multi-object fitter
    """
    def __init__(self, *args, **kw):

        self.mof_fitter=kw.pop('mof_fitter',None)

        super(MetacalFitter,self).__init__(*args, **kw)

        self.metacal_prior = self._get_prior(self['metacal'])
        self['metacal']['symmetrize_weight'] = \
            self['metacal'].get('symmetrize_weight',False)


    def go(self, mbobs_list_input):
        """
        do all fits and return fitter, data

        metacal data are appended to the mof data for each object
        """

        if self.mof_fitter is not None:
            # for mof fitting, we expect a list of mbobs_lists
            fitter, mof_data = self.mof_fitter.go(
                mbobs_list_input,
                get_fitter=True,
            )
            if mof_data is None:
                return None

            # this gets all objects, all bands in a list of MultiBandObsList
            mbobs_list = fitter.make_corrected_obs()

            if False:
                self._show_corrected_obs(mbobs_list_input, mbobs_list)
        else:
            mbobs_list = mbobs_list_input
            mof_data=None

        if self['metacal']['symmetrize_weight']:
            self._symmetrize_weights(mbobs_list)

        return self._do_all_metacal(mbobs_list, data=mof_data)

    def _symmetrize_weights(self, mbobs_list):
        for mbobs in mbobs_list:
            for obslist in mbobs:
                for obs in obslist:
                    self._symmetrize_weight(obs.weight)

        if False:
            from . import visualize
            visualize.view_mbobs_list(mbobs_list,title='symmetrized',weight=True)
            if 'q'==input('hit a key (q to quit): '):
                stop

    def _symmetrize_weight(self, wt):
        """
        symmetrize raw weight pixels in all of the maps
        """
        assert wt.shape[0] == wt.shape[1]

        for k in (1,2,3):
            wt_rot = np.rot90(wt, k=k)
            wzero  = np.where(wt_rot == 0.0)

            if wzero[0].size > 0:
                wt[wzero] = 0.0

    def _show_corrected_obs(self, mbobs_list, corrected_mbobs_list):
        import images

        if len(mbobs_list[0])==3:
            for i,mbobs in enumerate(corrected_mbobs_list):
                bim0=mbobs_list[i][0][0].image.transpose()
                gim0=mbobs_list[i][1][0].image.transpose()
                rim0=mbobs_list[i][2][0].image.transpose()
                bim=mbobs[0][0].image.transpose()
                gim=mbobs[1][0].image.transpose()
                rim=mbobs[2][0].image.transpose()
                mval=max(bim.max(), gim.max(), rim.max())
                rgb0=images.get_color_image(rim0/mval, gim0/mval, bim0/mval, nonlinear=0.1)
                rgb=images.get_color_image(rim/mval, gim/mval, bim/mval, nonlinear=0.1)
                #images.view(mbobs[0][0].image,title='%d' % i)
                imlist=[
                    rgb0/rgb0.max(), rgb/rgb.max(),
                    mbobs_list[i][0][0].weight, mbobs[0][0].weight,
                ]
                titles=['orig','corrected','weight orig','weight corr']
                images.view_mosaic(imlist, titles=titles)
        else:
            for i,mbobs in enumerate(corrected_mbobs_list):
                im0=mbobs_list[i][0][0].image
                im=mbobs[0][0].image
                #images.view(mbobs[0][0].image,title='%d' % i)
                imlist=[
                    im0, im, 
                    mbobs_list[i][0][0].weight, mbobs[0][0].weight,
                ]
                titles=['orig','corrected','weight orig','weight corr']
                images.view_mosaic(imlist, titles=titles)

        if 'q'==input('hit a key (q to quit): '):
            stop


    def _do_all_metacal(self, mbobs_list, data=None):
        """
        run metacal on all objects

        if some fail they will not be placed into the final output
        """

        nband=len(mbobs_list[0])

        datalist=[]
        for i,mbobs in enumerate(mbobs_list):
            if self._check_flags(mbobs):
                try:
                    boot=self._do_one_metacal(mbobs)
                    if isinstance(boot,dict):
                        res=boot
                    else:
                        res=boot.get_metacal_result()
                except (BootPSFFailure, BootGalFailure) as err:
                    logger.debug(str(err))
                    res={'mcal_flags':1}
                #except RuntimeError as err:
                #    # argh galsim and its generic errors
                #    logger.info('caught RuntimeError: %s' % str(err))
                #    res={'mcal_flags':1}

                if res['mcal_flags'] != 0:
                    logger.debug("        metacal fit failed")
                else:
                    # make sure we send an array
                    fit_data = self._get_metacal_output(res, nband, mbobs)
                    if data is not None:
                        odata = data[i:i+1]
                        fit_data = eu.numpy_util.add_fields(
                            fit_data,
                            odata.dtype.descr,
                        )
                        eu.numpy_util.copy_fields(odata, fit_data)

                    self._print_result(fit_data)
                    datalist.append(fit_data)

        if len(datalist) == 0:
            return None

        output = eu.numpy_util.combine_arrlist(datalist)
        return output


    def _do_one_metacal(self, mbobs):
        conf=self['metacal']

        psf_pars=conf['psf']
        max_conf=conf['max_pars']

        tpsf_obs=mbobs[0][0].psf
        if not tpsf_obs.has_gmix():
            _fit_one_psf(tpsf_obs, psf_pars)

        psf_Tguess=tpsf_obs.gmix.get_T()

        boot=self._get_bootstrapper(mbobs)
        if 'lm_pars' in psf_pars:
            psf_fit_pars=psf_pars['lm_pars']
        else:
            psf_fit_pars=None

        if self.metacal_prior is not None:
            prior=self.metacal_prior
            guesser=None
        else:
            mof_pars=mbobs.meta['fit_pars']
            prior=self._get_pars_prior(mof_pars,mbobs[0][0])
            guesser=ngmix.guessers.PriorGuesser(prior)

        boot.fit_metacal(

            psf_pars['model'],

            conf['model'],
            max_conf['pars'],

            psf_Tguess,
            psf_fit_pars=psf_fit_pars,
            psf_ntry=psf_pars['ntry'],

            prior=prior,
            guesser=guesser,
            ntry=max_conf['ntry'],

            metacal_pars=conf['metacal_pars'],
        )
        return boot

    def _get_pars_prior(sel, pars, obs):
        """
        create a joint prior based on the input parameters

        simple Normal used for all parameters except for
        the ellipticity
        """
        #ngmix.print_pars(pars,front='    pars for prior and guesser:')
        model=self['metacal']['model']
        assert model==self['mof']['model']

        scale=obs.jacobian.get_scale()

        ppars=self['metacal']['priors']
        cen_sigma=ppars['cen_sigma']
        cen_prior = ngmix.priors.CenPrior(
            pars[0],
            pars[1],
            cen_sigma,
            cen_sigma,
            rng=self.rng,
        )

        g_prior=ngmix.priors.GPriorBA(ppars['g_sigma'], rng=self.rng)

        # T changes very little with shear, we can use a
        # tight prior
        # T can be negative, so we need to choose the with
        # more carefully.  Use pixel scale
        T=pars[4]
        T_sigma = ppars['T_sigma_frac'] * (2*scale)**2
        #T_sigma = ppars['T_sigma_frac'] * abs(T)
        # this needs to be generalized
        #if T_sigma < 0.03:
        #    T_sigma=0.03

        T_prior=ngmix.priors.Normal(
            cen=T,
            sigma=T_sigma,
            rng=self.rng
        )

        if model=='bdf':
            Fstart=6
            fracdev=pars[5]
            fracdev_sigma=ppars['fracdev_sigma']
            fracdev_prior=ngmix.priors.Normal(
                cen=fracdev,
                sigma=fracdev_sigma,
                rng=self.rng
            )
        else:
            Fstart=5

        F_priors=[]
        for i in range(self.nband):
            F=pars[Fstart+i]
            F_sigma = abs(ppars['F_sigma_frac']*F)
            F_prior=ngmix.priors.Normal(
                cen=F,
                sigma=F_sigma,
                rng=self.rng
            )
            F_priors.append(F_prior)


        if model=='bdf':
            prior=ngmix.joint_prior.PriorBDFSep(
                cen_prior,
                g_prior,
                T_prior,
                fracdev_prior,
                F_priors,
            )
 
        else:
            prior=ngmix.joint_prior.PriorSimpleSep(
                cen_prior,
                g_prior,
                T_prior,
                F_priors,
            )
            
        return prior

    def _check_flags(self, mbobs):
        """
        only one epoch, so anything that hits an edge
        """
        flags=self['metacal'].get('bmask_flags',None)

        isok=True
        if flags is not None:
            for obslist in mbobs:
                for obs in obslist:
                    w=np.where( (obs.bmask & flags) != 0 )
                    if w[0].size > 0:
                        logger.info("   EDGE HIT")
                        isok = False
                        break

        return isok


    def _print_result(self, data):
        mess="        mcal s2n: %g Trat: %g"
        logger.debug(mess % (data['mcal_s2n'][0], data['mcal_T_ratio'][0]))

    def _get_metacal_dtype(self, npars, nband):
        dt=[
            ('x','f8'),
            ('y','f8'),
        ]
        for mtype in METACAL_TYPES:
            if mtype == 'noshear':
                back=None
            else:
                back=mtype

            n=Namer(front='mcal', back=back)
            if mtype=='noshear':
                dt += [
                    (n('psf_g'),'f8',2),
                    (n('psf_T'),'f8'),
                ]

            dt += [
                (n('nfev'),'i4'),
                (n('s2n'),'f8'),
                (n('s2n_r'),'f8'),
                (n('pars'),'f8',npars),
                (n('pars_cov'),'f8',(npars,npars)),
                (n('g'),'f8',2),
                (n('g_cov'),'f8',(2,2)),
                (n('T'),'f8'),
                (n('T_err'),'f8'),
                (n('T_ratio'),'f8'),
                (n('flux'),'f8',nband),
                (n('flux_cov'),'f8',(nband,nband)),
                (n('flux_err'),'f8',nband),
            ]

        return dt

    def _get_metacal_output(self, allres, nband, mbobs):
        # assume one epoch and line up in all
        # bands
        assert len(mbobs[0])==1,'one epoch only'



        npars=len(allres['noshear']['pars'])
        dt = self._get_metacal_dtype(npars, nband)
        data = np.zeros(1, dtype=dt)

        data0=data[0]
        data0['y'] = mbobs[0][0].meta['orig_row']
        data0['x'] = mbobs[0][0].meta['orig_col']

        for mtype in METACAL_TYPES:

            if mtype == 'noshear':
                back=None
            else:
                back=mtype

            n=Namer(front='mcal', back=back)

            res=allres[mtype]

            if mtype=='noshear':
                data0[n('psf_g')] = res['gpsf']
                data0[n('psf_T')] = res['Tpsf']

            for name in res:
                nn=n(name)
                if nn in data.dtype.names:
                    data0[nn] = res[name]

            # this relies on noshear coming first in the metacal
            # types
            data0[n('T_ratio')] = data0[n('T')]/data0['mcal_psf_T']

        return data

    def _get_bootstrapper(self, mbobs):
        from ngmix.bootstrap import MaxMetacalBootstrapper

        return MaxMetacalBootstrapper(
            mbobs,
            verbose=False,
        )

class MetacalAvgFitter(MetacalFitter):
    def _do_one_metacal(self, mbobs):
        nrand = self['metacal']['nrand']

        reslist=[]
        first=True
        for i in range(nrand):
            try:
                tboot=super(MetacalAvgFitter,self)._do_one_metacal(mbobs)

                res=tboot.get_metacal_result()
                reslist.append(res)
                if first:
                    boot=tboot
                    first=False

            except (BootPSFFailure, BootGalFailure) as err:
                logger.debug(str(err))
                res={'mcal_flags':1}
            except RuntimeError as err:
                # argh galsim and its generic errors
                logger.info('caught RuntimeError: %s' % str(err))
                res={'mcal_flags':1}

        if len(reslist)==0:
            raise BootGalFailure('none of the metacal fits worked')

        # this is the first, corresponds to the result in the
        # boot variable

        types=self['metacal']['metacal_pars']['types']
        dontavg=[
            'flags','nfev','ier','errmsg','model',
            'npix','lnprob','chi2per','dof','ntry',
        ]
        res=reslist[0]
        nkept = len(reslist)
        if nkept > 1:
            for tres in reslist[1:]:
                for type in types:
                    typeres=res[type]
                    ttyperes=tres[type]
                    for key in typeres:
                        if key not in dontavg:
                            #print('    key:',key)
                            typeres[key] += ttyperes[key]

            fac = 1.0/nkept
            for type in types:
                typeres=res[type]
                for key in typeres:
                    if key not in dontavg:
                        typeres[key] *= fac
            
        return boot

class AdmomMetacalFitter(MetacalFitter):
    #def __init__(self, *args, **kw):
    #    super(AdmomMetacalFitter,self).__init__(*args, **kw)
        
    def _do_one_metacal(self, mbobs):
        conf=self['metacal']

        boot=self._get_bootstrapper(mbobs)

        psf_Tguess=4.0*mbobs[0][0].jacobian.get_scale()**2

        boot.fit_metacal(
            psf_Tguess=psf_Tguess,
        )
        return boot

    def _get_bootstrapper(self, mbobs):
        from ngmix.bootstrap import AdmomMetacalBootstrapper

        return AdmomMetacalBootstrapper(
            mbobs,
            admom_pars=self['metacal'].get('admom_pars',None),
            metacal_pars=self['metacal']['metacal_pars'],
        )

class AdmomMetacalAvgFitter(AdmomMetacalFitter):
    def _do_one_metacal(self, mbobs):
        nrand = self['metacal']['nrand']

        reslist=[]
        first=True
        for i in range(nrand):
            try:
                tboot=super(AdmomMetacalAvgFitter,self)._do_one_metacal(mbobs)

                res=tboot.get_metacal_result()
                reslist.append(res)
                if first:
                    boot=tboot
                    first=False

            except (BootPSFFailure, BootGalFailure) as err:
                logger.debug(str(err))
                res={'mcal_flags':1}
            except RuntimeError as err:
                # argh galsim and its generic errors
                logger.info('caught RuntimeError: %s' % str(err))
                res={'mcal_flags':1}

        if len(reslist)==0:
            raise BootGalFailure('none of the metacal fits worked')

        # this is the first, corresponds to the result in the
        # boot variable

        types=self['metacal']['metacal_pars']['types']
        dontavg=[
            'flags','model',
            'npix','ntry',
            'numiter','nimage','flagstr',
        ]
        res=reslist[0]
        nkept = len(reslist)
        if nkept > 1:
            for tres in reslist[1:]:
                for type in types:
                    typeres=res[type]
                    ttyperes=tres[type]
                    for key in typeres:
                        if key not in dontavg:
                            #print('    key:',key)
                            typeres[key] += ttyperes[key]

            fac = 1.0/nkept
            for type in types:
                typeres=res[type]
                for key in typeres:
                    if key not in dontavg:
                        typeres[key] *= fac
            
        return boot


class MomentMetacalFitter(MetacalFitter):
    def __init__(self, *args, **kw):
        super(MomentMetacalFitter,self).__init__(*args, **kw)
        self._set_mompars()
        
    def _set_mompars(self):
        wpars=self['weight']

        T=ngmix.moments.fwhm_to_T(wpars['fwhm'])

        # the weight is always centered at 0, 0 or the
        # center of the coordinate system as defined
        # by the jacobian

        weight=ngmix.GMixModel(
            [0.0, 0.0, 0.0, 0.0, T, 1.0],
            'gauss',
        )

        # make the max of the weight 1.0 to get better
        # fluxes

        weight.set_norms()
        norm=weight.get_data()['norm'][0]
        weight.set_flux(1.0/norm)

        self.weight=weight

        wpars['use_canonical_center']=wpars.get('use_canonical_center',False)
        if wpars['use_canonical_center']:
            logger.info('using canonical center')

        if 'maxrad' in wpars:
            if wpars['maxrad']=='5sigma':
                sigma=np.sqrt(T/2)
                self.maxrad=5*sigma
            else:
                raise ValueError('bad maxrad type: %s' % wpars['maxrad'])
        else:
            self.maxrad=1.e9

        logger.info('using maxrad: %g arcsec' % self.maxrad)


    def _do_one_metacal(self, mbobs):
        #assert len(mbobs)==1
        #assert len(mbobs[0])==1
        obs=do_coadd_maybe(mbobs, self.rng)

        conf=self['metacal']

        mpars=conf['metacal_pars']

        odict=ngmix.metacal.get_all_metacal(
            obs,
            rng=self.rng,
            **mpars
        )


        res={}

        for type in mpars['types']:
            obs=odict[type]

            tres=self._measure_moments(obs)
            tres['g'] = tres['e']
            tres['g_cov'] = tres['e_cov']

            if type=='noshear':
                pres  = self._measure_moments(obs.psf)
                tres['gpsf'] = pres['e']
                tres['Tpsf'] = pres['T']

            res[type]=tres

        res['mcal_flags']=0
        boot=MomentBootstrapperFaker(res)
        return boot 



    """
    def _get_bootstrapper(self, mbobs):
        from ngmix.bootstrap import AdmomMetacalBootstrapper

        return AdmomMetacalBootstrapper(
            mbobs,
            admom_pars=self['metacal'].get('admom_pars',None),
            metacal_pars=self['metacal']['metacal_pars'],
        )
    """

    def _measure_moments(self, obs):
        """
        measure weighted moments
        """

        wpars=self['weight']

        if wpars['use_canonical_center']:
        
            ccen=(np.array(obs.image.shape)-1.0)/2.0
            jold=obs.jacobian
            jnew=jold.copy()
            jnew.set_cen(row=ccen[0], col=ccen[1])
            obs.jacobian = jnew

        res = self.weight.get_weighted_moments(obs=obs,maxrad=self.maxrad)

        if wpars['use_canonical_center']:
            obs.jacobian=jold

        if res['flags'] != 0:
            raise BootGalFailure("        moments failed")

        res['numiter'] = 1

        return res

class MomentBootstrapperFaker(object):
    def __init__(self, res):
        self.res=res
    def get_metacal_result(self):
        return self.res

class MaxFitter(FitterBase):
    """
    run a max like fitter
    """
    def __init__(self, *args, **kw):

        self.mof_fitter=kw.pop('mof_fitter',None)

        super(MaxFitter,self).__init__(*args, **kw)

        self.prior = self._get_prior(self['max'])


    def go(self, mbobs_list_input):
        """
        do all fits and return fitter, data

        metacal data are appended to the mof data for each object
        """

        if self.mof_fitter is not None:
            # for mof fitting, we expect a list of mbobs_lists
            fitter, mof_data = self.mof_fitter.go(
                mbobs_list_input,
                get_fitter=True,
            )
            if mof_data is None:
                return None

            # this gets all objects, all bands in a list of MultiBandObsList
            mbobs_list = fitter.make_corrected_obs()

        else:
            mbobs_list = mbobs_list_input
            mof_data=None

        return self._do_all_fits(mbobs_list, data=mof_data)


    def _do_all_fits(self, mbobs_list, data=None):
        """
        run metacal on all objects

        if some fail they will not be placed into the final output
        """

        nband=len(mbobs_list[0])

        datalist=[]
        for i,mbobs in enumerate(mbobs_list):
            if self._check_flags(mbobs):
                try:
                    boot, pres=self._do_one_fit(mbobs)
                    res=boot.get_max_fitter().get_result()
                except (BootPSFFailure, BootGalFailure):
                    res={'flags':1}

                if res['flags'] != 0:
                    logger.debug("        metacal fit failed")
                else:
                    # make sure we send an array
                    fit_data = self._get_output(res, pres, nband)
                    if data is not None:
                        odata = data[i:i+1]
                        fit_data = eu.numpy_util.add_fields(
                            fit_data,
                            odata.dtype.descr,
                        )
                        eu.numpy_util.copy_fields(odata, fit_data)

                    self._print_result(fit_data)
                    datalist.append(fit_data)

        if len(datalist) == 0:
            return None

        output = eu.numpy_util.combine_arrlist(datalist)
        return output


    def _do_one_fit(self, mbobs):
        conf=self['max']

        psf_pars=conf['psf']
        max_conf=conf['max_pars']

        tpsf_obs=mbobs[0][0].psf
        if not tpsf_obs.has_gmix():
            _fit_one_psf(tpsf_obs, psf_pars)

        psf_Tguess=tpsf_obs.gmix.get_T()

        boot=self._get_bootstrapper(mbobs)

        boot.fit_psfs(
            psf_pars['model'],
            psf_Tguess,
            fit_pars=psf_pars['lm_pars'],
            ntry=psf_pars['ntry'],
        )
        boot.fit_max(

            conf['model'],
            max_conf['pars'],

            prior=self.prior,
            ntry=max_conf['ntry'],
        )

        pres=self._get_object_psf_stats(boot.mb_obs_list)
        return boot, pres

    def _check_flags(self, mbobs):
        """
        only one epoch, so anything that hits an edge
        """
        flags=self['max'].get('bmask_flags',None)

        isok=True
        if flags is not None:
            for obslist in mbobs:
                for obs in obslist:
                    w=np.where( (obs.bmask & flags) != 0 )
                    if w[0].size > 0:
                        logger.info("   EDGE HIT")
                        isok = False
                        break

        return isok


    def _print_result(self, data):
        n=self._get_namer()
        mess="        s2n: %g Trat: %g"
        logger.debug(mess % (data[n('s2n')][0], data[n('T_ratio')][0]))

    def _get_namer(self):
        model=self['max']['model']
        return Namer(front=model)

    def _get_dtype(self, npars, nband):
        dt=[]

        n=self._get_namer()
        dt += [
            ('psf_g','f8',2),
            ('psf_T','f8'),
            (n('nfev'),'i4'),
            (n('s2n'),'f8'),
            (n('pars'),'f8',npars),
            (n('pars_cov'),'f8',(npars,npars)),
            (n('g'),'f8',2),
            (n('g_cov'),'f8',(2,2)),
            (n('T'),'f8'),
            (n('T_err'),'f8'),
            (n('T_ratio'),'f8'),
            (n('flux'),'f8',nband),
            (n('flux_cov'),'f8',(nband,nband)),
            (n('flux_err'),'f8',nband),
        ]

        return dt

    def _get_output(self, res, pres, nband):
        npars=len(res['pars'])
        dt = self._get_dtype(npars, nband)
        data = np.zeros(1, dtype=dt)

        n=self._get_namer()

        data0=data[0]

        data0['psf_g'] = pres['g']
        data0['psf_T'] = pres['T']

        for name in res:
            nn=n(name)
            if nn in data.dtype.names:
                data0[nn] = res[name]

        # this relies on noshear coming first in the metacal
        # types
        data0[n('T_ratio')] = data0[n('T')]/data0['psf_T']

        return data

    def _get_bootstrapper(self, mbobs):
        from ngmix.bootstrap import Bootstrapper

        return Bootstrapper(
            mbobs,
            verbose=False,
        )

    def _get_object_psf_stats(self, mbobs):
        """
        get the s/n for the given object.  This uses just the model
        to calculate the s/n, but does use the full weight map
        """
        g1sum=0.0
        g2sum=0.0
        Tsum=0.0
        wsum=0.0

        for band,obslist in enumerate(mbobs):
            for obsnum,obs in enumerate(obslist):
                twsum=obs.weight.sum()
                wsum += twsum

                tg1, tg2, tT = obs.psf.gmix.get_g1g2T()

                g1sum += tg1*twsum
                g2sum += tg2*twsum
                Tsum += tT*twsum

        g1 = g1sum/wsum
        g2 = g2sum/wsum
        T = Tsum/wsum

        return {
            'g':[g1,g2],
            'T':T,
        }


class Moments(FitterBase):
    def __init__(self, *args, **kw):
        super(Moments,self).__init__(*args, **kw)
        self._set_mompars()
 
    def go(self, mbobs_list):
        """
        run metacal on all objects

        if some fail they will not be placed into the final output
        """

        datalist=[]
        for i,mbobs in enumerate(mbobs_list):

            if self._check_flags(mbobs):
                obs=do_coadd_maybe(mbobs, self.rng)

                pres  = self._measure_moments(obs.psf)
                res   = self._measure_moments(obs)

                if res['flags'] != 0:
                    logger.debug("        moments failed: %s" % res['flags'])
                    print(res)

                if pres['flags'] != 0:
                    logger.debug("        psf moments failed: %s" % pres['flags'])
                    print(pres)

                if res['flags']==0 and pres['flags']==0:
                    # make sure we send an array
                    fit_data = self._get_output(res, pres)

                    self._print_result(fit_data)
                    datalist.append(fit_data)

                elif False:
                    import images
                    images.multiview(obs.image,title='im')
                    images.multiview(obs.psf.image,title='psf im')
                    if 'q'==input('hit a key (q to quit): '):
                        stop



        if len(datalist) == 0:
            return None

        output = eu.numpy_util.combine_arrlist(datalist)
        return output


    def _print_result(self, data):
        mess="        wmom s2n: %g Trat: %g"
        logger.debug(mess % (data['wmom_s2n'][0], data['wmom_T_ratio'][0]))

    def _measure_moments(self, obs):
        """
        measure weighted moments
        """


        wpars=self['weight']

        if wpars['use_canonical_center']:
            #logger.debug('        getting moms with canonical center')
        
            ccen=(np.array(obs.image.shape)-1.0)/2.0
            jold=obs.jacobian

            jnew=jold.copy()
            jnew.set_cen(row=ccen[0], col=ccen[1])
            obs.jacobian = jnew


        res = self.weight.get_weighted_moments(obs=obs,maxrad=self.maxrad)

        if wpars['use_canonical_center']:
            obs.jacobian=jold

        if res['flags'] != 0:
            return res

        res['numiter'] = 1
        res['g'] = res['e']
        res['g_cov'] = res['e_cov']

        return res

    def _get_dtype(self, model, npars):
        n=Namer(front=model)
        dt = [
            ('psf_g','f8',2),
            ('psf_T','f8'),
            (n('s2n'),'f8'),
            (n('pars'),'f8',npars),
            #(n('pars_cov'),'f8',(npars,npars)),
            (n('g'),'f8',2),
            (n('g_cov'),'f8',(2,2)),
            (n('T'),'f8'),
            (n('T_err'),'f8'),
            (n('T_ratio'),'f8'),
        ]

        return dt

    def _get_output(self, res, pres):

        npars=res['pars'].size

        model='wmom'
        n=Namer(front=model)

        dt=self._get_dtype(model, npars)
        output=np.zeros(1, dtype=dt)

        output['psf_g'] = pres['g']
        output['psf_T'] = pres['T']
        output[n('s2n')] = res['s2n']
        output[n('pars')] = res['pars']
        output[n('g')] = res['g']
        output[n('g_cov')] = res['g_cov']
        output[n('T')] = res['T']
        output[n('T_err')] = res['T_err']
        output[n('T_ratio')] = res['T']/pres['T']

        return output

    def _set_mompars(self):
        wpars=self['weight']

        T=ngmix.moments.fwhm_to_T(wpars['fwhm'])

        # the weight is always centered at 0, 0 or the
        # center of the coordinate system as defined
        # by the jacobian

        weight=ngmix.GMixModel(
            [0.0, 0.0, 0.0, 0.0, T, 1.0],
            'gauss',
        )

        # make the max of the weight 1.0 to get better
        # fluxes

        weight.set_norms()
        norm=weight.get_data()['norm'][0]
        weight.set_flux(1.0/norm)

        self.weight=weight

        wpars['use_canonical_center']=wpars.get('use_canonical_center',False)
        if wpars['use_canonical_center']:
            logger.info('using canonical center')

        if 'maxrad' in wpars:
            if wpars['maxrad']=='5sigma':
                sigma=np.sqrt(T/2)
                self.maxrad=5*sigma
            else:
                raise ValueError('bad maxrad type: %s' % wpars['maxrad'])
        else:
            self.maxrad=1.e9

        logger.info('using maxrad: %g arcsec' % self.maxrad)

    def _check_flags(self, mbobs):
        """
        only one epoch, so anything that hits an edge
        """
        flags=self['metacal'].get('bmask_flags',None)

        isok=True
        if flags is not None:
            for obslist in mbobs:
                for obs in obslist:
                    w=np.where( (obs.bmask & flags) != 0 )
                    if w[0].size > 0:
                        logger.info("   EDGE HIT")
                        isok = False
                        break

        return isok


class Metacal2CompFitter(MetacalFitter):
    """
    fit 2 component models to all the input observations
    """
    def _do_one_metacal(self, mbobs):
        """
        run the bootstrapper on these observations
        """

        prior=self._get_2comp_prior(mbobs)
        guesser=self._get_guesser(mbobs, prior)

        boot=Metacal2CompBootstrapper(
            mbobs,
            self['mof_2comp'],
            prior,
            guesser,
            metacal_pars=self['metacal']['metacal_pars'],
        )
        boot.go()
        return boot

    def _get_2comp_prior(self, mbobs):
        """
        we determine the prior from the parameters tht
        are in mbobs.meta['fit_pars']

        for now only the center information is used

        we assume the jacobian is centered on the
        best fit object center

        both components will have a prior centered
        on the best fit center, but we will set the
        width of this prior quite wide
        """
        n_components=2

        fit_pars=mbobs.meta['fit_pars']
        jacobian=mbobs[0][0].jacobian

        nband=len(mbobs)
        mofc=self['mof_2comp']
        ppars=mofc['priors']
        model=mofc['model']

        cen_priors=[]

        for i in range(n_components):

            # note this will be centered on 0,0; we assume the jacobian has
            # been set to center at the best fit
            p = self._get_prior_generic(ppars['cen'])
            cen_priors.append(p)

        gp = ppars['g']
        assert gp['type']=="ba"
        g_prior = self._get_prior_generic(gp)

        T_prior = self._get_prior_generic(ppars['T'])

        if ppars['flux']['type']=='from-pars-lognorm':
            if model=='bdf':
                fluxes = fit_pars[6:]
            else:
                fluxes = fit_pars[5:]
            flux_max=fluxes.max()
            sigma_frac=ppars['flux']['sigma_frac']
            F_priors=[]
            for flux in fluxes:
                if flux < 0.0:
                    flux = flux_max*0.1
                F_prior=ngmix.priors.LogNormal(flux, flux*sigma_frac, rng=self.rng)
                F_priors.append(F_prior)

        else:
            F_prior = self._get_prior_generic(ppars['flux'])
            F_priors = [F_prior]*nband

        if model=='bdf':
            assert 'fracdev' in ppars,"set fracdev prior for bdf model"
            fp = ppars['fracdev']
            assert fp['type'] == 'normal','only normal prior supported for fracdev'

            fracdev_prior = self._get_prior_generic(fp)

            return mof.priors.PriorBDFSepMulti(
                cen_priors,
                g_prior,
                T_prior,
                fracdev_prior,
                F_priors,
            )
        else:
            return mof.priors.PriorSimpleSepMulti(
                cen_priors,
                g_prior,
                T_prior,
                F_priors,
            )


    def _get_guesser(self, mbobs, prior):
        """
        we take the guesser from the parameters
        """
        n_components=2

        fit_pars=mbobs.meta['fit_pars']

        nband=len(mbobs)
        mofc=self['mof_2comp']
        ppars=mofc['priors']
        model=mofc['model']

        T = fit_pars[4]
        if model=='bdf':
            fluxes = fit_pars[6:]
        else:
            fluxes = fit_pars[5:]

        return TwoCompGuesser(
            model,
            T,
            fluxes,
            prior,
            rng=self.rng,
        )

class TwoCompGuesser(ngmix.guessers.GuesserBase):
    """
    Make guesses from the input T, fluxes and prior for center
    g is give a guess close to zero

    parameters
    ----------
    T: float
        Center for T guesses
    fluxes: float or sequences
        Center for flux guesses
    prior:
        cen, g drawn from this prior
    scaling: string
        'linear' or 'log'
    """
    def __init__(self, model, T, fluxes, prior, rng=None):
        if rng is None:
            rng=np.random.RandomState()

        self.rng=rng

        if np.isscalar(fluxes):
            fluxes=np.array(fluxes, dtype='f8', ndmin=1)

        self.model=model
        self.T=T
        self.fluxes=fluxes
        self.prior=prior

        self.g_halfwidth=0.02
        self.flux_halfwidth=0.40
        self.T_halfwidth=0.40

    def __call__(self, **keys):
        """
        center, shape are just distributed around zero

        if model is bdf, guess is also taken from prior
        """
        gw=self.g_halfwidth
        Tw=self.T_halfwidth
        Fw=self.flux_halfwidth

        ur = self.rng.uniform
        fluxes=self.fluxes

        nband=fluxes.size

        if self.model=='bdf':
            np=6+nband
            flux_start=6
        else:
            np=5+nband
            flux_start=5

        guess=self.prior.sample()
        for i in range(2):

            start = i*np
            # over-write g guesses
            guess[start+2],guess[start+3] = ur(low=-gw,high=gw,size=2)

            # over-write T guesses
            guess[start+4] = self.T*(1.0 + ur(low=-Tw,high=Tw))

            # over-write F guesses
            for band in range(nband):
                guess[start+flux_start+band] = fluxes[band]*(1.0 + ur(low=-Fw,high=Fw))

        self._fix_guess(guess, self.prior)

        return guess

    def _fix_guess(self, guess, prior, ntry=4):
        """
        just fix T and flux
        """

        n=guess.shape[0]
        for itry in range(ntry):
            try:
                lnp=prior.get_lnprob_scalar(guess)

                if lnp <= ngmix.priors.LOWVAL:
                    dosample=True
                else:
                    dosample=False
            except GMixRangeError as err:
                dosample=True

            if dosample:
                print_pars(guess, front="bad guess:")
                if itry < ntry:
                    tguess = prior.sample()
                    guess[4:] = tguess[4:]
                else:
                    # give up and just drawn a sample
                    guess = prior.sample()
            else:
                break



class Metacal2CompBootstrapper(object):
    """
    fit psfs and MOF
    """
    def __init__(self,
                 mbobs,
                 mof_conf,
                 prior,
                 guesser,
                 metacal_pars=None):

        self.mbobs=mbobs

        # includes psf and model fitting info
        self.mof_conf=mof_conf

        self.prior=prior
        self.guesser=guesser

        self._set_metacal_pars(metacal_pars)


    def go(self):
        """
        fit psfs and galaxy models for the sheared versions
        """
        odict=ngmix.metacal.get_all_metacal(
            self.mbobs,
            **self.metacal_pars
        )

        res={'mcal_flags':0}
        for key,mbobs in odict.items():
            tres = self._run_mof(mbobs)
            res[key] = tres
            res['mcal_flags'] |= tres['flags']

        self.metacal_result=res

    def get_metacal_result(self):
        return self.metacal_result

    def _run_mof(self, mbobs):
        n_components=2
        assert len(mbobs)==1,'not dealing with all bands in gmix checks etc.'

        try:
            _fit_all_psfs([mbobs], self.mof_conf['psf'])

            fitter = mof.MOF(
                mbobs,
                self.mof_conf['model'],
                n_components,
                prior=self.prior,
            )

            for i in range(self.mof_conf['ntry']):
                guess=self.guesser()
                fitter.go(guess)

                res=fitter.get_result()
                if res['flags']==0:
                    try:
                        gm = fitter.get_gmix(band=0)
                        g1,g2,T=gm.get_g1g2T()
                    except ngmix.GMixRangeError as err:
                        logger.info(str(err))
                        res['flags']=1

                    if res['flags']==0:
                        break

        except BootPSFFailure as err:
            print(str(err))
            res={'flags':1}

        if res['flags'] != 0:
            fitter=None
            data=None
        else:
            res = self._get_mof_output(mbobs, fitter, gm)

        return res

    def _get_mof_output(self, mbobs, fitter, gm):
        """
        add the combined ellipticity and T etc.
        """
        
        res=fitter.get_result()

        # don't care about band here, we are interested in the
        # structural parameters only
        #print('gm:')
        #print(gm)

        # this will properly account for offsets between
        # components
        g1,g2,T=gm.get_g1g2T()

        res['g'] = (g1,g2)
        res['T'] = T
        res['gpsf'], res['Tpsf'] = self._get_psf_stats(mbobs) 

        return res

    def _get_psf_stats(self, mbobs):
        wsum     = 0.0
        Tpsf_sum = 0.0
        gpsf_sum = np.zeros(2)
        for obslist in mbobs:
            for obs in obslist:
                g1,g2,T=obs.psf.gmix.get_g1g2T()

                twsum = obs.weight.sum()

                wsum += twsum
                gpsf_sum[0] += g1*twsum
                gpsf_sum[1] += g2*twsum
                Tpsf_sum += T*twsum

        g = gpsf_sum/wsum
        T = Tpsf_sum/wsum

        return g,T


    def _set_metacal_pars(self, metacal_pars_in):
        """
        make sure at least the step is specified
        """
        metacal_pars={
            'types':['noshear','1p','1m','2p','2m'],
        }

        if metacal_pars_in is not None:
            metacal_pars.update(metacal_pars_in)

        self.metacal_pars=metacal_pars



def _fit_all_psfs(mbobs_list, psf_conf):
    """
    fit all psfs in the input observations
    """
    fitter=AllPSFFitter(mbobs_list, psf_conf)
    fitter.go()
     


class AllPSFFitter(object):
    def __init__(self, mbobs_list, psf_conf):
        self.mbobs_list=mbobs_list
        self.psf_conf=psf_conf

    def go(self):
        for mbobs in self.mbobs_list:
            for obslist in mbobs:
                for obs in obslist:
                    psf_obs = obs.get_psf()
                    _fit_one_psf(psf_obs, self.psf_conf)

def _fit_one_psf(obs, pconf):
    Tguess=4.0*obs.jacobian.get_scale()**2

    if 'coellip' in pconf['model']:
        ngauss=ngmix.bootstrap.get_coellip_ngauss(pconf['model'])
        runner=ngmix.bootstrap.PSFRunnerCoellip(
            obs,
            Tguess,
            ngauss,
            pconf['lm_pars'],
        )


    else:
        runner=ngmix.bootstrap.PSFRunner(
            obs,
            pconf['model'],
            Tguess,
            pconf['lm_pars'],
        )

    runner.go(ntry=pconf['ntry'])

    psf_fitter = runner.fitter
    res=psf_fitter.get_result()
    obs.update_meta_data({'fitter':psf_fitter})

    if res['flags']==0:
        gmix=psf_fitter.get_gmix()
        obs.set_gmix(gmix)
    else:
        raise BootPSFFailure("failed to fit psfs: %s" % str(res))

class EMMetacalFitter(MetacalFitter):

    def _do_one_metacal(self, mbobs):
        res={'mcal_flags':0}

        assert len(mbobs)==1
        assert len(mbobs[0])==1
        obs=mbobs[0][0]

        # first do a fit to the original, which will be used as
        # a guess for the others
        boot=self._do_one_fit(obs)
        gmix_guess = boot.get_fitter().get_gmix()

        mpars=self['metacal']['metacal_pars']
        odict=ngmix.metacal.get_all_metacal(
            obs,
            rng=self.rng,
            **mpars
        )

        for key,tobs in odict.items():
            tboot = self._do_one_fit(tobs, gmix_guess=gmix_guess)
            res[key]=tboot.get_fitter().get_result()

        return res

    def _do_one_fit(self, obs, gmix_guess=None):
        pconf=self['metacal']['psf']
        oconf=self['metacal']['obj']

        psf_Tguess = 4.0*obs.jacobian.scale**2
        psf_runner=EMPSFRunner(
            obs.psf,
            psf_Tguess,
            pconf['ngauss'],
            pconf['em_pars'],
            ntry=pconf['ntry'],
            clip=pconf.get('clip',False),
        )

        obj_Tguess = 5.0*obs.jacobian.scale**2
        runner=EMRunner(
        #runner=EMPSFRunner(
            obs,
            obj_Tguess,
            oconf['ngauss'],
            oconf['em_pars'],
            ntry=oconf['ntry'],
            clip=oconf.get('clip',True),
        )

        boot=EMBootstrapper(runner, psf_runner=psf_runner)
        boot.go(gmix_guess=gmix_guess)
        return boot


class EMBootstrapper(object):
    def __init__(self, runner, psf_runner=None):
        """
        run EM.  Currently the total gmix is used to calculate
        g, s2n etc. We can open this up to splitting them

        parameters
        ----------
        obs: ngmix Observation
            The observation to fit
        runner: e.g. a EMRunner
            For fitting the primary observation
        psf_runner: optional, e.g. EMPSFRunner
            runner for fitting psf objservations in obs.psf
        """

        self.runner=runner
        self.psf_runner=psf_runner

    def go(self, gmix_guess=None):
        """
        run the runners
        """
        if self.psf_runner is not None:
            self._fit_psf()

        self._fit(gmix_guess=gmix_guess)

    def get_result(self):
        """
        get the result dict
        """
        return self.get_fitter().get_result()

    def get_fitter(self):
        """
        get the adaptive moments fitter
        """
        if not hasattr(self,'fitter'):
            raise RuntimeError("you need to run fit() successfully first")
        return self.fitter

    def _fit(self, gmix_guess=None):
        """
        pars controlling the EM fits are given on construction
        """
        self.runner.go(guess=gmix_guess)

        fitter=self.runner.get_fitter()
        res=fitter.get_result()

        if res['flags'] != 0:
            raise BootGalFailure("object fit failed")

        self.fitter=fitter
        try:
            self._set_stats()
        except GMixRangeError as err:
            raise BootGalFailure("object fit failed: %s" % str(err))


    def _set_stats(self):
        """
        add some statistics to the fitter result dict
        """

        res=self.fitter.get_result()
        gmix=self.fitter.get_gmix()

        g1,g2,T=gmix.get_g1g2T()

        obs_copy=self.runner.obs_orig.copy()
        obs_copy.set_gmix(gmix)

        tfitter=ngmix.fitting.TemplateFluxFitter(obs_copy)
        tfitter.go()
        tres=tfitter.get_result()

        # note original does not have flux set so we can
        # use it for guesses
        gmix.set_flux(tres['flux'])

        res['s2n'] = gmix.get_model_s2n(obs_copy)
        res['T'] = T
        res['g'] = (g1,g2)
        res['flux'] = tres['flux']
        res['flux_err'] = tres['flux_err']
        res['pars'] = gmix.get_full_pars()

        #logger.debug('    flux s2n: %g' % (res['flux']/res['flux_err']))
        logger.debug('    s2n: %g' % res['s2n'])

        if self.psf_runner is not None:
            pg1,pg2,pT = self.psf_runner.obs.gmix.get_g1g2T()
            #res['psf_g'] = (pg1, pg2)
            #res['psf_T'] = pT
            res['gpsf'] = (pg1, pg2)
            res['Tpsf'] = pT

    def _fit_psf(self):
        """
        Fit the psf observation
        """

        self.psf_runner.go()
        fitter=self.psf_runner.get_fitter()
        res=fitter.get_result()
        if res['flags'] != 0:
            raise BootPSFFailure("psf fit failed")

        gmix=fitter.get_gmix()
        gmix.set_flux(1.0)

        # if psf runner obs is the same as the runner obs.psf then
        # this will propagate
        self.psf_runner.obs.set_gmix(gmix)

    def _set_s2n(self, mb_obs_list, fitter):
        """
        do flux in each band separately
        """

        # for each band
        nband = len(mb_obs_list)
        res=fitter.get_result()
        res['flux'] = zeros(nband) - 9999
        res['flux_err'] = zeros(nband) + 9999
        res['flux_s2n'] = zeros(nband) - 9999

        try:
            gmix=fitter.get_gmix()

            for band,obs_list in enumerate(mb_obs_list):
                for obs in obs_list:
                    obs.set_gmix(gmix)

                flux_fitter=fitting.TemplateFluxFitter(obs_list)
                flux_fitter.go()

                fres=flux_fitter.get_result()
                if fres['flags'] != 0:
                    res['flags'] = fres
                    raise BootPSFFailure("could not get flux")

                res['flux'][band]=fres['flux']
                res['flux_err'][band]=fres['flux_err']

                if fres['flux_err'] > 0:
                    res['flux_s2n'][band]=fres['flux']/fres['flux_err']

        except GMixRangeError as err:
            raise BootPSFFailure(str(err))

class EMRunner(object):
    """
    wrapper to generate guesses and run the psf fitter a few times
    """
    def __init__(self, obs, Tguess, ngauss, em_pars, ntry=2, clip=False, rng=None):
        from functools import partial

        if rng is None:
            rng=np.random.RandomState()

        self.rng=rng
        self.ngauss = ngauss
        self.Tguess = Tguess
        self.sigma_guess = np.sqrt(Tguess/2.0)
        self.ntry=ntry

        self.set_obs(obs, clip=clip)

        self.em_pars=em_pars

    def set_obs(self, obsin, clip=False):
        """
        set a new observation with sky
        """
        assert isinstance(obsin,ngmix.Observation)

        if clip:
            im=obsin.image
            sky = im.max()/1000.0
            im_with_sky=im.clip(min=sky)
        else:
            im_with_sky, sky = ngmix.em.prep_image(obsin.image)

        self.obs_orig = obsin
        self.obs   = Observation(im_with_sky, jacobian=obsin.jacobian)
        self.sky   = sky

    def get_fitter(self):
        """
        get the GMixEM fitter
        """
        return self.fitter

    def go(self, guess=None):
        """
        the first guess can be taken from the input guess= keyword
        """

        fitter=ngmix.em.GMixEM(self.obs)

        for i in range(self.ntry):
            if i==0 and guess is not None:
                guess_i=guess
                #print("using input gmix guess")
                #print(guess_i)
            else:
                guess_i=self.get_guess()
                #print("using generated gmix guess")
                #print(guess_i)

            fitter.go(guess_i, self.sky, **self.em_pars)

            res=fitter.get_result()
            if res['flags']==0:
                break

        res['ntry'] = i+1
        self.fitter=fitter

    def get_guess(self):
        """
        Guess for the EM algorithm
        """

        rng=self.rng

        ngauss=self.ngauss
        Tguess=self.Tguess
        sigma=self.sigma_guess
        nper=6
        pars=np.zeros(ngauss*nper)


        flux_frac = 1.0/ngauss
        for i in range(ngauss):
            start=i*nper

            pars[start+0] = flux_frac*(1. + rng.uniform(low=-0.05, high=0.05))

            # relative to the jacobian center. Use our Tguess->sigma as a way
            # to distribute the positions
            pars[start+1] = rng.normal(scale=sigma*0.5)
            pars[start+2] = rng.normal(scale=sigma*0.5)

            pars[start+3] = 0.5*Tguess*(1.0 + rng.uniform(low=-0.1, high=0.1))
            pars[start+4] = Tguess*rng.uniform(low=-0.05, high=0.05)
            pars[start+5] = 0.5*Tguess*(1.0 + rng.uniform(low=-0.1, high=0.1))

        return ngmix.GMix(pars=pars)

from ngmix.bootstrap import (
    _em2_pguess, _em2_fguess,
    _em3_pguess, _em3_fguess,
    _em4_pguess, _em4_fguess,
)

class EMPSFRunner(EMRunner):
    """
    Runner for fitting a PSF with specialized guess generation
    """

    def get_guess(self):
        """
        Guess for the EM algorithm
        """

        if self.ngauss==1:
            return self._get_em_guess_1gauss()
        elif self.ngauss==2:
            return self._get_em_guess_2gauss()
        elif self.ngauss==3:
            return self._get_em_guess_3gauss()
        elif self.ngauss==4:
            return self._get_em_guess_4gauss()
        else:
            raise ValueError("bad ngauss: %d" % self.ngauss)

    def _get_em_guess_1gauss(self):
        
        ur=self.rng.uniform
        sigma2 = self.sigma_guess**2
        pars=[
            1.0 + ur(low=-0.1, high=0.1),
            ur(low=-0.1, high=0.1),
            ur(low=-0.1, high=0.1),
            sigma2*(1.0 + ur(low=-0.1, high=0.1)),
            sigma2*ur(low=-0.2, high=0.2),
            sigma2*(1.0 + ur(low=-0.1, high=0.1)),
        ]

        return ngmix.GMix(pars=pars)

    def _get_em_guess_2gauss(self):

        sigma2 = self.sigma_guess**2
        ur=self.rng.uniform

        pars=[
            _em2_pguess[0],
            ur(low=-0.1, high=0.1),
            ur(low=-0.1, high=0.1),
            _em2_fguess[0]*sigma2*(1.0 + ur(low=-0.1, high=0.1)),
            0.0,
            _em2_fguess[0]*sigma2*(1.0 + ur(low=-0.1, high=0.1)),

            _em2_pguess[1],
            ur(low=-0.1, high=0.1),
            ur(low=-0.1, high=0.1),
            _em2_fguess[1]*sigma2*(1.0 + ur(low=-0.1, high=0.1)),
            0.0,
            _em2_fguess[1]*sigma2*(1.0 + ur(low=-0.1, high=0.1)),
        ]

        return ngmix.GMix(pars=pars)

    def _get_em_guess_3gauss(self):

        sigma2 = self.sigma_guess**2
        ur=self.rng.uniform

        pars= [
            _em3_pguess[0]*(1.0+ur(low=-0.1, high=0.1)),
            ur(low=-0.1, high=0.1),
            ur(low=-0.1, high=0.1),
            _em3_fguess[0]*sigma2*(1.0 + ur(low=-0.1, high=0.1)),
            ur(low=-0.01, high=0.01),
            _em3_fguess[0]*sigma2*(1.0 + ur(low=-0.1, high=0.1)),

            _em3_pguess[1]*(1.0+ur(low=-0.1, high=0.1)),
            ur(low=-0.1, high=0.1),
            ur(low=-0.1, high=0.1),
            _em3_fguess[1]*sigma2*(1.0 + ur(low=-0.1, high=0.1)),
            ur(low=-0.01, high=0.01),
            _em3_fguess[1]*sigma2*(1.0 + ur(low=-0.1, high=0.1)),

            _em3_pguess[2]*(1.0+ur(low=-0.1, high=0.1)),
            ur(low=-0.1, high=0.1),
            ur(low=-0.1, high=0.1),
            _em3_fguess[2]*sigma2*(1.0 + ur(low=-0.1, high=0.1)),
            ur(low=-0.01, high=0.01),
            _em3_fguess[2]*sigma2*(1.0 + ur(low=-0.1, high=0.1)),
        ]


        return ngmix.GMix(pars=pars)

    def _get_em_guess_4gauss(self):

        sigma2 = self.sigma_guess**2
        ur=self.rng.uniform

        pars= [
            _em4_pguess[0]*(1.0+ur(low=-0.1, high=0.1)),
            ur(low=-0.1, high=0.1),
            ur(low=-0.1, high=0.1),
            _em4_fguess[0]*sigma2*(1.0 + ur(low=-0.1, high=0.1)),
            ur(low=-0.01, high=0.01),
            _em4_fguess[0]*sigma2*(1.0 + ur(low=-0.1, high=0.1)),

            _em4_pguess[1]*(1.0+ur(low=-0.1, high=0.1)),
            ur(low=-0.1, high=0.1),
            ur(low=-0.1, high=0.1),
            _em4_fguess[1]*sigma2*(1.0 + ur(low=-0.1, high=0.1)),
            ur(low=-0.01, high=0.01),
            _em4_fguess[1]*sigma2*(1.0 + ur(low=-0.1, high=0.1)),

            _em4_pguess[2]*(1.0+ur(low=-0.1, high=0.1)),
            ur(low=-0.1, high=0.1),
            ur(low=-0.1, high=0.1),
            _em4_fguess[2]*sigma2*(1.0 + ur(low=-0.1, high=0.1)),
            ur(low=-0.01, high=0.01),
            _em4_fguess[2]*sigma2*(1.0 + ur(low=-0.1, high=0.1)),

            _em4_pguess[2]*(1.0+ur(low=-0.1, high=0.1)),
            ur(low=-0.1, high=0.1),
            ur(low=-0.1, high=0.1),
            _em4_fguess[2]*sigma2*(1.0 + ur(low=-0.1, high=0.1)),
            ur(low=-0.01, high=0.01),
            _em4_fguess[2]*sigma2*(1.0 + ur(low=-0.1, high=0.1)),
        ]

        return ngmix.GMix(pars=pars)

def do_coadd_maybe(mbobs, rng):
    """
    coadd all images and psfs.  Assume perfect registration and
    same wcs
    """

    # note here assuming we can re-use the wcs etc.
    new_obs = mbobs[0][0].copy()

    if len(mbobs)==1 and len(mbobs[0])==1:
        return new_obs

    first=True
    wsum=0.0
    for obslist in mbobs:
        for obs in obslist:
            tim = obs.image
            twt = obs.weight
            tpsf_im = obs.psf.image
            tpsf_wt = obs.psf.weight

            medweight = np.median(twt)
            noise=np.sqrt(1.0/medweight)

            psf_medweight = np.median(tpsf_wt)
            psf_noise=np.sqrt(1.0/psf_medweight)

            tnim     = rng.normal(size=tim.shape, scale=noise)
            tpsf_nim = rng.normal(size=tpsf_im.shape, scale=psf_noise)

            wsum += medweight

            if first:
                im      = tim*medweight
                psf_im  = tpsf_im*medweight

                nim     = tnim * medweight
                psf_nim = tpsf_nim * medweight

                first=False
            else:
                im      += tim*medweight
                psf_im  += tpsf_im*medweight

                nim     += tnim * medweight
                psf_nim += tpsf_nim * medweight


    fac=1.0/wsum
    im *= fac
    psf_im *= fac

    nim *= fac
    psf_nim *= fac

    noise_var = nim.var()
    psf_noise_var = psf_nim.var()

    wt = np.zeros(im.shape) + 1.0/noise_var
    psf_wt = np.zeros(psf_im.shape) + 1.0/psf_noise_var

    new_obs.set_image(im, update_pixels=False )
    new_obs.set_weight(wt )

    new_obs.psf.set_image(psf_im, update_pixels=False )
    new_obs.psf.set_weight(psf_wt)

    """
    pconf={
        'model':'gauss',
        'lm_pars':{'maxfev':2000,'ftol':1.0e-5,'xtol':1.0e-5},
        'ntry':2,
    )
    _fit_one_psf(new_obs.psf, pconf)
    """

    if False:
        import images
        images.multiview(new_obs.image,title='im')
        images.multiview(new_obs.psf.image,title='psf im')
        if 'q'==input('hit a key (q to quit): '):
            stop

    return new_obs


