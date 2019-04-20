"""

for lots of crowding, the thing that dominates the chi squared
is the detections, not the model that's being fit or the
tolerance used for convergence

Using coarse tolerance helps a lot for getting the big groups,
and doesn't degrade the chi squared
{'ftol':tol, 'xtol':tol}

"""
import logging
import copy
import numpy as np
import ngmix
import esutil as eu
import galsim
import mof
from . import visualize
from . import pdfs

logger = logging.getLogger(__name__)

class Sim(dict):
    def __init__(self, config, rng):
        self.rng=rng

        self.update(config)

        self.g_pdf = self._make_g_pdf()

        self._make_primary_pdfs()

        if 'bulge' in self['pdfs']:
            self._make_bulge_pdfs()

        if 'knots' in self['pdfs']:
            self._make_knots_pdfs()

        self['coadd'] = self.get('coadd',False)
        self._set_position_pdf()

    def make_obs(self):
        self._set_bands()
        self._set_psf()
        self._set_objects()
        self._draw_objects()
        self._add_noise()

        # to get undetected objects
        if 'background' in self:
            if self['background']['measure']:
                self._subtract_backgrounds()

        if self['coadd']:
            self._do_coadd()

        self._make_obs()

    def get_truth_catalog(self):
        """
        get a copy of the truth catalog
        """
        return self._obj_data.copy()

    def get_obs(self):
        return self.obs

    def get_fofs(self,
                 fof_conf,
                 cat=None,
                 seg=None,
                 obs=None,
                 weight_type='weight',
                 show=False):
        """
        get lists of MultiBandObsList for each
        Friends of Friends group


        parameters
        ----------
        fof_conf: dict
            configuration for FOFs
        obs: observations
            If sent, these are used rather than
            the originals
        cat: catalog
            If sent, is used for making the MEDS files
        seg: seg map
            Should be sent with cat

        weight_type: string
            'weight' or 'uberseg'
        show: bool
            If True show the image used for making meds
        """
        #mm=self.get_multiband_meds(obs=obs, cat=cat, seg=seg)
        medser=self.get_medsifier(cat=cat, seg=seg, obs=obs)
        if medser.cat.size==0:
            return []

        mm=medser.get_multiband_meds()

        if obs is not None:
            #logger.info("assuming psfs are all the same")
            psf_obs=obs[0][0].psf
        else:
            psf_obs=None

        mn=mof.fofs.MEDSNbrs(
            mm.mlist,
            fof_conf,
        )

        nbr_data = mn.get_nbrs()

        nf = mof.fofs.NbrsFoF(nbr_data)
        fofs = nf.get_fofs()
        if fofs.size==0:
            return []

        if show:
            self._plot_fofs(mm, fofs)

        hist,rev=eu.stat.histogram(fofs['fofid'], rev=True)

        fof_mbobs_lists=[]
        nconf=self.get("noise_images",None)

        for i in range(hist.size):
            assert rev[i] != rev[i+1],'all fof groups should be populated'
            w=rev[ rev[i]:rev[i+1] ]

            # assuming number is index+1
            indices=fofs['number'][w]-1

            mbobs_list=[]
            for index in indices:
                mbobs=mm.get_mbobs(index, weight_type=weight_type)
                mbobs_list.append( mbobs )

            self._set_psfs(mbobs_list, psf_obs=psf_obs)

            if (nconf is not None 
                    and nconf['set_noise_images']):
                #logger.debug('setting noise images')
                self._set_noise_images(mbobs_list, medser.seg)

            fof_mbobs_lists.append( mbobs_list )

        return fof_mbobs_lists

    def _plot_fofs(self, mm, fofs):
        """
        make a plot of the fofs
        """
        mof.fofs.plot_fofs(
            mm.mlist[0],
            fofs,
            show=True,
            type='filled circle',
            orig_dims=self.obs[0][0].image.shape
        )


    def get_mbobs_list(self, cat=None, seg=None, obs=None, weight_type='weight'):
        """
        get a list of MultiBandObsList for every object or
        the specified indices

        this runs sep on the image
        """

        #mm=self.get_multiband_meds(cat=cat, seg=seg, obs=obs)
        medser=self.get_medsifier(cat=cat, seg=seg, obs=obs)
        if medser.cat.size==0:
            return []

        mm=medser.get_multiband_meds()

        mbobs_list = mm.get_mbobs_list(weight_type=weight_type)

        if obs is not None:
            logger.info("assuming psfs are all the same")
            psf_obs=obs[0][0].psf
        else:
            psf_obs=None

        self._set_psfs(mbobs_list, psf_obs=psf_obs)

        nconf=self.get("noise_images",None)
        if (nconf is not None 
                and nconf['set_noise_images']):
            #logger.debug('setting noise images')
            self._set_noise_images(mbobs_list, medser.seg)

        return mbobs_list

    def _set_noise_images(self, mbobs_list, seg):
        """
        find postage stamps with no objects in
        them and use them as noise images

        assume seg map same everywhere
        """

        nconf=self['noise_images']

        imcen=(np.array(self.obs[0][0].image.shape)-1.0)/2.0
        ncheck=nconf.get('ncheck',10)

        for mbobs in mbobs_list:
            # assume all same dimensions
            dims=mbobs[0][0].image.shape
            while True:
            #for i in range(ncheck):

                # choose a random position and see if
                # it is an empty patch

                cen = imcen + self.position_pdf.sample()
                subseg = self._get_patch(cen, seg, dims)

                # empty patches have segmap entirely
                # zero, unless we can't find one
                if nconf['check_seg']:
                    #if ( (i != ncheck-1)
                    #        and not np.all(subseg==0) ):
                    if not np.all(subseg==0):
                        continue

                for band,obslist in enumerate(mbobs):
                    for obsnum,obs in enumerate(obslist):
                        im=self.obs[band][obsnum].image
                        patch=self._get_patch(cen, im, dims)
                        
                        if nconf['negate']:
                            patch *= -1

                        obs.noise = patch

                # if we get here, we have set the noise image
                # for this object
                break

    def _get_patch(self, cen, im, dims):
        """
        extract a patch from the image

        leave it to the caller to make sure the position
        is what they want
        """

        row,col = cen.astype('i4')
        rowhalf = dims[0]//2
        colhalf = dims[1]//2
        start_row = row - rowhalf + 1
        start_col = col - colhalf + 1
        end_row   = row + rowhalf + 1 # plus one for slices
        end_col   = col + colhalf + 1

        subim=im[
            start_row:end_row,
            start_col:end_col,
        ].copy()

        return subim

    def _set_psfs(self, mbobs_list, psf_obs=None):
        for mbobs in mbobs_list:
            for olist in mbobs:
                for obs in olist:
                    if psf_obs is not None:
                        tpsf_obs=psf_obs.copy()
                    else:
                        tpsf_obs=self.get_psf_obs()
                    obs.set_psf(tpsf_obs)


    def get_multiband_meds(self, cat=None, seg=None, obs=None):
        """
        get a multiband MEDS instance
        """
        medser=self.get_medsifier(cat=cat, seg=seg, obs=obs)
        mm=medser.get_multiband_meds()
        return mm

    def get_medsifier(self, cat=None, seg=None, obs=None):
        """
        medsify the data
        """
        if obs is None:
            obs = self.obs

        dlist=[]
        for olist in obs:
            # assuming only one image per band
            tobs=olist[0]
            wcs=tobs.jacobian.get_galsim_wcs()

            dlist.append(
                dict(
                    image=tobs.image,
                    weight=tobs.weight,
                    wcs=wcs,
                )
            )

        sx_config=self.get('sx',None)
        meds_config=self.get('meds',None)
        peak_config=self.get('peaks',None)
        return mof.stamps.MEDSifier(
            dlist,
            cat=cat,
            seg=seg,
            sx_config=sx_config,
            meds_config=meds_config,
            peak_config=peak_config,
        )

    def show(self, **kw):
        """
        show a nice image of the simulation
        """
        return visualize.view_mbobs(self.obs, **kw)

    '''
    def show(self):
        """
        show a nice image of the simulation
        """
        import images
        #images.view(self.obs[2][0].image)
        rgb=self.get_color_image()
        images.view(rgb/rgb.max())

    def get_color_image(self):
        """
        get an rgb image of the sim
        """
        assert self['nband']==3,"must have 3 bands for color image"
        return visualize.make_rgb(
            self.obs[2][0].image,
            self.obs[1][0].image,
            self.obs[0][0].image,
        )
    '''

    def _make_g_pdf(self):
        if 'g' in self['pdfs']:
            c=self['pdfs']['g']
            return ngmix.priors.GPriorBA(c['sigma'], rng=self.rng)
        else:
            return None

    def _make_hlr_pdf(self):
        c=self['pdfs']['hlr']
        return self._get_generic_pdf(c)

    def _make_flux_pdf(self):
        c=self['pdfs']['flux']
        if isinstance(c,dict) and c['type']=='cosmos':
            return pdfs.CosmosFluxSampler(
                rng=self.rng,
                flux_range=c['flux_range'],
                r50_range=c['r50_range'],
                flux_mult=c.get('flux_mult',1.0),
            )
        else:
            return self._get_generic_pdf(c)

    def _make_primary_pdfs(self):
        if 'hlr_flux' in self['pdfs']:
            self.hlr_flux_pdf=self._make_hlr_flux_pdf()
        else:
            self.hlr_pdf = self._make_hlr_pdf()
            self.flux_pdf = self._make_flux_pdf()

    def _make_hlr_flux_pdf(self):
        c=self['pdfs']['hlr_flux']

        if c['type']=='cosmos':

            return pdfs.CosmosSampler(
                rng=self.rng,
                flux_range=c['flux_range'],
                r50_range=c['r50_range'],
                flux_mult=c['flux_mult'],
            )
        elif c['type'] == 'cosmos-extrap':
            return pdfs.CosmosExtrap(
                flux_mult=c['flux_mult'],
            )
        else:
            raise ValueError('bad hlr_flux type: "%s"' % c['type'])

    def _make_bulge_pdfs(self):
        self.bulge_hlr_frac_pdf=self._make_bulge_hlr_frac_pdf()
        self.fracdev_pdf=self._make_fracdev_pdf()

    def _make_bulge_hlr_frac_pdf(self):
        c=self['pdfs']['bulge']['hlr_fac']
        assert c['type'] == 'uniform'

        frng=c['range']
        return ngmix.priors.FlatPrior(frng[0], frng[1], rng=self.rng)

    def _make_fracdev_pdf(self):
        c=self['pdfs']['bulge']['fracdev']
        assert c['type'] == 'uniform'
        frng=c['range']
        return ngmix.priors.FlatPrior(frng[0], frng[1], rng=self.rng)

    def _get_bulge_stats(self):
        c=self['pdfs']['bulge']
        shift_width = c['bulge_shift']

        radial_offset = self.rng.uniform(
            low=0.0,
            high=shift_width,
        )
        theta = self.rng.uniform(low=0, high=np.pi*2)
        offset = (
            radial_offset*np.sin(theta),
            radial_offset*np.cos(theta),
        )

        hlr_fac = self.bulge_hlr_frac_pdf.sample()
        fracdev = self.fracdev_pdf.sample()
        grng=c['g_fac']['range']
        gfac = self.rng.uniform(
            low=grng[0],
            high=grng[1],
        )

        return hlr_fac, fracdev, gfac, offset


    def _get_knots_stats(self, disk_flux):
        c=self['pdfs']['knots']
        nrange = c['num']['range']
        num = self.rng.randint(nrange[0], nrange[1]+1)

        flux = num*c['flux_frac_per_knot']*disk_flux
        return num, flux

    def _make_knots_pdfs(self):
        c=self['pdfs']['knots']['num']
        assert c['type']=='uniform'

    def _get_generic_pdf(self, c):
        rng=self.rng

        if isinstance(c,dict):
            if c['type'] in ['lognormal','log-normal']:
                pdf = ngmix.priors.LogNormal(
                    c['mean'],
                    c['sigma'],
                    rng=rng,
                )
            elif c['type'] in ['uniform','flat']:
                pdf = ngmix.priors.FlatPrior(
                    c['range'][0],
                    c['range'][1],
                    rng=rng,
                )
            else:
                raise ValueError("bad pdf: '%s'" % c['type'])

            limits=c.get('limits',None)
            if limits is None:
                return pdf
            else:
                return ngmix.priors.LimitPDF(pdf, [0.0, 30.0])
        else:
            return pdfs.Constant(c)

    def _set_bands(self):
        nband=self.get('nband',None)

        if 'disk' not in self['pdfs']:
            self['pdfs']['disk'] = {}

        cdisk=self['pdfs']['disk']
        cbulge=self['pdfs'].get('bulge',None)
        cknots=self['pdfs'].get('knots',None)

        if nband is None or nband==1:
            self['nband']=1
            cdisk['color']=[1.0]
            if cbulge is not None:
                cbulge['color']=[1.0]
            if cknots is not None:
                cknots['color']=[1.0]


    #def _fit_psf_admom(self, obs):
    #    Tguess=4.0*self['pixel_scale']**2
    #    am=ngmix.admom.run_admom(obs, Tguess)
    #    return am.get_gmix()

    def get_psf_obs(self):
        kw={'scale':self['pixel_scale']}
        dims=self.get('psf_dims',None)
        if dims is not None:
            kw['nx'],kw['ny'] = dims[1],dims[0]

        psf_im = self.psf.drawImage(**kw).array

        dims = np.array(psf_im.shape)
        pcen=(dims-1.0)/2.0
        pjac = ngmix.DiagonalJacobian(
            row=pcen[0],
            col=pcen[1],
            scale=self['pixel_scale']
        )

        psf_im += self.rng.normal(
            scale=self['psf']['noise_sigma'],
            size=dims,
        )
        psf_wt=np.zeros(dims)+1.0/self['psf']['noise_sigma']**2

        return ngmix.Observation(
            psf_im,
            weight=psf_wt,
            jacobian=pjac,
        )

    def _set_position_pdf(self):
        type=self['positions']['type']
        if type=='cluster':
            sigma=self['positions']['scale']
            maxrad = 3*sigma

            pdf=ngmix.priors.TruncatedSimpleGauss2D(
                0.0,0.0,
                sigma, sigma,
                maxrad,
                rng=self.rng,
            )
        elif type=='uniform':
            half=self['positions']['width']/2.0
            pdf=pdfs.Flat2D(
                [-half, half],
                [-half, half],
                rng=self.rng,
            )
        else:
            raise ValueError("positions should be type 'cluster' or 'uniform'")

        self.position_pdf=pdf

    def _get_nobj(self):
        nobj=self['nobj']
        if isinstance(nobj,dict):
            nobj = self.rng.poisson(lam=nobj['mean'])
            if nobj < 1:
                nobj=1
        else:
            nobj=self['nobj']

        return nobj


    def _set_psf(self):
        import galsim
        model=self['psf']['model']
        assert model in ['gauss','moffat']

        if model=='gauss':
            self.psf = galsim.Gaussian(
                fwhm=self['psf']['fwhm'],
            )
        elif model=='moffat': 
            self.psf = galsim.Moffat(
                self['psf']['beta'],
                fwhm=self['psf']['fwhm'],
            )

    def _get_hlr_flux(self):
        if 'hlr_flux' in self['pdfs']:
            hlr, flux = self.hlr_flux_pdf.sample()
        else:
            hlr = self.hlr_pdf.sample()

            if self.flux_pdf=='track_hlr':
                flux = hlr**2 *self['pdfs']['F']['factor']
            else:
                flux = self.flux_pdf.sample()

        return hlr, flux

    def _get_object(self):

        hlr, flux = self._get_hlr_flux()

        disk_hlr = hlr

        if self.g_pdf is None:
            disk_g1,disk_g2 = 0.0, 0.0
        else:
            disk_g1,disk_g2 = self.g_pdf.sample2d()


        all_obj={}

        if 'bulge' in self['pdfs']:
            hlr_fac, fracdev, gfac, bulge_offset = self._get_bulge_stats()

            bulge_hlr = disk_hlr*hlr_fac
            bulge_g1,bulge_g2 = gfac*disk_g1, gfac*disk_g2

            disk_flux = (1-fracdev)*flux
            bulge_flux = fracdev*flux

            bulge=galsim.DeVaucouleurs(
                half_light_radius=bulge_hlr,
                flux=bulge_flux,
            ).shear(
                g1=bulge_g1, g2=bulge_g2,
            ).shift(
                dx=bulge_offset[1], dy=bulge_offset[0],
            )
            all_obj['bulge'] = bulge

        else:
            disk_flux = flux

        disk = galsim.Exponential(
            half_light_radius=disk_hlr,
            flux=disk_flux,
        ).shear(
            g1=disk_g1, g2=disk_g2,
        )
        all_obj['disk'] = disk

        if 'knots' in self['pdfs']:
            nknots, knots_flux = self._get_knots_stats(disk_flux)

            knots = galsim.RandomWalk(
                npoints=nknots,
                half_light_radius=disk_hlr,
                flux=knots_flux,
            ).shear(g1=disk_g1, g2=disk_g2)

            all_obj['knots'] = knots

        obj_cen1, obj_cen2 = self.position_pdf.sample()
        all_obj['cen'] = np.array([obj_cen1, obj_cen2])
        return all_obj

    def _set_objects(self):
        self.objlist=[]

        nobj = self._get_nobj()

        self._obj_data=np.zeros(
            nobj,
            dtype=[
                ('image_id','i4'),
                ('x','f8'),
                ('y','f8')
            ],
        )

        for i in range(nobj):

            obj = self._get_object()
            self.objlist.append(obj)

    def _draw_objects(self):
        """
        this is dumb, drawing into the full image when
        we don't need to
        """

        self.imlist=[]

        shear=self.get('shear',None)

        cdisk=self['pdfs']['disk']
        cbulge=self['pdfs'].get('bulge',None)
        cknots=self['pdfs'].get('knots',None)

        dims = self['dims']
        cen=(np.array(dims)-1.0)/2.0

        for band in range(self['nband']):
            objects=[]
            for iobj,obj_parts in enumerate(self.objlist):

                disk=obj_parts['disk']*cdisk['color'][band]
                tparts=[disk]

                if cbulge is not None:
                    bulge=obj_parts['bulge']*cbulge['color'][band]
                    tparts.append(bulge)

                if cknots is not None:
                    knots = obj_parts['knots']*cknots['color'][band]
                    tparts.append( knots )

                if len(tparts)==1:
                    obj = tparts[0]
                else:
                    obj = galsim.Sum(tparts)

                obj = obj.shift(
                    dx=obj_parts['cen'][1],
                    dy=obj_parts['cen'][0],
                )

                if shear is not None and not self['shear_all']:
                    obj = obj.shear(
                        g1=shear[0],
                        g2=shear[1],
                    )

                if band==0:
                    ocen=obj.centroid
                    row=cen[0]+ocen.y/self['pixel_scale']
                    col=cen[1]+ocen.x/self['pixel_scale']
                    self._obj_data['y'][iobj] = row
                    self._obj_data['x'][iobj] = col

                objects.append(obj)

            objects = galsim.Sum(objects)

            if shear is not None and self['shear_all']:
                objects = objects.shear(
                    g1=shear[0],
                    g2=shear[1],
                )

            convolved_objects = galsim.Convolve(objects, self.psf)

            kw={'scale':self['pixel_scale']}
            kw['method'] = self['draw_method']

            dims = self.get('dims',None)
            if dims is not None:
                kw['nx'],kw['ny'] = dims[1],dims[0]

            image =  convolved_objects.drawImage(**kw).array
            self.imlist.append(image)


    def _add_noise(self):
        for im in self.imlist:
            noise_image = self.rng.normal(
                scale=self['noise_sigma'],
                size=im.shape,
            )
            im += noise_image

    def _subtract_backgrounds(self):
        import sep
        c=self['background']['config']

        for im in self.imlist:
            bkg = sep.Background(
                im,
                bw=c['back_size'],
                bh=c['back_size'],
                fw=c['filter_width'],
                fh=c['filter_width'],
            )
            bkg_image = bkg.back()
            med=np.median(bkg_image)
            logger.debug('    bkg median: '
                         '%g rms: %g' % (med,bkg.globalrms))
            im -= bkg_image

    def get_all_metacal(self, **metacal_pars):
        """
        get metal versions of the big image
        """
        return ngmix.metacal.get_all_metacal(
            self.obs,
            **metacal_pars
        )

    def _do_coadd(self):
        """
        coadd all images. currently psf is same in all so
        no need to coadd it
        """
        assert len(self.imlist) > 1,"can't coadd one image"

        nim = len(self.imlist)
        self['coadd_noise_sigma'] = self['noise_sigma']*np.sqrt(nim)

        logger.debug('    doing coadd')
        im = self.imlist[0]

        for tim in self.imlist[1:]:
            im += tim

        self.imlist = [im]

    def _make_obs(self):

        mbobs=ngmix.MultiBandObsList()

        for im in self.imlist:
            jacobian = ngmix.DiagonalJacobian(
                row=0,
                col=0,
                scale=self['pixel_scale']
            )

            if self['coadd']:
                wt=np.zeros(im.shape) + 1.0/self['coadd_noise_sigma']**2
            else:
                wt=np.zeros(im.shape) + 1.0/self['noise_sigma']**2

            obs = ngmix.Observation(
                im,
                weight=wt,
                jacobian=jacobian,
                psf=self.get_psf_obs(),
            )
            olist=ngmix.ObsList()
            olist.append(obs)
            mbobs.append(olist)

        self.obs=mbobs

class PairSim(Sim):

    def _get_nobj(self):
        return 2

    def _set_objects(self):

        nobj = self._get_nobj()

        self._obj_data=np.zeros(
            nobj,
            dtype=[
                ('image_id','i4'),
                ('x','f8'),
                ('y','f8')
            ],
        )

        # these are dicts
        cen_obj = self._get_object()
        nbr_obj = self._get_object()

        # this is in addition to the sub-pixel offset
        # already present
        dx, dy = self._get_nbr_position()
        nbr_obj['cen'][1] += dx
        nbr_obj['cen'][0] += dy

        self.objlist = [
            cen_obj,
            nbr_obj,
        ]

    def _get_nbr_position(self):
        """
        place on a ring
        """
        posconf = self['positions']
        assert posconf['type']=='ring',('only center with nbr in ring '
                                        'supported for PairSim')

        if 'radius_pixels' in posconf:
            radarc = posconf['radius_pixels']*self['pixel_scale']
        elif 'radius_arcsec' in posconf:
            radarc = posconf['radius_arcsec']
        else:
            raise ValueError('radius_pixels or radius_arcsec should be '
                             'present in positions config')

        theta = self.rng.uniform(low=0, high=2*np.pi)

        dx = np.cos(theta)*radarc
        dy = np.sin(theta)*radarc
        return dx, dy

    def _set_position_pdf(self):
        """
        just pixel offsets; we do the ring offset elsewhere
        """
        half=self['pixel_scale']/2.0
        self.position_pdf=pdfs.Flat2D(
            [-half, half],
            [-half, half],
            rng=self.rng,
        )
