import numpy as np

def make_rgb_old(r, g, b):
    import images

    maxval=max( r.max(), g.max(), b.max() )
    scales = [1.0/maxval]*3
    rgb=images.get_color_image(
        r.transpose(),
        g.transpose(),
        b.transpose(),
        scales=scales,
        nonlinear=0.1,
    )

    return rgb

def make_rgb(mbobs):
    import images

    #SCALE=.015*np.sqrt(2.0)
    #SCALE=0.001
    # lsst
    SCALE=0.0005
    #relative_scales = np.array([1.00, 1.2, 2.0])
    relative_scales = np.array([1.00, 1.0, 2.0])

    scales= SCALE*relative_scales

    r=mbobs[2][0].image
    g=mbobs[1][0].image
    b=mbobs[0][0].image

    rgb=images.get_color_image(
        r.transpose(),
        g.transpose(),
        b.transpose(),
        scales=scales,
        nonlinear=0.12,
    )
    return rgb

def fake_rgb(mbobs):
    """
    fake rgb from single band
    """
    import images

    #SCALE=.015*np.sqrt(2.0)
    #SCALE=0.001
    SCALE=1.0
    relative_scales = np.array([1.0, 1.0, 1.0])
    scales= SCALE*relative_scales


    r=mbobs[0][0].image
    g=r
    b=r

    rgb=images.get_color_image(
        r.transpose(),
        g.transpose(),
        b.transpose(),
        #scales=scales,
        #nonlinear=0.12,
        nonlinear=0.07,
    )
    return rgb


def view_mbobs(mbobs, **kw):
    import images

    if len(mbobs)==3:
        rgb=make_rgb(mbobs)
        plt=images.view(rgb, **kw)
    else:
        #rgb=fake_rgb(mbobs)
        # just show the first one
        #plt = images.view(mbobs[0][0].image, **kw)
        #plt=images.view(rgb, **kw)
        #imc = mbobs[0][0].image.clip(min=1.0e-6)

        imin=-95.0
        imax=985.0
        imc = mbobs[0][0].image.copy()
        #imin, imax = imc.min(), imc.max()
        #print('min:',imin,'max:',imax)
        imc -= imin
        imc *= 1.0/imax
        plt = images.view(imc, nonlinear=0.3)

        #imstd=imc.std()
        #imc.clip(min=-, out=imc)
        #logim=np.log10(imc)

        #imc -= imc.min() + 1.0e-6
        #logim=np.log10(imc)
        #imc *= 1.0/imc.max()
        #logim *= 1.0/logim.max()
        #plt = images.view(logim, **kw)

    return plt


def view_mbobs_list(mbobs_list, **kw):
    import biggles
    import images
    import plotting

    weight=kw.get('weight',False)
    nband=len(mbobs_list[0])

    if weight:
        grid=plotting.Grid(len(mbobs_list))
        plt=biggles.Table(
            grid.nrow,
            grid.ncol,
        )
        for i,mbobs in enumerate(mbobs_list):
            if nband==3:
                im=make_rgb(mbobs)
            else:
                im=mbobs[0][0].image
            wt=mbobs[0][0].weight

            row,col = grid(i)

            tplt=images.view_mosaic([im, wt], show=False)
            plt[row,col] = tplt

        plt.show()
    else:
        if nband==3:
            imlist=[make_rgb(mbobs) for mbobs in mbobs_list]
        else:
            imlist=[mbobs[0][0].image for mbobs in mbobs_list]

        plt=images.view_mosaic(imlist, **kw)
    return plt
