import numpy as np
import seaborn as sns


def plot_psf_model(fwhms, g1, g2, im_width, axs):
    """Plot a PSF model.

    Parameters
    ----------
    fwhms : np.ndarray, shape (im_width, im_width)
        A grid of the FWHM of the PSF model.
    g1 : np.ndarray, shape (im_width, im_width)
        The 1-component of the shape of the PSF.
    g2 : np.ndarray, shape (im_width, im_width)
        The 2-component of the shape of the PSF.
    im_width : int
        The width of the image in pixels.
    axs : list of lists of axes
        A set of four axes to plot the PSF model statistics.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import seaborn as sns
    >>> sns.set()
    >>> fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 16))
    >>> plot_psf_model(fwhm, g1, g2, 225, axs)
    """
    n = fwhms.shape[0]
    xt = []
    for i, _x in enumerate(np.linspace(0, im_width, n)):
        if i % 10 == 0 or i == 0 or i == n - 1:
            xt.append("%0.0f" % _x)
        else:
            xt.append('')

    kwargs = {
        'xticklabels': xt,
        'yticklabels': xt
    }
    sns.heatmap(fwhms, square=True, ax=axs[0, 0], **kwargs)
    axs[0, 0].set_xlabel('column')
    axs[0, 0].set_ylabel('row')
    axs[0, 0].set_title('FWHM')

    nmod = 10
    g = np.sqrt(g1**2 + g2**2)
    g /= np.mean(g)
    loc = np.linspace(0, im_width, n)
    beta = np.arctan2(g2, g1)/2
    axs[0, 1].quiver(
        loc[::nmod],
        loc[::nmod],
        (g * np.cos(beta))[::nmod, ::nmod],
        (g * np.sin(beta))[::nmod, ::nmod],
        scale_units='xy',
        scale=0.075,
        headwidth=0,
        pivot='mid')
    axs[0, 1].set_xlabel('column')
    axs[0, 1].set_ylabel('row')
    axs[0, 1].set_title('PSF shape')

    sns.heatmap(g1, square=True, ax=axs[1, 0], **kwargs)
    axs[1, 0].set_xlabel('column')
    axs[1, 0].set_ylabel('row')
    axs[1, 0].set_title('g1')

    sns.heatmap(g2, square=True, ax=axs[1, 1], **kwargs)
    axs[1, 1].set_xlabel('column')
    axs[1, 1].set_ylabel('row')
    axs[1, 1].set_title('g2')
