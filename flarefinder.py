import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt
import theano
import exoplanet as xo
import pymc3 as pm
from scipy.signal import savgol_filter
import corner
from math import log, floor, sqrt
from astropy.io import fits as pyfits
from scipy.special import erf
from copy import copy, deepcopy
from scipy.interpolate import interp1d
import matplotlib.mlab as ml
import scipy.signal as signal
from stats.general import logtrapz

class Lightcurve():
    """
    A class designed to handle the day-to-day requirements for
    Kepler light curves, including removing DC offsets.

    Parameters
    ----------
    curve : string
       The file name for a light curves.
    """

    id = 0   # The TIC number of the star

    flux = np.array([])
    raw_flux = np.array([])
    time = np.array([])
    flux_error = np.array([])
    cadenceno = np.array([])

    running_median_dt = 0
    running_median_fit = np.array([])
    datagap = False

    def __init__(self, curve=None, maxgap=1):

        flux = None
        flux = np.array([])
        if curve != None:
            self.add_data(curve=curve, maxgap=1)

    def __str__(self):
        return "<bayesflare Lightcurve for TIC "+str(self.id)+">"

    def __unicode__(self):
        return self.__str__()

    def __repr__(self):
        return self.__str__()

    def identity_string(self):
        """
        Returns a string which identifies the lightcurve.

        Returns
        -------
        str
           An identifier of the light curve based on its lengt.
        """
        return "lc_len_"+str(len(self.flux))

    def dt(self):
        """
        Calculate the sample separation for the light curve.

        Returns
        -------
        float
           The separation time for data in the light curve.
        """
        return self.time[1] - self.time[0]

    def fs(self):
        """
        Calculate the sample frequency of the light curve.

        Returns
        -------
        float
           The sample frequency of the light curve.
        """
        return 1.0 / self.dt()

    def psd(self):
        """
        Calculate the one-sided non-windowed power spectrum of the light curve. This uses the
        :func:`matplotlib.mlab.psd` function for computing the power spectrum, with a single
        non-overlapping FFT.

        Returns
        -------
        sk : array-like
           The Power spectral density of the light curve.
        f  : array-like
           An array of the frequencies.
        """
        l = len(self.flux)

        # get the power spectrum of the lightcurve data
        sk, f = ml.psd(x=self.flux, window=signal.boxcar(l), noverlap=0, NFFT=l, Fs=self.fs(), sides='onesided')

        # return power spectral density and array of frequencies
        return sk, f

    def smooth(self,f,window=301,polyorder=3):
        """
        Smooth lightcurve w Savitzy-Golay filter

        Parameters
        -------
            f: the flux array
            window: window to smooth over
            polyorder: order of the polynomial fit

        Returns
        -------
        float
           The separation time for data in the light curve.
        """

        smooth = savgol_filter(f, window, polyorder=3)
        resid = f - smooth
        mask = resid < 2*np.sqrt(np.mean(resid**2))

        return smooth, resid, mask

    def add_data(self, target=None, maxgap=1):
        """
        Add light curve data to the object..

        Parameters
        ----------
        curvefile : string
           The file path file pointing to a light curve fits files.
        maxgap : int, optional, default+1
           The largest gap size (in bins) allowed before the light curve is deemed to contain gaps.

        Exceptions
        ----------
        NameError
           This needs to be replaced with an exception specific to the package!
           Error is raised if there is an I/O error while accessing a light curve file.
        """
        try:

            lcf = lk.search_lightcurvefile(target).download(quality_bitmask='hard')
            lc = lcf.get_lightcurve('PDCSAP_FLUX').remove_nans().normalize()

            raw_flux = lc.flux
            time = lc.time *24*3600 # convert to seconds
            norm_flux = (lc.flux-1)*1e3
            smooth_lc, resid, mask = self.smooth(norm_flux,window=301,polyorder=3)
            flux = resid
            flux_error = lc.flux_err
            cadenceno = lc.cadenceno


        except IOError:
            raise NameError("[Error] opening file")

        self.flux = np.append(self.flux, deepcopy(flux))
        self.raw_flux = np.append(self.raw_flux, deepcopy(raw_flux))
        self.time = np.append(self.time, deepcopy(time))
        self.flux_error = np.append(self.flux_error, deepcopy(flux_error))
        self.cadenceno = np.append(self.cadenceno, deepcopy(cadenceno))

        self.datagap = self.gap_checker(self.flux, maxgap=maxgap)
        self.interpolate()

    def gap_checker(self, d, maxgap=1):
        """
        Check for NaN gaps in the data greater than a given value.

        Parameters
        ----------
        d : :class:`numpy.ndarray`
           The array to check for gaps in the data.

        maxgap : int, optional, default: 1
           The maximum allowed size of gaps in the data.

        Returns
        -------
        bool
           ``True`` if there is a gap of maxgap or greater exists in ``d``, otherwise ``False``.
        """

        z = np.invert(np.isnan(d))
        y = np.diff(z.nonzero()[0])
        if len(y < maxgap+1) != len(y):
            return True
        else:
            return False

    def nan_helper(self, y):
        """
        Helper to handle indices and logical indices of NaNs.

        Parameters
        ----------
        y : ndarray
           An array which may contain NaN values.

        Returns
        -------
        nans : ndarray
          An array containing the indices of NaNs
        index : function
          A function, to convert logical indices of NaNs to 'equivalent' indices

        Examples
        --------

           >>> # linear interpolation of NaNs
           >>> spam = np.ones(100)
           >>> spam[10] = np.nan
           >>> camelot = bf.Lightcurve(curves)
           >>> nans, x = camelot.nan_helper(spam)
           >>> spam[nans]= np.interp(x(nans), x(~nans), spam[~nans])


        """

        return np.isnan(y), lambda z: z.nonzero()[0]

    def interpolate(self):
        """
        A method for interpolating the light curves, to compensate for NaN values.

        Examples
        --------

           >>> camelot = bf.Lightcurve(curves)
           >>> camelot.interpolate()

        """

        z = self.flux
        nans, za= self.nan_helper(z)
        z[nans]= np.interp(za(nans), za(~nans), z[~nans]).astype('float32')
        self.flux = z


    def plot(self, figsize=(10,3)):
        """
        Method to produce a plot of the light curve.

        Parameters
        ----------
        figsize : tuple
           The size of the output plot.

        """
        fig, ax = plt.subplots(1)
        plt.title('Lightcurve for TIC'+str(self.id))
        self.trace = ax.plot(self.time/(24*3600.0), self.flux)
        fig.autofmt_xdate()
        plt.xlabel('Time [days]')
        plt.ylabel('Luminosity')
        plt.show()

def estimate_noise_tv(d, sigma=1.0):
    """
    A method of estimating the noise, whilst ignoring large outliers.

    This uses the cumulative distribution of the data point and uses the probability
    contained within a Gaussian range (defined by sigma) to work out what the
    standard deviation is (i.e. it doesn't use tails of the distribution that
    contain large outliers, although the larger the sigma value to more outliers
    will effect the result.) This is mainly suitable to data in which the
    underlying noise is Gaussian.

    Parameters
    ----------
    d : array-like
        The time series of data (either a :class:`numpy.array` or a list).
    sigma: float
        The number of standard deviations giving the cumulative probability
        to be included in the noise calculation e.g. if sigma=1 then the central
        68% of the cumulative probability distribution is used.

    Returns
    -------
    std: float
        The noise standard deviation
    mean: float
        The value at the middle of the distribution
    """

    ld = len(d)

    # get normalised histogram
    n, bins = np.histogram(d, bins=ld, density=True)
    bincentres = (bins[:-1] + bins[1:])/2. # bin centres

    # get the cumulative probability distribution
    cs = np.cumsum(n*(bins[1]-bins[0]))

    # get unique values (helps with interpolation)
    csu, idx = np.unique(cs, return_index=True)
    binsu = bincentres[idx]

    # get the cumulative % probability covered by sigma
    cp = erf(sigma/np.sqrt(2.))

    interpf = interp1d(csu, binsu) # interpolation function

    # get the upper and lower interpolated data values that bound the range
    lowS = interpf(0.5 - cp/2.);
    highS = interpf(0.5 + cp/2.);

    # get the value at the middle of the distribution
    m = interpf(0.5);

    # get the standard deviation estimate
    std = (highS - lowS)/(2.*sigma)

    return std, m

def estimate_noise_ps(lightcurve, estfrac=0.5):
    """
    Use the high frequency part of the power spectrum of a light curve
    to estimate the time domain noise standard deviation of the
    data. This avoids the estimate being contaminated by low-frequency lines
    and flare signals.

    Parameters
    ----------
    lightcurve : :class:`.Lightcurve`
       A :class:`.Lightcurve` instance containing the time series data.
    estfrac : float, optional, default: 0.5
       The fraction of the spectrum (from the high frequency end)
       with which to estimate the noise. The default is 0.5 i.e. use
       the final half of the spectrum for the estimation.

    Returns
    -------
    sqrt(sk) : float
        The noise standard deviation
    sk : float
        The noise variance
    noise_v : :class:`numpy.array`
        A vector of noise variance values
    """
    l = len(lightcurve.flux)
    # get the power spectrum of the lightcurve data
    sk, f = lightcurve.psd()
    # get the mean of the final quarter of the data

    sk = np.mean(sk[int(np.floor((1.-estfrac)*len(sk))):]) #this was returning a float too I think
    # scale to give noise variance
    sk = sk * lightcurve.fs() / 2.
    noise_v = np.ones(nextpow2(2*len(lightcurve.flux)-1)) * sk

    return np.sqrt(sk), sk, noise_v

def nextpow2(i):
    """
    Calculates the nearest power of two to the inputed number.

    Parameters
    ----------
    i : int
       An integer.

    Output
    ------
    n : int
       The power of two closest to `i`.
    """
    n = 1
    while n < i: n *= 2
    return n

class Bayes():
    """
    The Bayes class contains the functions responsible for calculating the Bayesian odds ratios for
    a model given the light curve data.

    Parameters
    ----------
    lightcurve : :class:`.Lightcurve` object
       An instance of a :class:`.Lightcurve` which the Bayesian odds ratios will be calculated for.
    model : Model object
       An instance of a Model object which will be used to generate odds ratios.
    """

    # Object data
    premarg = {}         # Place to store the pre-marginalised bayes factor arrays

    def __init__(self, lightcurve, model):
        """
        The initiator method
        """
        self.lightcurve = lightcurve
        self.model      = deepcopy(model)
        self.ranges     = deepcopy(model.ranges)
        self.confidence = 0.999
        self.noise_ev = self.noise_evidence()

    def bayes_factors(self, **kwargs):
        """
        Work out the logarithm of the Bayes factor for a signal consisting of the model (e.g. a
        flare) in the light curve data compared with Gaussian noise (the model and the light curve
        must be defined on initialise of the class) for a range of model parameters. Of the model
        parameters the amplitude will be analytically marginalised over. The Bayes factor for each
        model time stamp (i.e. the central time of a flare) will be calculated over the parameter
        space containing the additional model parameters, as defined by the model. All these will
        require subsequent marginalisation if necessary.
        If the light curve has had detrending applied then the model will also get detrended in the
        same way.
        """

        model = self.model

        N = len(self.lightcurve.time)
        s = np.copy(model.shape)
        l = np.product(s)
        s = np.append(s,N)

        s = tuple(model.shape) + (N,)
        self.lnBmargAmp = -np.inf*np.ones(s)

        x = self.lightcurve.time
        z = self.lightcurve.flux
        sk = estimate_noise_ps(self.lightcurve)[1]

        for i in np.arange(l):
            i = int(i)
            q = np.unravel_index(i, model.shape)
            m = model(i)
            if m == None:
                # if the model is not defined (e.g. for the flare model when tau_g > tau_e)
                # set probability to zero (log probability to -inf)
                self.lnBmargAmp[q][:] = np.ones(N)*-np.inf
                continue
            # Generate the model flare
            m = m.flux

            # Run the xcorr and perform the analytical marginalisation over amplitude
            B = log_marg_amp(z, m, sk)

            priors = np.ndarray(tuple(model.shape)) #LW added
            mparams={} #LW added
            # get prior
            for k in range(len(model.shape)):
                # set parameter dict for prior function
                mparams[model.paramnames[k]] = self.ranges[model.paramnames[k]][q[k]]

            priors[q] = model.prior(mparams)


            # Apply Bayes Formula
            self.lnBmargAmp[q][:] = B + np.sum(priors[q])
            #self.lnBmargAmp[q][:] = B + np.sum(model.prior)

            self.premarg[id(model)] = self.lnBmargAmp

    def marginalise(self, pname):
        """
        Function to reduce the dimensionality of the `lnBmargAmp` :class:`numpy.ndarray` from `N` to
        `N-1` through numerical marginalisation (integration) over a given parameter.

        Parameters
        ----------
        axis: string
            The parameter name of the array that is to be marginalised.

        Returns
        -------
        B : :class:`Bayes`
            A :class:`Bayes` object in which the `lnBmargAmp` :class:`numpy.ndarray` has had one
            parameter marginalised over.
        """

        arr = self.lnBmargAmp
        places = self.ranges[pname]
        axis = self.model.paramnames.index(pname)
        if len(places) > 1:
            x = np.apply_along_axis(logtrapz, axis, arr, places)
        elif len(places) == 1:
            # no marginalisation required just remove the specific singleton dimension via reshaping
            z = arr.shape
            q = np.arange(0,len(z)).astype(int) != axis
            newshape = tuple((np.array(list(z)))[q])
            x = np.reshape(arr, newshape)

        model = copy(self.model)
        model.paramnames.remove(pname)

        B = Bayes(self.lightcurve, model)

        ranges = copy(self.ranges)
        del ranges[pname]
        B.ranges = ranges

        B.lnBmargAmp = x
        return B

    def marginalise_full(self):
        """
        Marginalise over each of the parameters in the `ranges` list in turn.

        Returns
        -------
        A : :class:`Bayes`
            A :class:`Bayes` object for which the `lnBmargAmp` array has been marginalised over all
            parameters in the `ranges` list
        """

        A = self
        for p in self.ranges:
            A = A.marginalise(p)

        return A

    def noise_evidence(self):
        """
        Calculate the evidence that the data consists of Gaussian noise. This calculates the noise
        standard deviation using the 'tailveto' method of :func:`.estimate_noise_tv`.

        Returns
        -------
        The log of the noise evidence value.

        .. note::
            In this the :func:`.estimate_noise_tv` method is hardcoded to use a `tvsigma` value of
            1.0.
        """
        var = estimate_noise_tv(self.lightcurve.flux, 1.0)[0]**2
        noise_ev = -0.5*len(self.lightcurve.flux)*np.log(2.*np.pi*var) - np.sum(self.lightcurve.flux**2)/(2.*var)

        return noise_ev


def log_marg_amp_full_model_wrapper(params):
    """
    Wrapper to :func:`.log_marg_amp_full_model` and :func:`.log_marg_amp_full_2Dmodel` function that
    takes in a tuple of all the required parameters. This is required to use the
    :mod:`multiprocessing` `Pool.map_async` function.

    Parameters
    ----------
    params : tuple
        A tuple of parameters required by :func:`.log_marg_amp_full_2Dmodel` or
        :func:`.log_marg_amp_full_model`

    Returns
    -------
    margamp : :class:`numpy.ndarray`
        An array containing the logarithm of the likelihood ratio.
    """
    shape = params[1]

    if len(shape) == 2: # specific case for a model with two parameters
        return log_marg_amp_full_2Dmodel(params[0], params[1], params[2], params[3], params[4],
                                         params[5], params[6], params[7], params[8], params[9],
                                         params[10])
    else:
        return log_marg_amp_full_model(params[0], params[1], params[2], params[3], params[4],
                                       params[5], params[6], params[7], params[8], params[9],
                                       params[10])

def log_likelihood_marg_background_wrapper(params):
    """
    Wrapper to :func:`.log_likelihood_marg_background` that takes a tuple of all the required
    parameters. This is required to use the :mod:`multiprocessing` `Pool.map_async` function.

    Parameters
    ----------
    params : tuple
        A tuple of parameters required by :func:`.log_likelihood_marg_background`.

    Returns
    -------
    margamp : :class:`numpy.ndarray`
        An array containing the logarithm of the likelihood ratio.
    """
    return log_likelihood_marg_background(params[0], params[1], params[2], params[3])


def contiguous_regions(condition):
        """
        Find contiguous regions for the condition e.g. array > threshold and return
        a list as two columns of the start and stop indices for each region
        (see http://stackoverflow.com/a/4495197/1862861)

        Parameters
        ----------
        condition : string
           A test condition (e.g. 'array > threshold') returning a :class:`numpy.array`.

        Returns
        -------
        idx : array-like
           A two column array containing the start and end indices of
           contiguous regions obeying the condition.

        """

        # Find the indicies of changes in "condition"
        d = np.diff(condition)
        idx, = d.nonzero()

        # We need to start things after the change in "condition". Therefore,
        # we'll shift the index by 1 to the right.
        idx += 1

        if condition[0]:
            # If the start of condition is True prepend a 0
            idx = np.r_[0, idx]

        if condition[-1]:
            # If the end of condition is True, append the length of the array
            idx = np.r_[idx, condition.size] # Edit

        # Reshape the result into two columns
        idx.shape = (-1,2)
        return idx


class SigmaThresholdMethod():
    """
    Search for points on a light curve that cross a threshold that is based on a number of
    standard deviations calculated from the data. This is based on the method used in [1]_.

    Parameters
    ----------
    lightcurve : :class`.Lightcurve`
       The light curve to be processed.
    detrendpoly : bool, optional, default: False
       Set to True to remove a second-order polynomial to fit the whole curve.
    detrendmedian : bool, optional, default: True
       Set to `True` to detrend the `lightcurve` data using a median filtering technique.
    noiseestmethod : {None, 'powerspectrum', 'tailveto'}, optional
       The method used to estimate the noise in the light curve. If `None` is chosen the noise will
       be estimated as the standard deviation of the entire light curve, including any signals.
       Defaults to `None`.
    psestfrac : float, optional, default: 0.5
       The fraction of the power spectrum to be used in estimating the noise, if
       `noiseestmethod=='powerspectrum'`.
    tvsigma : float, optional, default: 1.0
       The number of standard deviations giving the cumulative probability
       to be included in the noise calculation e.g. if sigma=1 then the central
       68% of the cumulative probability distribution is used.

    See Also
    --------

    estimate_noise_ps : The power spectrum noise estimator.
    estimate_noise_tv : The tail veto noise estimator.
    Lightcurve.running_median : The running median detrender.

    References
    ----------

    .. [1] Walkowicz *et al*, *AJ*, **141** (2011), `arXiv:1008.0853 <http://arxiv.org/abs/1008.0853>`_

    """

    sigma = 0
    nflares = 0
    flarelist = []

    def __init__(self, lightcurve, noiseestmethod=None, psestfrac=0.5, tvsigma=1.0):
        # detrend the lightcurve using a 2nd order polynomial fit to the whole curve
        # if required. Also detrend with a running median over 10 hr intervals.

        self.lightcurve = deepcopy(lightcurve)


        # get the standard deviation using the outlier removal method or not
        if noiseestmethod == None:
            self.sigma = np.std(self.lightcurve.flux)
        elif noiseestmethod == 'powerspectrum':
            self.sigma = estimate_noise_ps(self.lightcurve.flux, estfrac=peestfrac)[0]
        elif noiseestmethod == 'tailveto':
            self.sigma = estimate_noise_tv(self.lightcurve.flux, sigma=tvsigma)[0]
        else:
            print("Error... noise estimation method not recognised")

    def thresholder(self, sigmathresh=4.5, mincontiguous=8, usemedian=False, removeedges=False):
        """
        Perform the thresholding on the data.

        Parameters
        ----------
        sigmathresh : float, default: 4.5
           The number of standard deviations above which a value must be to count as a detection.
        mincontiguous : int, default: 3
           The number of contiguous above threshold values required to give a detection.
        usemedian : bool, default: False
           If True subtract the median value from the light curve, otherwise subtract the mean value.
        removeedges : bool, default: True
           If True remove the edges of the light curve with 5 hours (half the hardcoded running median
           window length) from either end of the data.

        Returns
        -------
        flarelist : list of tuples
           A list of tuples containing the star and end array indices of above-threshold regions
        len(flarelist)
           The number of 'flares' detected.
        """

        flarelist = []
        numcount = 0

        flux = copy(self.lightcurve.flux)

        if removeedges: # remove 5 hours (half the running median window) from either end of the data
            nremove = int(5.*60.*60./self.lightcurve.dt())
            flux = flux[nremove:-nremove]
        else:
            nremove = 0

        if usemedian:
            flux = flux - np.median(flux)
        else:
            flux = flux - np.mean(flux)

        condition = flux > sigmathresh*self.sigma

        # find contiguous regions
        for start, stop in contiguous_regions(condition):
            if stop-start > mincontiguous-1:
                # found a flare!
                flarelist.append((start+nremove, stop+nremove))

        self.nflares = len(flarelist)
        self.flarelist = flarelist

        return flarelist, len(flarelist) # return list of indices and number of flares


class OddsRatioDetector():
    """
    Class to produce the odds ratio detection statistic for a flare versus a selection of noise
    models. The class will also provides thresholding of the log odds ratio for the purpose of
    flare detection.

    Parameters
    ----------
    lightcurve : :class:`Lightcurve`
        The light curve data in which to search for signal.
    bglen : int, default: 55
        The length of the analysis window that slides across the data.
    bgorder : int, default: 4
        The order of the polynomial background variations used in the signal and noise models.
    nsinusoids : int, default: 0
        The number of sinusoids to find and use in the background variations.
    noiseestmethod : string, default: 'powerspectrum'
        The method used to estimate the noise standard deviation of the light curve data.
    psestfrac : float, default: 0.5
        If using the 'powerspectrum' method of noise estimation (:func:`.estimate_noise_ps`) this
        gives the fraction of the spectrum (starting from the high frequency end) used in the noise
        estimate. This value can be between 0 and 1.
    tvsigma : float, default: 1.0
        If using the 'tailveto' method of noise estimation (:func:`.estimate_noise_tv`) this given
        the standard deviation equivalent to the probability volume required for the noise estimate
        e.g. a value of 1.0 means the estimate is formed from the central 68% of the data's
        cumulative probability distribution. This value must be greater than 0.
    flareparams : dict, default: {'taugauss': (0, 1.5*60*60, 10), 'tauexp': (0.5*60*60, 3.*60*60, 10)}
        A dictionary containing the flare parameters 'tauexp' and 'taugauss' giving tuples of each of
        their lower and upper values (in seconds) along with the number of grid points spanning that
        range (if the tuple contains only a single value then this will be the fixed value of that
        parameter). These will be numerically marginalised over to produce the log odds ratio.
    noisepoly : bool, default: True
        If True then the noise model will include a polynomial background variation (with the same
        length and order as used in the signal model and set by `bglen` and `bgorder`.
    noiseimpulse : bool, default: True
        If True then the noise model will include an impulse model (:class:`.Impulse`) on top of a
        polynomial background variation.
    noiseimpulseparams : dict, default: {'t0', (0.,)}
        A dictionary containing the impulse parameters 't0' giving a tuple of its lower, and upper
        values (in seconds) and the number of grid points spanning that range (if a single value is
        given in the tuple then the parameter will be fixed at that value). This range will be
        numerically marginalised over. For the default values `0.` corresponds to the impulse being
        at the centre of the analysis window.
    noiseexpdecay : bool, default: True
        If True then the noise model will include a purely exponential decay model
        (:class:`Expdecay`) on top of a polynomial background variation.
    noiseexpdecayparams : dict, default: {'tauexp': (0.0, 0.25*60*60, 3)}
        A dictionary containing the exponential decay parameter 'tauexp' giving a tuples of its lower
        and upper values (in seconds) and the number of grid points spanning that range (if the
        tuple contains only a single value then this will be the fixed value of that parameter).
        This will be numerically marginalised over to produce the log odds ratio.
    noiseexpdecaywithreverse : bool, default: True
        If True then the noise model will include an exponential rise model (just the reverse of
        the exponential decay) on top of a polynomial background variation. This will have the same
        parameters as defined in `noiseexpdecayparams`.
    noisestep : bool, default: False
        If True then the noise model will include a step function model (:class:`.Step`) on top of a
        polynomial background variation.
    noisestepparams : dict, default: {'t0', (0.,)}
        A dictionary containing the step function parameters 't0' giving a tuple of its lower, and upper
        values (in seconds) and the number of grid points spanning that range (if a single value is
        given in the tuple then the parameter will be fixed at that value). This range will be
        numerically marginalised over. For the default values `0.` corresponds to the step being
        at the centre of the analysis window.
    ignoreedges : bool, default: True
        If this is true then any output log odds ratio will have the initial and final `bglen` /2
        values removed. This removes values for which the odds ratio has been calculated using
        fewer data points.

    Notes
    -----
    In the future this could be made more generic to allow any model as the signal model,
    rather than specifically being the flare model. Further noise models could also be added.
    """

    def __init__(self,
                 lightcurve,
                 bglen= 151, #721,
                 bgorder=0,
                 nsinusoids=0,
                 noiseestmethod='powerspectrum',
                 psestfrac=0.5,
                 tvsigma=1.0,
                 flareparams={'taugauss': (0, 1.5*60*60, 10), 'tauexp': (0.5*60*60, 3.*60*60, 10)},
                 noisepoly=False,
                 ignoreedges=False):

        self.lightcurve = deepcopy(lightcurve)
        self.bglen = bglen
        self.bgorder = bgorder
        self.nsinusoids = nsinusoids

        # set flare ranges
        self.set_flare_params(flareparams=flareparams)

        # set noise estimation method
        self.set_noise_est_method(noiseestmethod=noiseestmethod, psestfrac=psestfrac, tvsigma=tvsigma)

        # set noise models
        self.set_noise_poly(noisepoly=noisepoly) # polynomial background
        self.set_ignore_edges(ignoreedges=ignoreedges)

    def set_ignore_edges(self, ignoreedges=True):
        """
        Set whether to ignore the edges of the odds ratio i.e. points within half the
        background window of the start and end of the light curve.

        Parameters
        ----------
        ignoreedges : bool, default: True
            If True then the ends of the log odds ratio will be ignored.
        """
        self.ignoreedges = ignoreedges

    def set_flare_params(self, flareparams={'taugauss': (0, 1.5*60*60, 10), 'tauexp': (0.5*60*60, 3.*60*60, 10)}):
        """
        Set the Gaussian rise ('taugauss') and exponential decay ('tauexp') timescale parameters for the
        flare parameter grid. This can also contain parameter ranges for 't0' if required,
        but otherwise this will default to inf (which gives the centre of the time series).

        Parameters
        ----------
        flareparams : dict, default: {'taugauss': (0, 1.5*60*60, 10), 'tauexp': (0.5*60*60, 3.*60*60, 10)}
            A dictionary of tuples for the parameters 'taugauss' and 'tauexp'. Each must either be a
            single value of three values for the low end, high end (both in seconds) and number of
            parameter points.
        """

        if not 'taugauss' in flareparams:
            raise ValueError("Error... dictionary has no parameter 'taugauss'")
        if not 'tauexp' in flareparams:
            raise ValueError("Error... dictionary has no parameter 'tauexp'")

        if not 't0' in flareparams:
            flareparams['t0'] = (np.inf,)

        flareparams['amp'] = (1.,)

        self.flareparams = flareparams

    def set_noise_est_method(self, noiseestmethod='powerspectrum', psestfrac=0.5, tvsigma=1.0):
        """
        Set the noise estimation method and its parameters.

        Parameters
        ----------
        noiseestmethod : string, default: 'powerspectrum'
            The noise estimation method. Either 'powerspectrum' to use :func:`.estimate_noise_ps`, or
            'tailveto' to use :func:`.estimate_noise_tv`.
        psestfrac : float, default: 0.5
            The fraction of the upper end of the power spectrum to use for the 'powerspectrum'
            method (must be between 0 and 1).
        tvsigma : float, default: 1.0
            The number of 'standard deviations' corresponding to the central probability volume
            used in the 'tailveto' method.
        """
        self.psestfrac = None
        self.tvsigma = None
        self.noiseestmethod = noiseestmethod
        if noiseestmethod == 'powerspectrum':
            self.psestfrac = psestfrac
        elif noiseestmethod == 'tailveto':
            self.tvsigma = tvsigma
        else:
            print("Noise estimation method %s not recognised" % noiseestmethod)

    def set_noise_poly(self, noisepoly=False):
        """
        Set the noise model to include a polynomial background model.

        Parameters
        ----------
        noisepoly : bool, default: True
            Set to True if this model is to be used.
        """
        self.noisepoly = noisepoly


    def oddsratio(self):
        """
        Get a time series of log odds ratio for data containing a flare *and* polynomial background
        versus a selection of noise models. For the flare and noise models all parameter values
        (expect the central time of the model) are analytically, or numerically (using the
        trapezium rule) marginalised over.

        Each of the noise models, :math:`\\mathcal{O}^{\\textrm noise}_i`, in the denominator of
        the odds ratio are combined independently, such that

        .. math::

            \\mathcal{O} = \\frac{\\mathcal{O}^{\\textrm signal}}{\\sum_i \\mathcal{O}^{\\textrm noise}_i}

        where :math:`\\mathcal{O}^{\\textrm signal}` is the signal model.

        If no noise models are specified then the returned log odds ratio will be for the signal
        model versus Gaussian noise.
        """

        # get flare odds ratio
        Mf = Flare(self.lightcurve.time, amp=1, paramranges=self.flareparams)
        Bf = Bayes(self.lightcurve, Mf)
        Bf.bayes_factors()
        Of = Bf.marginalise_full()

        noiseodds = []
        # get noise odds ratios
        if self.noisepoly:
            Bg = Bf.bayes_factors_marg_poly_bgd_only(bglen=self.bglen,
                                                     bgorder=self.bgorder,
                                                     nsinusoids=self.nsinusoids,
                                                     noiseestmethod=self.noiseestmethod,
                                                     psestfrac=self.psestfrac, tvsigma=self.tvsigma)

            noiseodds.append(Bg)

        del Mf

        # get the total odds ratio
        if self.ignoreedges and self.bglen != None:
            valrange = np.arange(int(self.bglen/2), len(Of.lnBmargAmp)-int(self.bglen/2))
            ts = np.copy(self.lightcurve.time[valrange])
        else:
            valrange = range(0, len(Of.lnBmargAmp))
            ts = np.copy(self.lightcurve.time)

        lnO = []
        for i in valrange:
            denom = -np.inf
            for n in noiseodds:
                denom = logplus(denom, n[i])

            if len(noiseodds) > 0:
                lnO.append(Of.lnBmargAmp[i] - denom)
            else:
                lnO.append(Of.lnBmargAmp[i])

        return lnO, ts

    def impulse_excluder(self, lnO, ts, exclusionwidth=5):
        """
        Return a copy of the odds ratio time series with sections excluded based on containing features
        consistent with impulse artifacts. The type of feature is that which comes about due to impulses
        in the data ringing up the signal template as it moves onto and off-of the impulse. These give
        rise to a characteristic M-shaped feature with the middle dip (when the impulse model well
        matches the data) giving a string negative odds ratio.

        Parameters
        ----------
        lnO : list or :class:`numpy.array`
            A time series array of log odds ratios.
        exclusionwidth : int, default: 5
            The number of points either side of the feature to be excluded. In practice this should be
            based on the charactistic maximum flare width.
        """

        # find log odds ratios < -5 (i.e. favouring the impulse/noise model
        negidxs = np.arange(len(lnO))[np.copy(lnO) < -5.]

        idxarray = np.ones(len(lnO), dtype=np.bool) # array to say whether values should be excluded or not

        # check whether to exclude or not based on M shaped profile
        for idx in negidxs:
            if idx > 1 and idx < len(lnO)-2:
                c1 = False
                c2 = False
                # check previous value is positive and value before that is positive, but less than next one
                if lnO[idx-1] > 0 and lnO[idx-2] > 0 and lnO[idx-2] < lnO[idx-1]:
                    c1 = True
                # check next value is positive and value after that is positive, but less than previous one
                if lnO[idx+1] > 0 and lnO[idx+2] > 0 and lnO[idx+2] < lnO[idx+1]:
                    c2 = True

                # set exclusion is both these are true
                if c1 and c2:
                    stidx = idx - exclusionwidth
                    if stidx < 0:
                        stidx = 0
                    enidx = idx + exclusionwidth
                    if enidx > len(lnO)-1:
                        enidx = len(lnO)-1

                    idxarray[stidx:enidx] = False

        # return arrays with parts excluded
        return np.copy(lnO)[idxarray], np.copy(ts)[idxarray]


    def thresholder(self, lnO, thresh, expand=0, returnmax=True):
        """
        Output an list of array start and end indices for regions where the log odds ratio is
        greater than a given threshold `thresh`. Regions can be expanded by a given amount to allow
        close-by regions to be merged.

        This is used for flare detection.

        Parameters
        ----------
        lnO : list or :class:`numpy.array`
            A time series array of log odds ratios.
        thresh : float
            The log odds ratio threshold for "detections".
        expand : int, default:0
            Expand each contiguous above-threshold region by this number of indices at either side.
            After expansion any overlapping or adjacent regions will be merged into one region.
        returnmax : bool, default: True
            If True then return a list of tuples containing the maximum log odds ratio value in each
            of the "detection" segments and the index of that value.

        Returns
        -------
        flarelist : list of tuples
            A list of tuples of start and end indices of contiguous regions for the "detections".
        numflares : int
            The number of contiguous regions i.e. the number of detected flares.
        maxlist : list of tuples
            If `returnmax` is true then this contains a list of tuples with the maximum log
            odds ratio value in each of the "detection" segments and the index of that value.
        """

        # find contiguous regions
        flarelist = []
        for start, stop in contiguous_regions(np.copy(lnO) > thresh): #  make sure lnO is a numpy array by copying
            flarelist.append((start, stop))

        # expand segments if required, and then merge any adjacent or overlapping segments
        if expand > 0:
            if len(flarelist) == 1: # if only one flare
                segtmp = list(flarelist[0])
                segtmp[0] = segtmp[0]-expand    # expand the segment
                segtmp[-1] = segtmp[-1]+expand  # expand the segment

                # check if segment now goes out of range and if so correct it
                if segtmp[0] < 0:
                    segtmp[0] = 0
                if segtmp[-1] >= len(lnO):
                    segtmp[-1] = len(lnO)

                flarelist = [segtmp]
            elif len(flarelist) > 1:
                flisttmp = []

                # expand each segment
                for segn in flarelist:
                    segtmp = list(segn)
                    segtmp[0] = segtmp[0]-expand
                    segtmp[-1] = segtmp[-1]+expand

                    # check if segment now goes out of range and if so correct it
                    if segtmp[0] < 0:
                        segtmp[0] = 0
                    if segtmp[-1] >= len(lnO):
                        segtmp[-1] = len(lnO)

                    flisttmp.append(tuple(segtmp))

                flarelist = flisttmp

                # with expanded segments now check for overlapping or adjacent segments and merge
                j = 0
                newsegs = []
                while True:
                    thisseg = flarelist[j]
                    j = j+1
                    for k in range(j, len(flarelist)):
                        nextseg = flarelist[k]
                        if thisseg[-1] >= nextseg[0]: # overlapping or adjacent segment
                            thisseg = (thisseg[0], nextseg[-1])
                            j = j+1
                        else:
                            break

                    newsegs.append(thisseg)

                    # break from loop
                    if j >= len(flarelist):
                        break

                flarelist = list(newsegs)

        lnOc = np.copy(lnO)

        # return the list of maximum values and indices for the detections
        if returnmax:
            maxlist = []
            for segn in flarelist:
                v = np.arange(segn[0], segn[-1])
                i = np.argmax(lnOc[v])
                maxlist.append((lnOc[v[i]], v[i]))

            return flarelist, len(flarelist), maxlist
        else:
            return flarelist, len(flarelist) # return list of indices and number of flares

class Model():
    """
    A class with methods for a generic model.

    Parameters
    ----------
    mtype : string
       The model type, currently this can be 'flare', 'expdecay', or 'gaussian'
    ts : :class:`numpy.ndarray`
       A vector containing time stamps.
    amp : float, optional, default: 1
       The amplitude of the model.
    t0 : float, optional
       The central time of the model. Defaults to the centre of ``ts``.
    reverse : bool, optional, default: False
       A boolean flag. Set this to reverse the model shape.
    paramnames : list of strings
       A list with the names of each model parameter.
    paramranges : dict of tuples
       A dictionary of tuples defining the model parameter ranges.

    """

    amp = 0
    ts  = []
    t0  = None
    f   = []

    parameters = []
    paramnames = [] # names of valid parameters
    ranges = {}

    shape = []

    timeseries = []
    reverse=False

    modelname=None

    def __init__(self, ts, mtype, amp=1, t0=None, reverse=False, paramnames=None, paramranges=None):

        if t0 == None:
            t0 = ts[floor(len(ts)/2)]

        self.mtype = mtype.lower()
        self.paramnames = paramnames
        self.t0 = t0
        self.ts  = ts
        self.reverse = reverse
        self.shape = []
        self.ranges = {}

        # set default ranges
        if paramranges != None:
            self.set_params(paramranges)

    def __str__(self):
        return "<BayesFlare "+self.mtype+" model>"

    def __repr__(self):
        return self.__str__()

    def set_params(self, paramrangedict):
        """
        Set a grid of parameter ranges for the model.

        Parameters
        ----------
        paramrangedict : dict
           A dictionary of containing tuples for ranges of each of the parameters
           for the given model.

        """
        for p in self.paramnames:
            rangetuple = paramrangedict[p]

            if len(rangetuple) == 1:
                self.ranges[p] = np.array([rangetuple[0]])
            elif len(rangetuple) == 3:
                self.ranges[p] = np.linspace(rangetuple[0], rangetuple[1], rangetuple[2])
            else:
                raise ValueError("Error... range must either contain 1 or 3 values")
                return

            self.shape.append(len(self.ranges[p]))

    def filter_model(self, m, filtermethod='savitzkygolay', nbins=101, order=3, filterknee=(1./(0.3*86400.))):
        """
        Use the Savitzky-Golay smoothing (:func:`.savitzky_golay`) to high-pass filter the model m.
        Parameters
        ----------
        m : :class:`numpy.ndarray`
           An array containing the model.
        filtermethod : string, default: 'savitzkygolay'
           The method for filtering/detrending the model function. The default is
           the Savitzky-Golay method, but this can also be 'runningmedian' to use
           a running median detrending, or 'highpass' for a high-pass 3rd order Butterworth
           filter.
        nbins : int, optional, default: 101
           An odd integer width (in bins) for the Savitzky-Golay, or running median, filtering.
        order : int, optional, default: 3
           The polynomial order for the Savitzky-Golay filtering.
        filterknee : float, default: 1/(0.3*86400) Hz
           The filter knee frequency (in Hz) for the high-pass filter method.
        Returns
        -------
        The filtered model time series
        """
        if filtermethod == 'savitzkygolay':
            return (m - bf.savitzky_golay(m, nbins, order))
        elif filtermethod == 'runningmedian':
            return (m - bf.running_median(m, nbins))
        elif filtermethod == 'highpass':
            ml = bf.Lightcurve()
            ml.clc = np.copy(m)
            ml.cts = np.copy(self.ts)
            filtm = bf.highpass_filter_lightcurve(ml, knee=filterknee)
            del ml
            return filtm
        else:
            raise ValueError('Unrecognised filter method (%s) given' % filtermethod)

    def __call__(self, q, ts=None, filt=False, filtermethod='savitzkygolay', nbins=101, order=3, filterknee=(1./(0.3*86400.))):

        ts = self.ts
        idxtuple = np.unravel_index(q, self.shape)

        pdict = {}
        for i, p in enumerate(self.paramnames):
            pdict[p] = self.ranges[p][idxtuple[i]]

        f = self.model(pdict, ts=ts)

        if filt:
            f = self.filter_model(f, filtermethod=filtermethod, nbins=nbins, order=order, filterknee=filterknee)

        m = ModelCurve(ts, f)

        return m


class Flare(Model):
    """
    Creates an exponentially decaying flare model with a Gaussian rise.

    Parameters
    ----------
    ts : :class:`numpy.ndarray`, required
       A 1D array containing the times
    paramranges : dict, default: None
       A dictionary of the flare model parameter ranges. The default has no
       parameter grid set up.
    amp : float, default: 1
       The amplitude of the model
    t0 : float
       The central time of the model
    reverse : bool, default=False
       Reverse the model shape
    paramnames : list, default: ['t0' 'tauexp', 'taugauss', 'amp']
       The names of the flare model parameters

    Examples
    --------
    The flare model could be set up with the following parameter ranges (note
    that for the 't0' parameter a value of inf will just default to the centre of the
    time series):

       >>> ts = np.linspace(0., 30.*86400., 1500) # a time series (in seconds)
       >>> paramranges = { 't0': (np.inf,), ...
       >>>   'tauexp': (0., 10.*3600., 10), ...
       >>>   'taugauss': (0., 10.*3600., 10), ...
       >>>   'amp': (1.,)}
       >>> flare = Flare(ts, paramranges)
    """

    def __init__(self, ts, paramranges=None, amp=1, t0=None, reverse=False):
        Model.__init__(self, ts, mtype='flare', amp=amp, t0=t0, reverse=reverse,
                       paramnames=['t0', 'tauexp', 'taugauss', 'amp'],
                       paramranges=paramranges)

        self.modelname = 'flare'


    def model(self, pdict, ts=None):
        """
        The flare model.

        Parameters
        ----------
        pdict : dict,
           A dictionary of the flare model parameters ('t0', 'amp', 'taugauss', 'tauexp').
        ts : :class:`numpy.ndarray`, default: None
           A 1D set of time stamps (if 'None' the value of ts defined in the model is used).

        Returns
        -------
        f : :class:`numpy.ndarray`
           A 1D time series of values of the model evaluated at the set of parameters.
        """

        # check input values
        if not 't0' in pdict:
            raise ValueError("Error... no 't0' value in dictionary!")
        if not 'amp' in pdict:
            raise ValueError("Error... no 'amp' value in dictionary!")
        if not 'taugauss' in pdict:
            raise ValueError("Error... no 'taugauss' value in dictionary!")
        if not 'tauexp' in pdict:
            raise ValueError("Error... no 'tauexp' value in dictionary!")

        ts = self.ts

        t0 = pdict['t0']
        amp = pdict['amp']
        tauGauss = pdict['taugauss']
        tauExp = pdict['tauexp']

        # if t0 is inf then set it to the center of the time series
        if t0 == np.inf:
            t0 = ts[int(len(ts)/2.)]

        f = np.zeros(len(ts))
        f[ts == t0] = amp

        # avoid division by zero errors
        if tauGauss > 0:
            if self.reverse:
                f[ts > t0] = amp*np.exp(-(ts[ts > t0] - t0)**2 / (2*float(tauGauss)**2))
            else:
                f[ts < t0] = amp*np.exp(-(ts[ts < t0] - t0)**2 / (2*float(tauGauss)**2))

        if tauExp > 0:
            if self.reverse:
                f[ts < t0] = amp*np.exp((ts[ts < t0] - t0)/float(tauExp))
            else:
                f[ts > t0] = amp*np.exp(-(ts[ts > t0] - t0)/float(tauExp))

        return f

    def prior(self, pdict):
        """
        The prior function for the flare model parameters. This is a flat prior
        over the parameter ranges, but with :math:`\\tau_e \geq \\tau_g`.

        Parameters
        ----------
        pdict : dict
           A dictionary of the flare model parameters.


        Returns
        -------
        The log of the prior function.
        """

        # check input values
        if not 't0' in pdict:
            raise ValueError("Error... no 't0' value in dictionary!")
        if not 'amp' in pdict:
            raise ValueError("Error... no 'amp' value in dictionary!")
        if not 'taugauss' in pdict:
            raise ValueError("Error... no 'taugauss' value in dictionary!")
        if not 'tauexp' in pdict:
            raise ValueError("Error... no 'tauexp' value in dictionary!")

        t0 = pdict['t0']
        amp = pdict['amp']
        tauGauss = pdict['taugauss']
        tauExp = pdict['tauexp']

        t0range = self.ranges['t0']
        amprange = self.ranges['amp']
        taugrange = self.ranges['taugauss']
        tauerange = self.ranges['tauexp']

        t0prior = 0.
        if len(t0range) > 1:
            t0prior = -np.log(t0range[-1] - t0range[0])

        ampprior = 0.
        if len(amprange) > 1:
            ampprior = -np.log(amprange[-1] - amprange[0])

        # we the parameter space for which tauExp > tauGauss
        tauprior = 0.

        if tauGauss > tauExp or tauGauss > tauerange[-1]:
            tauprior = -np.inf # set prior to 0
        else:
            # get area

            taugmin = taugrange[0]
            taugmax = taugrange[-1]
            tauemin = tauerange[0]
            tauemax = tauerange[-1]

            dtaug = taugmax-taugmin
            dtaue = tauemax-tauemin

            if taugmin <= tauemin and taugmax <= tauemax:
                # get rectangle area and subtract the lower triangle
                parea = dtaue * dtaug - 0.5*(taugmax-tauemin)**2
            elif taugmin > tauemin and taugmax > tauemax:
                # get upper triangle area
                parea = 0.5*(tauemax-taugmin)**2
            elif taugmin > tauemin and taugmax < tauemax:
                # get upper trapezium area
                parea = 0.5*dtaug*((tauemax-taugmin)+(tauemax-taugmax))
            elif taugmin < tauemin and taugmax > tauemax:
                # get lower trapezium area
                parea = 0.5*dtaue*((tauemin-taugmin)+(tauemax-taugmin))

            tauprior = -np.log(parea)

        return (ampprior + t0prior + tauprior)


class Expdecay(Model):
    """
    Creates an exponential decay model.

    Parameters
    ----------
    ts : :class:`numpy.ndarray`, required
       A 1D array containing the times
    paramranges : dict, default: None
       A dictionary of the exponential decay model parameter ranges. The default has no
       parameter grid set up.
    amp : float, default: 1
       The amplitude of the model
    t0 : float
       The central time of the model
    reverse : bool, default=False
       Reverse the model shape
    paramnames : list, default: ['t0', 'amp', 'tauexp']
       The names of the exponential decay model parameters

    Examples
    --------
    The exponential decay model could be set up with the following parameter ranges (note
    that for the 't0' parameter a value of inf represents the centre of the
    time series):

       >>> ts = np.linspace(0., 30.*86400., 1500) # a time series (in seconds)
       >>> paramranges = { 't0': (np.inf,), ...
       >>>   'tauexp': (0., 2.*3600., 10), ...
       >>>   'amp': (1.,)}
       >>> expdecay = Expdecay(ts, paramranges)
    """

    def __init__(self, ts, amp=1, t0=None, reverse=False, paramranges=None):
        Model.__init__(self, ts, mtype='expdecay', amp=amp, t0=t0, reverse=reverse,
                       paramnames=['t0', 'amp', 'tauexp'],
                       paramranges=paramranges)

        self.modelname = 'expdecay'

    def model(self, pdict, ts=None):
        """
        The exponential decay model.

        Parameters
        ----------
        pdict : dict,
           A dictionary of the exponential decay model parameters ('t0', 'amp', 'tauexp').
        ts : :class:`numpy.ndarray`, default: None
           A 1D set of time stamps (if 'None' the value of ts defined in the model is used).

        Returns
        -------
        f : :class:`numpy.ndarray`
           A 1D time series of values of the model evaluated at the set of parameters.
        """
        # check input values
        if not 't0' in pdict:
            raise ValueError("Error... no 't0' value in dictionary!")
        if not 'amp' in pdict:
            raise ValueError("Error... no 'amp' value in dictionary!")
        if not 'tauexp' in pdict:
            raise ValueError("Error... no 'tauexp' value in dictionary!")

        if ts == None:
            ts = self.ts

        t0 = pdict['t0']
        amp = pdict['amp']
        tauExp = pdict['tauexp']

        # if t0 is inf then set it to the centre of the time series
        if t0 == np.inf:
            t0 = ts[int(len(ts)/2.)]

        f = np.zeros(len(ts))
        f[ts == t0] = amp

        reverse = self.reverse # get reverse (default to False)

        if tauExp > 0:
            if reverse:
                f[ts < t0] = amp*np.exp((ts[ts < t0] - t0)/float(tauExp))
            else:
                f[ts > t0] = amp*np.exp(-(ts[ts > t0] - t0)/float(tauExp))

        return f

    def prior(self, pdict):
        """
        The prior function for the exponential decay model parameters. This is a flat prior
        over the parameter ranges.

        Parameters
        ----------
        pdict : dict
           A dictionary of the transit model parameters.

        Returns
        -------
        The log of the prior function.
        """

        # check input values
        if not 't0' in pdict:
            raise ValueError("Error... no 't0' value in dictionary!")
        if not 'amp' in pdict:
            raise ValueError("Error... no 'amp' value in dictionary!")
        if not 'tauexp' in pdict:
            raise ValueError("Error... no 'tauexp' value in dictionary!")

        t0 = pdict['t0']
        amp = pdict['amp']
        tauExp = pdict['tauexp']

        t0range = self.ranges['t0']
        amprange = self.ranges['amp']
        tauexprange = self.ranges['tauexp']

        t0prior = 0.
        if len(t0range) > 1:
            t0prior = -np.log(t0range[-1] - t0range[0])

        ampprior = 0.
        if len(amprange) > 1:
            ampprior = -np.log(amprange[-1] - amprange[0])

        tauexpprior = 0.
        if len(tauexprange) > 1:
            tauexpprior = -np.log(tauexprange[-1] - tauexprange[0])

        return (t0prior + ampprior + tauexpprior)


class Gaussian(Model):
    """
    Creates a Gaussian profile model.

    Parameters
    ----------
    ts : :class:`numpy.ndarray`, required
       A 1D array containing the times
    paramranges : dict, default: None
       A dictionary of the delta-function model parameter ranges. The default has no
       parameter grid set up.
    amp : float, default: 1
       The amplitude of the model
    t0 : float
       The central time of the model
    paramnames : list, default: ['t0', 'amp', 'sigma']
       The names of the Gaussian model parameters

    Examples
    --------
    The Gaussian profile model could be set up with the following parameter ranges (note
    that for the 't0' parameter a value of inf represents the centre of the
    time series):

       >>> ts = np.linspace(0., 30.*86400., 1500) # a time series (in seconds)
       >>> paramranges = { 't0': (np.inf,), ...
       >>>   'sigma': (0., 3.*3600., 10), ...
       >>>   'amp': (1.,)}
       >>> gaussian = Gaussian(ts, paramranges)
    """

    def __init__(self, ts, amp=1, t0=None, paramranges=None):
        Model.__init__(self, ts, mtype='gaussian', amp=amp, t0=t0,
                       paramnames=['t0', 'sigma', 'amp'],
                       paramranges=paramranges)

        self.modelname = 'gaussian'

    def model(self, pdict, ts=None):
        """
        The Gaussian model.

        Parameters
        ----------
        pdict : dict,
           A dictionary of the Gaussian model parameters ('t0', 'amp', 'sigma').
        ts : :class:`numpy.ndarray`, default: None
           A 1D set of time stamps (if 'None' the value of ts defined in the model is used).

        Returns
        -------
        f : :class:`numpy.ndarray`
           A 1D time series of values of the model evaluated at the set of parameters.
        """
        # check input values
        if not 't0' in pdict:
            raise ValueError("Error... no 't0' value in dictionary!")
        if not 'amp' in pdict:
            raise ValueError("Error... no 'amp' value in dictionary!")
        if not 'sigma' in pdict:
            raise ValueError("Error... no 'sigma' value in dictionary!")

        if ts == None:
            ts = self.ts

        t0 = pdict['t0']
        amp = pdict['amp']
        sigma = pdict['sigma']

        # if t0 is inf then set it to the centre of the time series
        if t0 == np.inf:
            t0 = ts[int(len(ts)/2.)]

        # the Gaussian model for given parameters
        if sigma == 0: # if sigma is 0 then have delta function at point closest to t0
            f = np.zeros(len(ts))
            tm0 = ts-t0
            f[np.amin(tm0) == tm0] = amp
        else:
            f = amp*np.exp(-(ts - t0)**2/(2*float(sigma)**2))

        return f

    def prior(self, pdict):
        """
        The prior function for the Gaussian function model parameters. This is a flat prior
        over the parameter ranges.

        Parameters
        ----------
        pdict : dict
           A dictionary of the impulse model parameters.

        Returns
        -------
        The log of the prior function.
        """

        # check input values
        if not 't0' in pdict:
            raise ValueError("Error... no 't0' value in dictionary!")
        if not 'amp' in pdict:
            raise ValueError("Error... no 'amp' value in dictionary!")
        if not 'sigma' in pdict:
            raise ValueError("Error... no 'sigma' value in dictionary!")

        t0 = pdict['t0']
        amp = pdict['amp']
        sigma = pdict['sigma']

        t0range = self.ranges['t0']
        amprange = self.ranges['amp']
        sigmarange = self.ranges['sigma']

        t0prior = 0.
        if len(t0range) > 1:
            t0prior = -np.log(t0range[-1] - t0range[0])

        ampprior = 0.
        if len(amprange) > 1:
            ampprior = -np.log(amprange[-1] - amprange[0])

        sigmaprior = 0.
        if len(sigmaprior) > 1:
            sigmaprior = -np.log(sigmaprior[-1] - sigmaprior[0])

        return (t0prior + ampprior + sigmaprior)

class ModelCurve():

    def __init__(self, time, flux):
        self.flux = flux
        self.time = time

    def dt(self):
        """
        Calculates the time interval of the time series.

        Returns
        -------
        float
           The time interval.
        """
        return self.time[1] - self.time[0]

    def fs(self):
        """
        Calculates the sample frequency of the time series.

        Returns
        -------
        float
           The sample frequency.
        """
        return 1.0/self.dt()

def log_marg_amp(d, m, ss):
    """
    Calculate the logarithm of the likelihood ratio for the signal model
compared to a pure
    Gaussian noise model, but analytically marginalised over the unknown
model amplitude. This is
    calculated for each timestep in the data, and assumes a constant
noise level over the data.
    As this function uses :func:`numpy.correlate`, which works via FFTs,
the data should be
    contiguous and evenly spaced.

    Parameters
    ----------
    d : :class:`numpy.array`
        A 1D array containing the light curve time series data.
    m : :class:`numpy.array`
        A 1D array (of the same length as `d`) containing the model
function.
    ss : float or double
        The noise variance of the data (assumed constant)
    Returns
    -------
    B : double
        The logarithm of the likelihood ratio.
    """

    #get the data/model cross term
    dm = np.correlate(d, m, mode='same')

    # get the model autocorrelation term
    m2 = np.sum(m*m)

    # get the likelihood marginalised over the signal amplitude
    inside_erf = np.sqrt(0.5/(ss*m2))

    # use log of complementary error function from GSL
    logpart = 0.5*np.log(np.pi/(2.*m2))

    B = np.zeros(d.shape[0], dtype=np.float)
    logerf = 0.
    logerfc = 0.
    i = 0
    loopmax = len(B)
    k = 0
    for i in range(loopmax):
        k = dm[i] * inside_erf
        log_erf = np.log1p(erf( k ))
        B[i] = ((dm[i]*dm[i])/(2.*m2*ss)) + logpart + log_erf

    return B
