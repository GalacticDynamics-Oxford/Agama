#!/usr/bin/python
'''
This file is part of the Gaia Challenge, and contains the deprojection and resampling routine.
See gc_runfit.py for the overall description.

######################## SPHERICAL MODEL DEPROJECTION #########################
This module fills missing values for coordinate and velocity for the case of incomplete data.
On input, we have an array of particles with 9 parameters:
{x, y, [z?], [vx?], [vy?], [vz?], vx_err, vy_err, vz_err},
where [..?] denotes possibly missing data (denoted as NAN),
and velocities carry a Gaussian error with known dispersion (may be zero).
On output, an array of sub-samples and their weights is returned:
each particle is split into Nsamp sub-samples, where missing or imprecisely known data
are filled with randomly sampled values, so that in calculating the likelihood of a given
particle given a model, we sum up likelihoods of each of its subsamples with the provided
weights (this is essentially a Monte Carlo integration over the missing axes,
but always carried out using the same set of points for any model parameters, which
is necessary to remove sampling noise).
This random sampling may be uniform within a reasonable range (determined from the existing
input data), or according to some prior (in which case sub-sample weights will not be equal).
In case of missing z-coordinate, we construct a spherical deprojection of the density profile,
using the existing x- and y-coordinates, and sample the value of z-coordinate at each R
from this deprojected profile (thus having a non-uniform prior and placing sub-samples where
they are more likely to lie, that is, using a sort of importance sampling).
For missing velocity components, we sample from a heavy-tailed DF with a width estimated
from the known components of velocity of input points.

The spherical model deprojection and sampling the missing z coordinate is implemented in
the class ``SphericalModel'', and the resampling procedure for missing velocity components --
in the routine ``sampleMissingData''.
'''

import agama, numpy, scipy.optimize, scipy.special

def qrngHalton(size, base, scramble=True):
    """
    Generate an array of quasi-random numbers from the Halton sequence (from 0 to size-1).
    'base' should be a prime number, and sequences for different bases can be used for
    separate columns of a 2d array of quasi-random numbers.
    'scramble' turns on Owen scrambling (better statistical properties).
    """
    numiter = int(numpy.ceil(54 / numpy.log2(base)) - 1)
    vals = numpy.zeros(size)
    facs = numpy.ones(size)
    inds = numpy.arange(size)
    for j in range(numiter):
        perm = numpy.arange(base)
        if scramble:
            numpy.random.shuffle(perm)
        facs /= base
        vals += facs * perm[inds % base]
        inds //= base
    return vals

if True:  # use quasi-random numbers (recommended for a much lower variance in the Monte Carlo integral estimate)
    rng = qrngHalton
else:
    rng = lambda size,base: numpy.random.random(size=size)

def getstdnormal(x):
    """ Convert a uniformly distributed sample in [0:1) into a normally distributed one """
    return 2**0.5 * scipy.special.erfinv(x*2-1)


class SphericalModel:
    """
    Construct a spherically symmetric model using the information about projected particle positions;
    Arguments:
      Radii -- array of projected radii;
      show (optional, default False) -- flag requesting a diagnostic plot.
    """
    def __init__(self, Radii, show=False):
        # step 1: construct a smoothing spline S(log(R)) such that Sigma(R) = exp(S(log(R)) / (2*pi*R^2)
        RminSpl, RmaxSpl = numpy.percentile(Radii, [5, 95])
        gridSizeSpl  = max(int(round(numpy.log(len(Radii)))), 4)
        gridLogRSpl = numpy.linspace(numpy.log(RminSpl), numpy.log(RmaxSpl), gridSizeSpl)
        spl = agama.splineLogDensity(gridLogRSpl, numpy.log(Radii), infLeft=True, infRight=True)
        # step 2: approximate the surface density Sigma by a sum of K Gaussian components,
        # fitting it over the range of log-spaced grid points in radius from Rmin.
        # The reason for doing it this way is the following:
        # - it is trivial to deproject a Gaussian and sample missing z from it;
        # - one could fit the surface density profile by a multi-gaussian directly from the input points,
        # however, the fit turns out to be noisy, whereas fitting a smoothing spline first makes it, well, smoother.
        RminFit, RmaxFit = numpy.percentile(Radii, [0,100]) #[0.5, 99.5])
        K = max(10, int(numpy.log(RmaxFit/RminFit) * 2))
        gridSizeFit = 5*K-4
        gridLogRFit = numpy.linspace(numpy.log(RminFit)-0.5, numpy.log(RmaxFit)+0.5, gridSizeFit)
        gridRFit = numpy.exp(gridLogRFit)
        self.sigma = numpy.exp(numpy.linspace(numpy.log(RminFit)-0.5, numpy.log(RmaxFit)+0.5, K))
        basis = numpy.exp(-0.5 * gridRFit[:,None]**2 / self.sigma**2) / (2*numpy.pi * self.sigma**2)
        rhs = spl(gridLogRFit) - numpy.log(2*numpy.pi) - 2 * gridLogRFit  # log(Sigma(RFit))
        self.ampl = numpy.exp(scipy.optimize.leastsq(
            lambda params: numpy.log(numpy.sum(numpy.exp(params) * basis, axis=1)) - rhs,
            numpy.ones(K))[0])
        if show:
            print('number of Gaussian components: %i; amplitudes and sigmas:' % K)
            print(numpy.column_stack((self.ampl, self.sigma)))
            import matplotlib.pyplot as plt
            binsLogR = numpy.linspace(numpy.log(numpy.min(Radii)), numpy.log(numpy.max(Radii)), 2*int(len(Radii)**0.5))
            hist = numpy.histogram(numpy.log(Radii), bins=binsLogR)[0] / (len(Radii)*2*numpy.pi*(binsLogR[1:]-binsLogR[:-1]))
            plt.loglog(gridRFit, numpy.exp(spl(gridLogRFit)) / (2*numpy.pi), label='spline fit')
            plt.loglog(gridRFit, gridRFit**2 * self.surfaceDensity(gridRFit), label='multi-gaussian approx')
            plt.plot(numpy.exp(binsLogR).repeat(2)[1:-1], hist.repeat(2)+1e-100, label='histogram of input points')
            plt.xlim(numpy.exp(binsLogR[0])*0.9, numpy.exp(binsLogR[-1])*1.1)
            plt.ylim(0.5*min(hist[hist>0]), 2*max(hist))
            plt.legend(loc='upper left', frameon=False)
            plt.show()

    def surfaceDensity(self, R):
        """ Return surface density Sigma(R) """
        return numpy.sum(self.ampl * numpy.exp(-0.5 * R[:,None]**2 / self.sigma**2) / (2*numpy.pi * self.sigma**2), axis=1)

    def rho(self, r):
        """ Return 3d density rho(r) """
        return numpy.sum(self.ampl * numpy.exp(-0.5 * r[:,None]**2 / self.sigma**2) / (2*numpy.pi * self.sigma**2)**1.5, axis=1)

    def sampleZ(self, R, numSamples):
        r"""
        Sample the missing z-coordinate for an array of points in R
        Arguments:
            R: projected radii of points;
            numSamples: the number of z-values sampled for each point;
        Return:
            - z_{p,s}: array of shape (len(R), numSamples) with equally likely sampled z-values for each input point.
            - w_{p,s}: array of the same shape with associated weights for each sample, defined as follows:
            w[p,s] = numSamples^-1 Sigma(R_p) / rho(R_p, z_{p,s}) .
            In other words, the integral of an arbitrary function f(R,z) over z at a fixed R=R_p
            can be approximated by
            \int_{-\infty}^{\infty} f(R_p, z) dz = \sum_{s=1}^{numSamples} f(R_p,z_{p,s}) w_{p,s}
        """
        N = len(R)
        S = numSamples
        P = self.ampl * numpy.exp(-0.5 * R[:,None]**2 / self.sigma**2) / (2*numpy.pi * self.sigma**2)  # shape: (N,K)
        cs = numpy.cumsum(P, axis=1)
        surfaceDensity = cs[:,-1].copy()
        cs /= cs[:,-1:]  # shape: (N,K); cumulative probability of each point to belong to all previous components
        randi = rng(N*S, 2).reshape(N, S)
        randn = getstdnormal(rng(N*S, 3)).reshape(N, S)
        indComp = numpy.vstack([numpy.searchsorted(cs[p], randi[p]) for p in range(N)])
        zvalues = randn * self.sigma[indComp]
        r2 = numpy.repeat(R, S).reshape(N, S)**2 + zvalues**2
        rho = numpy.sum(self.ampl * numpy.exp(-0.5 * r2[:,:,None] / self.sigma**2) / (2*numpy.pi * self.sigma**2)**1.5, axis=2)
        weights = surfaceDensity[:,None] / rho / S
        return zvalues, weights


##################### RESAMPLING OF ORIGINAL DATA TO FILL MISSING VALUES #################

def sampleMissingData(particles, numSamples):
    """
    Split each input particle into numSamples samples, perturbing its velocity or
    assigning values for missing components of position/velocity.
    Arguments:
        particles -- array of Nx9 values (3 coordinates, 3 velocities, and 3 velocity errors),
        missing data (z, v_x, v_y, v_z) are indicated by NAN (may be different for each particle,
        but should have at least some particles with measured v_z, in order to estimate the velocity
        dispersion);
        numSamples -- number of samples of missing coordinates created from each particle.
    Return: two arrays -- NNx6 positions/velocities of samples and NN weights,
    where NN=len(particles)*numSamples.
    """
    numpy.random.seed(42)  # make resampling repeatable from run to run

    # duplicate the elements of the original particle array
    # (each particle is expanded into Nsubsamples identical samples)
    Radii      = (particles[:,0]**2+particles[:,1]**2)**0.5
    samples    = numpy.repeat(particles[:, 0:6], numSamples, axis=0)
    nparticles = particles.shape[0]
    nsamples   = samples.shape[0]  # total number of points
    assert nsamples == nparticles * numSamples
    weights    = numpy.ones(nsamples, dtype=numpy.float64) / numSamples
    vel_err    = particles[:, 6:9] if particles.shape[1]==9 else numpy.zeros((nparticles, 3))

    noz   = numpy.isnan(particles[:,2])
    novz  = numpy.isnan(particles[:,5])
    novxy = numpy.isnan(particles[:,3]+particles[:,4])
    # compute the velocity dispersion and max value for points with velocity data
    vmax  = numpy.nanmax (abs(particles[:,3:6]))
    vstd  = numpy.nanmean(particles[:,3:6]**2)**0.5
    vmad  = numpy.nanmean(abs(particles[:,3:6]))
    vmed  = numpy.median (abs(particles[:,3:6]).reshape(-1)[numpy.isfinite(particles[:,3:6].reshape(-1))])
    sigma = min(vmed*1.5, vmad*1.25, vstd)
    if not numpy.isfinite(sigma):
        raise ValueError('No valid velocity data in the input sample!')
    print('Resample %d input particles into %d internal samples (vmax=%g, vstd=%g, vmad=%g, vmed=%g, adopted sigma=%g)' %
        (nparticles, nsamples, vmax, vstd, vmad, vmed, sigma))

    # when using quasi-random numbers, it is crucial that the order of the sequence is the same between
    # different dimensions, so we have to create subsamples for all points, even if not all of them will be used.
    # dimensions of qrng with bases 2 and 3 will be used for assigning missing z.
    rand1, rand2, rand3 = rng(nsamples, base=5), rng(nsamples, base=7), rng(nsamples, base=11)

    if numpy.any(noz):   # z-coordinate is missing
        sphModel = SphericalModel(Radii)
        z, w = sphModel.sampleZ(Radii, numSamples)  # importance sampling from the deprojected density profile
        samples[noz.repeat(numSamples),2] = z[noz].reshape(-1)
        weights[noz.repeat(numSamples)  ] = w[noz].reshape(-1)

    # Sample missing vx,vy or add noise to existing measurements.
    # We sample the missing velocity components (vx,vy and/or vz) from specially designed distributions
    # that have both heavy tails and peaks around zero velocity, while their characteristic width
    # matches the estimated dispersion of available input velocity measurements.
    # Again, this is a kind of importance sampling with non-uniform weights assigned to the samples;
    # we use a heavy-tailed DF in order to reduce the Poisson noise from occasional (rare) outliers,
    # which are relatively more numerous (and hence less heavily weighted) in a heavy-tailed DF
    # compared to a Gaussian DF. Same motivation explains the additional peak at small velocities,
    # which may occur more frequently in the actual DF that in a Gaussian.
    # For particles with known (measured) components of velocity, these values are perturbed by
    # a Gaussian noise with the given dispersion (measurement errors), sampled with uniform weights.
    # Note that this would not be efficient if the errors (and measured values) are much larger than
    # the velocity dispersion -- in this case one would need to use importance sampling in order to
    # place more samples at smaller velocities, where we expect them to be located more probably.
    if numpy.any(novxy):  # (some) vx,vy are missing
        sampnovxy  = numpy.repeat(novxy,  numSamples)
        size = numpy.sum(sampnovxy)
        # resample vx,vy from a probability distribution P(|v|) ~ 1/(v^2+sigma^2)^{3/2},
        # which is broader than a 2d Gaussian ( ~ v exp[-1/2(v/sigma)^2] ) in both low and high-v regions
        cumul = rand1[sampnovxy]
        vtrans_ang = 2*numpy.pi * rand2[sampnovxy]
        vtrans_mag = 2*sigma * cumul / (1-cumul**2)**0.5
        weights[sampnovxy] /= (1-cumul**2)**2 / cumul / (2*numpy.pi) / (2*sigma)**2
        samples[sampnovxy,3] = vtrans_mag * numpy.cos(vtrans_ang)
        samples[sampnovxy,4] = vtrans_mag * numpy.sin(vtrans_ang)
    if numpy.any(~novxy):  # for points with vx,vy measurements, add gaussian errors
        sampyesvxy = numpy.repeat(~novxy, numSamples)
        samples[sampyesvxy,3] += getstdnormal(rand1[sampyesvxy]) * numpy.repeat(vel_err[~novxy,0], numSamples)
        samples[sampyesvxy,4] += getstdnormal(rand2[sampyesvxy]) * numpy.repeat(vel_err[~novxy,1], numSamples)
    # same for vz
    if numpy.any(novz):
        sampnovz  = numpy.repeat(novz,  numSamples)
        size = numpy.sum(sampnovz)
        # resample vz from the following distribution with a heavy tail and an integrable singularity at small |v|:
        # P(x=|v/s|) = x^{-1/2} (1+x^2)^{-5/4}
        rand  = rand3[sampnovz]
        sign  = numpy.where(rand >= 0.5, 1, -1)
        cumul = (rand*2) % 1
        value = 2*sigma * cumul**2 / (1-cumul**4)**0.5 * sign
        weights[sampnovz] /= 0.25 * (1-cumul**4)**1.5 / cumul / (2*sigma)
        samples[sampnovz,5] = value
    if numpy.any(~novz):
        sampyesvz = numpy.repeat(~novz, numSamples)
        samples[sampyesvz,5] += getstdnormal(rand3[sampyesvz]) * numpy.repeat(vel_err[~novz,2], numSamples)

    return samples, weights
