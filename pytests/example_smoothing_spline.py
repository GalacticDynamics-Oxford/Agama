#!/usr/bin/python
"""
This example demonstrates the use of splines for two separate tasks:
(1) constructing a non-parametric estimate for the probability density from discrete samples;
(2) constructing a non-parametric smooth approximation y=f(x) for a set of points (x,y)
In the first case, we generate sample points from a mixture of two normal distributions:
80% come from N(0,1) and 20% - from N(2,0.3).
We compare the recovered probability distribution function using the kernel density estimate
from scipy, and the penalized spline log-density estimate from Agama.
In the second case, we generate noisy "measurements": y[i] = f(x[i]) + g[i],
where f is a known function (sine in our example) and g is Gaussian noise with spatially
varying dispersion. We then construct a smooth approximation curve for these data points
with several choices of smoothing parameter.
"""
import agama,numpy
from matplotlib.pyplot import *
from scipy.stats import gaussian_kde

### example 1: density estimate
def gaussian(x, x0, sigma):
    return (2*numpy.pi)**-0.5 / sigma * numpy.exp( -0.5 * ( (x-x0) / sigma)**2 )
# original probability distribution is a mixture of two gaussians
def myfnc(x):
    return 0.8 * gaussian(x, 0, 1) + 0.2 * gaussian(x, 2, 0.3)
# original data points drawn from a mixture of two gaussians - traditional way
data = numpy.hstack(( numpy.random.normal (0, 1, 800), numpy.random.normal(2., 0.3, 200)))
# same effect could be achieved by Agama sampling routine (draw discrete samples from a N-dim distr.fnc.):
# data = agama.sampleNdim(myfnc, 1000, [-4.], [4.])[0].reshape(-1)
# simple-minded histogram
hist,bins = numpy.histogram(data, bins=30, range=(-3,3), density=True)
# kernel density estimate with an automatic choice of bandwidth (results in oversmoothing)
kde0 = gaussian_kde(data)
# same with a manually provided bandwidth (decreases bias but not completely, and also increases noise)
kde1 = gaussian_kde(data, 0.1)
# estimate of log(f(x)) using Agama routines
grid = numpy.linspace(-3, 3, 20)  # x-grid nodes for the spline
spl  = agama.splineLogDensity(grid, data, infLeft=True, infRight=True)
# plot the results
x = numpy.linspace(-4., 4., 201)  # interval for plotting
plot(x, myfnc(x), label='Original f(x)')
step(bins[:-1], hist, where='post', color='lightgrey', label='Simple histogram')
plot(x, kde0(x), color='g', label='Kernel density, auto bandwidth')[0].set_dashes([6,2])
plot(x, kde1(x), color='c', label='Kernel density, manual bandwidth')[0].set_dashes([4,1,1,1])
plot(x, numpy.exp(spl(x)), color='r', label='Spline density estimate',lw=1.5)[0].set_dashes([1,1])
legend(loc='upper left', frameon=False)
ylim(0, 0.5)
show()

### example 2: curve fitting with penalized splines
xp   = numpy.random.uniform(0, 5, 1000)**1.5  # x-values for points (more dense around origin)
sig  = 0.3 / (1 + 0.2*xp)  # spatially-variable amplitude of added noise
yp   = numpy.sin(xp) + numpy.random.normal(0, sig)
grid = numpy.linspace(0., 11., 100)  # x-grid nodes for the spline (deliberately more than needed)
# spline fit without any smoothing (results in wiggly overfitted curve)
spl0 = agama.splineApprox(grid, xp, yp, w=sig**-2, smooth=None)
# spline fit with optimal smoothing (default, recommended - result is insensitive to grid size)
spl1 = agama.splineApprox(grid, xp, yp, w=sig**-2)
# spline fit with more-than-optimal smoothing
spl2 = agama.splineApprox(grid, xp, yp, w=sig**-2, smooth=1)
# plot the results
x    = numpy.linspace(0., 11., 200)  # interval for plotting
plot(xp, yp,'.', label='original points', ms=3, c='c')
plot(x, spl0(x), label='spline fit, no smoothing', lw=1.5, c='b')[0].set_dashes([4,2])
plot(x, spl1(x), label='spline fit, optimal smoothing', c='r')
plot(x, spl2(x), label='spline fit, extra smoothing', lw=2, c='g')[0].set_dashes([1,1])
legend(loc='upper right', frameon=False, numpoints=1)
ylim(-1.5, 2.)
show()
