#!/usr/bin/python
"""
This example demonstrates the use of B-splines for fitting a distribution of points,
for instance, constructing a smooth velocity distribution function (VDF) from an array
of measured values, optionally deconvolving it from individual measurement errors.
We first create a plausible VDF from a certain tracer population mimicking a dwarf galaxy,
sample it with O(10^3) points to construct a mock kinematic dataset,
then launch a MCMC fit and show the results.
The fitted VDF is represented by a B-spline of degree 2 or 3 with nonnegative amplitudes.
"""
import sys, agama, numpy, matplotlib.pyplot as plt
numpy.random.seed(2)
addErrors = len(sys.argv)>1

print('Creating mock kinematic dataset...')
# the units are needed only to create the mock kinematic dataset
agama.setUnits(length=1, velocity=1, mass=1)
pot = agama.Potential(type='nfw', mass=1e8, scaleradius=1)
df = agama.DistributionFunction(type='quasiisothermal', mass=1, rdisk=0.4, hdisk=0.2,
    sigmar0=5, rsigmar=1.0, sigmamin=1.0, jmin=0.5, qjr=0.4, potential=pot)
# take VDF from a spatial region centered on (x=0.5, z=0) with a radius 0.25
selfnc = lambda x: (x[:,0]-0.5)**2 + x[:,2]**2 < 0.25**2
gm = agama.GalaxyModel(pot, df, sf=selfnc)
# sample a large number of points to construct a smooth "true" VDF,
# using the y-component of velocity (perpendicular to the sky plane, which is x,z in our case)
vlos_all = gm.sample(1000000)[0][:,4]
vmin = -15.0
vmax = +20.0
gridv_spline = numpy.linspace(vmin, vmax, 13)
log_vdf_true = agama.splineLogDensity(gridv_spline, vlos_all)
vdf_true = lambda x: numpy.exp(log_vdf_true(x))

# use a smaller subsample to create the mock catalogue
N = 500
vlos = vlos_all[numpy.random.choice(len(vlos_all), N)]
print('Mock dataset of %i stars created; mean v=%.2f, sigma=%.2f' % (N, numpy.mean(vlos), numpy.std(vlos)))
# assign individual measurement errors to each star, uniformly distributed between 2 and 4
if addErrors:
    vlos_err = numpy.random.random(size=N) * 2.0 + 2.0
else:
    vlos_err = numpy.zeros(N)
    print('Assuming no measurement errors; run the script with a non-empty command-line argument '
        'to add noise and run the fits with deconvolution (a few times slower)')
    assert all((vlos >= vmin) & (vlos <= vmax))  # if this fails, will need to extend the range of vmin..vmax
# add Gaussian noise to simulate measurement errors
vlos += numpy.random.normal(size=N) * vlos_err

# fit a K-th degree B-spline over the grid with ngrid points (K=2 or 3):
# such a function has ngrid+K-1 free parameters (amplitudes), but we fix the leftmost two and the rightmost two
# to be zero (i.e. the spline has zero value and zero derivative at endpoints), plus one more parameter
# is redundant because it is determined from the condition that the integral of the spline is unity.
ngrid = 11
degree = 3
nparams = ngrid + degree-1 - 4 - 1  # the actual number of free parameters in the fit
gridv_fit = numpy.linspace(vmin, vmax, ngrid)
# integrals of all basis functions, used to determine the "redundant" amplitude from the remaining ones
integ = agama.bsplineIntegrals(degree, gridv_fit)

def getampl(params):
    r"""
    Determine the full array of B-spline amplitudes from the provided array of parameters,
    from the condition  \sum_{k=1}^{N_ampl} integ_k ampl_k = 1.
    """
    # the "redundant" amplutide is the one for the most central element
    amid = (1 - integ[2:2+nparams//2].dot(params[0:nparams//2]) -
        integ[3+nparams//2:3+nparams].dot(params[nparams//2:])) / integ[nparams//2+3]
    ampl = numpy.hstack([0, 0, params[0:nparams//2], amid, params[nparams//2:], 0, 0])
    return ampl

gridv_check = numpy.linspace(vmin, vmax, 200)
def loglikelihood(params):
    """
    Construct the B-spline from the provided parameters and evaluate the log-likelihood of the dataset.
    """
    ampl = getampl(params)
    spl = agama.Spline(gridv_fit, ampl=ampl)
    # one could enforce the amplitudes to be nonnegative, which guarantees that the interpolated function
    # is also nonnegative. However, only a weaker condition is actually needed, that the values of
    # the function at its extremal points are nonnegative; these are provided by the Spline class.
    if any(spl(spl.extrema()) < 0):  # or any(ampl<0):
        return -numpy.inf
    ll = numpy.sum(numpy.log(spl(vlos, conv=vlos_err)))
    if not numpy.isfinite(ll):
        ll = -numpy.inf
    return ll

import emcee
params = numpy.ones(nparams) * 0.1 / (vmax-vmin)
nwalkers = 2*nparams
nsteps = 1000
walkers  = numpy.empty((nwalkers, len(params)))
for i in range(nwalkers):
    while True:   # ensure that we initialize walkers with feasible values
        walker = numpy.array(params) + (numpy.random.normal(size=nparams)*0.001 if i>0 else 0)
        prob = loglikelihood(walker)
        if numpy.isfinite(prob):
            walkers[i] = walker
            break
print('Start emcee...')
sampler = emcee.EnsembleSampler(nwalkers, len(params), loglikelihood)
prevmaxll  = -numpy.inf
prevmeanll = -numpy.inf
# perform several passes of length "nsteps", monitoring the convergence of the MC chain
while True:
    sampler.run_mcmc(walkers, nsteps)
    chain = sampler.chain
    walkers = chain[:,-1]
    loglike = sampler.lnprobability
    currmaxll  = numpy.max (loglike[:,-nsteps:])
    currmeanll = numpy.mean(loglike[:,-nsteps:])
    params = chain[:,-nsteps:].reshape(-1, nparams)[numpy.argmax(loglike[:,-nsteps:].reshape(-1))]
    print('%i steps; log-likelihood max: %.1f, mean: %.1f' % (chain.shape[1], currmaxll, currmeanll))
    if abs(currmaxll-prevmaxll) < 0.5 and abs(currmeanll-prevmeanll) < 0.5:
        break
    prevmaxll  = currmaxll
    prevmeanll = currmeanll

# use the last section of the chain with "nsteps" to plot the results
lastchain = chain[:,-nsteps:].reshape(-1, nparams)
lastloglike = loglike[:,-nsteps:].reshape(-1)

print('Plotting MCMC chain...')
# plot the evolution of parameters along the chain
fig, axes = plt.subplots(nparams+1, 1, sharex=True, figsize=(20,15))
for i in range(nparams):
    axes[i].plot(chain[:,:,i].T, color='b', alpha=0.3)
# last panel shows the evolution of log-likelihood for the ensemble of walkers
axes[-1].plot(loglike.T, color='b', alpha=0.3)
axes[-1].set_ylabel('log(L)')
maxloglike = numpy.max(loglike)
axes[-1].set_ylim(maxloglike-max(nparams*2,20), maxloglike)   # restrict the range of log-likelihood arount its maximum
fig.tight_layout(h_pad=0.)
plt.subplots_adjust(hspace=0, wspace=0)
plt.savefig('example_vdf_fit_bspline_emcee_chain.png')
plt.close()

# plot the posterior distributions of parameters
import corner, scipy.special
corner.corner(lastchain, quantiles=(0.16, 0.5, 0.84), show_titles=True)
# overplot the distribution of values of log-likelihood and compare it with the expected chi^2 distribution
plt.axes([0.7,0.7,0.25,0.25])
gridll = numpy.linspace(-14,2,33)
histll = numpy.histogram(lastloglike - (currmeanll+nparams*0.5), bins=gridll, density=True)[0]
#     / (gridll[1]-gridll[0]) / len(lastloglike)
plt.plot(gridll.repeat(2)[1:-1], histll.repeat(2))
plt.plot(gridll, 1/scipy.special.gamma(0.5*nparams) * numpy.maximum(0,-gridll)**(0.5*nparams-1) *
    numpy.exp(gridll), 'r', lw=2)
plt.xlabel(r'$\ln L - \ln L_{\sf max}$')
plt.savefig('example_vdf_fit_bspline_emcee_corner.png')
plt.close()

# finally, plot the most interesting thing - the resulting VDF profiles with uncertainties
plt.figure(figsize=(8,6))
gridv_plot = numpy.linspace(vmin, vmax, 81)
plt.plot(gridv_plot, vdf_true(gridv_plot), label='true f(v)', c='b')
hist = numpy.histogram(vlos, bins=gridv_plot[::2], density=True)[0]
plt.plot(gridv_plot[::2].repeat(2)[1:-1], hist.repeat(2), c='r',
    label='observed%s' % (' (with errors)' if addErrors else ''))
plt.plot(gridv_fit, gridv_fit*0, 'ko')  # show the nodes of the B-spline
plt.xlim(vmin, vmax)
# draw nsamples realizations of parameters from the chain, and show the 16/84 percentile of resulting VDF profiles
nsamples = 500
profs = numpy.zeros((nsamples, len(gridv_plot)))
meanv, sigmav = numpy.zeros((2, nsamples))
# integrals of all basis functions weighted by v and v^2, used to determine the mean and dispersion of the VDFs
integ1 = agama.bsplineIntegrals(degree, gridv_fit, power=1)
integ2 = agama.bsplineIntegrals(degree, gridv_fit, power=2)
for i in range(nsamples):
    params = lastchain[numpy.random.choice(len(lastchain),1)][0]
    ampl = getampl(params)
    meanv [i] = integ1.dot(ampl) / integ.dot(ampl)
    sigmav[i] = (integ2.dot(ampl) / integ.dot(ampl) - meanv[i]**2)**0.5
    spl = agama.Spline(gridv_fit, ampl=ampl)
    profs[i] = spl(gridv_plot)
plt.fill_between(gridv_plot, numpy.percentile(profs, 16, axis=0), numpy.percentile(profs, 84, axis=0),
    alpha=0.5, color='g', lw=0)
plt.plot([numpy.nan], [numpy.nan], c='g', label='B-spline fit%s' % (' (deconvolved)' if addErrors else ''))
plt.legend(loc='upper left', frameon=False)
plt.text(0.98, 0.98, r'$\overline{v}=%.3f\pm%.3f$' % (numpy.mean(meanv), numpy.std(meanv)),
    ha='right', va='top', transform=plt.gca().transAxes)
plt.text(0.98, 0.92, r'$\sigma=%.3f\pm%.3f$' % (numpy.mean(sigmav), numpy.std(sigmav)),
    ha='right', va='top', transform=plt.gca().transAxes)

plt.savefig('example_vdf_fit_bspline_final_profiles.png')
plt.show()
