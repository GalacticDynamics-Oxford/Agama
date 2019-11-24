#!/usr/bin/python

'''
    This example demonstrates how to find best-fit parameters of an action-based
    distribution function that matches the given N-body snapshot.

    The N-body model itself corresponds to a spherically-symmetric isotropic
    Hernquist profile, and we fit it with a double-power-law distribution function
    of Posti et al.2015. We use the exact potential (i.e., do not compute it
    from the N-body model itself, nor try to vary its parameters, although
    both options are possible), and compute actions for all particles only once.
    Then we scan the parameter space of DF, finding the maximum of the likelihood
    function with a multidimensional minimization algorithm.
    This takes a few hundred iterations to converge.

    This Python script is almost equivalent to the C++ test program example_df_fit.cpp,
    up to the difference in implementation of Nelder-Mead minimization algorithm.

    Additionally, we use the MCMC implementation EMCEE to explore the confidence
    intervals of model parameters around their best-fit values obtained at the first
    stage (deterministic search).
    Note that occasionally this first stage could get stuck in a local minimum,
    but the MCMC algorithm then usually finds its way towards the global minimum.
    In these cases one could see rather large fluctuations in the parameters
    explored by the chain.
'''
import agama, numpy
from scipy.optimize import minimize

labels = ['slopeIn', 'slopeOut', 'steepness', 'coef$J_r$', r'$\ln J_0$']

# convert from parameter space to DF params: note that we apply
# some non-trivial scaling to make the life easier for the minimizer
def dfparams(args):
    return dict(
        type = 'DoublePowerLaw',
        slopeIn   = args[0],
        slopeOut  = args[1],
        steepness = args[2],
        coefJrIn  = args[3],
        coefJzIn  = (3-args[3])/2,  # fix h_z=h_phi taking into account that h_r+h_z+h_phi=3
        coefJrOut = 1.,
        coefJzOut = 1.,
        J0        = numpy.exp(args[4]),
        norm = 1.)

# compute log-likelihood of DF with given params against an array of points
def model_likelihood(params, points):
    line = "J0=%6.5g, slopeIn=%6.5g, slopeOut=%6.5g, steepness=%6.5g, coefJrIn=%6.5g: " \
        % (params['J0'], params['slopeIn'], params['slopeOut'], params['steepness'], params['coefJrIn'])
    try:
        dpl = agama.DistributionFunction(**params)
        norm = dpl.totalMass()
        sumlog = numpy.sum( numpy.log(dpl(points)/norm) )
        print(line + ("LogL=%.8g" % sumlog))
        return sumlog
    except ValueError as err:
        print(line + "Exception " + str(err))
        return -1000.*len(points)

# function to minimize
def model_search_fnc(args, actions):
    return -model_likelihood(dfparams(args), actions)

# function to maximize
def model_search_emcee(args, actions):
    return model_likelihood(dfparams(args), actions)

# analytic expression for the ergodic distribution function f(E)
# in a Hernquist model with mass M, scale radius a, at energy E (it may be an array).
def dfHernquist(E):
    q = numpy.sqrt(-E)
    return 1 / (8 * 2**0.5 * numpy.pi**3) * (1-q*q)**-2.5 * \
        (3*numpy.arcsin(q) + q * numpy.sqrt(1-q*q) * (1-2*q*q) * (8*q**4 - 8*q*q - 3) )

# create an N-body representation of Hernquist model
def createHernquistModel(nbody):
    points      = numpy.zeros((nbody, 6))
    masses      = numpy.ones(nbody, dtype=numpy.float64) / nbody
    # 1. choose position:
    # assign the radius by inverting M(r), where M(r) is the enclosed mass - uniform in [0:1]
    radius      = 1 / (numpy.random.random(size=nbody)**-0.5 - 1)
    costheta    = numpy.random.uniform(-1, 1, size=nbody)
    sintheta    = (1-costheta**2)**0.5
    phi         = numpy.random.uniform(0, 2*numpy.pi, size=nbody)
    points[:,0] = radius * sintheta * numpy.cos(phi)
    points[:,1] = radius * sintheta * numpy.sin(phi)
    points[:,2] = radius * costheta
    # 2. assign velocity by rejection sampling
    potential   = -1 / (1 + radius)
    fmax        = 0.025 / radius**2 / (radius+3) # upper boundary on f(r,v) for rejection sampling
    velocity    = numpy.zeros(nbody)
    indices     = numpy.where(velocity == 0)[0]  # initially this contains all points
    while len(indices)>0:
        E       = numpy.random.random(size=len(indices)) * potential[indices]
        vel     = 2**0.5 * (E-potential[indices])**0.5
        fE      = dfHernquist(E) * vel * 2**-0.5
        f       = numpy.random.random(size=len(indices)) * fmax[indices]
        if(numpy.any(fE >= fmax[indices])):
            raise "Invalid upper boundary on f(E)"  # shouldn't happen
        assigned= numpy.where(f < fE)[0]
        velocity[indices[assigned]] = vel[assigned]
        indices = numpy.where(velocity == 0)[0]  # find out the unassigned elements
    costheta    = numpy.random.uniform(-1, 1, size=nbody)
    sintheta    = (1-costheta**2)**0.5
    phi         = numpy.random.uniform(0, 2*numpy.pi, size=nbody)
    points[:,3] = velocity * sintheta * numpy.cos(phi)
    points[:,4] = velocity * sintheta * numpy.sin(phi)
    points[:,5] = velocity * costheta
    return points, masses


def main():
    pot = agama.Potential(type="Dehnen", mass=1, scaleRadius=1.)
    actf= agama.ActionFinder(pot)
    particles, masses = createHernquistModel(100000)
    actions = actf(particles)

    # do a parameter search to find best-fit distribution function describing these particles
    initparams = numpy.array([2.0, 4.0, 1.0, 1.0, 0.0])
    result = minimize(model_search_fnc, initparams, args=(actions,), method='Nelder-Mead',
        options=dict(maxiter=1000, maxfev=1000, disp=True))

    # explore the parameter space around the best-fit values using the MCMC chain
    try:
        import matplotlib.pyplot as plt, emcee, corner
    except ImportError as ex:
        print(str(ex) + "\nYou need to install 'emcee' and 'corner' packages")
        exit()
    print("Starting MCMC")
    ndim = len(initparams)
    nwalkers = 16    # number of parallel walkers in the chain
    nsteps   = 300   # number of steps in MCMC chain
    nburnin  = 100   # number of initial steps to discard
    # initial coverage of parameter space - around the best-fit solution with a small dispersion
    initwalkers = [result.x + 0.01*numpy.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, model_search_emcee, args=(actions,))
    sampler.run_mcmc(initwalkers, nsteps)

    # show the time evolution of parameters carried by the ensemble of walkers (time=number of MC steps)
    fig,axes = plt.subplots(ndim+1, 1, sharex=True)
    for i in range(ndim):
        axes[i].plot(sampler.chain[:,:,i].T, color='k', alpha=0.5)
        axes[i].set_ylabel(labels[i])
    # last panel shows the evolution of log-likelihood for the ensemble of walkers
    axes[-1].plot(sampler.lnprobability.T, color='k', alpha=0.5)
    axes[-1].set_ylabel('log(L)')
    maxloglike = numpy.max(sampler.lnprobability)
    axes[-1].set_ylim(maxloglike-3*ndim, maxloglike)
    fig.tight_layout(h_pad=0.)
    plt.show()

    # show the posterior distribution of parameters
    samples = sampler.chain[:, nburnin:, :].reshape((-1, ndim))
    corner.corner(samples, labels=labels, quantiles=[0.16, 0.5, 0.84])
    plt.show()
    print("Acceptance fraction: %f" % numpy.mean(sampler.acceptance_fraction))  # should be in the range 0.2-0.5
    #print("Autocorrelation time: %f"% sampler.acor)  # should be considerably shorter than the total number of steps

main()
