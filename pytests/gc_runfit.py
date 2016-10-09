#!/usr/bin/python
'''
This file is part of the Gaia Challenge, and contains the main fitting routine.

The Gaia Challenge (or, more specifically, the "Spherical and triaxial" group)
presents the following task:
infer the gravitational potential, created entirely by the dark matter,
from the array of discrete tracers ("stars"), under the assumption of
spherical symmetry and dynamical equilibrium. The tracer population follows
a different density profile than the dark matter and may be anisotropic
in velocity space; the mock data was created by sampling from a physically valid
distribution function which we pretend not to know.
There are several different models (combinations of the potential and
the tracer distribution), and each one can be approached under three different
assumptions about the content of the data:
(1) full 6d phase-space coordinates of tracer particles with no errors;
(2) 5d (except for z-coordinate), velocities have a fixed Gaussian error of 2km/s;
(3) 3d (only x,y and v_z, the latter with an error of 2km/s);
of course, due to spherical symmetry, we only deal with the cylindrical radius,
not x and y separately.

This program, split into several python files, addresses this problem by
constructing a series of models with particular parameters of the gravitational
potential and the distribution function of tracers, and evaluating the likelihood
of each model against the provided data; we seek the parameters that maximize
the likelihood, and derive their uncertainties.

This module performs the actual fitting procedure for the given data file.
First we find a single maximum-likelihood solution with a deterministic
search algorithm, and then launch a Markov Chain Monte Carlo simulation to
explore the range of parameters that are still consistent with the data;
the latter stage uses the EMCEE algorithm with several independent 'walkers'
in the parameter space.
We perform several episodes in which the MCMC is run for several hundred steps,
and compare the mean and dispersion of parameters over the entire ensemble
between two consecutive episodes; when they no longer change, we declare
the procedure to be converged. Intermediate results are also stored in text
files, so that the fit may be restarted after each episode. We also display
the evolution of parameters along the chain, and the covariance plots.

The file "gc_modelparams.py" specifies the family of models,
defines the conversion between the scaled quantities that are explored
in the fitting process and the actual model parameters,
and determines the true profiles encoded in the name of the data file.

The file "gc_resample.py" deals with the missing data: each tracer particle
with missing or imprecise phase-space coordinates is split into a number of
subsamples, where this missing data is drawn from a non-uniform prior
distribution with a known sampling law. The likelihood of each tracer particle
is then computed as the weighted sum of likelihoods of all its subsamples.
In other words, this procedure performs the integration over missing coordinates
using a Monte Carlo approach with a fixed array of subsamples, which remains
the same throughout the entire fitting procedure -- this is necessary
to mitigate the impact of sampling noise (it is still present, but is the same
for all models).

The file "gc_plot.py" is a separate program that plots the summary of results for
a single model fitted under three different assumption about data completeness.
'''
import os, agama, numpy, scipy, matplotlib
matplotlib.use('Agg')
from gc_modelparams import ModelParams
from gc_resample import sampleMissingData
from scipy.optimize import minimize
import emcee, corner

###################$ GLOBAL CONSTANTS $###############

nsteps_deterministic  = 500   # number of steps per pass in deterministic minimizer
nsteps_mcmc           = 500   # number of MC steps per pass
nwalkers_mcmc         = 32    # number of independent MC walkers
nthreads_mcmc         = 1     # number of parallel threads in MCMC (note that each process is additionally openmp-parallelized internally)
initial_disp_mcmc     = 0.01  # initial dispersion of parameters carried by walkers around their best-fit values
phase_space_info_mode = 6     # mode=6: full phase-space information (3 positions and 3 velocities)
                              # mode=5: 5d phase-space (everything except z-coordinate)
                              # mode=3: 3d (x, y, vz)
num_subsamples        = 5000  # each input data point is represented by this number of samples
                              # which fill in the missing values of coordinate and velocity components
vel_error             = 2.0   # assumed observational error on velocity components (add noise to velocity if it is non-zero)


######################## MODEL-SEARCHER ####################################

def deterministicSearchFnc(params, obj):
    '''
    function to minimize using the deterministic algorithm (needs to be declared outside the class)
    '''
    loglike = obj.modelLikelihood(params)
    if not numpy.isfinite(loglike):
        loglike = -100*len(obj.particles)   # replace infinity with a very large negative number
    return -loglike


def monteCarloSearchFnc(params, obj):
    '''
    function to maximize using the monte carlo algorithm (needs to be declared outside the class)
    '''
    return obj.modelLikelihood(params)


class ModelSearcher:
    '''
    Class that encompasses the computation of likelihood for the given parameters,
    and implements model-searching algorithms (deterministic and MCMC)
    '''
    def __init__(self, filename):
        self.filename  = filename
        self.model     = ModelParams(filename)
        self.particles = numpy.loadtxt(filename)[:,0:6]
        if vel_error!=0:
            print "Assumed error of",vel_error,"km/s in velocity"
        if phase_space_info_mode <= 5:
            self.particles[:,2] *= numpy.nan    # remove z-coordinate
        if phase_space_info_mode <= 3:
            self.particles[:,3:5] *= numpy.nan  # remove vx and vy
        if phase_space_info_mode != 6 or vel_error != 0:
            self.samples, self.weights = sampleMissingData(
                numpy.hstack((self.particles, numpy.ones((self.particles.shape[0], 3)) * vel_error)),
                num_subsamples )

        # check if we may restart the search from already existing parameters
        try:
            self.values = numpy.loadtxt(self.filename+".best")
            if self.values.ndim==1:   # only one set of parameters - this occurs after the deterministic search
                self.values = self.values[:-1]  # the last column is the likelihood, strip it
            else:  # a number of MCMC walkers, each with its own set of parameters
                self.values = self.values[:,:-1]
            print "Loaded from saved file: (nwalkers,nparams)=",self.values.shape
        except:
            self.values = None
            return


    def modelLikelihood(self, params):
        '''
        Compute the likelihood of model (df+potential specified by scaled params)
        against the data (array of Nx6 position/velocity coordinates of tracer particles).
        This is the function to be maximized; if parameters are outside the allowed range, it returns -infinity
        '''
        prior = self.model.prior(params)
        if prior == -numpy.inf:
            print "Out of range"
            return prior
        try:
            # Compute log-likelihood of DF with given params against an array of actions
            pot     = self.model.createPotential(params)
            df      = self.model.createDF(params)
            if phase_space_info_mode == 6:  # actions of tracer particles
                if self.particles.shape[0] > 2000:  # create an action finder object for a faster evaluation
                    actions = agama.ActionFinder(pot)(self.particles)
                else:
                    actions = agama.actions(self.particles, pot)
                df_val  = df(actions)       # values of DF for these actions
            else:  # have full phase space info for resampled input particles (missing components are filled in)
                af      = agama.ActionFinder(pot)
                actions = af(self.samples)  # actions of resampled tracer particles
                # compute values of DF for these actions, multiplied by sample weights
                df_val  = df(actions) * self.weights
                # compute the weighted sum of likelihoods of all samples for a single particle,
                # replacing the improbable samples (with NaN as likelihood) with zeroes
                df_val  = numpy.sum(numpy.nan_to_num(df_val.reshape(-1, num_subsamples)), axis=1)

            loglike = numpy.sum( numpy.log( df_val ) )
            if numpy.isnan(loglike): loglike = -numpy.inf
            loglike += prior
            print "LogL=%.8g" % loglike
            return loglike
        except ValueError as err:
            print "Exception ", err
            return -numpy.inf


    def deterministicSearch(self):
        '''
        do a deterministic search to find the best-fit parameters of potential and distribution function.
        perform several iterations of search, to avoid getting stuck in a local minimum,
        until the log-likelihood ceases to improve
        '''
        if self.values is None:                   # just started
            self.values = self.model.initValues   # get the first guess from the model-scaling object
        elif self.values.ndim == 2:               # entire ensemble of values (after MCMC)
            self.values = self.values[0,:]        # leave only one set of values from the ensemble
        prevloglike = -deterministicSearchFnc(self.values, self)  # initial likelihood

        while True:
            print 'Starting deterministic search'
            result = scipy.optimize.minimize(deterministicSearchFnc, \
                self.values, args=(self,), method='Nelder-Mead', \
                options=dict(maxfev=nsteps_deterministic, disp=True))
            self.values = result.x
            loglike= -result.fun
            print 'result=', result.x, 'LogL=', loglike,
            # store the latest best-fit parameters and their likelihood
            numpy.savetxt(self.filename+'.best', numpy.hstack((self.values, loglike)).reshape(1,-1), fmt='%.8g')
            if loglike - prevloglike < 1.0:
                print 'Converged'
                return
            else:
                print 'Improved log-likelihood by', loglike - prevloglike
            prevloglike = loglike


    def monteCarloSearch(self):
        '''
        Explore the parameter space around the best-fit values using the MCMC method
        '''
        if self.values.ndim == 1:
            # initial coverage of parameter space (dispersion around the current best-fit values)
            nparams = len(self.values)
            ensemble = numpy.empty((nwalkers_mcmc, len(self.values)))
            for i in range(nwalkers_mcmc):
                while True:   # ensure that we initialize walkers with feasible values
                    walker = self.values + (numpy.random.randn(nparams)*initial_disp_mcmc if i>0 else 0)
                    prob   = monteCarloSearchFnc(walker, self)
                    if numpy.isfinite(prob):
                        ensemble[i,:] = walker
                        break
                    print '*',
            self.values = ensemble
        else:
            # check that all walkers have finite likelihood
            prob = numpy.zeros((self.values.shape[0],1))
            for i in range(self.values.shape[0]):
                prob[i,0] = monteCarloSearchFnc(self.values[i,:], self)
                if not numpy.isfinite(prob[i,0]):
                    print 'Invalid parameters for',i,'-th walker (likelihood is bogus)'
                else: print prob[i,0]

        nwalkers, nparams = self.values.shape
        sampler = emcee.EnsembleSampler(nwalkers, nparams, monteCarloSearchFnc, args=(self,), threads=nthreads_mcmc)
        prevmaxloglike = None
        while True:  # run several passes until convergence
            print 'Starting MCMC'
            sampler.run_mcmc(self.values, nsteps_mcmc)
            # restart the next pass from the latest values in the Markov chain
            self.values = sampler.chain[:,-1,:]

            # store the latest best-fit parameters and their likelihood, and the entire chain for the last nsteps_mcmc steps
            numpy.savetxt(self.filename+'.best', \
                numpy.hstack((self.values, sampler.lnprobability[:,-1].reshape(-1,1))), fmt='%.8g')
            numpy.savetxt(self.filename+".chain", \
                numpy.hstack((sampler.chain[-nsteps_mcmc:].reshape(-1,nparams),
                 sampler.lnprobability[-nsteps_mcmc:].reshape(-1,1))), fmt='%.8g')

            print "Acceptance fraction: ", numpy.mean(sampler.acceptance_fraction)  # should be in the range 0.2-0.5
            print "Autocorrelation time: ", sampler.acor  # should be considerably shorter than the total number of steps
            maxloglike = numpy.max(sampler.lnprobability[:,-nsteps_mcmc:])
            avgloglike = numpy.mean(sampler.lnprobability[:,-nsteps_mcmc:])  # avg.log-likelihood during the pass
            avgparams  = numpy.array([numpy.mean(sampler.chain[:,-nsteps_mcmc:,i]) for i in range(nparams)])
            rmsparams  = numpy.array([numpy.std (sampler.chain[:,-nsteps_mcmc:,i]) for i in range(nparams)])
            print "Max log-likelihood= %.8g, avg log-likelihood= %.8g" % (maxloglike, avgloglike)
            for i in range(nparams):
                sorted_values = numpy.sort(sampler.chain[:,-nsteps_mcmc:,i], axis=None)
                print "Parameter %20s  avg= %8.5g;  one-sigma range = (%8.5f, %8.5f)" \
                    % (self.model.labels[i], avgparams[i], \
                    sorted_values[int(len(sorted_values)*0.16)], sorted_values[int(len(sorted_values)*0.84)] )

            # plot the chain evolution and the posterior distribution + correlations between parameters
            self.plot(sampler.chain, sampler.lnprobability, self.model.labels)

            # check for convergence
            if not prevmaxloglike is None:
                if  maxloglike-prevmaxloglike  < 1.0 and \
                abs(avgloglike-prevavgloglike) < 1.0 and \
                numpy.all(avgparams-prevavgparams < 0.1) and \
                numpy.all(rmsparams-prevrmsparams < 0.1):
                    print "Converged"
                    return
            prevmaxloglike = maxloglike
            prevavgloglike = avgloglike
            prevavgparams  = avgparams
            prevrmsparams  = rmsparams


    def plot(self, chain, loglike, labels):
        '''
        Show the time evolution of parameters carried by the ensemble of walkers (time=number of MC steps),
        and the posterior distribution of parameters for the last nsteps_mcmc only
        '''
        ndim = chain.shape[2]
        fig,axes = matplotlib.pyplot.subplots(ndim+1, 1, sharex=True, figsize=(20,15))
        for i in range(ndim):
            axes[i].plot(chain[:,:,i].T, color='k', alpha=0.5)
            axes[i].set_ylabel(self.model.labels[i])
        # last panel shows the evolution of log-likelihood for the ensemble of walkers
        axes[-1].plot(loglike.T, color='k', alpha=0.5)
        axes[-1].set_ylabel('log(L)')
        maxloglike = numpy.max(loglike)
        axes[-1].set_ylim(maxloglike-3*ndim, maxloglike)   # restrict the range of log-likelihood arount its maximum
        fig.tight_layout(h_pad=0.)
        matplotlib.pyplot.savefig(self.filename+"_chain.png")

        try:
            corner.corner(chain[-nsteps_mcmc:].reshape((-1, chain.shape[2])), \
                quantiles=[0.16, 0.5, 0.84], labels=labels)
            matplotlib.pyplot.savefig(self.filename+"_posterior.png")
        except ValueError as err:
            print "Can't plot posterior distribution:", err


    def run(self):
        if self.values is None:   # first attempt a deterministic search to find the best-fit params
            self.deterministicSearch()
        self.monteCarloSearch()


################  MAIN PROGRAM  ##################

agama.setUnits(mass=1, length=1, velocity=1)

# get the directory name which is the same as the first part of the filename
basefilename = os.getcwd().split('/')[-1]
if basefilename.startswith('5'):
    phase_space_info_mode = 5
elif basefilename.startswith('6'):
    phase_space_info_mode = 6
    vel_error = 0
else:
    phase_space_info_mode = 3
basefilename = basefilename[1:]
modelSearcher = ModelSearcher(basefilename + "_1000_0" + ("_err" if vel_error>0 else "") + ".dat")
modelSearcher.run()
