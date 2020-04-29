#!/usr/bin/env python
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
To redo the plots without running the MCMC again, add a second command-line
argument "plot" (after the name of the data file).

The files "gc_modelparamsE.py" and "gc_modelparamsJ.py" specify two possible
families of models (only one of them should be selected to import).
They also define the conversion between the scaled quantities that are
explored in the fitting process and the actual model parameters,
and decodee the true parameters from the name of the data file.

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

To run the program, one needs to select one of the two possible families of
models: f(E,L) or f(J); select the type of available data (3, 5 or 6 known
phase-space coordinates), and provide the name of the data file as the argument.
If the program has been run previously, it will store the current set of best
parameters in a file <filename>.best, which can be used to hot-restart the fit.
However, if you switch to the other family of model, this file should be deleted,
as the model parameters are incompatible between the two families.
'''
from __future__ import print_function
import sys, numpy, scipy.optimize, scipy.special, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, emcee, corner
import agama

# a separate module for sampling over missing coordinates/velocities
from gc_resample import sampleMissingData

# a separate module which contains the description of the actual model
# here are two possible options: f(E,L) or f(J), uncomment one of the following lines:
from gc_modelparamsE import ModelParams
#from gc_modelparamsJ import ModelParams


###################$ GLOBAL CONSTANTS $###############

nsteps_deterministic  = 500   # number of steps per pass in deterministic minimizer
nsteps_mcmc           = 500   # number of MC steps per pass
nwalkers_mcmc         = 24    # number of independent MC walkers
nsamples_plot         = 200   # number of randomly chosen samples from the MCMC chain to plot
initial_disp_mcmc     = 0.01  # initial dispersion of parameters carried by walkers around their best-fit values
phase_space_info_mode = 3     # mode=6: full phase-space information (3 positions and 3 velocities)
                              # mode=5: 5d phase-space (everything except z-coordinate)
                              # mode=3: 3d (x, y, vz)
num_subsamples        = 1000  # each input data point is represented by this number of samples
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
    def __init__(self):
        try:
            self.filename  = sys.argv[1]
            self.particles = numpy.loadtxt(self.filename)[:,0:6]
            self.model     = ModelParams(self.filename)
        except Exception as ex:
            print(str(ex)+"\nNeed to provide input text file with stellar coordinates and velocities.")
            exit()
        if vel_error!=0:
            print("Assumed error of %f km/s in velocity" % vel_error)
        if phase_space_info_mode <= 5:
            self.particles[:,2] *= numpy.nan    # remove z-coordinate
        if phase_space_info_mode <= 3:
            self.particles[:,3:5] *= numpy.nan  # remove vx and vy
        if phase_space_info_mode != 6 or vel_error != 0:
            self.samples, self.weights = sampleMissingData(
                numpy.hstack((self.particles, numpy.ones((self.particles.shape[0], 3)) * vel_error)),
                num_subsamples )
        else:
            self.samples = None

        # check if we may restart the search from already existing parameters
        try:
            self.values = numpy.loadtxt(self.filename+".best")
            if self.values.ndim==1:   # only one set of parameters - this occurs after the deterministic search
                self.values = self.values[:-1]  # the last column is the likelihood, strip it
            else:  # a number of MCMC walkers, each with its own set of parameters
                self.values = self.values[:,:-1]
            print("Loaded from saved file: (nwalkers,nparams)=" + str(self.values.shape))
        except:
            self.values = None
            return


    def modelLikelihood(self, params):
        '''
        Compute the likelihood of model (df+potential specified by scaled params)
        against the data (array of Nx6 position/velocity coordinates of tracer particles).
        This is the function to be maximized; if parameters are outside the allowed range, return -infinity
        '''
        prior = self.model.prior(params)
        print(params, end=': ')
        if prior == -numpy.inf:
            print("Out of range")
            return prior
        try:
            # Compute log-likelihood of DF with given params against an array of actions
            pot, df = self.model.createModel(params)
            if self.samples is None:  # actions of tracer particles
                if self.particles.shape[0] > 2000:  # create an action finder object for a faster evaluation
                    actions = agama.ActionFinder(pot)(self.particles)
                else:
                    actions = agama.actions(pot, self.particles)
                df_val  = df(actions)       # values of DF for these actions
            else:  # have full phase space info for resampled input particles (missing components are filled in)
                af      = agama.ActionFinder(pot)
                actions = af(self.samples)  # actions of resampled tracer particles
                # compute values of DF for these actions, multiplied by sample weights
                df_vals = df(actions) * self.weights
                # compute the weighted sum of likelihoods of all samples for a single particle,
                # replacing the improbable samples (with NaN as likelihood) with zeroes
                df_val  = numpy.sum(numpy.nan_to_num(df_vals.reshape(-1, num_subsamples)), axis=1)
            loglike = numpy.sum( numpy.log( df_val ) )
            if numpy.isnan(loglike): loglike = -numpy.inf
            loglike += prior
            print("LogL=%.8g" % loglike)
            return loglike
        except ValueError as err:
            print("Exception "+str(err))
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
            print('Starting deterministic search')
            result = scipy.optimize.minimize(deterministicSearchFnc, \
                self.values, args=(self,), method='Nelder-Mead', \
                options=dict(maxfev=nsteps_deterministic, disp=True))
            self.values = result.x
            loglike= -result.fun
            print('result='+str(result.x)+' LogL='+str(loglike))
            # store the latest best-fit parameters and their likelihood
            numpy.savetxt(self.filename+'.best', numpy.hstack((self.values, loglike)).reshape(1,-1), fmt='%.8g')
            if loglike - prevloglike < 1.0:
                print('Converged')
                return
            else:
                print('Improved log-likelihood by %f' % (loglike - prevloglike))
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
                    print('*',end='')
            self.values = ensemble
        else:
            # check that all walkers have finite likelihood
            prob = numpy.zeros((self.values.shape[0],1))
            for i in range(self.values.shape[0]):
                prob[i,0] = monteCarloSearchFnc(self.values[i,:], self)
                if not numpy.isfinite(prob[i,0]):
                    print('Invalid parameters for %i-th walker (likelihood is bogus)' % i)
                else: print('%i-th walker: logL=%g' % (i, prob[i,0]))

        nwalkers, nparams = self.values.shape
        sampler = emcee.EnsembleSampler(nwalkers, nparams, monteCarloSearchFnc, args=(self,))
        prevmaxloglike = None
        while True:  # run several passes until convergence
            print('Starting MCMC')
            sampler.run_mcmc(self.values, nsteps_mcmc)
            # restart the next pass from the latest values in the Markov chain
            self.values = sampler.chain[:,-1,:]

            # store the latest best-fit parameters and their likelihood, and the entire chain for the last nsteps_mcmc steps
            numpy.savetxt(self.filename+'.best', \
                numpy.hstack((self.values, sampler.lnprobability[:,-1].reshape(-1,1))), fmt='%.8g')
            numpy.savetxt(self.filename+".chain", \
                numpy.hstack((sampler.chain[:,-nsteps_mcmc:].reshape(-1,nparams),
                 sampler.lnprobability[:,-nsteps_mcmc:].reshape(-1,1))), fmt='%.8g')

            print("Acceptance fraction: %g" % numpy.mean(sampler.acceptance_fraction))  # should be in the range 0.2-0.5
            try:
                print("Autocorrelation time: %g" % sampler.get_autocorr_time())
                # should be considerably shorter than the total number of steps
            except: pass  # sometimes it can't be computed, then ignore
            maxloglike = numpy.max(sampler.lnprobability[:,-nsteps_mcmc:])
            avgloglike = numpy.mean(sampler.lnprobability[:,-nsteps_mcmc:])  # avg.log-likelihood during the pass
            avgparams  = numpy.array([numpy.mean(sampler.chain[:,-nsteps_mcmc:,i]) for i in range(nparams)])
            rmsparams  = numpy.array([numpy.std (sampler.chain[:,-nsteps_mcmc:,i]) for i in range(nparams)])
            print("Max log-likelihood= %.8g, avg log-likelihood= %.8g" % (maxloglike, avgloglike))
            for i in range(nparams):
                sorted_values = numpy.sort(sampler.chain[:,-nsteps_mcmc:,i], axis=None)
                print("Parameter %20s  avg= %8.5g;  one-sigma range = (%8.5f, %8.5f)" % \
                    (self.model.labels[i], avgparams[i], \
                    sorted_values[int(len(sorted_values)*0.16)], \
                    sorted_values[int(len(sorted_values)*0.84)] ))

            # plot the chain evolution and the posterior distribution + correlations between parameters
            self.plot(sampler.chain, sampler.lnprobability, self.model.labels)

            # check for convergence
            if not prevmaxloglike is None:
                if  maxloglike-prevmaxloglike  < 1.0 and \
                abs(avgloglike-prevavgloglike) < 1.0 and \
                numpy.all(avgparams-prevavgparams < 0.1) and \
                numpy.all(rmsparams-prevrmsparams < 0.1):
                    print("Converged")
                    return
            prevmaxloglike = maxloglike
            prevavgloglike = avgloglike
            prevavgparams  = avgparams
            prevrmsparams  = rmsparams


    def plotProfiles(self, chain):
        '''
        plot the radial profiles of various quantities from the set of model in the MCMC chain,
        together with the true profiles.
        '''
        axes     = plt.subplots(2, 2, figsize=(12,8))[1].T.reshape(-1)
        rmin     = 0.01
        rmax     = 100.
        radii    = numpy.logspace(numpy.log10(rmin), numpy.log10(rmax), 41)
        midradii = (radii[1:] * radii[:-1])**0.5
        xyz      = numpy.column_stack((radii, radii*0, radii*0))

        # compute and store the profiles for each model in the chain, then take 68% and 95% percentiles
        dmdens   = numpy.zeros((chain.shape[0], len(radii)))
        dmslope  = numpy.zeros((chain.shape[0], len(midradii)))
        trdens   = numpy.zeros((chain.shape[0], len(radii)))
        trbeta   = numpy.zeros((chain.shape[0], len(radii)))
        print('Plotting profiles...')
        for i in range(len(chain)):
            pot, df    = self.model.createModel(chain[i])
            dmdens [i] = pot.density(xyz)
            dmslope[i] = numpy.log(dmdens[i,1:] / dmdens[i,:-1]) / numpy.log(radii[1:] / radii[:-1])
            trdens [i], trvel = agama.GalaxyModel(pot, df).moments(xyz, dens=True, vel=False, vel2=True)
            trbeta [i] = 1 - trvel[:,1] / trvel[:,0]

        # log-slope of the DM density profile  d(log rho) / d(log r)
        cntr = numpy.percentile(dmslope, [2.3, 15.9, 50, 84.1, 97.7], axis=0)
        axes[0].fill_between(midradii, cntr[0], cntr[4], color='lightgray')  # 2 sigma
        axes[0].fill_between(midradii, cntr[1], cntr[3], color='gray')       # 1 sigma
        axes[0].plot(midradii, cntr[2], color='k')  # median
        axes[0].set_xscale('log')
        axes[0].set_xlim(rmin, rmax)
        axes[0].set_ylim(-5, 1)
        axes[0].set_xlabel('$r$')
        axes[0].set_ylabel(r'$d(\ln\rho_{DM}) / d(\ln r)$')

        # DM density profile
        cntr = numpy.percentile(dmdens, [2.3, 15.9, 50, 84.1, 97.7], axis=0)
        axes[1].fill_between(radii, cntr[0], cntr[4], color='lightgray')  # 2 sigma
        axes[1].fill_between(radii, cntr[1], cntr[3], color='gray')       # 1 sigma
        axes[1].plot(radii, cntr[2], color='k')  # median
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')
        axes[1].set_xlim(rmin, rmax)
        axes[1].set_xlabel('$r$')
        axes[1].set_ylabel(r'$\rho_{DM}$')

        # velocity anisotropy coefficient (beta) of tracers
        cntr = numpy.percentile(trbeta, [2.3, 15.9, 50, 84.1, 97.7], axis=0)
        axes[2].fill_between(radii, cntr[0], cntr[4], color='lightgray')  # 2 sigma
        axes[2].fill_between(radii, cntr[1], cntr[3], color='gray')       # 1 sigma
        axes[2].plot(radii, cntr[2], color='k')  # median
        axes[2].set_xscale('log')
        axes[2].set_xlim(rmin, rmax)
        axes[2].set_ylim(-1, 1)
        axes[2].set_xlabel('$r$')
        axes[2].set_ylabel(r'$\beta_\star$')

        # 3d density profile of tracers
        cntr = numpy.percentile(trdens, [2.3, 15.9, 50, 84.1, 97.7], axis=0)
        axes[3].fill_between(radii, cntr[0], cntr[4], color='lightgray')  # 2 sigma
        axes[3].fill_between(radii, cntr[1], cntr[3], color='gray')       # 1 sigma
        axes[3].plot(radii, cntr[2], color='k')  # median
        axes[3].set_xscale('log')
        axes[3].set_yscale('log')
        axes[3].set_xlim(rmin, rmax)
        axes[3].set_xlabel('$r$')
        axes[3].set_ylabel(r'$\rho_\star$')

        # histogram of radial distribution of the original points on each of the four panels
        ptcount  = numpy.histogram((self.particles[:,0]**2 + self.particles[:,1]**2)**0.5, bins=radii)[0]
        for ax in axes:
            plt.twinx(ax)
            plt.plot(numpy.hstack(zip(radii[:-1], radii[1:])), numpy.repeat(ptcount, 2), 'g-', alpha=0.5)
            plt.ylim(0, 2*max(ptcount))

        try:
            true_dmdens = self.model.truePotential.density(xyz)
            true_dmslope= numpy.log(true_dmdens[1:] / true_dmdens[:-1]) / numpy.log(radii[1:] / radii[:-1])
            true_trdens = self.model.tracerDensity.density(xyz)
            true_trbeta = self.model.tracerBeta(radii)
            axes[0].plot(midradii, true_dmslope, color='r', lw=3, linestyle='--')
            axes[1].plot(   radii, true_dmdens,  color='r', lw=3, linestyle='--')
            axes[2].plot(   radii, true_trbeta,  color='r', lw=3, linestyle='--')
            axes[3].plot(   radii, true_trdens,  color='r', lw=3, linestyle='--')
            axes[1].set_ylim(true_dmdens[-1]*0.5, true_dmdens[0]*5)
            axes[3].set_ylim(true_trdens[-1]*0.5, true_trdens[0]*5)
        except AttributeError: pass  # no true values known

        plt.tight_layout()
        plt.savefig(self.filename+"_profiles.png")
        plt.close()


    def plot(self, chain, loglike, labels):
        '''
        Show the time evolution of parameters carried by the ensemble of walkers (time=number of MC steps),
        and the posterior distribution of parameters for the last nsteps_mcmc only
        '''
        ndim = chain.shape[2]
        fig,axes = plt.subplots(ndim+1, 1, sharex=True, figsize=(20,15))
        for i in range(ndim):
            axes[i].plot(chain[:,:,i].T, color='k', alpha=0.5)
            axes[i].set_ylabel(self.model.labels[i])
        # last panel shows the evolution of log-likelihood for the ensemble of walkers
        axes[-1].plot(loglike.T, color='k', alpha=0.5)
        axes[-1].set_ylabel('log(L)')
        maxloglike =numpy.max(loglike)
        maxexpected=numpy.median(loglike[:,-nsteps_mcmc:])+0.5*ndim-0.33   # expected max-likelihood for a chi2 distribution with ndim
        axes[-1].set_ylim(maxloglike-5-ndim, maxloglike)   # restrict the range of log-likelihood arount its maximum
        plt.tight_layout(h_pad=0.)
        plt.savefig(self.filename+"_chain.png")
        plt.close()

        latest_chain = chain[:,-nsteps_mcmc:].reshape(-1, chain.shape[2])
        try:
            trueParams = self.model.trueParams
        except AttributeError:
            trueParams = None
        try:
            corner.corner(latest_chain, quantiles=[0.16, 0.5, 0.84], labels=labels, truths=trueParams)
            # distribution of log-likelihoods - expected to follow the chi2 law with ndim degrees of freedom
            ax=plt.axes([0.64,0.64,0.32,0.32])
            bins=numpy.linspace(-4-ndim, 1, 101) + maxexpected
            ax.hist(loglike[:,-nsteps_mcmc:].reshape(-1), bins=bins, normed=True, histtype='step')
            xx=numpy.linspace(-4-ndim, 0, 101)
            ax.plot(xx + maxexpected, 1/scipy.special.gamma(0.5*ndim) * (-xx)**(0.5*ndim-1) * numpy.exp(xx), 'r', lw=2)
            ax.set_xlim(bins[0], bins[-1])
            ax.set_xlabel('log(L)')
            plt.savefig(self.filename+"_posterior.png")
            plt.close()
        except ValueError as err:
            print("Can't plot posterior distribution: "+str(err))

        try:
            self.plotProfiles(latest_chain[numpy.random.choice(len(latest_chain), nsamples_plot, replace=False)])
        except Exception as err:
            print("Can't plot profiles: "+str(err))


    def run(self):
        if self.values is None:   # first attempt a deterministic search to find the best-fit params
            self.deterministicSearch()
        self.monteCarloSearch()


################  MAIN PROGRAM  ##################

numpy.set_printoptions(precision=5, linewidth=200, suppress=True)
agama.setUnits(mass=1, length=1, velocity=1)
m=ModelSearcher()
if len(sys.argv)>2 and 'PLOT' in sys.argv[2].upper():
    chain = numpy.loadtxt(m.filename+'.chain')
    m.plotProfiles(chain[numpy.random.choice(len(chain), nsamples_plot, replace=False)])
else:
    m.run()
