#!/usr/bin/python
'''
This file is part of the Gaia Challenge, and defines the model
specified in terms of a "classical" DF f(E,L).
See gc_runfit.py for the overall description.
'''
import agama, numpy, re

class ModelParams:
    '''
    Class that represents the parameters space (used by the model-search routines)
    and converts from this space to the actual parameters of distribution function and potential,
    applying some non-trivial scaling to make the life easier for the minimizer.
    Parameters are, in order of appearance:
      -- potential params --
    lg(rhoscale) (density normalization at the scale radius, in units of Msun/Kpc^3 [log10-scaled])
    lg(Rscale) (scale radius for the two-power-law density profile, in units of Kpc [log10-scaled])
    gamma      (inner slope of the density profile)
    beta       (outer slope of the density profile)
    alpha      (steepness of the transition between the two asymptotic regimes)
      -- tracer DF params --
    lg(rstar)  (scale radius for the two-power-law density profile of stars [log10-scaled])
    Gamma      (inner slope of density profile of stars)
    Beta       (outer slope)
    Alpha      (steepness of the transition between the two asymptotic regimes)
    beta0      (central value of velocity anisotropy)
    lg(r_a)    (anisotropy radius beyond which beta approaches unity [log10-scaled])
    '''
    def __init__(self, filename):
        '''
        Initialize starting values of scaled parameters and and define upper/lower limits on them;
        also obtain the true values by analyzing the input file name
        '''
        self.initValues = numpy.array( [9.0, 0.1, 0.5, 4.0, 1.0,  0.1, 0.5, 4.0, 1.0,+0.1, 2.0])
        self.minValues  = numpy.array( [6.0,-1.0,-0.5, 2.2, 0.5, -2.0, 0.0, 3.5, 0.5,-0.5,-1.5])
        self.maxValues  = numpy.array( [10., 2.0, 1.9, 6.0, 2.5,  1.0, 1.9, 6.0, 3.0, 0.5, 5.0])
        self.labels     = ( \
            r'$\log_{10}(\rho_\mathrm{scale})$', r'$\log_{10}(R_\mathrm{scale})$', r'$\gamma$', r'$\beta$', r'$\alpha$', \
            r'$\log_{10}(R_\star)$', r'$\Gamma_\star$', r'$\mathrm{B}_\star$', r'$\mathrm{A}_\star$', r'$\beta_0$', r'$\log_{10}(r_a)$')
        self.numPotParams = 5  # potential params come first in the list
        self.numDFParams  = 6  # DF params come last in the list

        # parse true values
        m = re.search(r'gs(\d+)_bs(\d+)_rcrs([a-z\d]+)_rarc([a-z\d]+)_([a-z]+)_(\d+)mpc3', filename)
        n = re.search(r'data_([ch])_rh(\d+)_rs(\d+)_gs(\d+)', filename)
        if m:
            self.trueParams = (
                numpy.log10(float(m.group(6))*1e6),
                0.0,
                1.0 if m.group(5)=='cusp' else 0.0,
                3.0,
                1.0,
                numpy.log10(float(m.group(3))*0.01),
                float(m.group(1))*0.01,
                float(m.group(2))*0.10,
                2.0,
                0.0,
                numpy.log10(float(m.group(4))*0.01 * float(m.group(3))*0.01) if m.group(4)!='inf' else 5.0)
        elif n:
            self.trueParams = (
                numpy.log10(3.021516e7 if n.group(1)=='c' else 2.387329e7),
                numpy.log10(float(n.group(2))),
                0.0 if n.group(1)=='c' else 1.0,
                4.0,
                1.0,
                numpy.log10(1.75 if n.group(3)=='175' else 0.5),
                float(n.group(4))*0.1,
                5.0,
                2.0,
                -0.5,
                5.0)
        else:
            print("Can't determine true parameters!")
            return
        self.truePotential = agama.Potential(
            type  = 'Spheroid',
            densityNorm = 10**self.trueParams[0],
            scaleRadius = 10**self.trueParams[1],
            gamma = self.trueParams[2],
            beta  = self.trueParams[3],
            alpha = self.trueParams[4] )
        tracerParams = dict(
            type  = 'Spheroid',
            densityNorm = 1.0,
            scaleRadius = 10**self.trueParams[self.numPotParams+0],
            gamma       = self.trueParams[self.numPotParams+1],
            beta        = self.trueParams[self.numPotParams+2],
            alpha       = self.trueParams[self.numPotParams+3])
        # normalize the tracer density profile to have a total mass of unity
        tracerParams["densityNorm"] /= agama.Density(**tracerParams).totalMass()
        self.tracerDensity = agama.Density(**tracerParams)
        self.tracerBeta = lambda r: (self.trueParams[-2] + (r / 10**self.trueParams[-1])**2) / (1 + (r / 10**self.trueParams[-1])**2)


    def createModel(self, params):
        '''
        create a model (potential and DF) specified by the given [scaled] parameters
        '''
        potential       = agama.Potential(
            type        = 'Spheroid',
            densityNorm = 10**params[0],
            scaleRadius = 10**params[1],
            gamma       = params[2],
            beta        = params[3],
            alpha       = params[4] )
        density         = agama.Density(
            type        = 'Spheroid',
            scaleRadius = 10**params[self.numPotParams+0],
            gamma       = params[self.numPotParams+1],
            beta        = params[self.numPotParams+2],
            alpha       = params[self.numPotParams+3])
        df              = agama.DistributionFunction(
            type        = 'QuasiSpherical',
            potential   = potential,
            density     = density,
            beta0       = params[self.numPotParams+4],
            r_a         = 10**params[self.numPotParams+5])
        # check if the DF is everywhere nonnegative
        j = numpy.logspace(-5,10,200)
        if any(df(numpy.column_stack((j, j*0+1e-10, j*0))) <= 0):
            raise ValueError("Bad DF")
        return potential, df


    def prior(self, params):
        '''
        Return prior log-probability of the scaled parameters,
        or -infinity if they are outside the allowed range
        '''
        return 0 if all( params >= self.minValues ) and all( params <= self.maxValues ) else -numpy.inf
