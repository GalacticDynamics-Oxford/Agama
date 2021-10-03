#!/usr/bin/python
'''
This file is part of the Gaia Challenge, and defines the model
specified in terms of an action-based DF f(J).
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
    Beta       (DF slope at large J)
    Gamma      (DF slope at small J)
    eta        (steepness of the transition between two power-law regimes)
    g_r        (coefficient for the radial action in the linear combination of actions
               for the large J region; we set g_z = g_phi, and g_r+g_z+g_phi=3)
    h_r        (same for the small J region)
    lg(J0)     (boundary between small and large J, in units of Kpc*km/s)
    '''
    def __init__(self, filename):
        '''
        Initialize starting values of scaled parameters and and define upper/lower limits on them;
        also obtain the true values by analyzing the input file name
        '''
        self.initValues = numpy.array( [9.0, 0.1, 0.5, 4.0, 1.0,  6.0, 1.5, 1.0, 1.0, 1.0, 1.])
        self.minValues  = numpy.array( [6.0,-1.0,-0.5, 2.2, 0.5,  3.2, 0.0, 0.5, 0.1, 0.1,-1.])
        self.maxValues  = numpy.array( [10., 2.0, 1.9, 6.0, 2.5,  12., 2.8, 2.0, 2.8, 2.8, 3.])
        self.labels     = ( \
            r'$\log_{10}(\rho_\mathrm{scale})$', r'$\log_{10}(R_\mathrm{scale})$', r'$\gamma$', r'$\beta$', r'$\alpha$', \
            r'$B_\mathrm{DF}$', r'$\Gamma_\mathrm{DF}$', r'$\eta$', r'$g_r$', r'$h_r$', r'$\log_{10}(J_0)$')
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
                # true params of tracer DF do not belong to this family of models, so ignore them
                numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan)
            tracerParams = dict(
                type  = 'Spheroid',
                densityNorm = 1.0,
                scaleRadius = float(m.group(3))*0.01,
                gamma = float(m.group(1))*0.01,
                beta  = float(m.group(2))*0.1,
                alpha = 2.0 )
            beta0 = 0
            r_a = float(m.group(4))*0.01 * float(m.group(3))*0.01 if m.group(4)!='inf' else numpy.inf
        elif n:
            self.trueParams = (
                numpy.log10(3.021516e7 if n.group(1)=='c' else 2.387329e7),
                numpy.log10(float(n.group(2))),
                0.0 if n.group(1)=='c' else 1.0,
                4.0,
                1.0,
                numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan)
            tracerParams = dict(
                type  = 'Spheroid',
                densityNorm = 1.0,
                scaleRadius = 1.75 if n.group(3)=='175' else 0.5,
                gamma = float(n.group(4))*0.1,
                beta  = 5.0,
                alpha = 2.0 )
            beta0 = -0.5
            r_a   = numpy.inf
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
        # normalize the tracer density profile to have a total mass of unity
        tracerParams["densityNorm"] /= agama.Density(**tracerParams).totalMass()
        self.tracerDensity = agama.Density(**tracerParams)
        self.tracerBeta = lambda r: (beta0 + (r / r_a)**2) / (1 + (r / r_a)**2)

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
        # first create an un-normalized DF
        dfparams      = dict(
            type      =     'DoublePowerLaw',
            slopeOut  =     params[self.numPotParams+0],
            slopeIn   =     params[self.numPotParams+1],
            steepness =     params[self.numPotParams+2],
            coefJrOut =     params[self.numPotParams+3],
            coefJzOut =  (3-params[self.numPotParams+3])/2,
            coefJrIn  =     params[self.numPotParams+4],
            coefJzIn  =  (3-params[self.numPotParams+4])/2,
            j0        = 10**params[self.numPotParams+5],
            norm      = 1. )
        # compute its total mass
        totalMass = agama.DistributionFunction(**dfparams).totalMass()
        # and now normalize the DF to have a unit total mass
        dfparams["norm"] = 1. / totalMass
        df = agama.DistributionFunction(**dfparams)
        return potential, df


    def prior(self, params):
        '''
        Return prior log-probability of the scaled parameters,
        or -infinity if they are outside the allowed range
        '''
        return 0 if all( params >= self.minValues ) and all( params <= self.maxValues ) else -numpy.inf
