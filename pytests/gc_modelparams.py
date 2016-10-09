#!/usr/bin/python
'''
This file is part of the Gaia Challenge, and contains the description of the model.
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
    lg(rho_1)  (density normalization at a fixed radius r_1, in units of Msun/Kpc^3)
    lg(Rscale) (scale radius for the two-power-law density profile, in units of Kpc)
    gamma      (inner slope of the density profile)
    beta       (outer slope of the density profile)
    alpha      (steepness of transition between the two asymptotic regimes)
      -- DF params --
    Beta       (DF slope at large J)
    Gamma      (DF slope at small J)
    eta        (steepness of transition between two power-law regimes)
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
        self.initValues = numpy.array( [9.,  0., 0.5, 4.0, 1.0,  6.0, 1.5, 1.0, 1.0, 1.0, 1.])
        self.minValues  = numpy.array( [6., -1., 0.0, 2.2, .25,  3.2, 0.0, 0.5, 0.1, 0.1,-1.])
        self.maxValues  = numpy.array( [10., 2., 1.9, 6.0, 2.5,  12., 2.8, 2.0, 2.8, 2.8, 3.])
        self.labels     = ( \
            r'$\log_{10}(\rho_1)$', r'$\log_{10}(R_{scale})$', r'$\gamma$', r'$\beta$', r'$\alpha$', \
            r'$B_{DF}$', r'$\Gamma_{DF}$', r'$\eta$', r'$g_r$', r'$h_r$', r'$\log_{10}(J_0)$')
        self.numPotParams = 5  # potential params come first in the list
        self.numDFParams  = 6  # DF params come last in the list
        self.scaleRadius  = 1. # fixed radius r_1 at which the DM density is constrained (rho_1)

        # parse true values
        m = re.match(r'gs(\d+)_bs(\d+)_rcrs([a-z\d]+)_rarc([a-z\d]+)_([a-z]+)_(\d+)mpc3', filename)
        n = re.match(r'data_([ch])_rh(\d+)_rs(\d+)_gs(\d+)', filename)
        if m:
            self.truePotential = agama.Potential(
                type  = 'SpheroidDensity',
                densityNorm = float(m.group(6))*1e6,
                scaleRadius = 1.0,
                gamma = 1.0 if m.group(5)=='cusp' else 0.0,
                beta  = 3.0 )
            self.scaleRadius = float(m.group(3))*0.01
            self.tracerParams = dict(
                type  = 'SpheroidDensity',
                densityNorm = 1.0,
                scaleRadius = self.scaleRadius,
                gamma = float(m.group(1))*0.01,
                beta  = float(m.group(2))*0.1 )
            # normalize the tracer density profile to have a total mass of unity
            self.tracerParams["densityNorm"] /= agama.Density(**self.tracerParams).totalMass()
            self.tracerDensity = agama.Density(**self.tracerParams)
        elif n:
            self.truePotential = agama.Potential(
                type  = 'SpheroidDensity',
                densityNorm = 3.021516e7 if n.group(1)=='c' else 2.387329e7,
                scaleRadius = float(n.group(2)),
                gamma = 0.0 if n.group(1)=='c' else 1.0,
                beta  = 4.0 )
            self.scaleRadius = float(n.group(3))*0.01
            self.tracerParams = dict(
                type  = 'SpheroidDensity',
                densityNorm = 1.0,
                scaleRadius = self.scaleRadius,
                gamma = float(m.group(4))*0.1,
                beta  = 5.0 )
            self.tracerParams["densityNorm"] /= agama.Density(**self.tracerParams).totalMass()
            self.tracerDensity = agama.Density(**self.tracerParams)
        else:
            print "Can't determine true parameters!"


    def createPotential(self, params):

        rho1  = 10**params[0]
        r0    = 10**params[1]
        gamma = params[2]
        beta  = params[3]
        alpha = params[4]
        r1r0  = self.scaleRadius / r0
        rho0  = rho1 * (1 + r1r0**alpha)**((beta-gamma)/alpha) * r1r0**gamma
        return agama.Potential(
            type        = 'SpheroidDensity',
            densityNorm = rho0,
            scaleRadius = r0,
            gamma       = gamma,
            beta        = beta,
            alpha       = alpha )


    def createDF(self, params):

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
        return agama.DistributionFunction(**dfparams)


    def prior(self, params):
        '''
        Return prior log-probability of the scaled parameters,
        or -infinity if they are outside the allowed range
        '''
        return 0 if numpy.all( params >= self.minValues ) and numpy.all( params <= self.maxValues ) else -numpy.inf
