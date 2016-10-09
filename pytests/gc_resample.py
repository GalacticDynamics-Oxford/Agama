#!/usr/bin/python
'''
This file is part of the Gaia Challenge, and contains the deprojection routine.
See gc_runfit.py for the overall description.

######################## SPHERICAL MODEL DEPROJECTION #########################
This module fills missing values for coordinate and velocity for the case of incomplete data.
On input, we have an array of particles with 9 parameters:
{x, y, [z?], [vx?], [vy?], vz, vx_err, vy_err, vz_err},
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
For missing velocity components, we sample uniformly within the range of z-velocities
of input points.

The spherical model deprojection is implemented in the class ``SphericalModel'',
and the resampling procedure -- in the routine ``sampleMissingData''
'''

import agama, numpy, scipy
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import quad


def binIndices(length):
    '''
    Devise a scheme for binning the particles in projected radius,
    so that the inner- and outermost bins contain 10-15 particles,
    and the intermediate ones - Nptbin particles.
    '''
    Nptbin = numpy.maximum(length**0.5 * 2, 20)
    return numpy.hstack((0, 10, 25, \
        numpy.linspace(50, length-50, (length-100)/Nptbin).astype(int), \
        length-25, length-11, length-1))

class SphericalModel:
    '''
    Construct a spherically symmetric model using the information about
    projected particle positions and line-of-sight velocities;
    input: particles is a Nx6 array of coordinates and velocities;
    only the x,y coordinates and z-velocity are used
    '''
    def __init__(self, particles):
        # sort particles in projected radius
        particles_sorted = particles[ numpy.argsort((particles[:,0]**2 + particles[:,1]**2) ** 0.5) ]

        # create binning in projected radii
        Radii   = (particles_sorted[:,0]**2 + particles_sorted[:,1]**2) ** 0.5
        indices = binIndices(len(Radii))

        # compute the cumulative mass profile from input particles (M(R) - mass inside projected radius)
        # assuming equal-mass particles and total mass = 1
        cumulMass = numpy.linspace(1./len(Radii), 1., len(Radii))

        # create a smoothing spline for log(M/(Mtotal-M)) as a function of log(R), 
        # using points from the interval indices[1]..indices[-2], and spline knots at Radii[indices]
        self.spl_mass = agama.SplineApprox( \
            numpy.log(Radii[indices[1:-1]]), \
            numpy.log(Radii[indices[1]:indices[-2]]), \
            numpy.log(cumulMass[indices[1]:indices[-2]] / (1 - cumulMass[indices[1]:indices[-2]])), \
            smooth=2.0 )

        # compute 3d density at the same radial grid
        rho_grid   = numpy.log([self.rho_integr(R) for R in Radii[indices]])
        good_elems = numpy.where(numpy.isfinite(rho_grid))[0]
        if(len(good_elems)<len(rho_grid)):
            print "Invalid density encountered at r=", \
                Radii[indices[numpy.where(numpy.logical_not(numpy.isfinite(rho_grid)))]]

        # initialize an interpolating spline for 3d density (log-log scaled)
        self.spl_rho = scipy.interpolate.InterpolatedUnivariateSpline( \
            numpy.log(Radii[indices[good_elems]]), rho_grid[good_elems])

        # store the derivatives of the spline at endpoints (for extrapolation)
        self.logrmin = numpy.log(Radii[indices[ 0]])
        self.derrmin = self.spl_rho(self.logrmin,1)
        self.logrmax = numpy.log(Radii[indices[-1]])
        self.derrmax = self.spl_rho(self.logrmax,1)
        print 'Density slope: inner=',self.derrmin,', outer=',self.derrmax
        if self.derrmin>0: self.derrmin=0
        if self.derrmax>-3: self.derrmax=-3

        # cumulative kinetic energy (up to a constant factor) $\int_0^R \Sigma(R') \sigma_{los}^2(R') 2\pi R' dR'$
        cumulEkin  = numpy.cumsum(particles_sorted[:,5]**2) / len(Radii)
        self.total_Ekin = cumulEkin[-1]
        self.spl_Ekin = agama.SplineApprox( \
            numpy.log(Radii[indices]), \
            numpy.log(Radii[indices[0]:indices[-1]]), \
            numpy.log(cumulEkin[indices[0]:indices[-1]] / \
            (self.total_Ekin - cumulEkin[indices[0]:indices[-1]])), \
            smooth=2.0 )

    def cumul_mass(self, R):
        ''' Return M(<R) '''
        return 1 / (1 + numpy.exp(-self.spl_mass(numpy.log(R))))

    def surface_density(self, R):
        ''' Return surface density: Sigma(R) = 1 / (2 pi R)  d[ M(<R) ] / dR '''
        lnR = numpy.log(R)
        val = numpy.exp(self.spl_mass(lnR))
        return self.spl_mass(lnR, 1) * val / ( 2*3.1416 * R**2 * (1+val)**2 )

    def sigma_los(self, R):
        ''' Return line-of-sight velocity dispersion:
            sigma_los^2(R) = 1 / (2 pi R Sigma(R))  d[ cumulEkin(R) ] / dR '''
        lnR = numpy.log(R)
        val = numpy.exp(self.spl_Ekin(lnR))
        return ( self.total_Ekin * self.spl_Ekin(lnR, 1) * val / \
            ( 2*3.1416 * R**2 * (1 + val)**2 * self.surface_density(R) ) )**0.5

    def integrand(self, t, r, spl):
        ''' integrand in the density or pressure deprojection formula, depending on the spline passed as argument '''
        R = (r**2+(t/(1-t))**2) ** 0.5
        lnR = numpy.log(R)
        val = numpy.exp(spl(lnR))
        der = spl(lnR, 1)
        der2= spl(lnR, 2)
        dSdR= val / ( 2*3.1416 * R**3 * (1+val)**2 ) * (der2 - 2*der + der**2*(1-val)/(1+val) )
        return -1/3.1416 * dSdR / (1-t) / (r**2*(1-t)**2 + t**2)**0.5

    def rho_integr(self, r):
        ''' Return 3d density rho(r) computed by integration of the deprojection equation '''
        return scipy.integrate.quad(self.integrand, 0, 1, (r, self.spl_mass), epsrel=1e-3)[0]

    def rho(self, r):
        ''' Return 3d density rho(r) approximated by a spline '''
        logr   = numpy.log(r)
        result = self.spl_rho(numpy.maximum(self.logrmin, numpy.minimum(self.logrmax, logr))) \
            + self.derrmin * numpy.minimum(logr-self.logrmin, 0) \
            + self.derrmax * numpy.maximum(logr-self.logrmax, 0)
        return numpy.exp(result)

    def sigma_iso_integr(self, r):
        ''' Return isotropic velocity dispersion computed by integration of the deprojection equation '''
        return (scipy.integrate.quad(self.integrand, 0, 1, (r, self.spl_Ekin), epsrel=1e-3)[0] * \
            self.total_Ekin / self.rho_integr(r) )**0.5


##################### RESAMPLING OF ORIGINAL DATA TO FILL MISSING VALUES #################

def sampleZPosition(R, sph_model):
    '''
    Sample the missing z-component of particle coordinates
    from the density distribution given by the spherical model.
    input argument 'R' contains the array of projected radii,
    and the output will contain the z-values assigned to them,
    and the weights of individual samples.
    '''
    print 'Assigning missing z-component of position'
    rho_max = sph_model.rho(R)*2.0
    R0      = numpy.maximum(2., R)
    result  = numpy.zeros_like(R)
    weights = sph_model.surface_density(R)
    indices = numpy.where(result==0)[0]   # index array initially containing all input points
    while len(indices)>0:   # rejection sampling
        t = numpy.random.uniform(-1, 1, size=len(indices))
        z = R0[indices] * t/(1-t*t)**0.5
        rho = sph_model.rho( (R[indices]**2 + z**2)**0.5 )
        rho_bar = rho / (1-t*t)**1.5 / rho_max[indices]
        overflows = numpy.where(rho_bar>1)[0]
        if(len(overflows)>0):
            print 'Overflows:', len(overflows)
            #numpy.hstack((R[overflows].reshape(-1.1), t[overflows].reshape(-1.1), rho_bar[overflows].reshape(-1.1)))
        assigned = numpy.where(numpy.random.uniform(size=len(indices)) < rho_bar)[0]
        result [indices[assigned]]  = z[assigned]
        weights[indices[assigned]] /= rho[assigned]
        indices = numpy.where(result==0)[0]  # find out the unassigned elements
    return result, weights

def sampleMissingData(particles, Nsubsamples, fancy_z_assignment=True):
    '''
    Split each input particle into Nsamples samples, perturbing its velocity or
    assigning values for missing components of position/velocity.
    Input: particles -- array of Nx9 values (3 coordinates, 3 velocities, and 3 velocity errors),
    missing data is indicated by NAN columns
    (the case when only some particles have missing data is not currently supported);
    Nsubsamples -- number of sub-samples created from each particle;
    fancy_z_assignment -- if True, use importance sampling for z-coordinate
    from a deprojected density profile; otherwise sample uniformly
    '''
    numpy.random.seed(0)  # make resampling repeatable from run to run

    # duplicate the elements of the original particle array
    # (each particle is expanded into Nsamples identical samples)
    samples    = numpy.repeat(particles[:, 0:6], Nsubsamples, axis=0)
    nparticles = particles.shape[0]
    nsamples   = samples.shape[0]
    weights    = numpy.ones_like(nsamples, dtype=numpy.float64)
    vel_err    = particles[:, 6:9] if particles.shape[1]==9 else numpy.zeros((nparticles, 3))

    # compute maximum magnitude of distance and l.o.s. velocity used in assigning
    # missing z-coordinate and velocity components for resampled particles
    vmax = numpy.sort(abs(particles[:,5]))[int(nparticles*0.99)]  # take almost the maximal value
    Rmax = numpy.sort((particles[:,0]**2+particles[:,1]**2)**0.5)[int(nparticles*0.99)]  # same here
    print 'Resample %d input particles into %d internal samples (Rmax=%f, vmax=%f)' % \
        (nparticles, nsamples, Rmax, vmax)

    if numpy.any(numpy.isnan(samples[:,2])):   # z-coordinate is missing
        if fancy_z_assignment:  # use deprojection with unequal prior weights for resampled particles
            sph_model = SphericalModel(particles)
            samples_R = (samples[:,0]**2+samples[:,1]**2)**0.5
            samples[:,2], weights = sampleZPosition(samples_R, sph_model)
        else:                   # use uniformly distributed missing z component
            samples[:,2] = zmax*numpy.random.uniform(-1, 1, size=nsamples)
            weights *= 2*zmax

    if numpy.any(numpy.isnan(samples[:,3]+samples[:,4])):  # vx,vy are missing
        vtrans_mag = vmax * numpy.random.uniform(0, 1, size=nsamples)**0.5
        vtrans_ang = numpy.random.uniform(0, 2*numpy.pi, size=nsamples)
        samples[:,3] = vtrans_mag * numpy.cos(vtrans_ang)
        samples[:,4] = vtrans_mag * numpy.sin(vtrans_ang)
        vel_err[:,0:2] = 0    # don't add noise to vx,vy - they are already random

    if numpy.sum(vel_err)>0:  # add noise to the velocity components
        samples[:,3:6] += numpy.random.standard_normal((nsamples,3)) * \
            numpy.tile(vel_err, Nsubsamples).reshape(-1,3)

    #numpy.savetxt('samples.txt', numpy.hstack((samples, weights.reshape(-1,1))), fmt='%.7g')
    return samples, weights
