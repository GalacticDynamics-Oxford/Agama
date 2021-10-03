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

The spherical model deprojection is implemented in the class ``SphericalModel'',
and the resampling procedure -- in the routine ``sampleMissingData''
'''

import agama, numpy, scipy.integrate


class SphericalModel:
    '''
    Construct a spherically symmetric model using the information about projected particle positions;
    input: particles is a Nx(2 or more) array of coordinates, only the first two columns (x,y coordinates) are used
    '''
    def __init__(self, particles):
        Radii = (particles[:,0]**2 + particles[:,1]**2)**0.5
        Rmin,Rmax = numpy.percentile(Radii, [5, 95])
        GridSize  = max(int(round(numpy.log(len(Radii)))), 4)
        GridRadii = numpy.logspace(numpy.log10(Rmin), numpy.log10(Rmax), GridSize)
        self.spl_logSigma = agama.splineLogDensity(numpy.log(GridRadii), numpy.log(Radii), infLeft=True, infRight=True)

        def rho_integr(r):
            def integrand(xi):   # xi = ln(R)
                R2 = numpy.exp(2*xi)
                return numpy.exp(self.spl_logSigma(xi)) * (self.spl_logSigma(xi, 1)-2) / R2 / (R2-r**2)**0.5
            return -0.5 / numpy.pi**2 * scipy.integrate.quad(integrand, numpy.log(r), numpy.log(Radii[-1])+5., epsrel=1e-3)[0]

        # compute 3d density at the same radial grid
        rho_grid   = numpy.log([rho_integr(R) for R in GridRadii])
        good_elems = numpy.where(numpy.isfinite(rho_grid))[0]
        if(len(good_elems)<len(rho_grid)):
            print("Invalid density encountered at r=" + \
                str(GridRadii[numpy.where(numpy.logical_not(numpy.isfinite(rho_grid)))]))
        #print(numpy.column_stack((GridRadii, self.surface_density(GridRadii), numpy.exp(rho_grid))))
        rho_grid = rho_grid[good_elems]
        LogRadii = numpy.log(GridRadii[good_elems])

        # initialize an interpolating spline for 3d density (log-log scaled)
        spl_rho  = agama.CubicSpline(LogRadii, rho_grid)
        # check and correct endpoint log-slopes, if necessary
        slopein  = spl_rho(LogRadii[0], 1)
        slopeout = spl_rho(LogRadii[-1],1)
        SlopeIn  = max(-2.0, min(0.0, slopein))
        SlopeOut = min(-3.5, slopeout)
        print("Density slope: inner=%f [%f], outer=%f [%f]" % (SlopeIn, slopein, SlopeOut, slopeout))
        self.spl_rho = agama.CubicSpline(LogRadii, rho_grid, left=SlopeIn, right=SlopeOut)

    def surface_density(self, R):
        ''' Return surface density Sigma(R) '''
        return numpy.exp(self.spl_logSigma(numpy.log(R))) / (2*numpy.pi*R**2)

    def rho(self, r):
        ''' Return 3d density rho(r) '''
        return numpy.exp(self.spl_rho(numpy.log(r)))



##################### RESAMPLING OF ORIGINAL DATA TO FILL MISSING VALUES #################

def sampleZPosition(R, sph_model):
    '''
    Sample the missing z-component of particle coordinates
    from the density distribution given by the spherical model.
    input argument 'R' contains the array of projected radii,
    and the output will contain the z-values assigned to them,
    and the weights of individual samples.
    '''
    print('Assigning missing z-component of position')
    rho_max = sph_model.rho(R) * 2.0
    R0      = numpy.maximum(2., R)
    result  = numpy.zeros_like(R)
    weights = sph_model.surface_density(R)
    indices = numpy.where(result==0)[0]   # index array initially containing all input points
    while len(indices)>0:   # rejection sampling
        t = numpy.random.uniform(-1, 1, size=len(indices))
        z = R0[indices] * t/(1-t*t)**0.5
        rho = sph_model.rho( (R[indices]**2 + z**2)**0.5 )
        rho_bar = rho / (1-t*t)**1.5 / rho_max[indices]
        max_bar = numpy.amax(rho_bar)
        if max_bar>1:
            rho_max *= max_bar
            rho_bar /= max_bar
            print('Overflow by %f' % max_bar)
        assigned = numpy.where(numpy.random.uniform(size=len(indices)) < rho_bar)[0]
        #print('%i / %i' % (len(assigned), len(indices)))
        result [indices[assigned]]  = z[assigned]
        weights[indices[assigned]] /= rho[assigned]
        indices = numpy.where(result==0)[0]  # find out the unassigned elements
    return result, weights

def sampleMissingData(particles, Nsubsamples, fancy_z_assignment=True):
    '''
    Split each input particle into Nsamples samples, perturbing its velocity or
    assigning values for missing components of position/velocity.
    Input: particles -- array of Nx9 values (3 coordinates, 3 velocities, and 3 velocity errors),
    missing data (z, v_x, v_y, v_z) are indicated by NAN (may be different for each particle,
    but should have at least some particles with measured v_z, in order to estimate the velocity
    dispersion);
    Nsubsamples -- number of sub-samples created from each particle;
    fancy_z_assignment -- if True, use importance sampling for z-coordinate
    from a deprojected density profile; otherwise sample uniformly (not recommended)
    Return: two arrays -- NNx6 positions/velocities of subsamples and NN weights,
    where NN=len(particles)*Nsubsamples.
    '''
    numpy.random.seed(0)  # make resampling repeatable from run to run

    # duplicate the elements of the original particle array
    # (each particle is expanded into Nsubsamples identical samples)
    samples    = numpy.repeat(particles[:, 0:6], Nsubsamples, axis=0)
    nparticles = particles.shape[0]
    nsamples   = samples.shape[0]
    weights    = numpy.ones(nsamples, dtype=numpy.float64) / Nsubsamples
    vel_err    = particles[:, 6:9] if particles.shape[1]==9 else numpy.zeros((nparticles, 3))

    # compute maximum magnitude of distance and l.o.s. velocity used in assigning
    # missing z-coordinate and velocity components for resampled particles
    novz = numpy.isnan(particles[:,5])
    yesvz= numpy.logical_not(novz)
    novx = numpy.isnan(particles[:,3]+particles[:,4])
    yesvx= numpy.logical_not(novx)
    vmax = numpy.amax(numpy.abs(particles[yesvz,5]))
    vdisp= numpy.std(particles[yesvz,5])
    if not numpy.isfinite(vdisp):
        raise ValueError("No velocity (v_z) data!")
    Rmax = numpy.amax((particles[:,0]**2+particles[:,1]**2)**0.5)  # same here
    print('Resample %d input particles into %d internal samples (Rmax=%f, vmax=%f, sigma=%f)' % \
        (nparticles, nsamples, Rmax, vmax, vdisp))

    if numpy.any(numpy.isnan(samples[:,2])):   # z-coordinate is missing
        if fancy_z_assignment:  # use deprojection with unequal prior weights for resampled particles
            sph_model = SphericalModel(particles)
            samples_R = (samples[:,0]**2+samples[:,1]**2)**0.5
            samples[:,2], weights = sampleZPosition(samples_R, sph_model)
            weights  /= Nsubsamples
        else:                   # use uniformly distributed missing z component
            samples[:,2] = Rmax*numpy.random.uniform(-1, 1, size=nsamples)
            weights *= 2*Rmax

    # Sample missing vx,vy or add noise to existing measurements.
    # We sample the missing velocity components from a relatively heavy-tailed distribution
    # with the dispersion estimated from the known velocity (v_z) values.
    # Again, this is a kind of importance sampling with non-uniform weights assigned to the samples;
    # we use a heavy-tailed DF in order to reduce the Poisson noise from occasional (rare) outliers,
    # which are relatively more numerous (and hence less heavily weighted) in a heavy-tailed DF
    # compared to a Gaussian DF.
    # For particles with known (measured) components of velocity, these values are perturbed by
    # a Gaussian noise with the given dispersion (measurement errors), sampled with uniform weights.
    # Note that this would not be efficient if the errors (and measured values) are much larger than
    # the velocity dispersion -- in this case one would need to use importance sampling in order to
    # place more samples at smaller velocities, where we expect them to be located more probably.
    if numpy.any(novx):  # (some) vx,vy are missing
        sampnovx  = numpy.repeat(novx,  Nsubsamples)
        # resample vx,vy from a probability distribution ~ 1 / (vx^2+vy^2+1)^2
        cumul = numpy.random.uniform(0, 0.90, size=sum(sampnovx))  # 0 to 0.9 => 5.3-sigma region
        weights[sampnovx] *= (1-cumul)**-2
        vtrans_mag = 1.8 * vdisp * (cumul/(1-cumul))**0.5
        vtrans_ang = numpy.random.uniform(0, 2*numpy.pi, size=sum(sampnovx))
        samples[sampnovx,3] = vtrans_mag * numpy.cos(vtrans_ang)
        samples[sampnovx,4] = vtrans_mag * numpy.sin(vtrans_ang)
    if numpy.any(yesvx):
        sampyesvx = numpy.repeat(yesvx, Nsubsamples)
        samples[sampyesvx,3] += numpy.random.normal(size=sum(sampyesvx)) * numpy.repeat(vel_err[yesvx,0], Nsubsamples)
        samples[sampyesvx,4] += numpy.random.normal(size=sum(sampyesvx)) * numpy.repeat(vel_err[yesvx,1], Nsubsamples)
    # same for vz
    if numpy.any(novz):
        sampnovz  = numpy.repeat(novz,  Nsubsamples)
        # resample vz from Cauchy distribution
        cumul = numpy.random.uniform(0.05, 0.95, size=sum(sampnovz))  # +-6.3-sigma region
        value = numpy.tan( (cumul-0.5)*numpy.pi )
        weights[sampnovz] *= numpy.pi * (1 + value**2)
        samples[sampnovz,5] = value * vdisp
    if numpy.any(yesvz):
        sampyesvz = numpy.repeat(yesvz, Nsubsamples)
        samples[sampyesvz,5] += numpy.random.normal(size=sum(sampyesvz)) * numpy.repeat(vel_err[yesvz,2], Nsubsamples)

    return samples, weights/Nsubsamples
