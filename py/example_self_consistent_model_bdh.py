#!/usr/bin/python
"""
Example of construction of a three-component bulge-disk-halo equilibrium model of a disk galaxy.
This script is similar to example_self_consistent_model.py and example_self_consistent_model3.py
and uses the same iterative approach; the main differences are:

- The model has three components (though the bulge mass can be set to zero, making it just two),
and all components are specified by distribution functions (no static gas disk or central black hole).
Both spheroidal components (bulge and halo) use the QuasiSpherical DF constructed from their
density profiles with the (anisotropic) Eddington inversion formula, which after the iterations
produce a slightly flattened density distribution in response to the disk gravity, but have almost
the same mass profile as the initial one (see section 2.5.3 in reference.pdf for discussion).
The disk component is described by a QuasiIsothermal DF, and depending on its parameters,
may deviate somewhat from the initial density profile (more so if the disk is warmer).

- The parameters of models are provided directly in the script as arguments for createModel(),
rather than read from an INI file as in other examples.

- Instead of setting the physical length, velocity and mass scale of different components directly,
they are specified through derived properties, namely:
== the circular velocity vcirc = sqrt(dPhi/dR) at the fiducial radius 8 kpc is set to 240 km/s,
similar to the Milky Way.
== fracstars is the fraction of the acceleration dPhi/dR at this fiducial radius contributed by stars
(bulge+disk); it is similar but not identical to the fraction of stellar mass within this radius.
== fracbulge is the fraction of the bulge mass to the total stellar mass (not just within 8 kpc).
== the radial velocity dispersion of the disk component is determined by setting the lowest value
of the Toomre Q parameter, assuming an exponential profile of sigma_r = sigma_r,0 exp(-R/R_sigmaR).
In the final model, the surface density, velocity dispersion and epicyclic frequencies are somewhat
different from the assumed profiles, but the actual minimum value of Q is close to the prescribed one.
== the [constant] halo velocity anisotropy is given by beta0halo = 1 - sigma_t^2/(2 sigma_r^2),
while its rotation is controlled by the fraction of prograde orbits at large distances (rotfrac)
and the radius at which this fraction [nearly] reaches this asymptotic value (rhalorot),
which is translated into the Jphi0 parameter of the DF.
== other parameters, such as scale lengths and heights, are given directly.
The reason behind this choice is that it makes it easier to study the susceptibility of models
to bar instability; it is mainly driven by the stellar to total mass ratio (fracstars) and
depends on how warm the disk is (Qmin), and is also affected by the presence of the bulge
and the velocity structure of the halo.
See Fujii+2018 (MNRAS 477, 1451), Bland-Hawthorn+2023 (ApJ 947, 80), Chen&Shen 2025 (ApJ 990, 140),
Chen+2025 (ApJ 994, 124), Zheng+2603.12121, etc.

- The choice of units: without explicitly calling agama.setUnits, the code uses G=1,
and it is up to the user to interpret the models in physical units. Here we assume
a length unit = 1 kpc and velocity unit = 1 km/s, which implies a mass unit = 232500 Msun.

- An N-body representation of the self-consistent model is generated with the prescribed number
of stellar and halo particles, and the halo can optionally use a mass refinement scheme, in which
the inner regions have less massive particles than its outskirts (to reduce the overall number
of particles while keeping an adequate mass resolution in the disk-dominated region).
"""

import agama, numpy, os, sys, matplotlib.pyplot as plt

def createModel(
    fracstars,       # fraction of squared circular velocity at the fiducial radius contributed by stars (disk+bulge)
    fracbulge,       # fraction of stellar mass in the bulge component
    rbulge=1.0,      # cutoff radius for the bulge (if present)
    rdisk=2.5,       # scale radius for the disk
    hdisk=0.3,       # scale height of the sech^2 disk profile
    Qmin=1.5,        # minimum value of Toomre Q that indirectly sets the radial velocity dispersion in the disk
    rhalo=15.0,      # scale radius for the NFW halo
    rhalocutoff=200, # outer cutoff radius for the halo
    rhalorot=15.0,   # radius at which the rotation of the halo nearly reaches its asymptotic limit
    rotfrachalo=0.0, # amount of rotation in the halo (between -1 and +1) at large radii
    beta0halo=0.0,   # velocity anisotropy parameter of the halo (between -0.5 and 0.5)
    r0=8.0,          # fiducial radius
    vcirc0=240,      # circular velocity at the fiducial radius
    rsigmar=None,    # scale radius for the exponential decline of the radial velocity dispersion
    massrefinement=True, # whether to use multimass sampling for the halo to improve resolution in the inner part
    nstars = 200000, # number of particles in the stellar component (disk + bulge)
    nhalo  = 800000, # number of particles in the halo component
    ):
    name = 'fs%.2f_fb%.2f_Q%.1f' % (fracstars, fracbulge, Qmin)

    # DM halo is the standard NFW profile with a Gaussian cutoff to make its total mass finite
    haloparams  = dict(
    type              = 'Spheroid',
        gamma             = 1,
        beta              = 3,
        scaleRadius       = rhalo,
        outerCutoffRadius = rhalocutoff,
        cutoffStrength    = 2.0,
        densityNorm       = 1.0)

    # Bulge (if present) follows a simple power law profile rho~r^-1 with a Gaussian cutoff exp(-[r/rcut]^2)
    bulgeparams = dict(
        type              = 'Spheroid',
        gamma             = 1,    # inner slope
        beta              = 1,    # outer slope
        scaleRadius       = 1.0,  # does not matter since inner and outer slopes are identical
        outerCutoffRadius = rbulge,
        densityNorm       = 1.0)

    # Disk has a radially exponential, vertically sech^2 profile
    diskparams  = dict(
        type              = 'Disk',
        scaleRadius       = rdisk,
        scaleHeight       = -hdisk,
        surfaceDensity    = 1.0)

    # So far, the three components are "unnormalized", i.e. have arbitrary masses;
    # we now compute the circular velocity at the fiducial radius r0
    # and set the overall mass normalization so that this velocity equals vcirc0;
    # then normalize the halo such that it contributes a given fraction of the total
    # gravitational force at this radius, and split the remaining stellar mass
    # between the bulge and the disk in the given proportion.
    m0bulge = agama.Density(bulgeparams).totalMass()
    m0disk  = agama.Density(diskparams).totalMass()
    potbulge= agama.Potential(bulgeparams)
    potdisk = agama.Potential(diskparams)
    pothalo = agama.Potential(haloparams)
    g0stars = -r0 *(potdisk.force(r0,0,0)[0] / m0disk + potbulge.force(r0,0,0)[0] / m0bulge * fracbulge / (1-fracbulge))
    g0halo  = -r0 * pothalo.force(r0,0,0)[0]
    ahalo   = vcirc0**2 / g0halo * (1-fracstars)
    astars  = vcirc0**2 / g0stars* fracstars
    adisk   = astars / m0disk
    abulge  = astars / m0bulge * fracbulge / (1-fracbulge)
    haloparams ['densityNorm']   *= ahalo
    bulgeparams['densityNorm']   *= abulge
    diskparams['surfaceDensity'] *= adisk
    # create the total potential of all components, as well as potentials
    # of each component used only for visualizing the circular-velocity curve
    potbulge= agama.Potential(bulgeparams)
    potdisk = agama.Potential(diskparams)
    pothalo = agama.Potential(haloparams)
    pottotal= agama.Potential(bulgeparams, diskparams, haloparams)
    denstars= agama.Density(bulgeparams, diskparams)
    # recall that these numbers are given in fiducial N-body units of 2.325e5 Msun
    print('Component masses: bulge=%g, disk=%g, halo=%g (in units of 2.32e5 Msun)' %
        (potbulge.totalMass(), potdisk.totalMass(), pothalo.totalMass()))
    # show the radial profiles of the circular velocity (total and split by component) by dashed lines
    R = numpy.logspace(-1.0, 1.3, 32)
    xyz = numpy.column_stack((R, R*0, R*0))
    Sigma0 = denstars.projectedDensity(xyz[:,0:2])
    rho0 = denstars.density(xyz)
    ax = plt.subplots(2, 3, figsize=(12,6), dpi=100)[1].reshape(-1)
    plt.subplots_adjust(left=0.05, right=0.98, bottom=0.08, top=0.98, hspace=0.2, wspace=0.2)
    ax[0].plot(R, (-R * (potbulge.force(xyz)+potdisk.force(xyz))[:,0])**0.5, c='y', dashes=[4,2])
    ax[0].plot(R, (-R * potbulge.force(xyz)[:,0])**0.5, label='bulge', c='b', dashes=[4,2])
    ax[0].plot(R, (-R * potdisk .force(xyz)[:,0])**0.5, label='disk',  c='r', dashes=[4,2])
    ax[0].plot(R, (-R * pothalo .force(xyz)[:,0])**0.5, label='halo',  c='g', dashes=[4,2])
    ax[0].plot(R, (-R * pottotal.force(xyz)[:,0])**0.5, label='total', c='k', dashes=[4,2])
    ax[0].legend(loc='lower right', frameon=False, fontsize=10)
    ax[0].set_xlabel('R [kpc]')
    ax[0].set_ylabel('Vcirc [km/s]')
    ax[0].set_xlim(0, 20)
    ax[0].set_ylim(0, 250)
    plt.draw()
    plt.pause(0.1)

    # assume sigmar**2 = sigmar0**2 * exp(-r/rsigmar)
    sigmar0 = 1.0
    if rsigmar is None:
        rsigmar = 2*rdisk
    # determine the central velocity dispersion of the disk from requirement
    # that the Toomre parameter defined below (using only the disk component's density)
    # has a minimum value equal to the target Qmin.
    def toomreQ(r):
        force, deriv = pottotal.eval(numpy.column_stack([r, r*0, r*0]), acc=True, der=True)
        kappa = (-deriv.T[0] - 3*force.T[0]/r)**0.5
        Sigma = diskparams['surfaceDensity'] * numpy.exp(-r / diskparams['scaleRadius'])
        sigmar = sigmar0 * numpy.exp(-r / rsigmar)
        return sigmar * kappa / Sigma / 3.36
    r = numpy.logspace(-2, 2, 401)
    sigmar0 *= Qmin / min(toomreQ(r))

    # DFs of both spheroidal components are derived from their density profiles using the Eddington inversion
    dfbulge = agama.DistributionFunction(type='QuasiSpherical', potential=pottotal, density=agama.Density(bulgeparams))
    dfhalo  = agama.DistributionFunction(type='QuasiSpherical', potential=pottotal, density=agama.Density( haloparams),
        beta0=beta0halo, rotfrac=rotfrachalo, Jphi0=(-rhalorot**3*pottotal.force(rhalorot,0,0)[0])**0.5)
    # DF of the disk is a QuasiIsothermal family that aims to match the initial disk density profile as much as possible;
    # with an empirical correction factor for the disk scale radius in the DF,
    # which is set to be smaller than the original disk scale radius of the initial density profile.
    # This factor compensates the decrease in density in the inner part of the model,
    # compared to the initial density profile, which is stronger for warmer disks.
    rdisk_factor = 1.0 - 0.025 * Qmin
    dfdiskparams = dict(
        type     = 'QuasiIsothermal',
        potential= pottotal,
        Sigma0   = diskparams['surfaceDensity'],
        Rdisk    = rdisk * rdisk_factor,
        Hdisk    = hdisk,
        sigmar0  = sigmar0,
        Rsigmar  = rsigmar,
        sigmamin = sigmar0 * 0.01)
    # Because the DF does not exactly match the initial disk density profile,
    # we need to rescale its mass to match the target value
    dfdiskparams['Sigma0'] *= agama.Density(diskparams).totalMass() / agama.DistributionFunction(**dfdiskparams).totalMass()
    dfdisk = agama.DistributionFunction(**dfdiskparams)

    # Create the Self-Consistent Model object that implements the iterative construction process.
    # Its parameters need to be chosen so that the grids for representing the potential,
    # as well as corresponding grids for each component's density profile, cover the range of radii
    # spanned by these density profiles with a large safety factor (e.g. 10x the scale radius).
    model = agama.SelfConsistentModel(
    rminSph        = 0.10*rbulge,
    rmaxSph        = 500.0,
    sizeRadialSph  = 40,
    lmaxAngularSph = 6,
    RminCyl        = 0.05*rbulge,
    RmaxCyl        = 10*rdisk,
    sizeRadialCyl  = 25,
    zminCyl        = 0.2*hdisk,
    zmaxCyl        = 10*rdisk,
    sizeVerticalCyl= 25)
    model.potential  = pottotal  # initial potential
    if fracbulge>0:
        model.components.append(agama.Component(
            df             = dfbulge,
            disklike       = False,
            rminSph        = 0.10*rbulge,
            rmaxSph        = 5*rbulge,
            sizeRadialSph  = 20,
            lmaxAngularSph = 6))
    else:
        model.components.append(agama.Component(  # quasi-empty component
            density=agama.Density(type='plummer', mass=0),
            disklike=False))
    model.components.append(agama.Component(
        df              = dfdisk,
        disklike        = True,
        rminCyl         = 0.10*rbulge,
        rmaxCyl         = 8*rdisk,
        sizeRadialCyl   = 20,
        zminCyl         = 0.3*hdisk,
        zmaxCyl         = 10*hdisk,
        sizeVerticalCyl = 12))
    model.components.append(agama.Component(
        df             = dfhalo,
        disklike       = False,
        rminSph        = 0.10*rbulge,
        rmaxSph        = 500.0,
        sizeRadialSph  = 25,
        lmaxAngularSph = 6))

    # Perform a fixed number of iterations, which is usually sufficient for a good convergence of the model
    for iteration in range(4):
        print("\033[1;37mStarting iteration #%d\033[0m" % iteration)
        model.iterate()
        # print some diagnostic information after each iteration
        densBulge= model.components[0].density
        densDisk = model.components[1].density
        densHalo = model.components[2].density
        pt0 = (8.0, 0, 0.0)
        pt1 = (8.0, 0, 1.0)
        pt2 = (1.0, 0, 0.0)
        pt3 = (0.0, 0, 8.0)
        print("Disk  total mass=%g, rho(R=%g,z=%g)=%g, rho(R=%g,z=%g)=%g" %
            (densDisk.totalMass(), pt0[0], pt0[2], densDisk.density(pt0), pt1[0], pt1[2], densDisk.density(pt1)))
        print("Bulge total mass=%g, rho(R=%g,z=%g)=%g" %
            (densBulge.totalMass(), pt2[0], pt2[2], densBulge.density(pt2)))
        print("Halo  total mass=%g, rho(R=%g,z=%g)=%g, rho(R=%g,z=%g)=%g" %
            (densHalo.totalMass(), pt0[0], pt0[2], densHalo.density(pt0), pt3[0], pt3[2], densHalo.density(pt3)))
        print("Escape speed at origin=%g, total mass=%g" %
             ((-2*model.potential.potential(0,0,0))**0.5, model.potential.totalMass()))

    # replace the initial potential with the converged one and replot the circular-velocity curves
    pottotal = model.potential
    potbulge = agama.Potential(type='multipole', density=model.components[0].density)
    pothalo  = agama.Potential(type='multipole', density=model.components[2].density)
    potdisk  = pottotal[1]  # only the disk component / CylSpline
    ax[0].plot(R, (-R * (potbulge.force(xyz)+potdisk.force(xyz))[:,0])**0.5, c='y')
    ax[0].plot(R, (-R * potbulge.force(xyz)[:,0])**0.5, label='bulge', c='b')
    ax[0].plot(R, (-R * potdisk .force(xyz)[:,0])**0.5, label='disk',  c='r')
    ax[0].plot(R, (-R * pothalo .force(xyz)[:,0])**0.5, label='halo',  c='g')
    ax[0].plot(R, (-R * pottotal.force(xyz)[:,0])**0.5, label='total', c='k')
    ax[0].text(0.02, 0.02, 'dashed - init, solid - final', ha='left', va='bottom', transform=ax[0].transAxes, fontsize=10)
    plt.draw()
    plt.pause(0.1)

    print("\033[1;33mCreating an N-body representation of the model\033[0m")
    nbulge = int(round(nstars * fracbulge))
    ndisk  = nstars - nbulge
    print("Sampling disk DF")
    xvd, md = agama.GalaxyModel(potential=pottotal, df=dfdisk, af=model.af).sample(ndisk)
    if fracbulge>0:
        print("Sampling bulge DF")
        xvb, mb = agama.GalaxyModel(potential=pottotal, df=dfbulge, af=model.af).sample(nbulge)
    else:
        xvb = numpy.zeros((0,6))
        mb  = numpy.zeros(0)

    print("Sampling halo DF")
    # if requested, create a halo with several mass groups, with the central region occupied by
    # lighter and more numerous particles, and progressively coarse-graining them further out
    if massrefinement:
        # input range => output range;  mscale  count
        # 0     - 0.100  0     - 0.100   1      40%
        # 0.100 - 0.325  0.100 - 0.175   3      30%
        # 0.325 - 1.000  0.175 - 0.250   9      30%
        groups  = numpy.array([0.1, 0.225, 0.675])
        massmul = numpy.array([1.0, 3.000, 9.000])
        assert numpy.isclose(sum(groups), 1)
        nhalo_total = int(round(nhalo / sum(groups / massmul)))
        nhalo_groups_in  = numpy.round(nhalo_total * groups).astype(int)
        nhalo_groups_out = numpy.round(nhalo_total * groups / massmul).astype(int)
        # define the function used to sort particles into different mass groups:
        # here we use particle energy, but any other function of 6d phase-space coords can be used
        def sorterfnc(xv):
            return numpy.nan_to_num(pottotal.potential(xv[:,0:3]) + 0.5 * numpy.sum(xv[:,3:6]**2, axis=1))
        # mass refinement for halo particles based on energy:
        # first, sample a small number of particles, sort them in energy,
        # and determine the thresholds for all mass groups
        xvh = agama.GalaxyModel(potential=pottotal, df=dfhalo, af=model.af).sample(nhalo//10)[0]
        sorter = sorterfnc(xvh)
        thresholds = numpy.hstack((-numpy.inf, numpy.percentile(sorter, tuple(100*numpy.cumsum(groups))[:-1]), numpy.inf))
        # next, define a selection function that takes the particle coordinates as input
        # and returns values <=1, inversely proportional to mass multipliers for each particle's group
        def sf(xv):
            sorter = sorterfnc(xv)
            group = numpy.searchsorted(thresholds, sorter) - 1  # index of the mass group for each particle
            return 1 / massmul[group]
        xvh, mh = agama.GalaxyModel(potential=pottotal, df=dfhalo, af=model.af, sf=sf).sample(nhalo)
        # compensate the lower selection probability for particles in the outer region
        # by assigning a proportionally higher mass to these particles
        mh /= sf(xvh)
        mh_all = numpy.unique(mh)
        print('Refinement:   energy  massmul  count')
        for g in range(len(groups)):
            print('%8g to %8g %6g %8i' % (thresholds[g], thresholds[g+1], massmul[g], numpy.sum(mh == mh_all[g])))
    else:
        xvh, mh = agama.GalaxyModel(potential=pottotal, df=dfhalo, af=model.af).sample(nhalo)
    print('particle mass: bulge=%g, disk=%g, halo=%s' % ( (mb[0] if fracbulge>0 else 0), md[0], numpy.unique(mh)))

    # save the N-body snapshot in the NEMO format (one can also use text or Gadget formats)
    snap = (numpy.vstack((xvb, xvd, xvh)), numpy.hstack((mb, md, mh)))
    agama.writeSnapshot(name+'.nemo', snap, 'nemo')

    # the remaining part computes and plots various diagnostics
    print("\033[1;33mComputing density and velocity profiles\033[0m")
    gm = agama.GalaxyModel(potential=pottotal,
        df=agama.DistributionFunction(agama.DistributionFunction(dfdisk, dfbulge), dfhalo),  # stars and halo separately
        af=model.af)

    # rotation velocity and three components of velocity dispersion for disk+bulge
    Sigma = gm.moments(xyz[:,0:2], vel2=False, separate=True)[:,0]  # projected density moment for stars only
    rho,vel,sigma = gm.moments(xyz, dens=True, vel=True, vel2=True, separate=True)
    ax[1].plot(R, sigma[:,0,0]**0.5, label=r'$\sigma_{r}$', c='g')
    ax[1].plot(R,(sigma[:,0,1]-vel[:,0,1]**2)**0.5, label=r'$\sigma_{\phi}$', c='r')
    ax[1].plot(R, sigma[:,0,2]**0.5, label=r'$\sigma_{z}$', c='b')
    ax[1].plot(R, vel[:,0,1], label=r'$\overline{v}_\phi$', c='k')
    ax[1].legend(loc='center right', frameon=False, fontsize=10)
    ax[1].set_xlim(0, 20)
    ax[1].set_ylim(0, 250)
    ax[1].set_xlabel('R [kpc]')
    ax[1].text(0.5, 0.99, 'stars', ha='center', va='top', transform=ax[1].transAxes)

    # same for the halo
    ax[2].plot(R, sigma[:,1,0]**0.5, label=r'$\sigma_{r}$', c='g')
    ax[2].plot(R,(sigma[:,1,1]-vel[:,1,1]**2)**0.5, label=r'$\sigma_{\phi}$', c='r')
    ax[2].plot(R, sigma[:,1,2]**0.5, label=r'$\sigma_{z}$', c='b')
    ax[2].plot(R, abs(vel[:,1,1]), label=r'$\overline{v}_\phi$' if rotfrachalo>=0 else r'$-\overline{v}_\phi$', c='k')
    ax[2].legend(loc='lower right', frameon=False, fontsize=10)
    ax[2].set_xlim(0, 20)
    ax[2].set_ylim(0, 250)
    ax[2].set_xlabel('R [kpc]')
    ax[2].text(0.5, 0.99, 'halo', ha='center', va='top', transform=ax[2].transAxes)

    # 3d and surface density as functions of radius
    ax[3].plot(R, Sigma, c='r', label='surface density of stars')
    ax[3].plot(R, rho[:,0], c='b', label='3d density of stars @z=0')
    ax[3].plot(R, rho[:,1], c='g', label='3d density of halo @z=0')
    ax[3].legend(loc='lower left', frameon=False, fontsize=10)
    ax[3].set_xlabel('R [kpc]')
    ax[3].set_xlim(0, 20)
    ax[3].set_yscale('log')
    ax[3].set_ylim(1, 1e4)

    # ratios of resulting to "expected" (initial) values of 3d and surface density and radial velocity dispersion
    force, deriv = pottotal.eval(xyz, acc=True, der=True)
    kappa = (-deriv[:,0] - 3*force[:,0] / R)**0.5
    ToomreQ = sigma[:,0,0]**0.5 * kappa / 3.36 / Sigma
    ax[4].plot(R, Sigma  / Sigma0, c='r', label=r'$\Sigma/\Sigma_{\rm init}$')
    ax[4].plot(R, rho[:,0] / rho0, c='b', label=r'$\rho_{z=0} / \rho_{\rm init}$')
    ax[4].plot(R, sigma[:,0,0]**0.5 / (sigmar0 * numpy.exp(-R/rsigmar)), c='g', label=r'$\sigma_r / \sigma_{\rm init}$')
    ax[4].plot(R, ToomreQ, c='k', label=r'${\rm Toomre}\, Q$')
    ax[4].set_xlabel('R [kpc]')
    ax[4].set_xlim(0, 20)
    ax[4].set_ylim(0, 3)
    ax[4].legend(loc='upper right', frameon=False, fontsize=10)

    # radial distribution of particles of different masses
    # (all star particles have the same mass, but halo may be split into several mass groups when mass_refinement=True)
    rs = numpy.sum(numpy.vstack((xvb, xvd))[:,0:3]**2, axis=1)**0.5
    rh = numpy.sum(xvh[:,0:3]**2, axis=1)**0.5
    gridr = numpy.logspace(-1, 3, 81)
    hs = numpy.histogram(rs, bins=gridr, weights=numpy.hstack((mb, md)))[0]
    ax[5].plot(numpy.repeat(gridr, 2)[1:-1], numpy.repeat(hs, 2), label='stars (Mpart=%.3g)' % md[0], color='y')
    maxh = max(hs)
    for i,particle_mass in enumerate(numpy.unique(mh)):
        filt = mh == particle_mass
        hh = numpy.histogram(rh[filt], bins=gridr, weights=mh[filt])[0]
        maxh = max(maxh, max(hh))
        ax[5].plot(numpy.repeat(gridr, 2)[1:-1], numpy.repeat(hh, 2), label='halo (Mpart=%.3g)' % particle_mass, color='rgb'[i])
    ax[5].legend(loc='lower right', frameon=False, fontsize=10)
    ax[5].set_xscale('log')
    ax[5].set_yscale('log')
    ax[5].set_xlim(min(gridr), max(gridr))
    ax[5].set_ylim(maxh*2e-4, maxh*2)
    ax[5].set_xlabel('r [kpc]')
    ax[5].set_ylabel(r'$dM/d\log r$')

    plt.savefig(name+'.pdf')
    plt.draw()
    plt.pause(0.1)
    plt.ioff()
    plt.show()


# now can create a bunch of models with different properties and see how they evolve
# (in many cases, develop a bar, but this does not mean the model was not in equilibrium initially)

createModel(fracstars=0.50, fracbulge=0.1, Qmin=2.0, rsigmar=8.0, massrefinement=False)
#createModel(fracstars=0.40, fracbulge=0.0, Qmin=1.5, rsigmar=8.0, hdisk=0.2)
