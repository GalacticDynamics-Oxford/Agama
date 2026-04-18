#!/usr/bin/python
'''
Construct a self-consistent DF-based model of the Milky Way nuclear star cluster (NSC)
embedded into the nuclear stellar disk (NSD) and with a supermassive black hole at the center,
as described in Vasiliev+2026 (ApJ, in press; arXiv:2603.29502).
When this script is run for the first time, it creates the model described by the DF parameters
using the iterative self-consistent method, stores its potential and density profiles into files,
constructs 3-dimensional interpolators for the velocity distribution functions (VDFs) across
the image plane for each velocity dimension, and also stores them into a file.
On subsequent runs, these profiles are loaded from these files to skip the iterative procedure.
It then launches an interactive plot showing the model properties - the surface density profile
and velocity distributions (VDFs) in three dimensions, along with the stellar kinematic catalogues.
In the interactive plot, one can move the cursor around the 2d edge-on view of the NSC within 10 pc,
and the three panels on the right show the velocity distributions of the model at the given location,
separately for the NSC, NSD and large-scale bar components. The overplotted histograms in these panels
show velocities of stars in the circular region shown in gray in the main panel, which contains
the given number of stars (by default 100, but this number and thus the radius of the region can be
adjusted by scrolling the mouse wheel). The model VDFs are also averaged over this spatial region,
and if one reduces the number of stars to zero, histograms are removed and the model VDFs are shown
at a single point under the cursor. One can also toggle the vertical scale of these histograms between
linear and logarithmic by pressing "l" while the cursor is in any of the right panels.
Note that in the current version, the data shown in histograms are generated from a fiducial model
rather than represent the actual observational dataset, since the latter is not yet fully published.
Author: Eugene Vasiliev
Date: Mar 2026
'''
import agama, numpy, time, matplotlib, matplotlib.pyplot as plt, scipy.interpolate, scipy.spatial, scipy.stats

# Define units:
# [L] = pc
# [V] = km/s
# [M] = 10^6 Msun
unitMass = 1e6
agama.setUnits(length=1e-3, velocity=1, mass=unitMass)
distanceToGC = 8200.0  # distance to the Galactic centre in pc
masyr_to_kms = 4.74 * distanceToGC*1e-3  # ~40
kms_to_masyr = 1 / masyr_to_kms
arcsec_to_pc = distanceToGC / (180/numpy.pi * 3600)  # ~0.04
pc_to_arcsec = 1 / arcsec_to_pc


def createModel(params):
    """
    Construct a two-component self-consistent model of the NSC + NSD (plus the central black hole
    as the 0th component) with parameters provided as a dict (all of them except Mbh refer to the NSC DF).
    """
    def printoutInfo(it):
        print(('Iteration #%i. NSC / NSD density at 1 pc: %6.0f / %6.0f Msun/pc^3') % (it,
            scm.components[1].density.density(1,0,0) * 1e6,
            scm.components[2].density.density(1,0,0) * 1e6))

    # grid parameters for both NSC and NSD components
    paramsNSC = dict(
        rminSph        = 0.01,
        rmaxSph        = 1000,
        sizeRadialSph  = 26,
        lmaxAngularSph = 6,
        disklike       = False)
    paramsNSD = dict(
        rminSph        = 0.1,
        rmaxSph        = 1000,
        sizeRadialSph  = 25,
        lmaxAngularSph = 12,
        # although the NSD is certainly a disk, we put it into the same potential as the NSC by setting disklike=False,
        # and save 50% of computational cost by having only one potential (Multipole) instead of two (Multipole&CylSpline);
        # the errors in the potential from the sub-optimal potential of the disk are only noticeable at large radii,
        # where we don't have kinematic data anyway
        disklike       = False)

    # initialize the SelfConsistentModel object (only the potential expansion parameters)
    scm = agama.SelfConsistentModel(
        rminSph        = min(paramsNSC['rminSph'], paramsNSD['rminSph'])*0.5,
        rmaxSph        = max(paramsNSC['rmaxSph'], paramsNSD['rmaxSph'])*2.0,
        sizeRadialSph  = 50,
        lmaxAngularSph = max(paramsNSC['lmaxAngularSph'], paramsNSD['lmaxAngularSph']),
        verbose        = False,
    )
    t0 = time.time()

    ##############################
    # construct a three component model: NSD + NSC + BH
    # NSC -> generated self-consistently, fit best fit parameters
    # NSD -> generated self-consistently, best fit parameters from Sormani+22
    # BH  -> kept fixed as an external potential
    ##############################
    # note: (x,z) axis are projected axes on the plane of the sky, y axis is parallel to line-of-sight

    # NSC best-fitting model from Chatzopoulos+2015 (Equation 28 in Sormani+2020)
    density_NSC_init = agama.Density(type='Dehnen', mass=61, gamma=0.71, scaleRadius=5.9, axisRatioZ=0.73)

    # NSD model 3 from Sormani+2020 (Equation 24)
    density_NSD_init = agama.Density(
        agama.Density(type='Spheroid', DensityNorm=0.9*222.885e-5, gamma=0, beta=0, axisRatioZ=0.37, outerCutoffRadius=5.06, cutoffStrength=0.719),
        agama.Density(type='Spheroid', DensityNorm=0.9*169.975e-5, gamma=0, beta=0, axisRatioZ=0.37, outerCutoffRadius=24.6, cutoffStrength=0.793) )

    Mbh = params['Mbh']
    del params['Mbh']
    potential_BH_init = agama.Potential(type='Plummer', mass=Mbh, scaleRadius=0)
    scm.components.append(agama.Component(potential=potential_BH_init, disklike=False))

    # add NSC and NSD as static density profiles to begin with (this is needed to compute the initial potential)
    scm.components.append(agama.Component(density=density_NSC_init, disklike=False))
    scm.components.append(agama.Component(density=density_NSD_init, disklike=False))

    # compute the initial guess for the potential
    scm.iterate()

    dfNSC = agama.DistributionFunction(type='DoublePowerLaw', Jcutoff=1e4, **params)
    # replace the static density of the NSC with a DF-based component
    scm.components[1] = agama.Component(df=dfNSC, **paramsNSC)

    # NSD DF parameters are taken from Sormani+2022 without any changes
    massNSD  = 970.     #in 1e6 Msun
    Rdisk    = 75.      #in pc
    Hdisk    = 25.      #in pc
    sigmar0  = 75.0     #in km/s
    Rsigmar  = 1000.    #in pc
    sigmamin = 2.0      #in km/s
    Jmin     = 10000.   #in pc * km/s
    dfNSD = agama.DistributionFunction(potential=scm.potential, type='QuasiIsothermal',
        mass=massNSD, Rdisk=Rdisk, Hdisk=Hdisk, sigmar0=sigmar0, Rsigmar=Rsigmar, sigmamin=sigmamin, Jmin=Jmin)

    filename_pot = 'example_mw_nsc_potential.ini'
    filename_den = 'example_mw_nsc_density.ini'
    # if this script has been run previously, the final model can be initialized using these files, skipping iterations
    try:
        scm.potential = agama.Potential(filename_pot)
        densities = agama.Density(filename_den)
        scm.components[1] = agama.Component(density=densities[0], df=dfNSC, **paramsNSC)
        scm.components[2] = agama.Component(density=densities[1], df=dfNSD, **paramsNSD)
    except RuntimeError:
        # Use a fixed density profile for the NSD component for the first few iterations,
        # and then replace it with a DF-based component (best-fitting model of Sormani+2022) on the last iteration.
        # This profile is a better match for the NSD in the final model than the Chatzopoulos+2015 model defined earlier.
        numIter = 4
        density_NSD_init = agama.Density(
            agama.Density(type='spheroid', mass=40.0, outerCutoffRadius=33, cutoffStrength=1, gamma=1, beta=1),
            agama.Density(type='spheroid', mass=1060, outerCutoffRadius=80, cutoffStrength=1, gamma=0, beta=0, axisRatioZ=0.3) )
        scm.components[2] = agama.Component(density=density_NSD_init, disklike=False)
        # iterate to make NSC (and later NSD) DF and the total potential self-consistent
        for it in range(1, numIter+1):
            if it == numIter:
                scm.components[2] = agama.Component(df=dfNSD, **paramsNSD)
            scm.iterate()
            printoutInfo(it)
        # store the final model's potential and density profiles
        scm.potential.export(filename_pot)
        with open(filename_pot, 'r') as f:
            lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('#other parameters are not stored'):
                lines[i] = 'mass=%.15g\nscaleradius=%g\n' % (Mbh, 0)
        with open(filename_pot, 'w') as f:
            f.write(''.join(lines))
        agama.Density(scm.components[1].density, scm.components[2].density).export(filename_den)
        print('Self-consistent model constructed in %.3g seconds' % (time.time() - t0))

    return scm


def plotModel(scm):
    """
    Show various summary plots in four panels.
    first  panel: enclosed mass profiles for SMBH, NSC and NSD as functions of radius.
    second panel: 3d and surface density profiles of both NSC and NSD as functions of radius
                  (represent the same information in a different way).
    third  panel: radial profile of the axis ratio for the NSC.
    fourth panel: mean rotation and velocity dispersion profiles in three coordinates.
    """
    ax = plt.subplots(1, 4, figsize=(16, 4))[1]
    plotRadius = numpy.logspace(-2.0, 2.0, 25)
    xyz = numpy.column_stack([plotRadius, plotRadius*0, plotRadius*0])  # x,y,z along the major axis

    # enclosed mass profile (converted to Msun)
    massBH  = unitMass * scm.components[0].potential.totalMass()
    massNSC = unitMass * scm.components[1].density.enclosedMass(plotRadius)
    massNSD = unitMass * scm.components[2].density.enclosedMass(plotRadius)
    ax[0].plot(plotRadius, plotRadius*0 + massBH, label='SMBH', color='k')
    ax[0].plot(plotRadius, massNSC, label='NSC', color='r')
    ax[0].plot(plotRadius, massNSD, label='NSD', color='b')
    ax[0].plot(plotRadius, massBH + massNSC + massNSD, label='total', color='gray')
    ax[0].set_xscale('log')
    ax[0].set_xlim(min(plotRadius), max(plotRadius))
    ax[0].set_xlabel('radius [pc]')
    ax[0].set_ylabel(r'enclosed mass [$M_\odot$]')
    ax[0].set_yscale('log')
    ax[0].set_ylim(1e6, 2e8)
    ax[0].legend(loc='upper left', frameon=False, fontsize=10)

    # density profiles (3d and projected)
    densNSC = scm.components[1].density
    densNSD = scm.components[2].density
    ax[1].plot(plotRadius, unitMass * densNSC.density(xyz), label=r'$\rho_{\sf NSC}$', color='r')
    ax[1].plot(plotRadius, unitMass * densNSD.density(xyz), label=r'$\rho_{\sf NSD}$', color='b')
    ax[1].plot(plotRadius, unitMass * densNSC.projectedDensity(xyz[:,0:2], beta=numpy.pi/2), label=r'$\Sigma_{\sf NSC}$', color='r', dashes=[4,3])
    ax[1].plot(plotRadius, unitMass * densNSD.projectedDensity(xyz[:,0:2], beta=numpy.pi/2), label=r'$\Sigma_{\sf NSD}$', color='b', dashes=[4,3])
    ax[1].set_xscale('log')
    ax[1].set_xlim(min(plotRadius), max(plotRadius))
    ax[1].set_xlabel('radius [pc]')
    ax[1].set_ylabel(r'3d and surface density [$M_\odot/{\sf pc}^{\sf 3\,or\, 2}$]')
    ax[1].set_yscale('log')
    ax[1].set_ylim(1e2, 4e6)
    ax[1].plot([0.1, 1], [1e5*0.1**-1.5, 1e5], dashes=[1.5,1.0], color='k') #, label='$r^{-3/2}$')
    ax[1].text(0.2, 4e5, '$r^{-3/2}$', ha='center', va='center', rotation=-55)
    ax[1].legend(loc='upper right', frameon=False, fontsize=10)

    # NSC axis ratio
    principalAxes, principalAngles = densNSC.principalAxes(plotRadius)
    axisRatioNSC = (principalAxes[:,2] / principalAxes[:,0])**numpy.where(principalAngles[:,0]==0, 1, -1)
    ax[2].plot(plotRadius, axisRatioNSC, color='r', clip_on=False)
    ax[2].set_xscale('log')
    ax[2].set_xlim(min(plotRadius), max(plotRadius))
    ax[2].set_xlabel('radius [pc]')
    ax[2].set_ylabel('NSC axis ratio')
    ax[2].set_ylim(0, 1)

    # velocity disperion profiles
    gm = agama.GalaxyModel(scm.potential, agama.DistributionFunction(scm.components[1].df, scm.components[2].df))
    vel, vel2 = gm.moments(xyz[:,0:2], dens=False, vel=True, vel2=True, beta=numpy.pi/2)
    ax[3].plot(plotRadius, vel2[:,0]**0.5, color='b', label=r'$\sigma_{l}$')
    ax[3].plot(plotRadius, vel2[:,1]**0.5, color='r', label=r'$\sigma_{b}$')
    ax[3].plot(plotRadius,(vel2[:,2]-vel[:,2]**2)**0.5, color='g', label=r'$\sigma_{\rm los}$')
    ax[3].plot(plotRadius,-vel [:,2], color='c', label=r'$\overline{v_{\rm los}}$')
    ax[3].set_xscale('log')
    ax[3].set_xlim(min(plotRadius), max(plotRadius))
    ax[3].set_xlabel('radius [pc]')
    ax[3].set_ylabel('NSC velocity [km/s]')
    ax[3].set_yscale('log')
    ax[3].set_ylim(20, 500)
    ax[3].set_yticks([100], minor=False)
    ax[3].set_yticklabels(['100'], minor=False)
    ax[3].set_yticks([20,30,40,50,60,70,80,90,200,300,500], minor=True)
    ax[3].set_yticklabels(['20','30','','50','','','','','200','300','500'], minor=True)
    ax[3].legend(loc='upper right', frameon=False, fontsize=10)

    # show a secondary horizontal axis on top, marked in arcseconds instead of parsecs
    for a in ax:
        ta = a.twiny()
        ta.set_xscale('log')
        ta.set_xlim(min(plotRadius) * pc_to_arcsec, max(plotRadius) * pc_to_arcsec)
        ta.set_xlabel('radius [arcsec]')
    plt.tight_layout()
    plt.savefig('example_mw_nsc_profiles.pdf')


def fitMGE(density, sigma, plot=False):
    """
    Approximate an axisymmetric density profile with a multi-Gaussian expansion.
    Arguments:
        density:  instance of agama.Density.
        sigma:  lengths of major axes of the Gaussian components (typically should be log-spaced).
        plot:  a flag requesting to show a diagnostic plot with fit results (off by default).
    Return:
        amplitudes:  array of masses of Gaussian components.
        q:  array of corresponding flattenings (z/x axis ratios).
    """
    # nodes of GL quadrature for lmax=12
    angles = numpy.array([0, 0.232548650694442, 0.465078260546407, 0.697559716854733, 0.92993000032151, 1.161996089452876, 1.39270186866862])
    gridr  = numpy.logspace(numpy.log10(min(sigma))-0.2, numpy.log10(max(sigma))+0.1)
    xyz = numpy.vstack([numpy.column_stack((gridr * cosa, gridr*0, gridr * sina))
        for sina, cosa in zip(numpy.sin(angles), numpy.cos(angles))])
    log_rho_true = numpy.log(density.density(xyz))
    a,b,c = density.principalAxes(sigma)[0].T
    qinit = c/a
    ainit = numpy.log(sigma**3 * qinit * density.density(numpy.column_stack((sigma, sigma*0, sigma*0)))) + 3
    def logrhoappr(param):
        ampl = numpy.exp(param[:len(sigma)])
        q = param[len(sigma):] if len(param) > len(sigma) else qinit
        log_rho_appr = numpy.log(sum((
            ampl[i] * numpy.exp(-0.5 * (xyz[:,0]**2 + (xyz[:,2] / q[i])**2) / sigma[i]**2) / (sigma[i]**3 * abs(q[i]))
            for i in range(len(sigma)) )) / (2*numpy.pi)**1.5)
        return log_rho_appr
    def fitfnc(param):
        return numpy.nan_to_num(logrhoappr(param) - log_rho_true)
    result = scipy.optimize.leastsq(fitfnc, ainit)[0]  # first fit amplitudes only, with fixed q
    result = scipy.optimize.leastsq(fitfnc, numpy.hstack((result, qinit)))[0]  # then release q and optimize further
    ampl = numpy.exp(result[:len(sigma)])
    q = abs(result[len(sigma):])
    if plot:
        log_rho_appr = logrhoappr(result)
        ax0, ax1 = plt.subplots(1, 2, figsize=(12,5))[1]
        for i in range(3):
            color = 'rgb'[i]
            j = i*3
            ax0.plot(gridr, numpy.exp(log_rho_true[j*len(gridr):(j+1)*len(gridr)]), color=color, dashes=[4,3])
            ax0.plot(gridr, numpy.exp(log_rho_appr[j*len(gridr):(j+1)*len(gridr)]), color=color)
        for k in range(len(sigma)):
            ax0.plot(gridr, ampl[k] / ((2*numpy.pi)**1.5 * sigma[k]**3 * q[k]) *
                numpy.exp(-0.5 * (xyz[:len(gridr),0]**2 + (xyz[:len(gridr),2] / q[k])**2)/sigma[k]**2),
                color='r', lw=0.5)
        ax0.set_xscale('log')
        ax0.set_yscale('log')
        ax0.set_ylim(1e-7,1e3)
        ax1.loglog(sigma, ampl, color='r')
        ax2 = ax1.twinx()
        ax2.plot(sigma, q, color='b')
        plt.show()
    return ampl, q


def sampleZfromMGE(sigma, ampl, q, X, Y, eta):
    r"""
    Sample the missing Z-coordinate for a point in the sky plane.
    Arguments:
        sigma, ampl, q:  parameters of the K-component axisymmetric MGE density profile to sample from,
        assumed to be observed edge-on.
        X, Y:  coordinates in the sky plane (scalars).
        eta:  array of random numbers between 0 and 1 (length S), which represent the fraction
        of cumulative density along the LoS for each point.
    Return:
        - Z_s: array of length S with equally likely sampled Z-values.
        - W_s: array of the same length with associated weights for each sample, defined as follows:
        W[s] = S^-1 Sigma(R) / rho(R, Z_{s}) .
        In other words, the integral of an arbitrary function f(R, Z) over z at a fixed R
        can be approximated by
        \int_{-\infty}^{\infty} f(R, z) dz = \sum_{s=0}^{S-1} f(R, Z_s) W_s
    """
    K = len(sigma)
    assert len(ampl) == K
    Sigma = ampl * numpy.exp(-0.5 * (X**2 + (Y/q)**2) / sigma**2) / (2*numpy.pi * sigma**2 * q)
    csum  = numpy.cumsum(Sigma)
    k = numpy.searchsorted(csum, eta * csum[-1])
    v = (eta * csum[-1] - numpy.hstack([0, csum[:-1]])[k]) / Sigma[k]
    assert numpy.all( (v>=0) & (v<=1) )
    # convert a uniformly distributed sample in [0:1) into a normally distributed one
    Z = sigma[k] * 2**0.5 * scipy.special.erfinv(v*2-1)
    rho = numpy.sum(ampl * numpy.exp(-0.5 * (X**2 + (Y/q)**2 + Z[:,None]**2) / sigma**2) /
        (2*numpy.pi * sigma**2)**1.5 / q, axis=1)
    return Z, csum[-1] / rho


def qrngHalton(size, base, scramble=True):
    """
    Generate an array of quasi-random numbers from the Halton sequence (from 0 to size-1).
    'base' should be a prime number, and sequences for different bases can be used for
    separate columns of a 2d array of quasi-random numbers.
    'scramble' turns on randomization and Owen scrambling (better statistical properties).
    """
    numiter = int(numpy.ceil(54 / numpy.log2(base)) - 1)
    vals = numpy.zeros(size)
    facs = numpy.ones(size)
    inds = numpy.arange(size)
    for j in range(numiter):
        perm = numpy.arange(base)
        if scramble:
            numpy.random.shuffle(perm)
        facs /= base
        vals += facs * perm[inds % base]
        inds //= base
    return vals


def createVDFinterpolators(scm, numSamplesVDF=100000, gridSizeR=15, gridSizeA=4):
    """
    Collect data for plotting the velocity distribution functions at any point in the sky plane.
    The first step is to create a grid in the X-Y (sky) plane, which is equivalent to the y-z plane
    of the model, with the Z direction (line of sight) corresponding to the x direction in the model.
    At each point in this 15 x 4 grid, a large number of samples (10^5) cover the remaining dimensions:
    distance (Z-coordinate) and all three velocity components. They are sampled using quasi-random 
    (i.e. low-discrepancy) numbers, and the distance dimension uses the importance sampling approach,
    with Z values drawn from the actual density distribution of the model conditioned on X,Y.
    The contribution of each sampled point to the LOSVD is a product of its prior weight (coming
    from the importance sampling) and the value of the model DFs at this 6d phase-space point.
    The VDFs at each X,Y point of the grid are represented by cubic splines in each velocity dimension
    (V_{X,Y,Z}), and their normalization is essentially the surface density at this point X,Y.
    The amplitudes of these splines, as well as grid parameters, are stored in a .npz file
    to avoid recomputing them on subsequent runs.
    This routine returns a dict with all necessary elements for a subsequent construction of
    3d interpolators for VDFs f(X,Y,V_i), i={X,Y,Z}; X=hor, Y=ver, Z=los.
    """
    filename = 'example_mw_nsc_vdfs.npz'
    try:  # loading cached VDFs
        result = numpy.load(filename)
        return result
    except Exception:
        pass  # VDFs not yet created

    # convert the self-consistent model to an instance of GalaxyModel for the two stellar components
    gm = agama.GalaxyModel(scm.potential, agama.DistributionFunction(scm.components[1].df, scm.components[2].df))
    # construct velocity distributions using quasi-Monte Carlo samples
    innerRadius = 0.90
    outerRadius = 240.0
    gridr = numpy.logspace(numpy.log10(innerRadius)-0.1, numpy.log10(outerRadius)+0.3, gridSizeR) * arcsec_to_pc
    grida = numpy.pi/2 * numpy.array([0, 0.3, 0.7, 1.0])
    gridv = agama.symmetricGrid(35, 30, 1000)
    assert len(grida) == gridSizeA
    X = numpy.repeat(gridr, len(grida)) * (1 - numpy.sin(numpy.tile(grida, len(gridr)))**2)**0.5
    Y = numpy.repeat(gridr, len(grida)) * numpy.sin(numpy.tile(grida, len(gridr)))
    velocities = numpy.zeros((len(X) * numSamplesVDF, 3))
    weights = numpy.zeros((len(velocities), 2))
    degree = 3
    mat = agama.bsplineMatrix(degree, gridv)
    xgridv = numpy.hstack((numpy.repeat(gridv[0], degree), gridv, numpy.repeat(gridv[-1], degree)))
    vdfvals = numpy.zeros((3, len(X), len(gridv), 2))
    norm = numpy.zeros((len(X), 2))
    sigma = numpy.logspace(-3, 2.5, 15) # widths of Gaussian components for the MGE approximation of the density profile
    ampl, q = fitMGE(scm.components[1].density + scm.components[2].density, sigma, plot=False)
    ts = ta = ti = 0
    for i in range(len(X)):
        t0 = time.time()
        try:
            q0, q1, q2, q3 = scipy.stats.qmc.Halton(4).random(numSamplesVDF).T
        except Exception as e:   # not available in older versions of scipy
            q0, q1, q2, q3 = (qrngHalton(numSamplesVDF, base) for base in (2, 3, 5, 7))
        Z, WZ = sampleZfromMGE(sigma, ampl, q, X[i], Y[i], q0)  # values and weight factors for distance samples
        positions = numpy.column_stack((Z, numpy.repeat(X[i], numSamplesVDF), numpy.repeat(Y[i], numSamplesVDF)))
        eta  = q1
        phi  = q2 * numpy.pi*2
        cth  = q3 * 2 - 1
        sth  = (1 - cth**2)**0.5
        vesc = (-2 * gm.potential.potential(positions))**0.5
        vmag = (1 - (1-eta)**(2./3))**(1./3) * vesc
        WV   = numpy.ones(numSamplesVDF) / (9./8/numpy.pi * (1-eta)**(1./3)) * vesc**3  # weight factors for velocity samples
        irange = slice(i*numSamplesVDF, (i+1)*numSamplesVDF)
        velocities[irange, 0] = vmag * sth * numpy.cos(phi)
        velocities[irange, 1] = vmag * sth * numpy.sin(phi)
        velocities[irange, 2] = vmag * cth
        t1 = time.time()
        acts = gm.af(numpy.column_stack((positions, velocities[irange])))
        acts[:,2] *= -1  # flip the sign of angular momentum, since the actual NSC rotates in the opposite sense to the model
        acts[acts[:,0]==0] = numpy.nan  # guard against rare cases when the action finder incorrectly returns Jr=0 instead of a very large value
        weights[irange] = numpy.nan_to_num((WZ * WV)[:,None] * numpy.column_stack((gm.df[0](acts), gm.df[1](acts))))
        norm[i] = numpy.sum(weights[irange], axis=0)
        t2 = time.time()
        ts += t1-t0
        ta += t2-t1
    # even though this second loop can be absorbed into the previous one, this somehow makes execution slower
    for i in range(len(X)):
        irange = slice(i*numSamplesVDF, (i+1)*numSamplesVDF)
        t2 = time.time()
        for d in range(3):
            vels_d = velocities[irange, d]
            use = (vels_d >= gridv[0]) * (vels_d <= gridv[-1])
            design_matrix = scipy.interpolate.BSpline.design_matrix(vels_d[use], xgridv, degree, extrapolate=False).T
            for c in range(2):
                rhs = numpy.nan_to_num(design_matrix.dot(weights[irange,c][use]))
                spl = agama.Spline(gridv, ampl=numpy.linalg.solve(mat, rhs) / norm[i,c])
                vals = spl(gridv)
                if d==1 or d==2:
                    vals = 0.5 * (vals + vals[::-1])  # symmetrize vhor, vver, but not vlos which has rotation
                vdfvals[d,i,:,c] = vals
        t3 = time.time()
        ti += t3-t2
    norm /= numSamplesVDF
    print('VDF interpolators created in %.3g+%.3g+%.3g seconds' % (ts, ta, ti))
    result = dict(gridr=gridr, grida=grida, gridv=gridv,
        vdf=vdfvals.reshape(3,len(gridr),len(grida),len(gridv),2).astype(numpy.float32),
        norm=norm.reshape(len(gridr),len(grida),2) )
    numpy.savez_compressed(filename, **result)
    return result


def showVDFs(vdfs):
    """
    The VDFs can then be interpolated at any other point X',Y' using 3d cubic splines.
    """
    stars = numpy.load('example_mw_nsc_obs.npz')
    no_pm = ~numpy.isfinite(stars['VX'])
    pm_fritz = ~no_pm & ~stars['virac']
    pm_virac = ~no_pm &  stars['virac']
    kdt = scipy.spatial.cKDTree(numpy.column_stack((stars['X'], stars['Y'])))

    gridr = vdfs['gridr']
    grida = vdfs['grida']
    gridv = vdfs['gridv']
    norm  = vdfs['norm']
    gridSizeA = len(grida)

    # extend the grid in angle from 1st quadrant to all four quadrants (note that X axis points left):
    #            ^ Y
    #         ...|...
    #        / 1 | 2 \
    # X <-- |----+----|
    #        \ 4 | 3 /
    #         '''''''
    # the line-of-sight velocity distribution is flipped in Q2 w.r.t Q1, i.e. f(-X, -V_los) = f(X, V_los),
    # while Q3 and Q4 are identical to Q2 and Q1 respectively, but are still provided to the interpolator
    # to ensure that it is (nearly) symmetric w.r.t. flipping the sign of Y,
    # i.e. derivatives w.r.t. Y are nearly zero on the X axis.
    # The interpolator is then only used in the upper half-plane (Q1, Q2);
    # the lower half-plane is only needed to initialize the Y-derivatives on the X axis.
    gridadup = numpy.hstack([-grida[1:][::-1], grida, numpy.pi-grida[:-1][::-1], grida[1:]+numpy.pi])
    vdfvals = numpy.zeros((3, len(gridr), len(gridadup), len(gridv), 2))
    Sigvals = numpy.zeros((len(gridr), len(gridadup), 2))
    vdfvals[:, :, : gridSizeA                ] = vdfs['vdf'][:,:,::-1]
    vdfvals[:, :,   gridSizeA-1:2*gridSizeA-1] = vdfs['vdf']
    vdfvals[:, :, 2*gridSizeA-2:3*gridSizeA-2] = vdfs['vdf'][:,:,::-1,::-1]
    vdfvals[:, :, 3*gridSizeA-3:4*gridSizeA-3] = vdfs['vdf'][:,:,:   ,::-1]
    Sigvals[   :, : gridSizeA                ] = vdfs['norm'][:,::-1]
    Sigvals[   :,   gridSizeA-1:2*gridSizeA-1] = vdfs['norm']
    Sigvals[   :, 2*gridSizeA-2:3*gridSizeA-2] = vdfs['norm'][:,::-1]
    Sigvals[   :, 3*gridSizeA-3:4*gridSizeA-3] = vdfs['norm']
    basis = (numpy.log(gridr), gridadup, gridv)
    opts  = dict(rtol=1e-10, atol=0)
    vdfc = [scipy.interpolate.RegularGridInterpolator(basis, vdfvals[d,:,:,:,0], method='cubic', solver_args=opts)
        for d in range(3)]
    vdfd = [scipy.interpolate.RegularGridInterpolator(basis, vdfvals[d,:,:,:,1], method='cubic', solver_args=opts)
        for d in range(3)]
    Sigsplc = scipy.interpolate.RectBivariateSpline(numpy.log(gridr), gridadup, numpy.log(Sigvals[:,:,0]))
    Sigspld = scipy.interpolate.RectBivariateSpline(numpy.log(gridr), gridadup, numpy.log(Sigvals[:,:,1]))
    def surfaceDensityNSC(XY):
        logr = numpy.log(numpy.sum(XY.T**2, axis=0)**0.5)
        ang  = numpy.arctan2(abs(XY[:,1]), abs(XY[:,0]))
        return numpy.exp(Sigsplc(logr, ang, grid=False))
    def surfaceDensityNSD(XY):
        logr = numpy.log(numpy.sum(XY.T**2, axis=0)**0.5)
        ang  = numpy.arctan2(abs(XY[:,1]), abs(XY[:,0]))
        return numpy.exp(Sigspld(logr, ang, grid=False))

    gridvplot = numpy.linspace(-500, 500, 251)

    # treatment of bar contamination
    def makeTwoGaussianDF(v1, s1, v2, s2, a1):
        a2 = 1-a1
        def VDF(v, v_err=0):
            return (2*numpy.pi)**-0.5 * (
            a1 * numpy.exp(-0.5 * (v-v1)**2 / (s1**2 + v_err**2)) / (s1**2 + v_err**2)**0.5 +
            a2 * numpy.exp(-0.5 * (v-v2)**2 / (s2**2 + v_err**2)) / (s2**2 + v_err**2)**0.5 )
        return VDF

    barVDFlos = makeTwoGaussianDF( 42.0, 40.0, -1.0, 146.0,0.037)(gridvplot)
    barVDFhor = makeTwoGaussianDF(-46.0, 53.0, -2.0, 169.0, 0.28)(gridvplot)
    barVDFver = makeTwoGaussianDF(  4.0, 25.0,  2.0, 131.0, 0.21)(gridvplot)
    vdfbar = [barVDFlos, barVDFhor, barVDFver]
    Sigma_bar = 0.01  # in units of 1e6 Msun/pc^2

    def onscroll(event):
        if event.inaxes not in (axi, slider): return
        delta = +1 if event.button=='up' else -1
        sqrtnumstars.set_val(min(max(sqrtnumstars.val+delta, sqrtnumstars.valmin), sqrtnumstars.valmax))
        onmove(event)

    def getvdf(xy, vdf_nsc, vdf_nsd, vdf_bar):
        if isinstance(xy, tuple) and len(xy)==2:
            xy = numpy.array([xy])  # 1x2
        r = numpy.sum(xy**2, axis=1)**0.5
        logr = numpy.log(numpy.clip(r, min(gridr), max(gridr)))
        ang  = numpy.arctan2(abs(xy[:,1]), xy[:,0])
        result_nsc = vdf_nsc(numpy.column_stack((
            numpy.tile(logr, len(gridvplot)), numpy.tile(ang, len(gridvplot)), numpy.repeat(gridvplot, len(xy))
            ))).reshape(len(gridvplot), len(xy))
        result_nsd = vdf_nsd(numpy.column_stack((
            numpy.tile(logr, len(gridvplot)), numpy.tile(ang, len(gridvplot)), numpy.repeat(gridvplot, len(xy))
            ))).reshape(len(gridvplot), len(xy))
        Sigma_nsc = surfaceDensityNSC(xy)
        Sigma_nsd = surfaceDensityNSD(xy)
        result_total = (result_nsc * Sigma_nsc + result_nsd * Sigma_nsd + (vdf_bar * Sigma_bar)[:,None]) / (Sigma_nsc + Sigma_nsd + Sigma_bar)
        return (numpy.mean(result_total, axis=1),
            numpy.mean(result_nsc * Sigma_nsc / (Sigma_nsc + Sigma_nsd + Sigma_bar), axis=1),
            numpy.mean(result_nsd * Sigma_nsd / (Sigma_nsc + Sigma_nsd + Sigma_bar), axis=1),
            numpy.mean( (vdf_bar * Sigma_bar)[:,None] / (Sigma_nsc + Sigma_nsd + Sigma_bar), axis=1),
        )

    def onmove(event):
        if not hasattr(event, 'xdata') or event.inaxes not in (axi, slider): return
        if event.inaxes == slider:
            x, y = area.get_center()
        else:
            x, y = event.xdata, event.ydata
        if not numpy.isfinite(x): return
        ngbr = int(sqrtnumstars.val)**2
        if ngbr>0:
            dist, indx = kdt.query((x, y), ngbr)
        else:
            dist, indx = 0.1, []
        for d in range(3):
            key  = ['VZ', 'VX', 'VY'][d]
            vels = stars[key][indx]
            vels = vels[numpy.isfinite(vels)]
            hist = numpy.histogram(vels, bins=bins)[0]
            norm = 1.0 / len(vels) / (bins[1]-bins[0]) if len(vels)>0 else 0
            hists[d][0].set_offsets(numpy.column_stack((bcen, hist * norm)))
            hists[d][1].set_segments([ [[x,yt], [x, yb]] for x, yt, yb in zip(bcen, (hist + hist**0.5) * norm, (hist - hist**0.5) * norm) ])
            if len(vels) < 1:
                points = (x,y)
            else:
                points = numpy.column_stack((stars['X'], stars['Y']))[indx][numpy.isfinite(stars[key][indx])]
            vdfTotal, vdfNSC, vdfNSD, vdfBar = getvdf(points, vdfc[d], vdfd[d], vdfbar[d])
            curves_tot[d].set_ydata(vdfTotal)
            curves_NSC[d].set_ydata(vdfNSC)
            curves_NSD[d].set_ydata(vdfNSD)
            curves_bar[d].set_ydata(vdfBar)
        area.set_center((x,y))
        area.set_radius(dist if ngbr<=1 else dist[-1])
        fig.canvas.draw()

    fig = plt.figure(figsize=(15,10), dpi=75)
    axi = plt.axes([0.06, 0.06, 0.6, 0.9])
    axl = plt.axes([0.73, 0.71, 0.25, 0.25])
    axh = plt.axes([0.73, 0.395,0.25, 0.25], sharey=axl)
    axv = plt.axes([0.73, 0.06, 0.25, 0.25], sharey=axl)
    axes= [axl, axh, axv]
    curves_tot = [ax.plot(gridvplot, gridvplot*0, color='gray',  label='total')[0] for ax in axes]
    curves_NSC = [ax.plot(gridvplot, gridvplot*0, color='c',       label='NSC')[0] for ax in axes]
    curves_NSD = [ax.plot(gridvplot, gridvplot*0, color='m',       label='NSD')[0] for ax in axes]
    curves_bar = [ax.plot(gridvplot, gridvplot*0, color='y', label='bar')[0] for ax in axes]
    bins = numpy.linspace(-500, 500, 26)
    bcen = (bins[1:] + bins[:-1]) * 0.5
    hists = []
    for a in axes:
        pt = a.scatter(bcen, bcen*0, color='k', marker='o', linewidths=0, edgecolors='none', s=10, label='data')
        eb = matplotlib.collections.LineCollection([[[x,0],[x,0]] for x in bcen], linewidths=1.0, colors='k')
        a.add_artist(eb)
        hists.append((pt,eb))
        a.set_xlim(min(bins), max(bins))

    area = matplotlib.patches.Circle((numpy.inf, numpy.inf), 0.0, fill=True, color='black', alpha=0.25)
    axi.add_artist(area)
    axi.add_artist(matplotlib.patches.Circle((0,0), 10.0, fill=False, color='k'))
    axi.add_artist(matplotlib.patches.Circle((0,0), min(gridr), fill=True,  color='gray', alpha=0.5))
    for ax in axes:
        ax.set_xlim(min(gridvplot), max(gridvplot))
        ax.set_ylim(1e-5, 0.01)
    axi.set_xlim(10, -10)
    axi.set_ylim(-10, 10)
    axi.set_aspect('equal')
    axi.set_xlabel('X [pc]', fontsize=14)
    axi.set_ylabel('Y [pc]', fontsize=14)
    axl.set_xlabel(r'$v_{\rm los}$ [km/s]', fontsize=16)
    axh.set_xlabel(r'$v_{\rm hor}$ [km/s]', fontsize=16)
    axv.set_xlabel(r'$v_{\rm ver}$ [km/s]', fontsize=16)
    axi.scatter(stars['X'][no_pm],    stars['Y'][no_pm],    marker='o', linewidths=0, edgecolors='none', color='b', s=4, label='no PM')
    axi.scatter(stars['X'][pm_fritz], stars['Y'][pm_fritz], marker='x', linewidths=0.5, color='r', s=6, label='PM Fritz+16')
    axi.scatter(stars['X'][pm_virac], stars['Y'][pm_virac], marker='+', linewidths=0.5, color='g', s=8, label='PM VIRAC2')
    # overplot the surface density contours of both NSC and NSD
    gridX = numpy.sinh(numpy.linspace(-5, 5, 100)) / numpy.sinh(5) * 10
    gridY = gridX.copy()
    gridXY = numpy.column_stack((numpy.tile(gridX, len(gridY)), numpy.repeat(gridY, len(gridX))))
    Sigma_nsc = surfaceDensityNSC(gridXY).reshape(len(gridY), len(gridX))
    axi.contour(gridX, gridY, numpy.log10(Sigma_nsc), cmap='Blues', vmin=-2, vmax=0.8, levels=numpy.linspace(-2, 0.8, 8), zorder=-2)
    axi.legend(loc='upper left', frameon=False, fontsize=14, markerscale=2, numpoints=1, scatterpoints=1)
    axl.legend(loc='upper left', frameon=False, fontsize=14, handlelength=1.2, handletextpad=0.2, scatterpoints=1, labelspacing=0.2)

    slider = plt.axes([0.86, 0.97, 0.1, 0.02])
    class MyFormatter(matplotlib.ticker.ScalarFormatter):
        def __call__(self, value, *args):
            return '%i' % (value**2)
    slider.xaxis.set_major_formatter(MyFormatter())
    sqrtnumstars = matplotlib.widgets.Slider(slider, 'number of stars in VDF', 0, 20, valinit=10, valstep=1)
    sqrtnumstars.on_changed(onmove)
    fig.canvas.mpl_connect('motion_notify_event', onmove)
    fig.canvas.mpl_connect('scroll_event', onscroll)
    print('Move around the image plane to display VDFs for stars in the shaded region; '
        'see the text at the beginning of this file for more information.')
    plt.show()


params = dict(
    mass      = 60.0, # [1e6 Msun] total mass of the NSC
    J0        = 240., # [pc*km/s] characteristic action
    slopeIn   =-1.00, # [dimensionless] must be <3, power-law index \Gamma in the inner part of the model - indirectly controls the density slope
    slopeOut  = 4.00, # [dimensionless] must be >3, power-law index B in the outer part of the model - also related to density slope
    steepness = 0.70, # [dimensionless] parameter controlling the steepness of the transition between the two asymptotic regimes
    coefJrIn  = 1.60, # [dimensionless] coefficient hr in the linear combination of actions, controlling the anisotropy in the inner part of the model
    coefJzIn  = 0.40, # [dimensionless] coefficient hz, responsible for flattening in z direction
    coefJrOut = 1.35, # [dimensionless] similar coefficients gr; gz for the outer part
    coefJzOut = 1.32, # [dimensionless]
    rotFrac   = 0.95, # [dimensionless] fraction of rotation in the model (0 is no rotation, +-1 is maximum rotation)
    Jphi0     = 160., # [pc*km/s] extent of central core with suppressed rotation
    Mbh       = 4.30, # [1e6 Msun] mass of the SMBH
)

# construct an instance of agama.SelfConsistentModel describing the entire system (NSC + NSD + SMBH)
scm = createModel(params)

# show 1d profiles of various quantities in the model
plotModel(scm)

# prepare data for the interactive visualization of velocity distributions
vdfs = createVDFinterpolators(scm)

# and show them
showVDFs(vdfs)
