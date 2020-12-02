#!/usr/bin/python
'''
This program is an example of running observationally-constrained Schwarzschild models
(the FORSTAND code, Vasiliev&Valluri 2020).
It has several modes of operation:

1.  Run a model for a particular choice of parameters for the potential, orbit library, etc.
    (actually, a series of models with the same potential/orbit, but different mass-to-light ratios,
    in which the velocities are rescaled before fitting to observational constraints).
    Normally one would launch several copies of the script with different parameters (e.g. Mbh),
    the results will be collected in a summary text file,
    and each model's best-fit LOSVD and orbit library will be stored in separate files.

2.  Display an interactive plot with several panels:
  - kinematic maps (v,sigma,Gauss-Hermite moments) of the data or the model(s),
    one may click on any aperture and examine the LOSVD profile in both the data and the current model;
  - a 2d plot of chi2 values as a function of potential parameters for a grid of models,
    one may choose a model from the grid, load its LOSVD and display corresponding kinematic maps;
  - LOSVD in the given aperture (both data constraints with uncertainties and the current model);
  - distribution of orbit weights of the current model in two projections of integral space:
    mean radius vs. normalized squared total angular momentum [L/Lcirc(E)]^2, or
    mean radius vs. orbit inclination Lz/L; the size of symbols indicates orbit weights.

3.  Prepare mock data for running the first two tasks.
    For this, one needs an N-body snapshot - in this example it should be produced by running
    a separate script  example_self_consistent_model3.py, which creates a three-component galaxy model
    (disk + bulge + DM halo) with a central black hole.
    It should be straightforward to feed in any other N-body snapshot, adjusting a few parameters.

This script uses various routines from the submodule agama.schwarzlib, which are general enough
to be used in any model fitting task. The specifics of each particular galaxy are encoded as numerous
parameters scattered througout this file.
Almost all needed parameters have reasonable default values in this example, but of course these are
not necessarily optimal for other purposes.
When adapting this example to your particular dataset, you will need to modify/adjust many parameters;
those which likely need to be changed are marked by [REQ],
other parameters which may be adjusted but have reasonable default values are marked by [OPT].
It is convenient to assign default values at the beginning of the script, and optionally change
some of them by providing command-line arguments in the form  name=value.

To run a complete example of constructing a grid of Schwarzschild models, do the following:

1.  Construct an N-body snapshot which will be used for the mock data, by running
    example_self_consistent_model3.py
    Among other things, it will produce the two files model_disk_final, model_bulge_final,
    which together contain the stellar particles of the model galaxy.

2.  Prepare the mock photometric and kinematic datasets, by running
    example_forstand.py do=mock
    It requires two Python modules 'mgefit' and 'vorbin'.
    This will produce an MGE file with surface density, and two kinematic datasets
    (low-resolution covering a large part of the galaxy, and high-res for the central region).
    Kinematic data (LOSVDs) in ~200 Voronoi bins are provided in two alternative forms:
  - histogrammed LOSVDs with values and their uncertainties in ~15 bins across the velocity axis;
  - Gauss-Hermite moments of LOSVDs (v, sigma, h3, h4, h5, h6 and their uncertainties).
    The models could be run on either of these alternative datasets, the choice is controlled
    by the command-line argument  hist=[y/n]  (the mock data are always generated for both cases).

3.  Now run several series of models with different values of Mbh:
    example_forstand.py do=run Mbh=...
    (of course, one may also adjust many other parameters, including hist=[true/false])
    Each series of models has the same gravitational potential, scaled by several values of M/L;
    the LOSVD and the orbit properties of each model are written into separate files,
    and the summary of all models (grid of parameters and chi2 values) are stored in a single file
    results***.txt (*** is either GH or Hist).
    The true value of Mbh is 1e8, and M/L is 1; it makes sense to explore at least a few series of
    models with Mbh ranging from 0 to ~3-5 times the true value.

4.  Finally, one may explore the grid of models for all values of Mbh and M/L by running
    example_forstand.py do=plot [hist=... and other params]

When adapting this script to a particular galaxy with existing observational datasets,
start from step 4 (to make sure that the observed kinematic maps look reasonable and
geometric parameters of the model, such as viewing angles and grids, are correct),
and then go back to step 3 (run several series of models) before going to step 4 again.
'''

import sys, numpy, agama

############### parse parameters from command-line arguments or assign default values #############
arglist = []
for arg in sys.argv[1:]:
    nameval = arg.split('=')
    if len(nameval)!=2:
        raise ValueError('Command-line arguments should be in the form  name=value')
    arglist.append([nameval[0].upper(), nameval[1]])
args = dict(arglist)

distance  = float(args.get('DISTANCE', 20626))  # [REQ] assumed distance [kpc]
arcsec2kpc= distance * numpy.pi / 648000        # conversion factor (number of kiloparsecs in one arcsecond)
agama.setUnits(mass=1, length=arcsec2kpc, velocity=1)  # [OPT] units: mass = 1 Msun, length = 1", velocity = 1 km/s
Mbh       = float(args.get('MBH', 0))           # [REQ] mass of the central black hole  [Msun]
Omega     = float(args.get('OMEGA', 0))         # [REQ] pattern speed (relevant only for non-axisymmetric models) [km/s/length_unit]
halotype  =       args.get('HALOTYPE', 'nfw')   # [OPT] halo type: 'LOG' or 'NFW'
vhalo     = float(args.get('VHALO', 190))       # [OPT] asymptotic (LOG) or peak (NFW) circular velocity of the halo [km/s]
rhalo     = float(args.get('RHALO', 150))       # [OPT] core (LOG) or scale (NFW) radius of the halo [lenth_unit]
Upsilon   = float(args.get('UPSILON', 1.0))     # [OPT] initial value of mass-to-light ratio in the search
multstep  = float(args.get('MULTSTEP', 1.02))   # [OPT] multiplicative step for increasing/decreasing Upsilon during grid search
numOrbits = int  (args.get('NUMORBITS', 20000)) # [OPT] number of orbit in the model (size of orbit library)
intTime   = float(args.get('INTTIME', 100.0))   # [OPT] integration time in units of orbital period
regul     = float(args.get('REGUL', 1. ))       # [OPT] regularization parameter (larger => more uniform orbit weight distribution in models)
incl      = float(args.get('INCL', 60.0))       # [REQ] inclination angle (0 is face-on, 90 is edge-on) [degrees]
beta      = incl * numpy.pi/180                 # same in radians
alpha     = float(args.get('ALPHA', 0.0))       # [REQ] azimuthal angle of viewing direction in the model coordinates (relevant only for non-axisym)
degree    = int  (args.get('DEGREE', 2))        # [OPT] degree of B-splines  (0 means histograms, 2 or 3 is preferred)
symmetry  = 'a'                                 # [OPT] symmetry of the model ('s'pherical, 'a'xisymmetric, 't'riaxial)
command   = args.get('DO', 'run').upper()       # [REQ] operation mode: 'RUN' - run a model, 'PLOT' - show the model grid and maps
usehist   = args.get('HIST', 'n')[0] in 'yYtT1' # [OPT] whether to use LOSVD histograms as input (default 'no' is to use GH moments)
variant   = 'Hist' if usehist else 'GH'         # suffix for disinguishing runs using histogramed LOSVDs or GH moments
fileResult= 'results%s.txt' % variant           # [OPT] filename for collecting summary results for the entire model grid
numpy.random.seed(42)  # make things repeatable
numpy.set_printoptions(precision=4, linewidth=200, suppress=True)

### parameters for the density dataset
densityParams = dict(
    type  = 'DensityCylindricalLinear',   # [REQ]: variant of density discretization grid; remaining parameters depend on this choice
    gridr = agama.nonuniformGrid(nnodes=20, xmin=0.2, xmax=100.),  # [REQ] grid in cylindrical radius (TODO: determine automatically?)
    gridz = agama.nonuniformGrid(nnodes=15, xmin=0.2, xmax=15.0),  # [REQ] grid in vertical coordinate
    mmax  = 0  # [OPT] number of azimuthal-harmonic coefficients (0 for axisymmetric systems)
)
filenameMGE = 'mge.txt'       # [REQ] file with parameters of the MGE model for the surface density profile (if MGE is used)

### common parameters for kinematic datasets (though in principle they may also differ between them)
gridv  = numpy.linspace(-500, 500, 26)  # [REQ] the grid in model velocity space (will be multiplied by sqrt(Upsilon) when comparing to data)
velpsf = 0.0                  # [OPT] velocity-space PSF (usually not needed, as the spectroscopic fits produce deconvolved LOSVDs)
# [OPT] define the degree and velocity grid for the observed LOSVD provided as histograms or (less likely) higher-degree B-splines;
# the following two lines are needed [REQ] only if the input is provided in the form of binned LOSVDs (usehist=True),
# but we also use these parameters to generate mock LOSVD histograms if command=='MOCK'
hist_degree = 0               # [OPT] B-spline degree for the observed LOSVDs (0 means histogram)
hist_gridv  = numpy.linspace(-400, 400, 17)  # [OPT] velocity grid for the observed LOSVDs (boundaries of velocity bins, not centers!)

### parameters of the 1st kinematic dataset
gamma1 = 25.0 * numpy.pi/180  # [REQ] CW rotation angle of the image-plane X axis relative to the line of nodes (=major axis for axisym.systems)
psf1   = 1.0                  # [REQ] width of the Gaussian PSF ( may use more than one component: [ [width1, weight1], [width2, weight2] ] )
kinemParams1 = dict(          # parameters passed to the constructor of the Target class
    type     = 'LOSVD',
    symmetry = symmetry,      # symmetry properties of the potential
    alpha    = alpha,         # two angles defining the orientation of the model
    beta     = beta,          # w.r.t. image plane (same for all kinematic datasets)
    gamma    = gamma1,        # third angle is the rotation of the image plane, may be different for each dataset
    psf      = psf1,          # spatial PSF
    velpsf   = velpsf,        # velocity-space PSF
    degree   = degree,        # parameters for the internal datacube represented by B-splines:
    gridv    = gridv,         # usually will be identical for all datasets (except gridx,gridy which is determined by apertures)
)
filenameVorBin1 = 'voronoi_bins_lr.txt'  # [REQ] Voronoi binning scheme for this dataset
filenameHist1   = 'kinem_hist_lr.txt'    # [REQ] histogrammed representation of observed LOSVDs
filenameGH1     = 'kinem_gh_lr.txt'      # [REQ] Gauss-Hermite parametrization of observed LOSVDs (usually only one of these two files is given)

### same for the 2nd kinematic dataset [OPT] - may have only one dataset, or as many as needed
gamma2 = -10.0 * numpy.pi/180
psf2   = 0.1                  # in this case it's a high-resolution IFU datacube
kinemParams2 = dict(
    type     = 'LOSVD',
    symmetry = symmetry,
    alpha    = alpha,
    beta     = beta,
    gamma    = gamma2,
    psf      = psf2,
    velpsf   = velpsf,
    degree   = degree,
    gridv    = gridv,
)
filenameVorBin2 = 'voronoi_bins_hr.txt'
filenameHist2   = 'kinem_hist_hr.txt'
filenameGH2     = 'kinem_gh_hr.txt'


# generate mock observations from an N-body model ([OPT] - of course this section is not needed when running the script on actual observations)
if command == 'MOCK':
    # here we use the N-body model generated by another example program:  example_self_consistent_model3.py
    # among other things, it outputs two N-body snapshot files - one for the disk, the other for the bulge component
    try:
        print('Reading input snapshot')
        snapshot1 = agama.readSnapshot('model_disk_final')
        snapshot2 = agama.readSnapshot('model_bulge_final')
        posvel    = numpy.vstack((snapshot1[0], snapshot2[0]))  # 2d Nx6 array of positions and velocities
        mass      = numpy.hstack((snapshot1[1], snapshot2[1]))  # 1d array of N particle masses
        # if your N-body snapshot is contained in a single file, just load it and assign posvel,mass arrays as specified above
    except:
        print('You need to generate N-body snapshots by running example_self_consistent_model3.py')
        exit()
    # convert the N-body model (which was set up in N-body units with G=1) to observational units defined at the beginning of this script
    rscale = 30.0   # [REQ] 1 length unit of the N-body snapshot corresponds to this many length units of this script (arcseconds)
    mscale = 4e10   # [REQ] 1 mass unit of the snapshot corresponds to this many mass units of this script (solar masses)
    vscale = (agama.G * mscale / rscale)**0.5  # same for the N-body velocity unit => km/s
    posvel[:, 0:3] *= rscale
    posvel[:, 3:6] *= vscale
    mass *= mscale

    # pre-step ([OPT] - can use only for axisymmetric systems): create several rotated copies of the input snapshot to reduce Poisson noise
    nrot = 9  # [OPT] number of rotation angles
    posvel_stack = []
    print('Creating %d rotated copies of input snapshot' % nrot)
    for ang in numpy.linspace(0, numpy.pi, nrot):
        sina, cosa = numpy.sin(ang), numpy.cos(ang)
        posvel_stack.append( numpy.column_stack((
            posvel[:,0] * cosa + posvel[:,1] * sina,
            posvel[:,1] * cosa - posvel[:,0] * sina,
            posvel[:,2],
            posvel[:,3] * cosa + posvel[:,4] * sina,
            posvel[:,4] * cosa - posvel[:,3] * sina,
            posvel[:,5] )) )
    posvel = numpy.vstack(posvel_stack)
    mass   = numpy.tile(mass, nrot) / nrot

    # 0th step: construct an MGE parametrization of the density (note: this is a commonly used, but not necessarily optimal approach)
    print('Creating MGE')
    mge = agama.schwarzlib.makeMGE(posvel, mass, beta, distance, plot=True)
    numpy.savetxt(filenameMGE, mge, fmt='%12.6g %11.3f %11.4f',
        header='MGE file\nsurface_density  width  axis_ratio\n[Msun/pc^2]   [arcsec]')

    # 1st step: construct Voronoi bins for kinematic datasets
    print('Creating Voronoi bins')
    xc, yc, bintags = agama.schwarzlib.makeVoronoiBins(
        posvel,
        gridx = numpy.linspace(-30.0, 30.0, 61),   # [REQ] X-axis pixel boundaries for the 1st (LR) dataset
        gridy = numpy.linspace(-30.0, 30.0, 61),   # [REQ] same for Y axis
        nbins = 150,     # [REQ] desired number of Voronoi bins (the result may differ somewhat)
        alpha = alpha,   # orientation angles - same as in kinemParams1
        beta  = beta,
        gamma = gamma1
    )
    # save the binning scheme to text file
    numpy.savetxt(filenameVorBin1, numpy.column_stack((xc, yc, bintags)), fmt='%7.3f %7.3f %7d')

    # 2st step: construct the LOSVD target and apply it to the N-body snapshot
    print('Computing LOSVDs of input snapshot')
    apertures    = agama.schwarzlib.getBinnedApertures(xc, yc, bintags)      # obtain boundary polygons from Voronoi bins
    gridx, gridy = agama.schwarzlib.makeGridForTargetLOSVD(apertures, psf1)  # construct a suitable image-plane grid
    target       = agama.Target(apertures=apertures, gridx=gridx, gridy=gridy, **kinemParams1)
    datacube     = target((posvel, mass)).reshape(len(apertures), -1)
    # assign errors/noise on the computed values from the Poisson noise estimate of B-spline amplitudes
    particlemass = numpy.mean(mass)
    noisecube    = (numpy.maximum(datacube, particlemass) * particlemass)**0.5

    # 3rd step: convert the B-spline LOSVDs to GH moments
    print('Computing Gauss-Hermite moments and their error estimates')
    ghorder = 6  # [OPT] order of GH expansion
    ghm_val, ghm_err = agama.schwarzlib.ghMomentsErrors(degree=degree, gridv=gridv, values=datacube, errors=noisecube, ghorder=ghorder)
    ind = (1,2,6,7,8,9)  # keep only these columns, corresponding to v,sigma,h3,h4,h5,h6
    numpy.savetxt(filenameGH1, numpy.dstack((ghm_val, ghm_err))[:,ind,:].reshape(len(apertures), -1), fmt='%8.3f',
        header='v        v_err    sigma    sigma_err h3       h3_err    h4       h4_err    h5       h5_err    h6       h6_err')

    # 4th step: convert the B-splines to histograms, which are in fact 0th-degree B-splines;
    # we might have constructed model LOSVDs in terms of histograms directly, but this would have been less accurate
    # than rebinning the model LOSVDs onto the new velocity grid
    conv = numpy.linalg.solve(   # conversion matrix from the model B-splines into observed histograms
        agama.bsplineMatrix(hist_degree, hist_gridv),
        agama.bsplineMatrix(hist_degree, hist_gridv, degree, gridv) ).T
    hist_val = datacube.dot(conv)
    hist_err = (numpy.maximum(hist_val, particlemass) * particlemass)**0.5  # again estimate errors from Poisson noise
    # save the interleaved values and error estimates of the B-spline amplitudes in each aperture to a text file
    numpy.savetxt(filenameHist1, numpy.dstack((hist_val, hist_err)).reshape(len(apertures), -1), fmt='%9.3g')

    # repeat for the 2nd (HR) dataset)
    print('Same steps for the 2nd dataset')
    xc, yc, bintags = agama.schwarzlib.makeVoronoiBins(
        posvel,
        gridx = numpy.linspace(-1.0, 1.0, 21),
        gridy = numpy.linspace(-1.0, 1.0, 21),
        nbins = 50,
        alpha = alpha,
        beta  = beta,
        gamma = gamma2
    )
    numpy.savetxt(filenameVorBin2, numpy.column_stack((xc, yc, bintags)), fmt='%7.3f %7.3f %7d')

    apertures    = agama.schwarzlib.getBinnedApertures(xc, yc, bintags)
    gridx, gridy = agama.schwarzlib.makeGridForTargetLOSVD(apertures, psf2)
    target       = agama.Target(apertures=apertures, gridx=gridx, gridy=gridy, **kinemParams2)
    datacube     = target((posvel, mass)).reshape(len(apertures), -1)
    noisecube    = (numpy.maximum(datacube, particlemass) * particlemass)**0.5

    ghm_val, ghm_err = agama.schwarzlib.ghMomentsErrors(degree=degree, gridv=gridv, values=datacube, errors=noisecube, ghorder=ghorder)
    numpy.savetxt(filenameGH2, numpy.dstack((ghm_val, ghm_err))[:,ind,:].reshape(len(apertures), -1), fmt='%8.3f',
        header='v        v_err    sigma    sigma_err h3       h3_err    h4       h4_err    h5       h5_err    h6       h6_err')

    hist_val = datacube.dot(conv)
    hist_err = (numpy.maximum(hist_val, particlemass) * particlemass)**0.5
    numpy.savetxt(filenameHist2, numpy.dstack((hist_val, hist_err)).reshape(len(apertures), -1), fmt='%9.3g')


    print('Finished creating mock datasets, now you may run this script with the argument  do=plot  or  do=run')
    exit()


### assemble the datasets (Targets and constraints)
datasets = []

### 0: photometry => 3d density profile and its discretization scheme for a density Target

# read the input MGE file, skipping the first three lines as comments, deproject it and construct the Density object
try:
    mge = numpy.loadtxt(filenameMGE, skiprows=3)   # [REQ] file with MGE parametrization of surface density profile
except:
    print('%s not found; you need to generate the mock data first, as explained at the beginning of this file' % filenameMGE)
    exit()

densityStars = agama.schwarzlib.makeDensityMGE(mge, distance, arcsec2kpc, beta)
# note: one may use any alternative method for specifying the density profile of stars, not necessarily MGE
#densityStars = agama.Density(agama.Density('dens_disk'), agama.Density('dens_bulge'))

datasets.append(agama.schwarzlib.DensityDataset(
    density=densityStars,
    tolerance=0.0,   # [OPT] fractional tolerance (e.g., 0.01) on the values of density constraints; may be 0 -- satisfy them exactly
    **densityParams  # remaining parameters set above
) )


### 1: 1st kinematic dataset

# read the Voronoi binning scheme and convert it to polygons (aperture boundaries)
vorbin    = numpy.loadtxt(filenameVorBin1)
apertures = agama.schwarzlib.getBinnedApertures(xcoords=vorbin[:,0], ycoords=vorbin[:,1], bintags=vorbin[:,2])
# note that when using real observational data, the coordinate system in the image plane is usually
# right-handed, with Y pointing up and X pointing right. This is different from the convention used
# in Agama, where X points left. Therefore, one will need to invert the X axis of the observed dataset:
# getBinnedApertures(xcoords=-vorbin[:,0], ...)

# use either histograms or GH moments as input data
if usehist:
    # [REQ] read the input kinematic data in the form of histograms;
    # if using the mock data as produced by this script, each line contains both values and errors for each velocity bin
    # in a given aperture, but when using data coming from other sources, may need to adjust the order of columns below
    kindat = numpy.loadtxt(filenameHist1)
    datasets.append(agama.schwarzlib.KinemDatasetHist(
        density   = densityStars,
        tolerance = 0.01,              # [REQ] relative error in fitting aperture mass constraints
        obs_val   = kindat[:, 0::2],   # [REQ] values of velocity histograms
        obs_err   = kindat[:, 1::2],   # [REQ] errors in these values
        obs_degree= hist_degree,
        obs_gridv = hist_gridv,
        apertures = apertures,
        **kinemParams1
    ) )
else:
    # [REQ] read the input kinematic data (V, sigma, higher Gauss-Hermite moments);
    # if using the mock data produced by this script, each line contains interleaved values and errors of v,sigma,h3...h6,
    # but when using data coming from other sources, may need to adjust the order of columns below
    kindat = numpy.loadtxt(filenameGH1)
    datasets.append(agama.schwarzlib.KinemDatasetGH(
        density   = densityStars,
        tolerance = 0.01,              # [REQ] relative error in fitting aperture mass constraints
        ghm_val   = kindat[:, 0::2],   # [REQ] values of v,sigma,h3,h4...
        ghm_err   = kindat[:, 1::2],   # [REQ] errors in the same order
        apertures = apertures,
        **kinemParams1
    ) )


### 2: [OPT] same for the 2nd kinematic dataset (and similarly for all subsequent ones)
vorbin    = numpy.loadtxt(filenameVorBin2)
apertures = agama.schwarzlib.getBinnedApertures(xcoords=vorbin[:,0], ycoords=vorbin[:,1], bintags=vorbin[:,2])
if usehist:
    kindat = numpy.loadtxt(filenameHist2)
    datasets.append(agama.schwarzlib.KinemDatasetHist(
        density   = densityStars,
        tolerance = 0.01,
        obs_val   = kindat[:, 0::2],
        obs_err   = kindat[:, 1::2],
        obs_degree= hist_degree,
        obs_gridv = hist_gridv,
        apertures = apertures,
        **kinemParams2
    ) )
else:
    kindat = numpy.loadtxt(filenameGH2)
    datasets.append(agama.schwarzlib.KinemDatasetGH(
        density   = densityStars,
        tolerance = 0.01,
        ghm_val   = kindat[:, 0::2],
        ghm_err   = kindat[:, 1::2],
        apertures = apertures,
        **kinemParams2
    ) )


### finally, decide what to do
if command == 'RUN':

    # create a dark halo according to the provided parameters (type, scale radius and circular velocity)
    if rhalo>0 and vhalo>0:
        if   halotype.upper() == 'LOG':
            densityHalo = agama.schwarzlib.makeDensityLogHalo(rhalo, vhalo)
        elif halotype.upper() == 'NFW':
            densityHalo = agama.schwarzlib.makeDensityNFWHalo(rhalo, vhalo)
        else:
            raise ValueError('Invalid halo type')
    else:
        densityHalo = agama.Density(type='Plummer', mass=0, scaleRadius=1)  # no halo

    # additional density component for constructing the initial conditions:
    # create more orbits at small radii to better resolve the kinematics around the central black hole
    densityExtra = agama.Density(type='Dehnen', scaleradius=1)

    # fiducialMbh: Mbh used to construct initial conditions (may differ from Mbh used to integrate orbits;
    # the idea is to keep fiducialMbh fixed between runs with different Mbh, so that the initial conditions
    # for the orbit library are the same, compensating one source of noise in chi2 due to randomness)
    fiducialMbh = densityStars.totalMass() * 0.01

    # potential of the galaxy, excluding the central BH
    pot_gal   = agama.Potential(type='Multipole',
        density=agama.Density(densityStars, densityHalo),  # all density components together
        lmax=32, mmax=0, gridSizeR=40)  # mmax=0 means axisymmetry; lmax is set to a large value to accurately represent the disk
    # potential of the central BH
    pot_bh    = agama.Potential(type='Plummer', mass=Mbh, scaleRadius=1e-4)
    # same for the fiducial BH
    pot_bhfidu= agama.Potential(type='Plummer', mass=fiducialMbh, scaleRadius=1e-4)
    # total potential of the model (used to integrate the orbits)
    pot_total = agama.Potential(pot_gal, pot_bh)
    # total potential used to generate initial conditions only
    pot_fidu  = agama.Potential(pot_gal, pot_bhfidu)

    # prepare initial conditions - use the same total potential independent of the actual Mbh
    # [OPT]: choose the sampling method: isotropic IC drawn from Eddington DF are created by
    #   density.sample(numorbits, potential)
    # while IC with preferential rotation (for disky models) are constructed from Jeans eqns by
    #   density.sample(numorbits, potential, beta={0-0.5}, kappa={1 or -1, depending on sign of rotation})
    # Here we add together two sets of IC - the majority of orbits sampled with Jeans eqns,
    # plus a small fraction additionally sampled from the central region to improve coverage
    ic = numpy.vstack((
        densityStars.sample(int(numOrbits*0.85), potential=pot_fidu, beta=0.3, kappa=1)[0],
        #densityStars.sample(int(numOrbits*0.85), potential=pot_fidu)[0],
        densityExtra.sample(int(numOrbits*0.15), potential=pot_fidu)[0] ))

    # launch the orbit library and perform fits for several values of Upsilon;
    agama.schwarzlib.runModel(datasets=datasets, potential=pot_total, ic=ic, intTime=intTime, Upsilon=Upsilon, multstep=multstep, regul=regul,
        # [OPT] prefix - common part of the file name storing LOSVDs of each model in this series;
        # the value of Upsilon is appended to each filename;  here one may adjust the string format or the list of parameters to store
        filePrefix = 'M%.3g_O%.3g_Rh%.3g_Vh%.3g_N%d_R%.2f_%s' % (Mbh, Omega, rhalo, vhalo, numOrbits, regul, variant),
        # [OPT] data stored at the beginning of each line (= a separate model with a given Upsilon) in the results/summary file;
        # usually should contains the same parameters as in filePrefix, but separated by tabs.
        # Keep track of the order of parameters - when reading the results file in the plotting part of this script, the order should be the same.
        # After the linePrefix, each line in the result file will contain the value of Upsilon, values of chi2 for each dataset,
        # regularization penalty, and the name of the file with LOSVD of that model.
        linePrefix = '\t'.join([ '%.3g' % Mbh, '%.3g' % Omega, '%.3g' % rhalo, '%.3g' % vhalo, '%d' % numOrbits, '%.2f' % regul ]),
        # [OPT] results/summary file
        fileResult = fileResult )

elif command == 'PLOT':

    try:
        tab = numpy.loadtxt(fileResult, dtype=str)
        # keep only the models with the same parameters as given in the command-line arguments (or their default values),
        # except the two parameters (Mbh and M/L) that are shown on the chi2 plane 
        # [OPT]: may choose different fixed/free params, but make sure the order of columns corresponds to that provided to runModel
        filt = (
            (tab[:,1].astype(float) == Omega) *
            (tab[:,2].astype(float) == rhalo) *
            (tab[:,3].astype(float) == vhalo) *
            (tab[:,4].astype(int  ) == numOrbits) *
            (tab[:,5].astype(float) == regul)
        )
        tab = tab[filt]
        if len(tab) == 0:
            print('No models satisfying all criteria are found in %s' % fileResult)
    except:
        print('File not found: %s' % fileResult)
        tab = numpy.zeros((0,10))
    filenames = tab[:,-1]                 # last column is the filename of LOSVD file for each model
    tab = tab[:,:-1].astype(float)        # remaining columns are numbers
    Mbh = tab[:,0] * tab[:,6]             # the order of parameters is the same as in linePrefix provided to runModel
    ML  = tab[:,6]                        # Upsilon is appended as the first column after those provided in linePrefix
    chi2= numpy.sum(tab[:,8:-1], axis=1)  # chi2 values are stored separately for each dataset, but here we combine all of them except regularization penalty
    # launch interactive plot with [OPT] Mbh vs M/L as the two coordinate axes displayed in chi2 plane (may choose a different pair of parameters)
    agama.schwarzlib.runPlot(datasets=datasets, aval=Mbh, bval=ML, chi2=chi2, filenames=filenames,
        # [OPT] various adjustable parameters for the plots (ranges, names, etc.) - most have reasonable default values
        alabel='Mbh', blabel='M/L', alim=(0, 4e8), blim=(0.9, 1.1), vlim=(-500,500),
        v0lim=(-150,150), sigmalim=(40,160), v0err=15.0, sigmaerr=15.0)

else:
    print('Nothing to do!')
