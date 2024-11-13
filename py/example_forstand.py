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

3.  Display some diagnostic plots useful for assessing the overall setup before running any models:
    projected and 3d density profiles along major and minor axes, location and masses of spatial bins,
    circular-velocity curve of the potential, and the observed values of v0 and sigma.
    The surface density and especially the deprojected 3d density help to check the photometric model;
    they should be smooth and have a sensible shape. In the 3d density plot, we also show the nodes of
    the spatial discretization grid, which should cover the region of interest, especially the central
    cusp (if we are interested in measuring Mbh, we need to have at least a few bins in the density
    profile within the expected radius of influence).
    The top right panel shows the values of density constraints (essentially, cell masses);
    ideally they should be quite uniform (with the exception of the innermost ones, which may be
    smaller if the grid is deliberately made finer in the central region). If the dynamical range
    spans more than 3-4 orders of magnitude, there is a risk that the smallest bins don't get any
    suitable orbits passing through them, so these density constraints would be impossible to satisfy,
    and the model either will be infeasible (if the density tolerance is set to zero) or will have
    a large and unpredictably varying penalty for violating these density constraints, which is also
    bad. In these cases one would need to adjust the grid parameters or even change the density
    discretization scheme to a different kind (e.g. classic instead of cylindrical, or vice versa).
    The bottom right panel shows the circular-velocity curve (split between mass components)
    for the model with the currently chosen parameters (Mbh, stellar M/L, etc.). For comparison,
    the values of kinematic constraints (v0 and sigma) in all bins are plotted against radius.
    In a thin and cold disk observed edge-on, the maximum value of |v0| should be close to the
    amplitude of the total circular velocity, but in general it will be smaller by some factor.
    This plot may be used to get a rough idea about the expected M/L: the amplitude of Vcirc in
    the potential is scaled by sqrt(Upsilon), where Upsilon (provided as a command-line argument)
    is the starting value for scanning the M/L axis automatically performed by the code.

4.  Prepare mock data for running the first two tasks.
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

3.  Examine the model setup by running
    example_forstand.py do=test
    (this is less relevant for the mock dataset, but could be quite helpful when working
    with real data, to check if the parameters are sensible, or to diagnose possible problems).

4.  Now run several series of models with different values of Mbh:
    example_forstand.py do=run Mbh=...
    (of course, one may also adjust many other parameters, including hist=[true/false])
    Each series of models has the same gravitational potential, scaled by several values of M/L;
    the LOSVD and the orbit properties of each model series are written into separate .npz files,
    and the summary of all models (grid of parameters and chi2 values) are stored in a single file
    results***.txt (*** is either GH or Hist).
    For each series of models with a given potential, the one with a M/L that gives the lowest chi2
    is converted into an N-body representation and written into a separate text file.
    The true value of Mbh is 1e8, and M/L is 1; it makes sense to explore at least a few series of
    models with Mbh ranging from 0 to ~3-5 times the true value.

5.  Finally, one may explore the grid of models for all values of Mbh and M/L by running
    example_forstand.py do=plot [hist=... and other params]

When adapting this script to a particular galaxy with existing observational datasets,
start from step 4 (to make sure that the observed kinematic maps look reasonable and
geometric parameters of the model, such as viewing angles and grids, are correct),
and then go back to step 3 (run several series of models) before going to step 4 again.

This script is mainly tailored to axisymmetric systems, although the Forstand code is applicable
in a more general context (e.g., to rotating triaxial barred galaxies).
The main limitation is the lack of suitable deprojection methods: in this example we use
the Multi-Gaussian expansion to fit the 2d surface density profile of the N-body snapshot
and then deproject it into an ellipsoidally stratified 3d density profile.
In the case of triaxial systems, especially with radially varying axis ratios, this procedure
is much less reliable and may even fail to produce a reasonable deprojection if the actual
3d shape is not well described by ellipsoids.
For triaxial systems, there are two rather than one angle specifying the orientation:
inclination (beta) and the angle alpha between the major axis and the line of nodes,
and in the rotating case, the pattern speed Omega is also a free parameter.

The model parameters and corresponding chi^2 values are stored in a single file
resultsGH.txt (for Gauss-Hermite parametrization of the LOSVD) or
resultsHist.txt (for histograms), and the kinematic maps and orbit distribution of each series
of models with the same potential and varying M/L are stored in a separate .npz archive.
In the interactive plotting regime, the likelihood surface is shown as a function of two
model parameters (in this example, Mbh and M/L), but one may choose another pair of parameters
by providing different columns of the results file as "aval", "bval" arguments for
agama.schwarzlib.runPlot(...); the values of remaining parameters should then be fixed and
specified as command-line arguments.
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

distance  = float(args.get('DISTANCE', 20626.5))# [REQ] assumed distance [kpc]
arcsec2kpc= distance * numpy.pi / 648000        # conversion factor (number of kiloparsecs in one arcsecond)
agama.setUnits(mass=1, length=arcsec2kpc, velocity=1)  # [OPT] units: mass = 1 Msun, length = 1", velocity = 1 km/s
Mbh       = float(args.get('MBH', 1e8))         # [REQ] mass of the central black hole  [Msun]
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
alpha_deg = float(args.get('ALPHA', 0.0))       # [REQ] azimuthal angle of viewing direction in the model coordinates (relevant only for non-axisym)
alpha     = alpha_deg * numpy.pi/180            # same in radians
degree    = int  (args.get('DEGREE', 2))        # [OPT] degree of B-splines  (0 means histograms, 2 or 3 is preferred)
symmetry  = 'a'                                 # [OPT] symmetry of the model ('s'pherical, 'a'xisymmetric, 't'riaxial)
addnoise  =      (args.get('ADDNOISE', 'True')  # [OPT] whether to add a realistic amount of noise in generating mock datacubes
    .upper() in ('TRUE', 'T', 'YES', 'Y'))
nbody     = int  (args.get('NBODY', 100000))    # [OPT] number of particles for the N-body representation of the best-fit model
nbodyFormat = args.get('NBODYFORMAT', 'text')   # [OPT] format for storing N-body snapshots (text/nemo/gadget)
command   = args.get('DO', '').upper()          # [REQ] operation mode: 'RUN' - run a model, 'PLOT' - show the model grid and maps, 'TEST' - show diagnostic plots, 'MOCK' - create mock maps
variant   = args.get('VARIANT', 'GH').upper()   # [OPT] choice between three ways of representing and fitting LOSVDs (see below)
if 'HIST' not in variant and 'GH' not in variant and 'VS' not in variant:
    raise RuntimeError('parameter "variant" should be one of "GH" (Gauss-Hermite moments), "VS" (classical moments - v & sigma), or "HIST" (LOSVD histograms)')
fileResult= 'results%s.txt' % variant           # [OPT] filename for collecting summary results for the entire model grid
seed      = int  (args.get('SEED', 99))         # [OPT] random seed (different values will create different realizations of mock data when do=MOCK, or initial conditions for the orbit library when do=RUN)
agama.setRandomSeed(seed)                       # note that Agama has its own
numpy.random.seed(seed)                         # make things repeatable when generating mock data (*not* related to the seed for the orbit library)
numpy.set_printoptions(precision=4, linewidth=9999, suppress=True)

# In this example, we use the Multi-Gaussian Expansion to parametrize
# the surface density profile and deproject it into the 3d density profile,
# but the code works with any other choice of 3d density model.
filenameMGE = 'mge_i%.0f.txt' % incl    # [REQ] file with parameters of the MGE model for the surface density profile (if MGE is used)

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
filenameVorBin1 = 'voronoi_bins_i%.0f_lr.txt' % incl # [REQ] Voronoi binning scheme for this dataset
filenameHist1   = 'kinem_hist_i%.0f_lr.txt'   % incl # [REQ] histogrammed representation of observed LOSVDs
filenameGH1     = 'kinem_gh_i%.0f_lr.txt'     % incl # [REQ] Gauss-Hermite parametrization of observed LOSVDs (usually only one of these two files is given)
filenameVS1     = 'kinem_vs_i%.0f_lr.txt'     % incl # [REQ] 

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
filenameVorBin2 = 'voronoi_bins_i%.0f_hr.txt' % incl
filenameHist2   = 'kinem_hist_i%.0f_hr.txt'   % incl
filenameGH2     = 'kinem_gh_i%.0f_hr.txt'     % incl
filenameVS2     = 'kinem_vs_i%.0f_hr.txt'     % incl


def makeMockKin(kinemParams, gridxy, nbins, filenameVorBin, filenameHist, filenameGH, filenameVS):
    # 1st step: construct Voronoi bins for kinematic datasets
    print('Creating Voronoi bins')
    xc, yc, bintags = agama.schwarzlib.makeVoronoiBins(
        posvel,
        gridx = gridxy,   # [REQ] X-axis pixel boundaries for the 1st (LR) dataset
        gridy = gridxy,   # [REQ] same for Y axis
        nbins = nbins,    # [REQ] desired number of Voronoi bins (the result may differ somewhat)
        alpha = kinemParams['alpha'],  # orientation angles
        beta  = kinemParams['beta'],
        gamma = kinemParams['gamma']
    )
    # save the binning scheme to text file
    numpy.savetxt(filenameVorBin, numpy.column_stack((xc, yc, bintags)), fmt='%8.4f %8.4f %7d')

    # 2st step: construct the LOSVD target and apply it to the N-body snapshot
    print('Computing LOSVDs of input snapshot')
    apertures    = agama.schwarzlib.getBinnedApertures(xc, yc, bintags)      # obtain boundary polygons from Voronoi bins
    gridx, gridy = agama.schwarzlib.makeGridForTargetLOSVD(apertures, kinemParams['psf'])  # construct a suitable image-plane grid
    target       = agama.Target(apertures=apertures, gridx=gridx, gridy=gridy, **kinemParams)
    datacube     = target((posvel, mass)).reshape(len(apertures), -1)

    # datacube now contains the amplitudes of B-spline representation of LOSVDs in each aperture.
    # Assign the uncertainties on each amplitude assuming the Poisson noise;
    # for this we need to know the "typical" amplitude produced by one particle (e.g., placed at the center),
    # which is the particle mass divided by the bin size of the velocity grid.
    oneparticle = numpy.mean(mass) / (gridv[1]-gridv[0])
    noise = numpy.maximum(1, datacube / oneparticle)**0.5
    # it turns out that this noise level is very low in our example, so we multiply it by some factor >1
    if '_lr' in filenameGH: noise *= 8
    else: noise *= 2
    # to propagate the uncertainties throughout subsequent computations, construct "nboot" realizations
    # of the original datacube perturbed by Gaussian noise
    nboot = 16
    datacubes = datacube + numpy.random.normal(size=(nboot,)+datacube.shape) * noise * oneparticle
    if addnoise:
        datacube = datacubes[0]   # take one perturbed realization as the input (noisy) data
    # else keep datacube as computed originally

    # 3th step, variant A: convert the B-splines to histograms, which are in fact 0th-degree B-splines;
    # we might have constructed model LOSVDs in terms of histograms directly, but this would have been less accurate
    # than rebinning the model LOSVDs onto the new velocity grid
    conv = numpy.linalg.solve(   # conversion matrix from the model B-splines into observed histograms
        agama.bsplineMatrix(hist_degree, hist_gridv),
        agama.bsplineMatrix(hist_degree, hist_gridv, degree, gridv) ).T
    hist_val = datacube.dot(conv)
    hist_err = numpy.std(datacubes.dot(conv), axis=0)
    hist_norm= numpy.sum(hist_val, axis=1)[:,None]
    # save the interleaved values and error estimates of the B-spline amplitudes in each aperture to a text file
    numpy.savetxt(filenameHist,
        numpy.dstack((hist_val/hist_norm, hist_err/hist_norm)).reshape(len(apertures), -1),
        fmt='%9.3g')

    # 3rd step, variant B: convert the B-spline LOSVDs to GH moments
    print('Computing Gauss-Hermite moments and their error estimates')
    ghorder = 6  # [OPT] order of GH expansion
    ghm_val = agama.ghMoments(degree=degree, gridv=gridv, ghorder=ghorder, matrix=datacube)
    ghm_err = numpy.std(
        agama.ghMoments(degree=degree, gridv=gridv, ghorder=ghorder,
            matrix=datacubes.reshape(-1, datacubes.shape[2])).
        reshape(nboot, len(apertures), -1),
        axis=0)
    ind = (1,2,6,7,8,9)  # keep only these columns, corresponding to v,sigma,h3,h4,h5,h6
    numpy.savetxt(filenameGH,
        numpy.dstack((ghm_val, ghm_err))[:,ind,:].reshape(len(apertures), -1),
        fmt='%8.3f', header='v        v_err    sigma    sigma_err '+
        'h3       h3_err    h4       h4_err    h5       h5_err    h6       h6_err')

    # 3th step, variant C: convert the B-splines to V and sigma
    i0 = agama.bsplineIntegrals(degree, gridv)
    i1 = agama.bsplineIntegrals(degree, gridv, power=1)
    i2 = agama.bsplineIntegrals(degree, gridv, power=2)
    datacube0  = datacube.dot(i0)
    datacubes0 = datacubes.dot(i0)
    meanv_val  = (datacube.dot(i1) / datacube0)
    sigma_val  = (datacube.dot(i2) / datacube0 - meanv_val**2)**0.5
    meanv_vals = (datacubes.dot(i1) / datacubes0)
    sigma_vals = (datacubes.dot(i2) / datacubes0 - meanv_vals**2)**0.5
    meanv_err  = numpy.std(meanv_vals, axis=0)
    sigma_err  = numpy.std(sigma_vals, axis=0)
    numpy.savetxt(filenameVS,
        numpy.column_stack((meanv_val, meanv_err, sigma_val, sigma_err)),
        fmt='%8.3f', header='v v_err sigma sigma_err')


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
        exit('You need to generate N-body snapshots by running example_self_consistent_model3.py')
    # convert the N-body model (which was set up in N-body units with G=1) to observational units defined at the beginning of this script
    rscale = 30.0   # [REQ] 1 length unit of the N-body snapshot corresponds to this many length units of this script (arcseconds)
    mscale = 4e10   # [REQ] 1 mass unit of the snapshot corresponds to this many mass units of this script (solar masses)
    vscale = (agama.G * mscale / rscale)**0.5  # same for the N-body velocity unit => km/s
    print('Scaling N-body model to physical units: 1 length unit = %g arcsec = %g kpc, 1 velocity unit = %g km/s, 1 mass unit = %g Msun' %
        (rscale, rscale * arcsec2kpc, vscale, mscale))
    posvel[:, 0:3] *= rscale
    posvel[:, 3:6] *= vscale
    mass *= mscale

    # 0th step: construct an MGE parameterization of the density (note: this is a commonly used, but not necessarily optimal approach)
    print('Creating MGE')
    mge = agama.schwarzlib.makeMGE(posvel, mass, beta, distance, plot=True)
    numpy.savetxt(filenameMGE, mge, fmt='%12.6g %11.3f %11.4f',
        header='MGE file\nsurface_density  width  axis_ratio\n[Msun/pc^2]   [arcsec]')

    # Low-resolution dataset with a FoV 1x1' and pixel size 1" (comparable to ground-based IFU such as SAURON).
    # Note: make sure that the pixel size passed to makeVoronoiBins is rounded to at most 4 significant digits,
    # since this is the precision with which we save it later; otherwise the subsequent reading of Voronoi bins will fail
    makeMockKin(kinemParams1, numpy.linspace(-30.0, 30.0, 61), 150, filenameVorBin1, filenameHist1, filenameGH1, filenameVS1)
    # High-resolution dataset similar to AO-assisted IFU such as NIFS (2x2", pixel size 0.1")
    makeMockKin(kinemParams2, numpy.linspace(- 1.0,  1.0, 21),  50, filenameVorBin2, filenameHist2, filenameGH2, filenameVS2)

    print('Finished creating mock datasets, now you may run this script with the argument  do=plot  or  do=run')
    exit()


### assemble the datasets (Targets and constraints)
datasets = []

### 0: photometry => 3d density profile and its discretization scheme for a density Target

# read the input MGE file, skipping the first three lines as comments, deproject it and construct the Density object.
# Instead of MGE, one may use any other parametric density profile, e.g. one or more Sersic components with parameters
# determined by photometric fitting software such as Galfit
try:
    mge = numpy.loadtxt(filenameMGE)   # [REQ] file with MGE parametrization of surface density profile
except:
    print('%s not found; you need to generate the mock data first, as explained at the beginning of this file' % filenameMGE)
    exit()

densityStars = agama.schwarzlib.makeDensityMGE(mge, distance, arcsec2kpc, beta)
#densityStars = agama.Density(agama.Density('dens_disk'), agama.Density('dens_bulge'))  # true profiles of this mock dataset

### parameters for the density dataset
# the choice of discretization scheme depends on the morphological type of the galaxy being modelled:
# for disky systems, DensityCylindrical[TopHat/Linear] is preferred, either in the axisymmetric regime
# (mmax=0), or more generally with mmax>0;
# for spheroidal systems, DensityClassic[TopHat/Linear] or DensitySphHarm may be more suitable,
# and in any case, the choice of radial [and possibly vertical] grid requires careful consideration.
# Here we do it automatically to ensure that the grid covers almost the entire model
# and has roughly equal mass in each shell (for spherical) or slab (for cylindrical grids),
# but this might not be suitable for every case; in particular, one may wish to make the grids
# denser in the central region to better constrain the 3d density profile near the black hole.
densityParams = dict(type = (
    'DensityClassicTopHat',
    'DensityClassicLinear',
    'DensitySphHarm',
    'DensityCylindricalTopHat',
    'DensityCylindricalLinear')[4])   # [REQ] choose one of these types!
# use the discrete samples from the density profile to choose the grid parameters
samples = densityStars.sample(10000)[0]
if densityParams['type'] == 'DensityClassicTopHat' or densityParams['type'] == 'DensityClassicLinear':
    # create a grid in elliptical radius with axis ratio chosen to [roughly] match those of the density profile
    axes = numpy.percentile(numpy.abs(samples), 90, axis=0)  # three principal axes in the outer part of the profile
    axes/= numpy.exp(numpy.mean(numpy.log(axes)))  # normalize so that the product of three axes is unity
    ellrad = numpy.sum((samples / axes)**2, axis=1)**0.5
    # [OPT] make the inner grid segment contain 1% of the total mass
    # (to better constrain the density near the black hole, though this may need some further tuning),
    # and the remaining segments contain roughly equal fractions of mass up to 99% of the total mass
    densityParams['gridr'] = numpy.hstack([0, numpy.percentile(ellrad, tuple(numpy.linspace(1, 99, 24))) ])
    densityParams['axisRatioY'] = axes[1] / axes[0]
    densityParams['axisRatioZ'] = axes[2] / axes[0]
    print('%s grid in elliptical radius: %s, axis ratios: y/x=%.3g, z/x=%.3g' %
        (densityParams['type'], densityParams['gridr'], densityParams['axisRatioY'], densityParams['axisRatioZ']))
    # [OPT] each shell in the elliptical radius is divided in three equal 'panes'
    # adjacent to each of the principal axes, and then each pane is further divided
    # into a square grid of cells with stripsPerPane elements on each side
    densityParams['stripsPerPane'] = 2
elif densityParams['type'] == 'DensitySphHarm':
    # this discretization scheme uses a grid in spherical radius and a spherical-harmonic expansion in angles
    sphrad = numpy.sum(samples**2, axis=1)**0.5
    # [OPT] same procedure as above, using roughly equal-mass bins in spherical radius except the innermost one
    densityParams['gridr'] = numpy.hstack([0, numpy.percentile(sphrad, tuple(numpy.linspace(1, 99, 24))) ])
    # [OPT] order of angular spherical-harmonic expansion in theta and phi (must be even)
    densityParams['lmax'] = 0 if symmetry[0]=='s' else 8
    densityParams['mmax'] = 0 if symmetry[0]!='t' else 6
    print('%s grid in spherical radius: %s, lmax=%i, mmax=%i' %
        (densityParams['type'], densityParams['gridr'], densityParams['lmax'], densityParams['mmax']))
elif densityParams['type'] == 'DensityCylindricalTopHat' or densityParams['type'] == 'DensityCylindricalLinear':
    sampleR = (samples[:,0]**2 + samples[:,1]**2)**0.5
    samplez = abs(samples[:,2])
    # [OPT] choose the grids in R and z so that each 'slab' (1d projection along the complementary coordinate)
    # contains approximately equal mass, though this doesn't guarantee that the 2d cells would be even roughly balanced
    densityParams['gridR'] = numpy.hstack([0, numpy.percentile(sampleR, tuple(numpy.linspace(1, 99, 20))) ])
    densityParams['gridz'] = numpy.hstack([0, numpy.percentile(samplez, tuple(numpy.linspace(1, 99, 15))) ])
    # [OPT] number of azimuthal-harmonic coefficients (0 for axisymmetric systems)
    densityParams['mmax']  = 0 if symmetry[0]!='t' else 6
    print('%s grid in R: %s, z: %s, mmax=%i' %
        (densityParams['type'], densityParams['gridR'], densityParams['gridz'], densityParams['mmax']))

datasets.append(agama.schwarzlib.DensityDataset(
    density=densityStars,
    # [OPT] fractional tolerance (e.g., 0.01) on the values of density constraints;
    # may be 0, requiring to satisfy them exactly, but in this case the solution may be infeasible
    tolerance=0.01,
    alpha=alpha,     # the orientation of intrinsic model coordinates w.r.t. the observed ones,
    beta=beta,       # specified by two Euler angles (used only for plotting the projected density)
    **densityParams  # remaining parameters set above
) )


### 1: 1st kinematic dataset

# read the Voronoi binning scheme and convert it to polygons (aperture boundaries)
vorbin1    = numpy.loadtxt(filenameVorBin1)
apertures1 = agama.schwarzlib.getBinnedApertures(xcoords=vorbin1[:,0], ycoords=vorbin1[:,1], bintags=vorbin1[:,2])
# note that when using real observational data, the coordinate system in the image plane is usually
# right-handed, with Y pointing up and X pointing right. This is different from the convention used
# in Agama, where X points left. Therefore, one will need to invert the X axis of the observed dataset:
# getBinnedApertures(xcoords=-vorbin[:,0], ...)

# use either histograms, GH moments, or classical moments (v & sigma) as input data
if 'HIST' in variant:
    # [REQ] read the input kinematic data in the form of histograms;
    # if using the mock data as produced by this script, each line contains both values and errors for each velocity bin
    # in a given aperture, but when using data coming from other sources, may need to adjust the order of columns below
    kindat1 = numpy.loadtxt(filenameHist1)
    datasets.append(agama.schwarzlib.KinemDatasetHist(
        density   = densityStars,
        tolerance = 0.01,              # [REQ] relative error in fitting aperture mass constraints
        obs_val   = kindat1[:, 0::2],  # [REQ] values of velocity histograms
        obs_err   = kindat1[:, 1::2],  # [REQ] errors in these values
        obs_degree= hist_degree,
        obs_gridv = hist_gridv,
        apertures = apertures1,
        **kinemParams1
    ) )
elif 'GH' in variant:
    # [REQ] read the input kinematic data (V, sigma, higher Gauss-Hermite moments);
    # if using the mock data produced by this script, each line contains interleaved values and errors of v,sigma,h3...h6,
    # but when using data coming from other sources, may need to adjust the order of columns below
    kindat1 = numpy.loadtxt(filenameGH1)
    datasets.append(agama.schwarzlib.KinemDatasetGH(
        density   = densityStars,
        tolerance = 0.01,              # [REQ] relative error in fitting aperture mass constraints
        ghm_val   = kindat1[:, 0::2],  # [REQ] values of v,sigma,h3,h4...
        ghm_err   = kindat1[:, 1::2],  # [REQ] errors in the same order
        apertures = apertures1,
        **kinemParams1
    ) )
elif 'VS' in variant:
    # [REQ] read the input kinematic data (v and sigma, which have a different meaning here than in the case of
    # Gauss-Hermite moments above; specifically, v is the mean velocity and sigma is its standard deviation,
    # while for the GH parameterization, v and sigma are the center and width of the best-fit Gaussian);
    # data format used in this script: v, v_error, sigma, sigma_error, one line per aperture
    kindat1 = numpy.loadtxt(filenameVS1)
    datasets.append(agama.schwarzlib.KinemDatasetVS(
        density   = densityStars,
        tolerance = 0.01,              # [REQ] relative error in fitting aperture mass constraints
        vs_val   = kindat1[:, 0::2],   # [REQ] values of v,sigma
        vs_err   = kindat1[:, 1::2],   # [REQ] errors in the same order
        apertures = apertures1,
        **kinemParams1
    ) )


### 2: [OPT] same for the 2nd kinematic dataset (and similarly for all subsequent ones)
vorbin2     = numpy.loadtxt(filenameVorBin2)
apertures2  = agama.schwarzlib.getBinnedApertures(xcoords=vorbin2[:,0], ycoords=vorbin2[:,1], bintags=vorbin2[:,2])
if 'HIST' in variant:
    kindat2 = numpy.loadtxt(filenameHist2)
    datasets.append(agama.schwarzlib.KinemDatasetHist(
        density   = densityStars,
        tolerance = 0.01,
        obs_val   = kindat2[:, 0::2],
        obs_err   = kindat2[:, 1::2],
        obs_degree= hist_degree,
        obs_gridv = hist_gridv,
        apertures = apertures2,
        **kinemParams2
    ) )
elif 'GH' in variant:
    kindat2 = numpy.loadtxt(filenameGH2)
    if True: datasets.append(agama.schwarzlib.KinemDatasetGH(
        density   = densityStars,
        tolerance = 0.01,
        ghm_val   = kindat2[:, 0::2],
        ghm_err   = kindat2[:, 1::2],
        apertures = apertures2,
        **kinemParams2
    ) )
elif 'VS' in variant:
    kindat2 = numpy.loadtxt(filenameVS2)
    datasets.append(agama.schwarzlib.KinemDatasetVS(
        density   = densityStars,
        tolerance = 0.01,
        vs_val   = kindat2[:, 0:4:2],
        vs_err   = kindat2[:, 1:4:2],
        apertures = apertures2,
        **kinemParams2
    ) )


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
    lmax=32,  # lmax is set to a large value to accurately represent a disky density profile
    mmax=0 if symmetry[0]!='t' else 6, gridSizeR=40)  # mmax>0 only for triaxial systems
# potential of the central BH
pot_bh    = agama.Potential(type='Plummer', mass=Mbh, scaleRadius=1e-4)
# same for the fiducial BH
pot_bhfidu= agama.Potential(type='Plummer', mass=fiducialMbh, scaleRadius=1e-4)
# total potential of the model (used to integrate the orbits)
pot_total = agama.Potential(pot_gal, pot_bh)
# total potential used to generate initial conditions only
pot_fidu  = agama.Potential(pot_gal, pot_bhfidu)


### finally, decide what to do
if command == 'RUN':

    # prepare initial conditions - use the same total potential independent of the actual Mbh
    # [OPT]: choose the sampling method: isotropic IC drawn from Eddington DF are created by
    #   density.sample(numorbits, potential)
    # while IC with preferential rotation (for disky models) are constructed from axisymmetric Jeans eqns by
    #   density.sample(numorbits, potential, beta={0-0.5}, kappa={1 or -1, depending on sign of rotation})
    # Here we add together two sets of IC - the majority of orbits sampled with axisymmetric Jeans eqns,
    # plus a small fraction additionally sampled from the central region to improve coverage.
    ic = numpy.vstack((
        densityStars.sample(int(numOrbits*0.85), potential=pot_fidu, beta=0.3, kappa=1)[0],
        densityExtra.sample(int(numOrbits*0.15), potential=pot_fidu)[0] ))


    # launch the orbit library and perform fits for several values of Upsilon;
    agama.schwarzlib.runModel(datasets=datasets, potential=pot_total, ic=ic,
        intTime=intTime, Upsilon=Upsilon, multstep=multstep, regul=regul, Omega=Omega,
        # [OPT] prefix - common part of the file name storing LOSVDs of each model in this series;
        # the value of Upsilon is appended to each filename;  here one may adjust the string format or the list of parameters to store
        filePrefix = 'M%.3g_O%.3g_Rh%.3g_Vh%.3g_i%.0f_a%.0f_N%d_R%.2f_%s' %
            (Mbh, Omega, rhalo, vhalo, incl, alpha_deg, numOrbits, regul, variant),
        # [OPT] data stored at the beginning of each line (= a separate model with a given Upsilon) in the results/summary file;
        # usually should contains the same parameters as in filePrefix, but separated by tabs.
        # Keep track of the order of parameters - when reading the results file in the plotting part of this script, the order should be the same.
        # After the linePrefix, each line in the result file will contain the value of Upsilon, values of chi2 for each dataset,
        # regularization penalty, and the name of the file with LOSVD of that model.
        linePrefix = '\t'.join([ '%.3g' % Mbh, '%.3g' % Omega, '%.3g' % rhalo, '%.3g' % vhalo,
            '%.0f' % incl, '%.0f' % alpha_deg, '%d' % numOrbits, '%.2f' % regul ]),
        # [OPT] results/summary file
        fileResult = fileResult,
        # [OPT] parameters for the N-body snapshot representing the best-fit model
        nbody = nbody, nbodyFormat = nbodyFormat )

elif command == 'TEST':

    # plot various diagnostics to check if the parameters of the model are reasonable
    import matplotlib.pyplot as plt
    ax = plt.subplots(2, 2, figsize=(12,8), dpi=100)[1].reshape(-1)
    # radial range of these plots is somewhat arbitrary, but should encompass the extent of kinematic data (may need adjustment)
    gridrmajor = numpy.logspace(numpy.log10(0.05), numpy.log10(200))
    gridrminor = numpy.logspace(numpy.log10(0.05), numpy.log10(100))
    # surface density along the major and minor axes
    Sigmamajor = densityStars.projectedDensity(numpy.column_stack((gridrmajor, gridrmajor*0)), beta=beta, alpha=alpha)
    Sigmaminor = densityStars.projectedDensity(numpy.column_stack((gridrminor*0, gridrminor)), beta=beta, alpha=alpha)
    ax[0].loglog(gridrmajor, Sigmamajor, color='b', label=r'$\Sigma(R)$ major')
    ax[0].loglog(gridrminor, Sigmaminor, color='r', label=r'$\Sigma(R)$ minor')
    ax[0].set_xlabel('projected radius')
    ax[0].set_ylabel('surface density')
    ax[0].legend(loc='lower left', frameon=False)
    ax[0].set_xlim(min(gridrmajor), max(gridrmajor))
    ax[0].set_ylim(min(Sigmamajor), max(Sigmamajor))

    # deprojected 3d density along the major (x) and minor (z) axes
    rhomajor = densityStars.density(numpy.column_stack((gridrmajor, gridrmajor*0, gridrmajor*0)))
    rhominor = densityStars.density(numpy.column_stack((gridrminor*0, gridrminor*0, gridrminor)))
    ax[2].loglog(gridrmajor, rhomajor, color='b', label=r'$\rho(R,z=0)$')
    ax[2].loglog(gridrminor, rhominor, color='r', label=r'$\rho(R=0,z)$')
    ax[2].set_xlabel('radius')
    ax[2].set_ylabel('3d density')
    ax[2].legend(loc='lower left', frameon=False)
    ax[2].set_xlim(min(gridrmajor), max(gridrmajor))
    ax[2].set_ylim(min(rhomajor)/5, max(rhomajor))
    # also show the location of nodes of the 3d density discretization grid
    if 'gridR' in densityParams: gridnodesmajor = densityParams['gridR']
    if densityParams['type'] == 'DensityCylindricalTopHat' or densityParams['type'] == 'DensityCylindricalLinear':
        ax[2].plot(densityParams['gridR'], numpy.interp(densityParams['gridR'], gridrmajor, rhomajor), 'b|', ms=6)
        ax[2].plot(densityParams['gridz'], numpy.interp(densityParams['gridz'], gridrminor, rhominor), 'r|', ms=6)
    else:
        if 'axisRatioZ' in densityParams:   # elliptical grid in DensityClassic[TopHat/Linear]
            multX = (densityParams['axisRatioY'] * densityParams['axisRatioZ'])**(-1./3)  # scaling along the X (major) axis
            multZ =  densityParams['axisRatioZ'] * multX                                  # scaling along the Z (minor) axis
        else:
            multX = multZ = 1
        ax[2].plot(densityParams['gridr']*multX, numpy.interp(densityParams['gridr']*multX, gridrmajor, rhomajor), 'b|', ms=6)
        ax[2].plot(densityParams['gridr']*multZ, numpy.interp(densityParams['gridr']*multZ, gridrminor, rhominor), 'r|', ms=6)

    # values of discretized density constraints: this is rather technical, but a large spread in values
    # (more than a few orders of magnitude) may present trouble for the solution - then one would need
    # to change some grid parameters, so that the distribution of cell masses is more uniform
    ax[1].plot(datasets[0].cons_val[1:])
    ax[1].set_xlabel('constraint index')
    ax[1].set_ylabel('density constraint value')
    ax[1].set_yscale('log')

    # plot observed parameters of GH expansion v0 and sigma against radius
    if 'HIST' not in variant:
        aperture_radii1 = numpy.array([numpy.mean(ap[:,0]**2+ap[:,1]**2)**0.5  for ap in apertures1])
        aperture_radii2 = numpy.array([numpy.mean(ap[:,0]**2+ap[:,1]**2)**0.5  for ap in apertures2])
        ax[3].scatter(aperture_radii1, abs(kindat1[:,0]), label='v', c='y', linewidths=0)
        ax[3].scatter(aperture_radii1, kindat1[:,2],  label='sigma', c='c', linewidths=0)
        ax[3].scatter(aperture_radii2, abs(kindat2[:,0]), c='y', marker='x')
        ax[3].scatter(aperture_radii2, kindat2[:,2],      c='c', marker='x')
    # plot circular-velocity curves of each potential component
    for name, pot in [
        ['stars', agama.Potential(type='Multipole', density=densityStars, mmax=0, lmax=32)],
        ['halo',  agama.Potential(type='Multipole', density=densityHalo)],
        ['BH',    pot_bh],
        ['total', pot_total] ]:
        vcirc = (-gridrmajor * pot.force(numpy.column_stack((gridrmajor, gridrmajor*0, gridrmajor*0)))[:,0] * Upsilon)**0.5
        ax[3].plot(gridrmajor, vcirc, label=name)
    ax[3].legend(loc='upper left', scatterpoints=1, ncol=(2 if 'HIST' in variant else 3), frameon=False)
    ax[3].set_xscale('log')
    ax[3].set_xlim(min(gridrmajor), max(gridrmajor))
    ax[3].set_ylim(0, max(vcirc)*1.25)
    ax[3].set_xlabel('radius')
    ax[3].set_ylabel('circular velocity')

    plt.tight_layout()
    plt.show()

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
            (tab[:,4].astype(float) == incl) *
            (tab[:,5].astype(float) == alpha_deg) *
            (tab[:,6].astype(int  ) == numOrbits) *
            (tab[:,7].astype(float) == regul)
        )
        tab = tab[filt]
        if len(tab) == 0:
            print('No models satisfying all criteria are found in %s' % fileResult)
    except:
        print('File not found: %s' % fileResult)
        tab = numpy.zeros((0,12))
    filenames = tab[:,-1]                 # last column is the filename of LOSVD file for each model
    tab = tab[:,:-1].astype(float)        # remaining columns are numbers
    Mbh = tab[:,0] * tab[:,8]             # the order of parameters is the same as in linePrefix provided to runModel
    ML  = tab[:,8]                        # Upsilon is appended as the first column after those provided in linePrefix
    chi2= numpy.sum(tab[:,10:-1], axis=1) # chi2 values are stored separately for each dataset, but here we combine all of them except regularization penalty
    # launch interactive plot with [OPT] Mbh vs M/L as the two coordinate axes displayed in chi2 plane (may choose a different pair of parameters)
    agama.schwarzlib.runPlot(datasets=datasets, aval=Mbh, bval=ML, chi2=chi2, filenames=filenames,
        # [OPT] various adjustable parameters for the plots (ranges, names, etc.) - most have reasonable default values
        alabel='Mbh', blabel='M/L', alim=(0, 4e8), blim=(0.9, 1.2), vlim=(-500,500),
        v0lim=(-150,150), sigmalim=(40,160), v0err=15.0, sigmaerr=15.0, potential=pot_total)

else:
    exit('Nothing to do!')
