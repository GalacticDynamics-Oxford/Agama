'''
This is a collection of routines and classes for dealing with observationally-constrained
Schwarzchild models. This submodule contains general routines, while the specifics of
each particular galaxy model and observational dataset may be kept in user scripts;
see  example_forstand.py  for a complete example of fitting procedures.
'''

import numpy as _numpy, agama as _agama

### -------------------------------------------- ###
### Specific tools for Multi-Gaussian expansions ###

def getDensityParamsMGE(mass, Sx, Sy, Sz):
    '''
    return a dictionary containing parameters for creating agama.Density object
    corresponding to a single Gaussian component of an MGE
    '''
    return dict( \
        density = 'Spheroid',
        axisRatioY  = Sy/Sx,
        axisRatioZ  = Sz/Sx,
        scaleRadius = 1,
        gamma       = 0,
        beta        = 0,
        alpha       = 1,
        outerCutoffRadius = 2**0.5 * Sx,
        cutoffStrength = 2,
        densityNorm    = mass / ((2*_numpy.pi)**1.5 * Sx * Sy * Sz) )


def makeDensityMGE(tab, distance, length_unit, beta):
    '''
    Construct an agama.Density object corresponding to an axisymmetric MGE read from a text file
    and deprojected assuming the given inclination angle.
    Input:
    tab - array with 3 columns, as read from a text file produced by MGE fitting routines;
    each row contains data for one Gaussian components, columns are:
    central luminosity [Lsun/pc^2], width of the major axis [arcsec], flattening (q<=1).
    distance - assumed distance to the object [kpc], needed to convert arcseconds to parsecs.
    length_unit - the length unit of the script [kpc]
    beta - inclination angle [radians]
    '''
    arcsec2kpc = distance * _numpy.pi / 648000   # conversion factor from arcseconds to kpc
    if 1 - min(tab[:,2])**2 > _numpy.sin(beta)**2:
        raise ValueError('Deprojection is impossible for the given inclination')
    # construct a Density object from each component, and finally the total composite density
    components = [
        _agama.Density(**getDensityParamsMGE(
            # for each Gaussian, the central surface density Sigma0 (1st column) expressed in Msun/pc^2
            # is converted to the total mass  M = 2 pi Sigma0 [pc/"]^2 L^2 q,
            # where L is the major axis length in arcsec (2nd column), q is the axis ratio (3rd column),
            # and an extra conversion factor is needed to transform from 1/pc^2 to 1/"^2
            # (why did ever anyone think of providing the input in these mixed-up units?)
            mass = 2*_numpy.pi * t[0] * (1000*arcsec2kpc * t[1])**2 * t[2],
            # convert scale radii in arcseconds into the model length units
            Sx   = t[1] * arcsec2kpc / length_unit,
            Sy   = t[1] * arcsec2kpc / length_unit,   # two axes are identical
            Sz   = t[1] * arcsec2kpc / length_unit * (1 - (1-t[2]**2) / _numpy.sin(beta)**2)**0.5)  # third is smaller
        ) for t in tab]
    return _agama.Density(*components)


def surfaceDensityMGE(tab, xp, yp):
    '''
    Evaluate the surface density specified by a MGE at a given set of points xp,yp in the image plane
    Input:
    tab - array with 3 columns, as read from a text file produced by MGE fitting routines
    each row contains data for one Gaussian components, columns are:
    central luminosity (Lsun/pc^2), width of the major axis (arcsec), flattening (q<=1).
    xp, yp - two arrays of equal length, specifying the image plane coordinates of points
    where the surface density should be computed
    '''
    return _numpy.sum([comp[0] * _numpy.exp( -0.5 * (xp**2 + (yp/comp[2])**2) / comp[1]**2) for comp in tab], axis=0)


def makeDensityLogHalo(rcore, vcirc, rcutoff=None):
    '''
    Construct an agama.Density object corresponding to a core-isothermal profile
    with the given core radius (rcore) and asymptotic circular velocity (vcirc).
    There is a dedicated Logarithmic potential model, but it has infinite mass and does not tend to zero at r=infinity.
    Therefore, we replace it by a combination of two Spheroid models with the same density profile in the inner part,
    but a cutoff at large radii (by default, rcutoff=100*rcore).
    '''
    if rcutoff is None: rcutoff = 100*rcore
    rho0 = (vcirc/rcore)**2 / (4*_numpy.pi*_agama.G)
    return _agama.Density(
        dict(type='spheroid', alpha=2, beta=2, gamma=-2, densitynorm=  rho0, scaleradius=rcore, outercutoffradius=rcutoff),
        dict(type='spheroid', alpha=2, beta=4, gamma= 0, densitynorm=3*rho0, scaleradius=rcore, outercutoffradius=rcutoff) )


def makeDensityNFWHalo(rscale, vcirc, rcutoff=None):
    '''
    Construct an agama.Density object corresponding to a truncated NFW profile
    with the given scale radius (rscale) and peak circular velocity (vcirc, reached at radius ~2.2 rscale).
    There is a dedicated NFW potential model, but it has infinite mass; therefore, we replace it
    by a Spheroid model with the same density profile in the inner part, but a cutoff at large radii.
    '''
    if rcutoff is None: rcutoff = 100*rscale
    rho0 = (vcirc/rscale/0.465)**2 / (4*_numpy.pi*_agama.G)
    return _agama.Density(type='spheroid', alpha=1, beta=3, gamma=1, densitynorm=rho0, scaleradius=rscale, outercutoffradius=rcutoff)


def makeMGE(particles, masses, beta, distance, plot=True):
    '''
    Construct an MGE fit to an N-body snapshot
    Arguments:
      particles:  Nx3 array of particle coordinates;
      masses:     array of particle masses;
      beta:       inclination angle [rad];
      distance:   distance to the galaxy [Kpc], needed to convert units;
      plot:       if true, show plots of fitted models for visual inspection.
    Return:
      an array of shape Kx3, with K MGE components; columns are:
      central luminosity (Lsun/pc^2), width of the major axis (arcsec), flattening (q<=1).
    '''
    # use Cappellari's mgefit package
    from mgefit.find_galaxy        import find_galaxy
    from mgefit.sectors_photometry import sectors_photometry
    from mgefit.mge_fit_sectors    import mge_fit_sectors
    from mgefit.mge_print_contours import mge_print_contours

    X, Y  = _numpy.dot(_agama.makeRotationMatrix(alpha=0, beta=beta, gamma=0)[0:2], particles[:,:3].T)[0:2]  # image-plane coords
    xmax  = _numpy.percentile(_numpy.abs(X), 99)
    ymax  = _numpy.percentile(_numpy.abs(Y), 99)
    pixel = max(
        _numpy.percentile( (X*X+Y*Y)**0.5, 100./len(particles) ),  # choose the pixel size to contain at least ~100 particles in the center,
        max(xmax,ymax) / 2000 )    # but don't make it too small, or else we run out of memory in constructing the histogram
    gridx = _numpy.linspace(-xmax, xmax, 2*int(round(xmax/pixel)))
    gridy = _numpy.linspace(-ymax, ymax, 2*int(round(ymax/pixel)))
    hist  = _numpy.histogram2d(X, Y, bins=(gridx, gridy), weights=masses)[0].T
    fgal  = find_galaxy(hist, fraction=0.1)
    fgal.theta = 0   # force the position angle to align with the line of nodes (major axis of the projected disk)
    xcenter = len(gridy)*0.5
    ycenter = len(gridx)*0.5
    spho  = sectors_photometry(hist, fgal.eps, fgal.theta, xcenter, ycenter, minlevel=-10, plot=plot)
    mge   = mge_fit_sectors(spho.radius, spho.angle, spho.counts, fgal.eps, ngauss=15, qbounds=(_numpy.cos(beta)*1.001,1),
        sigmapsf=0.0, scale=pixel, linear=False, plot=plot, fignum=2)
    # rescale:
    arcsec2kpc = distance * _numpy.pi / 648000   # conversion factor from arcseconds to kpc
    q        = mge.sol[2]
    sigma    = mge.sol[1] * pixel
    surfdens = mge.sol[0] / (2*_numpy.pi * (1000 * arcsec2kpc * sigma)**2 * q)
    print('Total mass in N-body snapshot: %.4g;  in the image: %.4g;  in MGE: %.4g' % (_numpy.sum(masses), _numpy.sum(hist), sum(mge.sol[0])))

    if plot:
        import matplotlib.pyplot as plt, scipy.ndimage
        plt.figure()
        # several levels of smoothing (for display purposes only):
        # for each histogram bin, choose the smoothing kernel that better matches the Poisson noise in the bin
        meanm= _numpy.mean(masses)
        for w in range(6):
            smooth = 2.0**w
            smhist = scipy.ndimage.gaussian_filter(hist, sigma=smooth**0.5)
            mixfrac= _numpy.tanh(0.1 * smooth * smhist / meanm)**2
            hist   = hist * mixfrac + smhist * (1-mixfrac)
        mge_print_contours(hist, fgal.theta, xcenter, ycenter, mge.sol, binning=4, sigmapsf=1.0, magrange=12, scale=pixel)
        print('*** Close all figures to continue ***')
        plt.show()

    return _numpy.column_stack((surfdens, sigma, q))


### ------------------------------------------------------------------------------- ###
### routines for dealing with 2d IFU data, represented on a regular 2d grid of spaxels,
### several spaxels could be Voronoi-binned together into apertures.
### Agama uses arbitrary polygons that define the boundaries of each aperture,
### whether it is just a square pixel or a more complicated region.

def makeVoronoiBins(particles, gridx, gridy, nbins, alpha=0, beta=0, gamma=0, plot=True):
    '''
    Construct Voronoi binning scheme by maintaining nearly-constant signal-to-noise ratio in each aperture
    Arguments:
      particles - Nx3 array of particle coordinates;
      gridx, gridy - grids (pixel boundaries, not centers) in the image plane;
      nbins - desired number of Voronoi bins, the actual result may differ by 10-20%;
      alpha, beta, gamma - Euler angles specifying the orientation of the image plane in the coordinate system of the N-body snapshot.
    Return:
      three arrays of size nbinx*nbiny: x- and y-coordinates of pixel centers, and bin index for each pixel
    '''
    # use Cappellari's vorbin package
    from vorbin.voronoi_2d_binning import voronoi_2d_binning
    X, Y = _numpy.dot(_agama.makeRotationMatrix(alpha, beta, gamma)[0:2], particles[:,:3].T)[0:2]  # image-plane coords
    hist = _numpy.histogram2d(X, Y, bins=(gridx, gridy))[0].reshape(-1)
    xc   = _numpy.repeat(0.5*(gridx[1:]+gridx[:-1]), len(gridy)-1)
    yc   = _numpy.tile  (0.5*(gridy[1:]+gridy[:-1]), len(gridx)-1)
    snr  = (hist+1)**0.5
    bintags, xNode, yNode, xBar, yBar, sn, nPixels, scale = \
        voronoi_2d_binning(xc, yc, hist, hist**0.5, (1.*_numpy.sum(hist)/nbins)**0.5, plot=plot, quiet=0)
    if plot:
        import matplotlib.pyplot as plt
        plt.gcf().axes[0].set_xlim(plt.gcf().axes[0].get_xlim()[::-1])  # invert X axis as per the convention used in Agama
        print('*** Close all figures to continue ***')
        plt.show()
    return xc, yc, bintags


def getRegularApertures(xcoords, ycoords):
    '''
    Construct boundary polygons corresponding to a regular separable grid in x,y:
    each aperture is a rectangular region spanning x[i]..x[i+1], y[j]..y[j+1].
    Input:  grid nodes in x and y; the number of pixels is (len(xcoords)-1) * (len(ycoords)-1)
    Output: list of pixel boundaries (each element is a 4x2 array of four corner points)
    '''
    ii = range(len(xcoords)-1)
    jj = range(len(ycoords)-1)
    return _numpy.array([ ((xcoords[i],ycoords[j]), (xcoords[i+1],ycoords[j]),
        (xcoords[i+1],ycoords[j+1]), (xcoords[i],ycoords[j+1]))
        for i,j in zip(_numpy.tile(ii, len(jj)), _numpy.repeat(jj, len(ii))) ])


def getBinnedApertures(xcoords, ycoords, bintags):
    '''
    Convert the data for Voronoi binned pixels into the polygons
    describing the boundary of each connected Voronoi region.
    Input: three 1d arrays of equal length: coord x and y of the center of each pixel,
    and bin index for each pixel.
    Output contains a list of length Nbins, each element is the list of vertices
    of the boundary polygon for each bin (a 2d array with 2 columns (x,y) and Nvert rows).
    '''
    binTags = _numpy.unique(bintags)   # list of bin tags (indices)
    xCoords = _numpy.unique(_numpy.round(xcoords*1e6)*1e-6)    # list of possible x coordinates
    yCoords = _numpy.unique(_numpy.round(ycoords*1e6)*1e-6)    # same for y; rounded appropriately
    xpixel  = xCoords[1]-xCoords[0]    # size of a single pixel (assuming they all are equal)
    ypixel  = yCoords[1]-yCoords[0]    # same in y direction
    xcount  = int(round((xCoords[-1]-xCoords[0]) / xpixel)+1)  # total number of pixels in x direction
    ycount  = int(round((yCoords[-1]-yCoords[0]) / ypixel)+1)  # same for y
    if xcount > 10000 or ycount > 10000:
        raise ValueError("Can't determine pixel size: "+str(xpixel)+" * "+str(ypixel)+" doesn't seem right")
    polygons= []        # list of initially empty polygons
    matrix  = _numpy.ones((xcount,ycount), dtype=_numpy.int32) * -1  # 2d array of binTag for each pixel
    for (p,x,y) in zip(bintags, xcoords, ycoords):  # assign bin tags to each pixel in the 2d array
        matrix[int(round((x-xCoords[0]) / xpixel)), int(round((y-yCoords[0]) / ypixel)) ] = int(p)
    for b in binTags:
        # obtain the table of x- and y-indices of all elements with the same binTag
        ix, iy = (matrix==b).nonzero()
        if len(ix)==0:
            print('Empty bin %i' % b)
            continue
        minx   = min(ix)-1
        maxx   = max(ix)+2
        miny   = min(iy)-1
        maxy   = max(iy)+2
        pixels = _numpy.zeros((maxx-minx, maxy-miny), dtype=_numpy.int32)
        for kx,ky in zip(ix,iy):
            pixels[kx-minx, ky-miny] = 1
        # start at the top left corner and move counterclockwise along the boundary
        vertices = []
        cx, cy = ix[0]-minx, iy[0]-miny
        direc  = 'd'  # first go down
        edges  = 0
        while edges < 1000:
            if direc == 'd':
                cx += 1
                if   pixels[cx,   cy-1]: direc = 'l'
                elif pixels[cx,   cy  ]: direc = 'd'
                elif pixels[cx-1, cy  ]: direc = 'r'
                else: direc = '?'
            elif direc == 'r':
                cy += 1
                if   pixels[cx,   cy  ]: direc = 'd'
                elif pixels[cx-1, cy  ]: direc = 'r'
                elif pixels[cx-1, cy-1]: direc = 'u'
                else: direc = '?'
            elif direc == 'u':
                cx -= 1
                if   pixels[cx-1, cy  ]: direc = 'r'
                elif pixels[cx-1, cy-1]: direc = 'u'
                elif pixels[cx  , cy-1]: direc = 'l'
                else: direc = '?'
            elif direc == 'l':
                cy -= 1
                if   pixels[cx-1, cy-1]: direc = 'u'
                elif pixels[cx,   cy-1]: direc = 'l'
                elif pixels[cx,   cy  ]: direc = 'd'
                else: direc = '?'
            if direc == '?':
                raise ValueError("Can't construct boundary polygon for bin "+str(b))
                break
            else:
                vertices.append( \
                    ( xpixel * (cx+minx-0.5) + xCoords[0], ypixel * (cy+miny-0.5) + yCoords[0] ))
            if cx+minx == ix[0] and cy+miny == iy[0]:
                break   # reached the initial point
            edges += 1
        if edges>=1000:
            raise ValueError('Lost in the way for bin '+str(b))
        polygons.append( _numpy.array(vertices) )
    return _numpy.array(polygons)


def writeApertures(filename, polygons):
    '''
    Write the list of polygons serving as aperture boundaries to a text file
    '''
    with open(filename, 'w') as dfile:
        for i,polygon in enumerate(polygons):
            for vertex in polygon:  dfile.write('%f %f\n' % (vertex[0], vertex[1]))
            if i<len(polygons)-1:   dfile.write('\n')


def readApertures(filename):
    '''
    Read the list of polygons from a text file
    '''
    with open(filename) as dfile:
        return _numpy.array([
            _numpy.array([float(a) for a in b.split()]).reshape(-1,2)
            for b in dfile.read().split('\n\n')
        ])


def makeGridForTargetLOSVD(polygons, psf, psfwingfrac=0.99, psfcorefrac=0.2, pixelmult=1.0):
    '''
    Construct a suitable regular grid for a LOSVD Target, which covers all apertures with some extra margin
    beyond the outer boundaries, to account for PSF smoothing.
    Arguments:
      polygons:  list of Voronoi-binned polygons with coordinates aligned with the X,Y grid;
      the pixel size of the input grid is determined automatically.
      psf:       a single float (width of the Gaussian PSF) or a list of Gaussian components [width, weight].
      psfwingfrac [default 0.99]: fraction of PSF light covered by the internal grid; determines
      the width of additional margins around the input grid to account for PSF wings.
      psfcorefrac [default 0.20]: fraction of PSF light in the core covered by one pixel of the internal grid;
      the pixel size determined from the input grid may be increased to contain at least this fraction of PSF.
      pixelmult [default 1.0]: coefficient of proportionality between the pixel size of the internal grid
      and the pixel size of the input grid (smaller => higher-resolution internal grid);
      1.0 is good enough (accuracy ~1%) for 2nd or 3rd-degree B-splines.
    '''
    import scipy.special, scipy.optimize
    # first, determine the min/max coords and the pixel size of the voronoi-binned regular grid
    pixel = _numpy.inf
    xmin  = _numpy.inf
    xmax  =-_numpy.inf
    ymin  = _numpy.inf
    ymax  =-_numpy.inf
    for p in polygons:
        delta = ( (p[1:,0]-p[:-1,0])**2 + (p[1:,1]-p[:-1,1])**2 )**0.5  # lengths of polygon edges
        pixel = min(pixel, _numpy.amin(delta[delta!=0]))
        xmin  = min(xmin,  _numpy.amin(p[:,0]))
        xmax  = max(xmax,  _numpy.amax(p[:,0]))
        ymin  = min(ymin,  _numpy.amin(p[:,1]))
        ymax  = max(ymax,  _numpy.amax(p[:,1]))
    # second, determine the "safety margin" -- how much extra space one needs to reserve beyond
    # the extent of the input polygons, in order to account for PSF smoothing.
    # we require that >=psfwingfrac of PSF light should be within the grid boundaries [ADJUSTABLE].
    # also determine the minimum pixel size containing a given fraction of PSF in the core [ADJUSTABLE].
    if isinstance(psf, (int,float)):
        margin = scipy.special.erfinv(psfwingfrac) * 2**0.5 * psf
        minpix = scipy.special.erfinv(psfcorefrac) * 2**1.5 * psf
    else:  # psf is an array with more than one element
        fnc = lambda x: sum([p[1] * scipy.special.erf(x / p[0] / 2**0.5) for p in psf]) - psfwingfrac
        margin = scipy.optimize.brentq(fnc, 0, 10*max([p[0] for p in psf]))
        fnc = lambda x: sum([p[1] * scipy.special.erf(x / p[0] / 2**1.5) for p in psf]) - psfcorefrac
        minpix = scipy.optimize.brentq(fnc, 0, 10*max([p[0] for p in psf]))
    # assign the pixel size for the internal grid to be proportional [ADJUSTABLE]
    # to the smallest resolution element of the input voronoi-binned dataset,
    # but no smaller than minpix (the width of the PSF core).
    pix  = max(pixel * pixelmult, minpix)
    Xmin = int(_numpy.floor((xmin-margin) / pix))
    Xmax = int(_numpy.ceil ((xmax+margin) / pix))
    Ymin = int(_numpy.floor((ymin-margin) / pix))
    Ymax = int(_numpy.ceil ((ymax+margin) / pix))
    # if |Xmin| ~= Xmax, make the grid fully symmetric (this saves some cpu time), similarly for Y
    if Xmin<0 and Xmax>0 and (abs(Xmin+Xmax) <= 3 or (-Xmin/Xmax>=0.8) and (-Xmin/Xmax<=1.25)):
        Xmax = max(Xmax, -Xmin)
        Xmin = -Xmax
    if Ymin<0 and Ymax>0 and (abs(Ymin+Ymax) <= 3 or (-Ymin/Ymax>=0.8) and (-Ymin/Ymax<=1.25)):
        Ymax = max(Ymax, -Ymin)
        Ymin = -Ymax
    # sanity check
    if Xmax-Xmin > 1000 or Ymax-Ymin > 1000:
        raise ValueError("Can't construct a suitable grid (pixel size "+str(pix)+" doesn't seem right")
    # finally, construct two regularly-spaced grids
    gridx = _numpy.linspace(Xmin*pix, Xmax*pix, Xmax-Xmin+1)
    gridy = _numpy.linspace(Ymin*pix, Ymax*pix, Ymax-Ymin+1)
    print(('Kinematic spaxel size is %.2f; %d apertures in the region [%.2f:%.2f, %.2f:%.2f]; '+
        'PSF is %s; internal kinematic %dx%d grid is [%.2f:%.2f, %.2f:%.2f] with pixel size %.2f') %
        (pixel, len(polygons), xmin, xmax, ymin, ymax,
        str(psf), len(gridx), len(gridy), min(gridx), max(gridx), min(gridy), max(gridy), pix))
    return gridx, gridy


def ghMomentsErrors(degree, gridv, values, errors, ghorder, nboot=100):
    '''
    Compute the Gauss-Hermite moments and their error estimates from the provided values and errors
    of B-spline representation of LOSVDs by bootstapping
    '''
    ghm_values = _agama.ghMoments(degree=degree, gridv=gridv, matrix=values, ghorder=ghorder)
    bootstraps = _numpy.vstack([values] * nboot)
    bootstraps+= _numpy.vstack([errors] * nboot) * _numpy.random.normal(size=bootstraps.shape)
    boot_ghm   = _agama.ghMoments(degree=degree, gridv=gridv, ghorder=ghorder, matrix=_numpy.maximum(0, bootstraps))
    ghm_errors = _numpy.std(boot_ghm.reshape(nboot, -1), axis=0).reshape(ghm_values.shape)
    return ghm_values, ghm_errors


### --------------------------------------------------------------------- ###
### Dataset classes - combination of Target objects and constraint values ###

class DensityDataset:
    '''
    Class for representing the Target object and constraints for the 3d density profile
    '''
    def __init__(self, density, tolerance=0, **target_params):
        '''
        Arguments:
          density:   3d density profile of stars
          tolerance: relative error on density constraints
          target_params:  all other parameters of the Target object (including type='Density***')
        '''
        self.target_params = target_params
        self.target = _agama.Target(**target_params)
        # 0th constraint is the total mass, with zero tolerance; remaining ones are cell masses; everything normalized to totalMass
        self.cons_val = _numpy.hstack(( 1, self.target(density) / density.totalMass() ))
        self.cons_err = _numpy.hstack(( 0, abs(self.cons_val[1:]) * tolerance ))
        print('Fraction of mass in %i density bins: %g' % (len(self.target), sum(self.cons_val[1:])))

    def getOrbitMatrix(self, density_matrix, Upsilon):
        '''
        Produce the matrix of orbit contributions to each of the constraints
        from the matrix recorded during orbit integration by the Target object
        '''
        # 0th column is the contribution of each orbit to the total mass, i.e. 1
        return _numpy.column_stack(( _numpy.ones(len(density_matrix)), density_matrix ))

    def getPenalty(self, model_dens, Upsilon):
        '''
        Compute the penalty for all density constraints from the vector of values
        for the best-fit model
        '''
        # only compute the penalty for constraints that had nonzero associated errors
        use = self.cons_err[1:]!=0
        dif = ((model_dens[use] - self.cons_val[1:][use]) / self.cons_err[1:][use])**2
        return _numpy.sum(dif)


class KinemDatasetGH:
    '''
    Class for representing the Target(type='LOSVD') and observational constraints in the form of Gauss-Hermite moments
    '''
    def __init__(self, density, ghm_val, ghm_err, tolerance=0, **target_params):
        '''
        Arguments:
          ghm_val:  N x M array of observational constraints provided as Gauss-Hermite moments:
            N is the number of apertures,
            M is the order of GH expansion = number of constraints in each aperture
            columns are:  v, sigma, h_3, h_4, ... h_M
          ghm_err:  N x M array of observational error estimates, in the same order
          density:  3d density profile of stars, needed to compute the normalizations of LOSVDs (aperture masses)
          tolerance:  fractional error on aperture masses
          target_params:  all other parameters of the Target(type='LOSVD', ...),
            except that when gridx, gridy are not provided, they will be constructed automatically
        '''
        if not 'gridx' in target_params:
            target_params['gridx'], target_params['gridy'] = makeGridForTargetLOSVD(
                target_params['apertures'], target_params.get('psf', 0))
        self.target_params = target_params
        self.target = _agama.Target(**target_params)
        self.mod_degree = target_params['degree']
        self.mod_gridv  = target_params['gridv']
        self.num_bsplines = len(self.mod_gridv) + self.mod_degree - 1
        self.num_aper, self.num_cons = ghm_val.shape  # N, M; num_cons = order of GH expansion
        if ghm_val.shape != ghm_err.shape:
            raise ValueError('ghm_val and ghm_err must have the same shape')
        # surface density convolved with PSF and integrated over the area of each aperture, normalized to totalMass
        self.aperture_mass = self.target(density) / density.totalMass()
        self.aperture_mass_err = self.aperture_mass * tolerance
        if len(self.aperture_mass) != self.num_aper:
            raise ValueError('ghm_val should have the same number of rows as the number of apertures')
        # contributions of all even GH moments to the overall normalization
        norm = _numpy.ones(self.num_aper)
        mult = 0.5**0.5
        for m in range(4, self.num_cons+1, 2):
            mult *= (1-1./m)**0.5
            norm += ghm_val[:, m-1] * mult
        # parameters of GH series in each aperture: amplitude, center, width
        v0    = ghm_val[:,0]
        sigma = ghm_val[:,1]
        self.ghbasis  = _numpy.column_stack(( self.aperture_mass / norm, v0, sigma ))
        # constraint values: aperture masses and GH moments h_1..h_M, with h_1 = h_2 = 0
        self.cons_val = _numpy.column_stack((
            self.aperture_mass,
            ghm_val[:,0:2]*0,
            ghm_val[:,2:]
        )).reshape(-1)
        # constraint errors: translate errors in v0, sigma into errors in h_1,h_2
        self.cons_err = _numpy.column_stack((
            self.aperture_mass_err,
            ghm_err[:,0:2] / sigma[:,None] / 2**0.5,
            ghm_err[:,2:]
        )).reshape(-1)
        # also store the original arrays (v,sigma,h_3...h_M) and their error estimates
        self.ghm_val  = ghm_val
        self.ghm_err  = ghm_err

    def getOrbitMatrix(self, kinem_matrix, Upsilon):
        '''
        Produce the matrix of orbit contributions to each of the kinematic constraints (incl. aperture mass)
        from the matrix recorded during orbit integration by the LOSVD Target object.
        The LOSVDs of each orbit are converted into GH moments _in the observational GH basis_ (v,sigma),
        so that the values h1,h2,h3... can be compared with the observational constraints.
        Upsilon is the mass-to-light ratio of the model, so that its velocity grid is scaled by sqrt(Upsilon)
        before computing the GH moments.
        Returns: the matrix to be provided to the optimization solver
        '''
        num_orbits = len(kinem_matrix)
        mod_bsint  = _agama.bsplineIntegrals(self.mod_degree, self.mod_gridv)
        return _numpy.dstack((
            # 0th column is the contribution of each orbit to aperture masses
            kinem_matrix.reshape(num_orbits, self.num_aper, self.num_bsplines).dot(mod_bsint),
            # remaining columns are GH moments h_1..h_M computed in the observed GH basis
            # after rescaling the model velocity grid by sqrt(Upsilon) and
            # multiplying the values of coefficients by 1/sqrt(Upsilon)
            _agama.ghMoments(
            degree=self.mod_degree, gridv=self.mod_gridv * Upsilon**0.5,
            matrix=kinem_matrix, ghorder=self.num_cons, ghbasis=self.ghbasis).
            reshape(num_orbits, self.num_aper ,  self.num_cons+1)[:,:,1:] * Upsilon**-0.5
        )). reshape(num_orbits, self.num_aper * (self.num_cons+1))

    def getPenalty(self, model_losvd, Upsilon):
        '''
        Compute the penalty for kinematic constraints from the array of LOSVDs of the model.
        Arguments:
          model_losvd:  the array of length num_aper * num_bsplines  containing the LOSVDs of
          the entire model (i.e., the sum of LOSVDs of each orbit multiplied by orbit weights).
          Upsilon:  mass-to-light ratio of the model, used to scale the velocity grid.
          Now the GH moments for the model are computed in the model's best-fit GH basis,
          so that h1=h2=0, and we compare the values of v,sigma,h3...hM with the observed ones.
        Returns: the error-weighted penalty (chi^2) separately for each GH term
          (including the aperture mass as the 0th element) summed over all apertures.
        '''
        ghm_mod   = _agama.ghMoments(
            degree=self.mod_degree, gridv=self.mod_gridv * Upsilon**0.5,
            matrix=model_losvd * Upsilon**-0.5, ghorder=self.num_cons). \
            reshape(self.num_aper,  self.num_cons+4)
        # keep only the columns v,sigma,h_3...h_M
        ghm_mod   = _numpy.column_stack(( ghm_mod[:,(1,2)], ghm_mod[:,6:] ))
        # add the penalty for aperture mass, if the tolerance was not zero
        mod_bsint = _agama.bsplineIntegrals(self.mod_degree, self.mod_gridv)
        apmass_mod= model_losvd.reshape(self.num_aper, self.num_bsplines).dot(mod_bsint)
        use       = self.aperture_mass_err != 0
        return _numpy.hstack((
            # penalty for aperture mass
            _numpy.sum( ((apmass_mod - self.aperture_mass)[use] / self.aperture_mass_err[use])**2 ),
            # chi2 w.r.t. originally provided data (separately for v,sigma and each GH term)
            _numpy.sum( ((ghm_mod - self.ghm_val) / self.ghm_err)**2, axis=0 )  # array of length M
        ))

    def getGHMoments(self):
        '''
        Return v,sigma,h3..hM of the observational dataset
        '''
        return self.ghm_val, self.ghm_err

    def getLOSVD(self, gridv):
        '''
        Construct the interpolated LOSVD profiles from the observed GH moments and their uncertainties (used in runPlot).
        Arguments:
          gridv:  grid for computing the profiles
        Returns:  array of shape  3 x num_aper x len(gridv)  containing 16, 50 and 84th percentiles of LOSVDs in each aperture
        '''
        nboot = 100
        result= _numpy.zeros((3, self.num_aper, len(gridv)))
        for n in range(self.num_aper):
            boots = _agama.ghInterp(
                self.ghbasis[n,0],  # amplitude
                self.ghm_val[n,0] + _numpy.random.normal(size=nboot) * self.ghm_err[n,0],  # center
                self.ghm_val[n,1] + _numpy.random.normal(size=nboot) * self.ghm_err[n,1],  # width
                _numpy.vstack((_numpy.ones(nboot), _numpy.zeros((2,nboot)),  # coefs: h0=1, h1=h2=0, remaining are perturbed by noise
                    self.ghm_val[n,2:][:,None] + _numpy.random.normal(size=(self.num_cons-2, nboot)) * self.ghm_err[n,2:][:,None])),
                gridv)
            result[:,n] = _numpy.percentile(boots, axis=1, q=[16,50,84])
        return result


class KinemDatasetHist:
    '''
    Class for representing the Target(type='LOSVD') and observational constraints in the form of (generalized) LOSVD histograms
    '''
    def __init__(self, density, obs_degree, obs_gridv, obs_val, obs_err, tolerance=0, **target_params):
        '''
        Arguments:
          obs_degree:  B-spline degree of the observational dataset;
            in the case of histograms, degree=0, but in general, the data may be provided as
            linearly-interpolated LOSVDs (degree=1) or even higher-degree B-splines
            (although the errors are still assumed to be uncorrelated, which is likely incorrect for degree>1)
          obs_gridv:  velocity grid defining the histogram boundaries or, more generally, breakpoints of B-splines;
            in particular, for histograms these are boundaries, not centers of velocity bins.
          obs_val:  N x M array of observational constraints provided as values of LOSVD histograms:
            N apertures, M values in each LOSVD,  with M = len(obs_gridv) + obs_degree - 1
          obs_err:  N x M array of observational error estimates, in the same order;
          density:  3d density profile of stars, needed to compute the normalizations of LOSVDs (aperture masses);
          tolerance:  fractional error on aperture masses;
          target_params:  all other parameters of the Target(type='LOSVD', ...),
            except that when gridx, gridy are not provided, they will be constructed automatically
        '''
        if not 'gridx' in target_params:
            target_params['gridx'], target_params['gridy'] = makeGridForTargetLOSVD(
                target_params['apertures'], target_params.get('psf', 0))
        self.target_params = target_params
        self.target = _agama.Target(**target_params)
        self.mod_degree = target_params['degree']
        self.mod_gridv  = target_params['gridv']
        self.num_bsplines = len(self.mod_gridv) + self.mod_degree - 1
        self.num_aper, self.num_cons = obs_val.shape  # N, M; num_cons = number of histogram bins/coefs

        if obs_val.shape != obs_err.shape:
            raise ValueError('obs_val and obs_err must have the same shape')
        # surface density convolved with PSF and integrated over the area of each aperture, normalized to totalMass
        self.aperture_mass = self.target(density) / density.totalMass()
        self.aperture_mass_err = self.aperture_mass * tolerance
        if len(self.aperture_mass) != self.num_aper:
            raise ValueError('obs_val should have the same number of rows as the number of apertures')
        if self.num_cons != len(obs_gridv) + obs_degree - 1:
            raise ValueError('the number of columns in obs_val conflicts with obs_degree and obs_gridv')
        self.obs_degree, self.obs_gridv = obs_degree, obs_gridv
        # row-normalize the provided histograms
        obs_bsint = _agama.bsplineIntegrals(self.obs_degree, self.obs_gridv)
        obs_norm  = obs_val.dot(obs_bsint)
        # constraint values and errors: provided histograms normalized by aperture masses
        self.obs_val = obs_val * (self.aperture_mass / obs_norm)[:,None]
        self.obs_err = obs_err * (self.aperture_mass / obs_norm)[:,None]
        self.cons_val = _numpy.column_stack((self.aperture_mass, self.obs_val)).reshape(-1)
        self.cons_err = _numpy.column_stack((self.aperture_mass_err, self.obs_err)).reshape(-1)

    def _getConvMatrix(self, Upsilon):
        # conversion matrix from the amplitudes of B-splines in the model basis
        # into the values of histograms in the observed basis plus one column for aperture mass constraints
        return _numpy.column_stack((
            # 0th column is the contribution of all B-splines in the model to aperture masses
            _agama.bsplineIntegrals(self.mod_degree, self.mod_gridv),
            # remaining columns are contributions of each B-spline term in the model
            # to each velocity bin in the histogram
            Upsilon**-0.5 * _numpy.linalg.solve(
            _agama.bsplineMatrix(self.obs_degree, self.obs_gridv),
            _agama.bsplineMatrix(self.obs_degree, self.obs_gridv, self.mod_degree, self.mod_gridv * Upsilon**0.5) ).T
        ))

    def getOrbitMatrix(self, kinem_matrix, Upsilon):
        '''
        Produce the matrix of orbit contributions to each of the kinematic constraints (incl. aperture mass)
        from the matrix recorded during orbit integration by the LOSVD Target object.
        The B-splines of model LOSVDs are converted into the amplitudes of histograms in the observational velocity grid.
        '''
        num_orbits = len(kinem_matrix)
        return kinem_matrix. \
            reshape(num_orbits, self.num_aper, self.num_bsplines). \
            dot(self._getConvMatrix(Upsilon)). \
            reshape(num_orbits, self.num_aper * (self.num_cons+1))

    def getPenalty(self, model_losvd, Upsilon):
        '''
        Compute the penalty for kinematic constraints from the array of LOSVDs of the model.
        Arguments:
          model_losvd:  the array of length num_aper * num_bsplines  containing the LOSVDs of
          the entire model (i.e., the sum of LOSVDs of each orbit multiplied by orbit weights).
          Upsilon:  mass-to-light ratio of the model, used to scale the velocity grid.
          As in getOrbitMatrix, the B-splines are converted/rebinned into the observational velocity grid.
        Returns: the error-weighted penalty (chi^2) separately for each bin in the histogram
          (including the aperture mass as the 0th element) summed over all apertures.
        '''
        cons_mod = model_losvd.reshape(self.num_aper, self.num_bsplines). \
            dot(self._getConvMatrix(Upsilon)). \
            reshape(-1)
        use   = self.cons_err != 0
        error = _numpy.zeros(self.num_aper * (self.num_cons+1))
        error[use] = ( (cons_mod[use] - self.cons_val[use]) / self.cons_err[use] )**2
        return _numpy.sum(error.reshape(self.num_aper, self.num_cons+1), axis=0)

    def getGHMoments(self, ghorder=6):
        '''
        Compute GH moments from the histogrammed LOSVDs
        Return: array of size num_aper x ghorder containing the values  v,sigma,h3..hM  in each aperture
        '''
        ghm_val, ghm_err = ghMomentsErrors(degree=self.obs_degree, gridv=self.obs_gridv,
            values=self.obs_val, errors=self.obs_err, ghorder=ghorder)
        ind = tuple([1,2]+range(6,ghorder+4))  # indices of columns containing v,sigma,h3...hM in the matrix returned by ghMoments
        return ghm_val[:,ind], ghm_err[:,ind]

    def getLOSVD(self, gridv):
        '''
        Construct the interpolated LOSVD profiles from the histograms and their uncertainties (used in runPlot).
        Arguments:
          gridv:  grid for computing the profiles
        Returns:  array of shape  3 x num_aper x len(gridv)  containing 16, 50 and 84th percentiles of LOSVDs in each aperture
        '''
        nboot = 100
        result= _numpy.zeros((3, self.num_aper, len(gridv)))
        for n in range(self.num_aper):
            boots = _agama.bsplineInterp(
                self.obs_degree, self.obs_gridv,
                _numpy.column_stack([self.obs_val[n]] * nboot) +
                _numpy.column_stack([self.obs_err[n]] * nboot) * _numpy.random.normal(size=(self.num_cons, nboot)),
                gridv)
            result[:,n] = _numpy.percentile(boots, axis=1, q=[16,50,84])
        return result



def runModel(datasets, potential, ic, Omega=0, intTime=100.0,
    Upsilon=1.0, regul=1.0, multstep=2**(1./10), deltaChi2=100.0, filePrefix='', linePrefix='', fileResult='results.dat'):
    '''
    Construct the orbit library for the given potential and datasets/constraints,
    and solve the optimization problem multiple times with varying values of mass-to-light ratio.
    Arguments:
      datasets:   a list of objects containing targets and constraints
      potential:  total potential in which to integrate the orbits
      ic:         initial conditions for the orbits (Nx6 array)
      Omega:      pattern speed
      intTime:    integration time in units of dynamical time of each orbit
      regul:      regularization parameter for the solution
      Upsilon:    initial value of M/L for the search
      multstep:   multiplicative increment/decrement for Upsilon during the search
      deltaChi2:  search stops when the chi2 value of the best-fit model is bracketed from both larger and smaller Upsilon values by at least that much
      filePrefix: common prefix for filenames storing the model LOSVDs and orbit parameters; the value of Upsilon is appended to the filename
      linePrefix: data written at the beginning of each line in the results file, before the values of Upsilon and chi2 for each dataset are appended
      fileResult: the name of the text file storing the summary information
    Returns: chi2 of the best-fit model in this series (also stores all models in the result file)
    '''

    try: len(datasets)   # ok if it is a tuple/list
    except: datasets = (datasets,)  # if not, make it a tuple
    if len(datasets)==0:
        raise ValueError('Need at least one dataset for the model')

    # build orbit library
    matrices = _agama.orbit(potential=potential, ic=ic, time=potential.Tcirc(ic) * intTime, Omega=Omega,
        targets=[d.target for d in datasets], trajsize=1000)
    trajs    = matrices[-1]   # list of orbit trajectories
    matrices = matrices[:-1]  # matrices corresponding to datasets
    assert(len(matrices) == len(datasets))

    # record various structural properties of orbits
    Rm = _numpy.array([ _numpy.mean( _numpy.sum(t[:,0:3]**2,axis=1)**0.5 ) for t in trajs[:,1] ])  # mean radius of each orbit
    Lz = _numpy.array([ _numpy.mean( t[:,0]*t[:,4]-t[:,1]*t[:,3] ) for t in trajs[:,1] ])
    L2 = _numpy.array([ _numpy.mean((t[:,0]*t[:,4]-t[:,1]*t[:,3])**2 + (t[:,1]*t[:,5]-t[:,2]*t[:,4])**2 + (t[:,2]*t[:,3]-t[:,0]*t[:,5])**2) for t in trajs[:,1] ])
    E  = potential.potential(ic[:,0:3]) + _numpy.sum(ic[:,3:6]**2, axis=1) * 0.5  # total energy of each orbit
    Rc = potential.Rcirc(E=E)                      # radius of a circular orbit with the given energy
    Lc = 2*_numpy.pi * Rc**2 / potential.Tcirc(E)  # angular momentum of a circular orbit with this energy
    Ci = Lz / L2**0.5  # Lz/L = cos(incl)
    Lr = L2 / Lc**2    # (L/Lc)^2 = 1-ecc^2 (at least for a Kepler orbit)

    def solve(Upsilon):
        '''
        Solve the optimization problem for the previously recorded orbit library.
        Arguments:
            Upsilon: mass-to-light ratio, which determines the velocity scaling (~Upsilon**0.5)
            regul:   regularization parameter, values of order unity provide a good balance between smoothing and bias
        Returns:     combined chi^2 for all datasets (excluding regularization penalty)
        '''
        num_dof   = sum([ sum(d.cons_err>0) for d in datasets ])
        # scaling factor for RHS and weights - needed because CVXOPT is not entirely scale-invariant
        # and may fail to find the solution if the order of magnitude of the solution vector is too large.
        # therefore, we multiply the r.h.s. by 1/mult, and the weights (solution) by mult, with the magnitude
        # of mult chosen empirically to speed up convergence but without compromising the accuracy.
        # the absolute values reported by the solver are chi^2/Ndof and should be around unity for a decent fit.
        mult      = num_dof**0.5   * 10  # may make it somewhat larger, but not too much as it ruins accuracy

        # prepare the matrix equation
        matrix    = [ d.getOrbitMatrix(m, Upsilon).T for d,m in zip(datasets, matrices) ]
        rhs       = [ d.cons_val/mult for d in datasets ]
        pen_cons  = [ 2*d.cons_err**-2 for d in datasets ]
        # regularization penalty
        numOrbits = len(ic)
        totalMass = 1.0  # weights are normalized to total mass of unity
        pen_reg   = 2. * regul * _numpy.ones(numOrbits) * numOrbits / totalMass**2

        print('%s, Upsilon=%5.3f: solving...' % (filePrefix, Upsilon))
        # solve the matrix equation
        try:
            weights = _agama.solveOpt(matrix=matrix, rhs=rhs, rpenq=pen_cons, xpenq=pen_reg) * mult
        except Exception as e:
            print('Error! %s' % str(e))
            # arbitrarily set uniform weights
            weights = _numpy.ones(numOrbits) * totalMass / numOrbits

        # analyze the differences in rhs and print out the results
        superpositions = [ weights.dot(m) for m in matrices ]
        penalties = [ d.getPenalty(s, Upsilon) for d,s in zip(datasets, superpositions) ]
        penReg  = 0.5 * _numpy.sum(weights**2 * pen_reg)
        entropy = -sum(weights * _numpy.log(weights+1e-100)) / totalMass + _numpy.log(totalMass / numOrbits)
        # number of orbits contributing 0.999 of total mass
        numUsed =  sum(_numpy.cumsum(_numpy.sort(weights)) > 0.001 * totalMass)
        losvd   = []
        penalty = []  # penalties for all datasets and for regularization
        print('%s, Upsilon=%5.3f: results:' % (filePrefix, Upsilon))
        for d,p,s in zip(datasets, penalties, superpositions):
            print('Penalty for %i %s constraints: %s %g' %
                (len(d.cons_val), str(d.target), str(p), _numpy.sum(p)))
            penalty.append(_numpy.sum(p))
            try:
                # note: we assume that gridv is the same for all kinematic datasets!
                loshdr = 'Degree: %i Grid: %s' % \
                    (d.mod_degree, '\t'.join(['%.1f' % x for x in d.mod_gridv * Upsilon**0.5]))
                losvd.extend(s.reshape(d.num_aper, d.num_bsplines) * Upsilon**-0.5)
            except AttributeError:
                pass  # not a kinematic dataset

        print('Penalty for regularization: %7.2f;  entropy: %.3g;  # of useful orbits: %i / %i' %
            (penReg, entropy, numUsed, numOrbits))

        # write out model LOSVD
        fileNameLOS = filePrefix + '_Y%5.3f.los' % Upsilon
        _numpy.savetxt(fileNameLOS, _numpy.vstack(losvd), fmt='%.4g', header=loshdr)

        # save the orbit weights and properties (radius, eccentricity, inclination)
        fileNameORB = filePrefix + '_Y%5.3f.npz' % Upsilon
        _numpy.savez_compressed(fileNameORB,
            weights=weights.astype(_numpy.float32),
            Rm=Rm.astype(_numpy.float32),
            Lr=Lr.astype(_numpy.float32),
            Ci=Ci.astype(_numpy.float32))

        # append results to the summary file
        with open(fileResult, 'a') as fileout:
            fileout.write('\t'.join(
                [linePrefix, '%5.3f' % Upsilon] +
                ['%7.2f' % _numpy.sum(p) for p in penalty+[penReg] ] +
                [fileNameLOS] ) + '\n')

        return sum(penalty)  # chi2 for all datasets, excluding regularization penalty

    # Solve the optimization problem multiple times with different values of mass-to-light ratio,
    # starting from 'Upsilon' and going up or down in [logarithmic] steps `multstep',
    # until the best-fit model is bracketed from both ends by model with at least
    # 'deltaChi2' difference in fit quality.
    best = solve(Upsilon)
    chia = best
    chib = best
    Upsa = Upsilon
    Upsb = Upsilon
    while min(chia, chib) <= best+deltaChi2:
        if chia<=chib:
            Upsa *= multstep
            chia = solve(Upsa)
            best = min(best, chia)
        else:
            Upsb /= multstep
            chib = solve(Upsb)
            best = min(best, chib)
    return best


def runPlot(datasets,                           # list of [kinematic] datasets to be plotted
    aval=[], bval=[], chi2=[], filenames=[],    # data for the chi2(a,b) plot, and the list of associated LOSVD files
    interp='linear', alabel='', blabel='',      # parameters  for the chi2(a,b) plot
    alim=None, blim=None, deltaChi2lim=100,     # axes ranges for the chi2(a,b) plot
    xlim=None, ylim=None,                       # ranges for x,y coordinates on kinematic maps
    v0lim=None, sigmalim=None, hlim=(-0.1,0.1), # ranges for color axes on kinematic maps for v0,sigma,h3..h6
    v0err=None, sigmaerr=None, herr=0.1,        # ranges for color axes on maps of errors in  v0,sigma,h3..h6
    vlim=None                                   # range of the velocity axis for the LOSVD plot
    ):
    '''
    Show an interactive plot with several panels:
    - kinematic maps (v,sigma and Gauss-Hermite moments up to h6) with the choice between
      measured values (data), errors (data err), model values (model) and
      differences between model and data, normalized by data errors (model err);
      the measurements are provided in spatial regions (apertures) which can be examined
      interactively.
    - line-of-sight velocity distribution in the selected aperture on the map.
    - two-dimensional plot of Delta chi^2 contours in a grid of models (the difference in
      fit quality between the best-fit model and all other models); one can pick the model
      from the grid and load its kinematic maps.
    - distributions of orbit weights of the selected model in the space of mean radius vs.
      normalized squared angular momentum [L/L_circ(E)]^2 or inclination L_z/L.

    Arguments:
      datasets:  a list of Dataset objects defining the apertures and measured values/errors.
      aval, bval, chi2, filenames:  arrays of equal length specifying the horizontal (a) and
      vertical (b) coordinates of the model grid on the chi2 plot (e.g., Mbh vs M/L),
      the corresponding chi2 values, and the list of LOSVD files corresponding to each model.
      interp:  choice of interpolation method for plotting the chi2 contours, possible values:
      'linea' (default), 'cubic' and 'rbf' (radial basis functions).
      alabel, blabel:  the names of the physical quantities on the two axes of the chi2 plot.
      alim, blim:  display ranges for the two axes of the chi2 plot (chosen automatically
      if not provided).
      deltaChi2lim:  color scale range for the delta chi^2 values on this plot.
      xlim, ylim:  display ranges for the kinematic maps (chosen automatically if not given).
      v0lim, sigmalim, hlim:  color scale ranges for the kinematic maps (v, sigma and four
      higher GH moments sharing the same range), each one is a tuple of lower and upper limit
      (chosen automatically if not provided).
      v0err, sigmaerr, herr:  color scale ranges for the maps of measurement errors
      (single value for each of the three arguments, by default taken from the data errors).
      vlim:  velocity range for the LOSVD plot (autodetect if not provided).
    '''

    import re, scipy.interpolate, matplotlib, matplotlib.pyplot

    # placeholder for data shared between subroutines
    class this: pass

    def loadModel(modelIndex):
        '''
        load the model LOSVD file, compute GH moments
        '''
        modelfilename = filenames[modelIndex]
        this.modellabel.set_text(modelfilename)
        this.selected.set_data([aval[modelIndex]], [bval[modelIndex]])
        # read the model LOSVDs
        try:
            with open(modelfilename) as lfile:
                header = lfile.readline()
                this.mod_degree= int(re.search('Degree: (\d+)', header).group(1))
                this.mod_gridv = _numpy.array([float(a) for a in re.search('Grid: (.*)$', header).group(1).split()])
            this.los_mod = _numpy.loadtxt(modelfilename)
            this.ghm_mod = _agama.ghMoments(matrix=this.los_mod, gridv=this.mod_gridv, degree=this.mod_degree, ghorder=6)[:,(1,2,6,7,8,9)]
            dif = _numpy.sum(_numpy.nan_to_num((this.ghm_mod - ghm_val) / ghm_err)**2, axis=0)
            print('Loaded %s, chi2 for v=%.2f, sigma=%.2f, h3=%.2f, h4=%.2f, h5=%.2f, h6=%.2f, total=%.2f, in file=%.2f' %
                (modelfilename, dif[0], dif[1], dif[2], dif[3], dif[4], dif[5], sum(dif), chi2[modelIndex]) )
        except Exception as e:
            print("Can't read %s: %s" % (modelfilename, str(e)))
            try: del this.los_mod
            except: pass
            try: del this.ghm_mod
            except: pass
        try:
            plotWeights(modelfilename.replace('.los','.npz'))
        except Exception as e:
            print(e)

    def plotMaps():
        '''
        replot kinematic maps depending on the current display mode
        '''
        for b in buttons: b.set_lw(0)
        this.mode.set_lw(2)
        for p, patch in enumerate(patchcoll):
            if this.mode == buttons[0] or this.mode == buttons[2]:
                patch.set_cmap('breeze')
                if this.mode == buttons[0]:
                    data = ghm_val[:,p]
                else:
                    try: data = this.ghm_mod[:,p]
                    except AttributeError: data = ghm_val[:,0]*0
                patch.set_array(data)
                patch.set_clim(panel_params[p]['data_range'])
            elif this.mode == buttons[1]:
                patch.set_cmap('PuBu')
                patch.set_array(ghm_err[:,p])
                patch.set_clim(0, panel_params[p]['error_range'])
            elif this.mode == buttons[3]:
                patch.set_cmap('RdBu_r')
                try: patch.set_array((this.ghm_mod[:,p] - ghm_val[:,p]) / ghm_err[:,p])
                except AttributeError: patch.set_array(ghm_val[:,0]*0)
                patch.set_clim([-3,3])
            else: raise ValueError('Unknown mode')

    def plotLOSVD():
        '''
        plot the data and model LOSVDs in the currently selected aperture
        '''
        try: ind = this.ind_aper
        except AttributeError: return
        losvdplot.cla()
        losvdplot.fill_between(plot_gridv, obs_losvd[0,ind], obs_losvd[2,ind], facecolor='r', alpha=0.33, lw=0)
        # plot the model LOSVD
        try: losvdplot.plot(plot_gridv, _agama.bsplineInterp(this.mod_degree, this.mod_gridv, this.los_mod[ind], plot_gridv), 'k')[0].set_dashes([5,2])
        except AttributeError: pass
        losvdplot.set_xlim(min(plot_gridv), max(plot_gridv))
        losvdplot.set_yticklabels([])
        losvdplot.set_xlabel('v')
        losvdplot.set_ylabel('f(v)')
        # print some useful info
        coefs = ['v0', 'sigma', 'h3', 'h4', 'h5', 'h6']
        text  = 'Aperture #%i centered at x=%.3f, y=%.3f: ' % (ind, _numpy.mean(apertures[ind][:,0]), _numpy.mean(apertures[ind][:,1]))
        for i in range(6):
            text += '%s=%.3f +- %.3f ' % (coefs[i], ghm_val[ind,i], ghm_err[ind,i])
            try:
                err = (this.ghm_mod[ind,i]-ghm_val[ind,i]) / ghm_err[ind,i]
                if err < -1.:  text += '[\033[1;31m %.3f \033[0m] ' % this.ghm_mod[ind,i]
                elif err> 1.:  text += '[\033[1;34m %.3f \033[0m] ' % this.ghm_mod[ind,i]
                else: text += '[ %.3f ] ' % this.ghm_mod[ind,i]
            except AttributeError: pass
        print(text)
        # highlight the selected polygon in all panels (make its boundary thicker)
        lw = _numpy.ones(len(apertures))
        lw[ind] = 3.
        for p in patchcoll: p.set_linewidths(lw)

    def onclick(event):
        '''
        handle interactive user input
        '''
        if event.artist == modelgrid:
            loadModel(event.ind[0])
            plotMaps()
            plotLOSVD()
            fig.canvas.draw()
            return
        if event.artist in buttons:
            this.mode = event.artist
            plotMaps()
            fig.canvas.draw()
            return
        has_poly, ind_poly = event.artist.contains(event.mouseevent)
        if has_poly:
            this.ind_aper = ind_poly['ind'][-1]
            plotLOSVD()
            fig.canvas.draw()

    def plotWeights(filename):
        '''
        plot orbit weight distributions
        '''
        orbits = _numpy.load(filename)
        R=orbits['Rm']
        L=orbits['Lr']
        I=orbits['Ci']
        weights=orbits['weights']
        axo.cla()
        axp.cla()
        # plot all orbits in gray, and orbits which have significant weight in the model in color with larger points
        axo.scatter(R, L, s=2, marker='o', color='#E0E0E0', edgecolors='none')
        axo.scatter(R, L, s=2*weights*len(weights), marker='o', c=I, cmap='mist', vmin=-1, vmax=1, alpha=0.5, edgecolors='none')
        axp.scatter(R, I, s=2, marker='o', color='#E0E0E0', edgecolors='none')
        axp.scatter(R, I, s=2*weights*len(weights), marker='o', c=L, cmap='mist', vmin= 0, vmax=1, alpha=0.5, edgecolors='none')
        axo.set_xscale('log')
        axo.set_xlim(0.02, 200)
        axo.set_ylim(0, 1)
        axo.set_xlabel('$R_\mathrm{circ}(E)$', labelpad=-2, fontsize=12)
        axo.set_ylabel(' $[L/L_\mathrm{circ}(E)]^2$', labelpad=0, fontsize=12)
        axp.set_xscale('log')
        axp.set_xlim(0.02, 200)
        axp.set_ylim(-1, 1)
        axp.set_xlabel('$R_\mathrm{circ}(E)$', labelpad=-2, fontsize=12)
        axp.set_ylabel(' $L_z/L$', labelpad=-5, fontsize=12)


    # main section of the runPlot routine
    fig = matplotlib.pyplot.figure(figsize=(24,10))
    fig.canvas.mpl_connect('pick_event', onclick)

    # parse and combine all kinematic datasets
    apertures = []
    obs_degree= []
    obs_gridv = []
    ghm_val   = []
    ghm_err   = []
    for d in datasets:
        try:
            sing = _numpy.sin(d.target_params['gamma'])
            cosg = _numpy.cos(d.target_params['gamma'])
            for i,a in enumerate(d.target_params['apertures']):
                apertures.append(_numpy.column_stack(( a[:,0] * cosg - a[:,1] * sing, a[:,1] * cosg + a[:,0] * sing )))
            gv, ge = d.getGHMoments()
            # we use exactly 6 GH moments, even if the data have fewer or more
            if gv.shape[1]<6:
                gv = _numpy.column_stack((gv, _numpy.zeros((gv.shape[0], 6-gv.shape[1])) ))
                ge = _numpy.column_stack((ge, _numpy.zeros((ge.shape[0], 6-ge.shape[1])) ))
            elif gv.shape[1]>6:
                gv = gv[:,0:6]
                ge = ge[:,0:6]
            ghm_val.extend(gv)
            ghm_err.extend(ge)
        except: pass  # not the right type of dataset
    ghm_val = _numpy.vstack(ghm_val)
    ghm_err = _numpy.vstack(ghm_err)
    if xlim is None: xlim = (min([_numpy.amin(p[:,0]) for p in apertures]), max([_numpy.amax(p[:,0]) for p in apertures]))
    if ylim is None: ylim = (min([_numpy.amin(p[:,1]) for p in apertures]), max([_numpy.amax(p[:,1]) for p in apertures]))
    if vlim is None: vlim = (min(ghm_val[:,0] - ghm_val[:,1] * 3.0), max(ghm_val[:,0] + ghm_val[:,1] * 3.0))
    plot_gridv = _numpy.linspace(vlim[0], vlim[1], 201)

    # parameters of kinematic maps; the default ranges are taken from the input data,
    # but the user may adjust the range of data values and errors
    if v0lim    is None: v0lim   = max(abs(ghm_val[:,0])+ghm_err[:,0]) * _numpy.array([-1,1])  # symmetric range
    if sigmalim is None: sigmalim=(min(ghm_val[:,1]-ghm_err[:,1]), max(ghm_val[:,1]+ghm_err[:,1]))
    if v0err    is None: v0err   = max(ghm_err[:,0])
    if sigmaerr is None: sigmaerr= max(ghm_err[:,1])
    panel_params = [
        dict(title='$v_0$',   data_range=v0lim,   error_range=v0err,   extent=[0.20, 0.55, 0.19, 0.45]),
        dict(title='$\sigma$',data_range=sigmalim,error_range=sigmaerr,extent=[0.20, 0.05, 0.19, 0.45]),
        dict(title='$h_3$',   data_range=hlim,    error_range=herr,    extent=[0.40, 0.55, 0.19, 0.45]),
        dict(title='$h_4$',   data_range=hlim,    error_range=herr,    extent=[0.40, 0.05, 0.19, 0.45]),
        dict(title='$h_5$',   data_range=hlim,    error_range=herr,    extent=[0.60, 0.55, 0.19, 0.45]),
        dict(title='$h_6$',   data_range=hlim,    error_range=herr,    extent=[0.60, 0.05, 0.19, 0.45]),
    ]

    # collect the data for plotting the observed LOSVDs in each aperture
    obs_losvd = []
    for d in datasets:
        try: obs_losvd.append(d.getLOSVD(plot_gridv))
        except AttributeError:  pass  # not the right type of dataset
    obs_losvd = _numpy.hstack(obs_losvd)

    ##### likelihood surface #####
    if len(aval)>0:
        print('%i models available' % len(aval))
        ax = fig.add_axes([0.02, 0.55, 0.17, 0.45])
        modelgrid = ax.plot(aval, bval, 'o', c='g', ms=5, picker=5, mew=0, alpha=0.75)[0]
        this.selected  = ax.plot([_numpy.nan], [_numpy.nan], marker='o', c='r', ms=8, mew=0)[0]
        this.modellabel= ax.text(0.01, 0.01, '', color='r', ha='left', va='bottom', transform=ax.transAxes, fontsize=10)
        ax.text(0.5, 1.0, '$\mathrm{min}\,\chi^2=%.2f$' % min(chi2), color='r', ha='center', va='top', transform=ax.transAxes)
        ax.set_xlabel(alabel, labelpad=0)
        ax.set_ylabel(blabel, labelpad=0)
        if alim is None: alim = (min(aval), max(aval))
        if blim is None: blim = (min(bval), max(bval))
        anorm = alim[1]-alim[0]
        bnorm = blim[1]-blim[0]
        ax.set_xlim(alim[0], alim[1])
        ax.set_ylim(blim[0]-0.05*bnorm, blim[1]+0.05*bnorm)
        try:
            a, b  = _numpy.meshgrid(_numpy.linspace(alim[0], alim[1], 201), _numpy.linspace(blim[0], blim[1], 201))
            if interp=='linear' or interp=='cubic':
                c = scipy.interpolate.griddata((aval/anorm, bval/bnorm), chi2-min(chi2), (a/anorm, b/bnorm), method=interp)
            elif interp=='rbf':
                c = scipy.interpolate.Rbf(aval/anorm, bval/bnorm, chi2-min(chi2), function='thin_plate')(a/anorm, b/bnorm)
            else: raise ValueError('Invalid interpolation method')
            # 1-sigma, 2-sigma, 3-sigma, etc. confidence intervals for a 2-dof chi2 distribution
            cntr = _numpy.array([2.30,6.18,11.83] + [x**2 + _numpy.log(x**2*_numpy.pi/2) + 2*x**-2 for x in range(4,33)])
            ax.contourf(a, b, c, cntr, cmap='hell_r', vmin=0, vmax=deltaChi2lim, alpha=0.75)
            ax.clabel(ax.contour(a, b, c, cntr, cmap='Blues_r', vmin=0, vmax=deltaChi2lim), fmt='%.0f', fontsize=10, inline=1)
        except Exception as e:
            print(e)
    else:
        modelgrid = None

    ##### four buttons determining which map to display #####
    radioplot = fig.add_axes([0.02, 0.4, 0.17, 0.1])
    radioplot.set_axis_off()
    buttons = [
        matplotlib.patches.Rectangle((-0.46, 0.05), 0.45,0.4, color='#60f080', picker=True, ec='k'),
        matplotlib.patches.Rectangle(( 0.01, 0.05), 0.45,0.4, color='#ffe000', picker=True, ec='k'),
        matplotlib.patches.Rectangle((-0.46,-0.45), 0.45,0.4, color='#80a0ff', picker=True, ec='k'),
        matplotlib.patches.Rectangle(( 0.01,-0.45), 0.45,0.4, color='#ff80a0', picker=True, ec='k') ]
    for b in buttons: radioplot.add_artist(b)
    radioplot.text(-0.25, 0.25, 'data',      ha='center', va='center')
    radioplot.text( 0.25, 0.25, 'data err',  ha='center', va='center')
    radioplot.text(-0.25,-0.25, 'model',     ha='center', va='center')
    radioplot.text( 0.25,-0.25, 'model err', ha='center', va='center')
    radioplot.set_xlim(-0.5,0.5)
    radioplot.set_ylim(-0.5,0.5)
    this.mode = buttons[0]

    ##### LOSVD in the selected aperture #####
    losvdplot = fig.add_axes([0.02, 0.05, 0.17, 0.35])
    losvdplot.set_yticklabels([])
    losvdplot.set_xlim(min(plot_gridv), max(plot_gridv))
    losvdplot.set_xlabel('v')
    losvdplot.set_ylabel('f(v)')

    ##### maps of v,sigma and higher Gauss-Hermite moments #####
    patchcoll = []
    panels = []
    for param in panel_params:
        patches = matplotlib.collections.PatchCollection([matplotlib.patches.Polygon(p, True) for p in apertures], picker=0.0, edgecolor=(0.5,0.5,0.5,0.5))
        patchcoll.append(patches)
        ax=fig.add_axes(param['extent'])
        ax.add_collection(patches)
        ax.set_xlim(xlim[1], xlim[0])  # note the inverted X axis!
        ax.set_ylim(ylim[0], ylim[1])
        ax.text(0.02, 0.98, param['title'], fontsize=16, transform=ax.transAxes, ha='left', va='top')
        panels.append(ax)

    plotMaps()
    # add colorbars to data maps
    for patch, param in zip(patchcoll, panel_params):
        cax = fig.add_axes([param['extent'][0], param['extent'][1]-0.03, param['extent'][2], 0.01])
        fig.colorbar(patch, cax=cax, orientation='horizontal')
    # make sure that pan/zoom is synchronized between kinematic maps
    for p in panels[1:]:
        p.get_shared_x_axes().join(*panels)
        p.get_shared_y_axes().join(*panels)
    # enforce a correct aspect ratio for kinematic maps
    for p in panels:
        p.set_aspect('equal')#, 'datalim')

    ##### two panels with orbit weights #####
    axo=fig.add_axes([0.82, 0.05, 0.18, 0.45])
    axa=fig.add_axes([0.995,0.15,0.005,0.25])  # colorbar showing the third variable
    axa.imshow(_numpy.linspace(0,1,256).reshape(-1,1), extent=[0,1,-1,1], origin='lower', interpolation='nearest', aspect='auto', cmap='mist')
    axa.set_xticks([])
    axa.set_ylabel('$L_z/L$', fontsize=10, labelpad=-10)
    axp=fig.add_axes([0.82, 0.55, 0.18, 0.45])
    axa=fig.add_axes([0.995,0.65,0.005,0.25])  # colorbar
    axa.imshow(_numpy.linspace(0,1,256).reshape(-1,1), extent=[0,1,0,1], origin='lower', interpolation='nearest', aspect='auto', cmap='mist')
    axa.set_xticks([])
    axa.set_ylabel(' $[L/L_\mathrm{circ}(E)]^2$', fontsize=10, labelpad=-2)

    # start the interactive plot
    matplotlib.pyplot.show()
