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
    return dict(
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
    # construct the total composite density from the list of parameters of each component
    components = [getDensityParamsMGE(
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
        for t in tab]
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
    bintags, xNode, yNode, xBar, yBar, sn, nPixels, scale = voronoi_2d_binning(
            xc, yc, hist, (hist+0.1)**0.5, (1.*_numpy.sum(hist)/nbins)**0.5,
            plot=plot, quiet=False, pixelsize=min(min(gridx[1:]-gridx[:-1]), min(gridy[1:]-gridy[:-1])))
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
    xpixel  = _numpy.min(xCoords[1:]-xCoords[:-1])    # size of a single pixel (assuming they all are equal)
    ypixel  = _numpy.min(yCoords[1:]-yCoords[:-1])    # same in y direction
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
                vertices.append(
                    ( xpixel * (cx+minx-0.5) + xCoords[0], ypixel * (cy+miny-0.5) + yCoords[0] ))
            if cx+minx == ix[0] and cy+miny == iy[0]:
                break   # reached the initial point
            edges += 1
        if edges>=1000:
            raise ValueError('Lost in the way for bin '+str(b))
        polygons.append( _numpy.array(vertices) )
    result = _numpy.empty(len(polygons), dtype=object)
    result[:] = polygons
    return result #_numpy.array(polygons, dtype=object)  <-- this doesn't work when all arrays are of the same size


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
        polygons = [ _numpy.array([float(a) for a in b.split()]).reshape(-1,2)
            for b in dfile.read().split('\n\n') ]
    result = _numpy.empty(len(polygons), dtype=object)
    result[:] = polygons
    return result


def makeGridForTargetLOSVD(polygons, psf, psfwingfrac=0.99, psfcorefrac=0.2, pixelmult=1.0, maxsize=100):
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
      maxsize [default 100]: maximum number of grid segments per dimension; if the ratio of the grid extent
      to pixel size exceeds this limit, a non-uniform grid will be created, such that central segments have
      the size determined from the above algorithm, but they become progressively larger further out.
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
    xmin = xmin - margin
    xmax = xmax + margin
    ymin = ymin - margin
    ymax = ymax + margin
    Xmin = int(_numpy.floor(xmin / pix))
    Xmax = int(_numpy.ceil (xmax / pix))
    Ymin = int(_numpy.floor(ymin / pix))
    Ymax = int(_numpy.ceil (ymax / pix))
    # if |Xmin| ~= Xmax, make the grid fully symmetric (this saves some cpu time), similarly for Y
    if Xmin<0 and Xmax>0 and (abs(Xmin+Xmax) <= 3 or (-1.*Xmin/Xmax>=0.8) and (-1.*Xmin/Xmax<=1.25)):
        Xmax = max(Xmax, -Xmin)
        Xmin = -Xmax
    if Ymin<0 and Ymax>0 and (abs(Ymin+Ymax) <= 3 or (-1.*Ymin/Ymax>=0.8) and (-1.*Ymin/Ymax<=1.25)):
        Ymax = max(Ymax, -Ymin)
        Ymin = -Ymax
    # sanity check
    if Xmax-Xmin > 1000 or Ymax-Ymin > 1000:
        raise ValueError("Can't construct a suitable grid (pixel size "+str(pix)+" doesn't seem right")
    # finally, construct two regularly-spaced grids if they do not exceed maxsize [ADJUSTABLE]
    if max(Xmax-Xmin, Ymax-Ymin) <= maxsize:
        gridx = _numpy.linspace(Xmin*pix, Xmax*pix, Xmax-Xmin+1)
        gridy = _numpy.linspace(Ymin*pix, Ymax*pix, Ymax-Ymin+1)
    else:  # otherwise concoct a non-uniformly spaced grid of that size
        rmax = max(abs(xmin), abs(ymin), abs(xmax), abs(ymax))
        # if the grid extent is not symmetric, and the grid is non-uniform,
        # it is not trivial to determine its stretch factor that produces exactly maxsize segments..
        nnodes = maxsize
        for nnodes in range(maxsize, 2*maxsize+1):
            tempgrid = _agama.symmetricGrid(nnodes, pix, rmax)
            # cut the resulting grid to "barely" cover the desired grid extent
            tempgridx = tempgrid[min(_numpy.where(tempgrid > xmin)[0]) - 1 : max(_numpy.where(tempgrid < xmax)[0]) + 2]
            tempgridy = tempgrid[min(_numpy.where(tempgrid > ymin)[0]) - 1 : max(_numpy.where(tempgrid < ymax)[0]) + 2]
            if max(len(tempgridx), len(tempgridy)) <= maxsize:
                gridx = tempgridx
                gridy = tempgridy
            else:
                break
        pix = (pix, max(max(gridx[1:]-gridx[:-1]), max(gridy[1:]-gridy[:-1])))

    print(('Kinematic spaxel size is %.2f; %d apertures in the region [%.2f:%.2f, %.2f:%.2f]; '+
        'PSF is %s; internal kinematic %dx%d grid is [%.2f:%.2f, %.2f:%.2f] with pixel size %s') %
        (pixel, len(polygons), xmin, xmax, ymin, ymax,
        str(psf), len(gridx), len(gridy), min(gridx), max(gridx), min(gridy), max(gridy),
        ('%.2f-%.2f' % pix) if isinstance(pix, tuple) else ('%.2f' % pix)))
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
    def __init__(self, density, tolerance=0, alpha=0, beta=0, gamma=0, **target_params):
        '''
        Arguments:
          density:   3d density profile of stars;
          tolerance: relative error on density constraints;
          alpha, beta, gamma: three Euler angles specifying the orientation of the intrinsic model
              model coordinate system w.r.t. the image plane (*not* used for modelling, only for plotting);
          target_params:  all other parameters of the Target object (including type='Density***').
        '''
        # extract some important arguments from the dictionary in a case-insensitive way
        targetType = None
        gridr = None
        gridz = None
        for k in target_params:
            if k.upper() == 'TYPE':  targetType = target_params[k]
            if k.upper() == 'GRIDR': gridr = target_params[k]
            if k.upper() == 'GRIDZ': gridz = target_params[k]
        if targetType is None:
            raise TypeError('DensityDataset should be constructed with a Density*** target type')
        self.target_params = target_params
        self.target  = _agama.Target(**target_params)
        self.density = density
        self.alpha   = alpha
        self.beta    = beta
        self.gamma   = gamma
        # 0th constraint is the total mass, with zero tolerance; remaining ones are cell masses;
        # both constraints and their tolerances are normalized to totalMass
        self.totalMass = density.totalMass()
        self.cons_val = _numpy.hstack(( 1, self.target(density) / self.totalMass ))
        self.cons_err = _numpy.hstack(( 0, abs(self.cons_val[1:]) * tolerance ))
        # constraints with very small (or zero) absolute values are assigned a zero tolerance
        self.cons_err[self.cons_err < 1e-12 * _numpy.max(abs(self.cons_val[1:]))] = 0.
        # compute the fraction of total mass of the density model within the extent of the grid;
        # the method for obtaining this fraction depends on the discretization scheme:
        if 'DENSITYCLASSIC' in targetType.upper():
            ncons = len(self.target)   # all constraints describe masses in 3d grid cells
        elif 'DENSITYCYLINDRICAL' in targetType.upper():
            ncons = len(gridr) * len(gridz)   # take only the m=0 harmonic term in the array of constraints
        elif 'DENSITYSPHHARM' == targetType.upper():
            ncons = len(gridr)+1   # same but for the l=0,m=0 harmonic term only
        else:
            ncons = 0    # unknown discretization scheme, should have failed at the earlier stage
        print('%s with %i constraints; total mass: %g; fraction of mass in %i density bins: %g' %
            (targetType, len(self.target), self.totalMass, ncons, sum(self.cons_val[1:ncons+1])))

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
        dif = ((model_dens[use] - self.cons_val[1:][use]) / self.cons_err[1:][use])
        # print the density constraints which have too large deviations in the solution
        header_shown = False
        for i in range(len(self.cons_val)-1):
            if use[i] and abs(model_dens[i]-self.cons_val[i+1]) > 3*self.cons_err[i+1]:
                if not header_shown:
                    print('3d density constraint:                    required        actual   deviation/sigma')
                    header_shown = True
                print('%-36s  %12.4g  %12.4g  %8.4f' %
                (self.target[i], self.cons_val[i+1], model_dens[i], (model_dens[i]-self.cons_val[i+1])/self.cons_err[i+1]))
        return _numpy.sum(dif**2)

    def projectedDensity(self, gridx, gridy):
        '''
        Compute the surface density of the model at a given orientation, using the provided 1d grids
        in the observed coordinate system (used only in the interactive plotting script)
        '''
        xy = _numpy.column_stack((_numpy.repeat(gridx, len(gridy)), _numpy.tile(gridy, len(gridx))))
        return (self.density.projectedDensity(xy, alpha=self.alpha, beta=self.beta, gamma=self.gamma).
            reshape(len(gridx), len(gridy)))


class KinemDatasetVS:
    '''
    Class for representing the Target(type='LOSVD') and observational constraints in the form of first two moments
    '''
    def __init__(self, density, vs_val, vs_err, tolerance=0, **target_params):
        '''
        Arguments:
          ghm_val:  N x 2 array of observational constraints - mean velocity (v) and its dispersion (sigma),
            where N is the number of apertures.
          ghm_err:  N x 2 array of observational uncertainties for v and sigma
          density:  3d density profile of stars, needed to compute the normalizations of LOSVDs (aperture masses)
          tolerance:  fractional error on aperture masses
          target_params:  all other parameters of the Target(type='LOSVD', ...),
            except that when gridx, gridy are not provided, they will be constructed automatically
        '''
        gridx = None
        for k in target_params:
            if k.upper() == 'GRIDX': gridx = target_params[k]
        if gridx is None:
            target_params['gridx'], target_params['gridy'] = makeGridForTargetLOSVD(
                target_params['apertures'], target_params.get('psf', 0))
        self.target_params = target_params
        self.target = _agama.Target(**target_params)
        self.mod_degree = target_params['degree']
        self.mod_gridv  = target_params['gridv']
        self.num_bsplines = len(self.mod_gridv) + self.mod_degree - 1
        self.num_aper = vs_val.shape[0]
        if vs_val.shape != vs_err.shape:
            raise ValueError('vs_val and vs_err must have the same shape')
        # surface density convolved with PSF and integrated over the area of each aperture, normalized to totalMass
        self.aperture_mass = self.target(density) / density.totalMass()
        self.aperture_mass_err = self.aperture_mass * tolerance
        if len(self.aperture_mass) != self.num_aper:
            raise ValueError('vs_val should have the same number of rows as the number of apertures')
        # constraint values: aperture masses multiplied by 0th, 1st and 2nd velocity moments
        self.cons_val = _numpy.column_stack((
            self.aperture_mass,
            self.aperture_mass *  vs_val[:,0],
            self.aperture_mass * (vs_val[:,0]**2 + vs_val[:,1]**2 + vs_err[:,0]**2 + vs_err[:,1]**2),
        )).reshape(-1)
        # constraint errors: translate errors in sigma into errors in v^2+sigma^2
        self.cons_err = _numpy.column_stack((
            self.aperture_mass_err,
            self.aperture_mass * vs_err[:,0],
            self.aperture_mass * (
                vs_err[:,0]**2 * (2 * vs_err[:,0]**2 + 4 * vs_val[:,0]**2) +
                vs_err[:,1]**2 * (2 * vs_err[:,1]**2 + 4 * vs_val[:,1]**2) )**0.5
        )).reshape(-1)
        # also store the original arrays (v,sigma) and their error estimates
        self.vs_val = vs_val[:,0:2]
        self.vs_err = vs_err[:,0:2]

    def getOrbitMatrix(self, kinem_matrix, Upsilon):
        '''
        Produce the matrix of orbit contributions to each of the kinematic constraints (incl. aperture mass)
        from the matrix recorded during orbit integration by the LOSVD Target object.
        The LOSVDs of each orbit are converted into 0th, 1st and 2nd velocity moments.
        Upsilon is the mass-to-light ratio of the model, so that its velocity grid is scaled by sqrt(Upsilon).
        Returns: the matrix to be provided to the optimization solver
        '''
        num_orbits = len(kinem_matrix)
        mod_bsint0 = _agama.bsplineIntegrals(self.mod_degree, self.mod_gridv)
        mod_bsint1 = _agama.bsplineIntegrals(self.mod_degree, self.mod_gridv, power=1)
        mod_bsint2 = _agama.bsplineIntegrals(self.mod_degree, self.mod_gridv, power=2)
        return _numpy.dstack((
            # 0th column is the contribution of each orbit to aperture masses
            kinem_matrix.reshape(num_orbits, self.num_aper, self.num_bsplines).dot(mod_bsint0),
            # 1st and 2nd columns are mean v and mean v^2, after rescaling the model velocity grid by sqrt(Upsilon)
            kinem_matrix.reshape(num_orbits, self.num_aper, self.num_bsplines).dot(mod_bsint1) * Upsilon**0.5,
            kinem_matrix.reshape(num_orbits, self.num_aper, self.num_bsplines).dot(mod_bsint2) * Upsilon**1.0,
        )). reshape(num_orbits, self.num_aper * 3)

    def getPenalty(self, model_losvd, Upsilon):
        '''
        Compute the penalty for kinematic constraints from the array of LOSVDs of the model.
        Arguments:
          model_losvd:  the array of length num_aper * num_bsplines  containing the LOSVDs of
          the entire model (i.e., the sum of LOSVDs of each orbit multiplied by orbit weights).
          Upsilon:  mass-to-light ratio of the model, used to scale the velocity grid.
        Returns: the error-weighted penalty (chi^2) separately for each type of constraint, summed over all apertures;
          an array of 3 elements [aperture mass, mean velocity, velocity dispersion (or rather, standard deviation)].
        '''
        mod_bsint0 = _agama.bsplineIntegrals(self.mod_degree, self.mod_gridv)
        mod_bsint1 = _agama.bsplineIntegrals(self.mod_degree, self.mod_gridv, power=1)
        mod_bsint2 = _agama.bsplineIntegrals(self.mod_degree, self.mod_gridv, power=2)
        a_mod = model_losvd.reshape(self.num_aper, self.num_bsplines).dot(mod_bsint0)
        v_mod = model_losvd.reshape(self.num_aper, self.num_bsplines).dot(mod_bsint1) / a_mod * Upsilon**0.5
        s_mod =(model_losvd.reshape(self.num_aper, self.num_bsplines).dot(mod_bsint2) / a_mod * Upsilon - v_mod**2)**0.5
        # add the penalty for aperture mass, if the tolerance was not zero
        use = self.aperture_mass_err != 0
        return _numpy.array([
            # penalty for aperture mass
            _numpy.sum( ((a_mod - self.aperture_mass)[use] / self.aperture_mass_err[use])**2 ),
            # chi2 w.r.t. originally provided data (v and sigma)
            _numpy.sum( ((v_mod - self.vs_val[:,0]) / self.vs_err[:,0])**2 ),
            _numpy.sum( ((s_mod - self.vs_val[:,1]) / self.vs_err[:,1])**2 ),
        ])

    def getGHMoments(self, model_losvd=None, Upsilon=None):
        '''
        Return v & sigma of the observational dataset (if model_losvd is None) or the model LOSVD
        '''
        if model_losvd is None:
            return self.vs_val, self.vs_err
        else:
            mod_bsint0 = _agama.bsplineIntegrals(self.mod_degree, self.mod_gridv)
            mod_bsint1 = _agama.bsplineIntegrals(self.mod_degree, self.mod_gridv, power=1)
            mod_bsint2 = _agama.bsplineIntegrals(self.mod_degree, self.mod_gridv, power=2)
            a_mod = model_losvd.dot(mod_bsint0)
            v_mod = model_losvd.dot(mod_bsint1) / a_mod * Upsilon**0.5
            s_mod =(model_losvd.dot(mod_bsint2) / a_mod * Upsilon - v_mod**2)**0.5
            return _numpy.column_stack([v_mod, s_mod, _numpy.zeros((len(v_mod), 4))])

    def getLOSVD(self, gridv):
        '''
        Construct the LOSVD profiles (assuming a Gaussian shape) from the observed v and sigma and their uncertainties.
        Arguments:
          gridv:  grid for computing the profiles
        Returns:  array of shape  3 x num_aper x len(gridv)  containing 16, 50 and 84th percentiles of LOSVDs in each aperture
        '''
        nboot = 100
        result= _numpy.zeros((3, self.num_aper, len(gridv)))
        for n in range(self.num_aper):
            boots = _agama.ghInterp(
                self.aperture_mass[n],  # amplitude
                self.vs_val[n,0] + _numpy.random.normal(size=nboot) * self.vs_err[n,0],  # center
                self.vs_val[n,1] + _numpy.random.normal(size=nboot) * self.vs_err[n,1],  # width
                _numpy.array([1,0,0]),
                gridv)
            result[:,n] = _numpy.percentile(boots, axis=1, q=[16,50,84])
        return result


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
        gridx = None
        for k in target_params:
            if k.upper() == 'GRIDX': gridx = target_params[k]
        if gridx is None:
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
        # constraint values: 0th column stands for aperture mass constraints (normalized to unity),
        # remaining columns are GH moments h_1..h_M, with h_1 = h_2 = 0
        self.cons_val = _numpy.column_stack((
            _numpy.ones(self.num_aper),
            ghm_val[:,0:2]*0,
            ghm_val[:,2:]
        )).reshape(-1)
        # constraint errors: translate errors in v0, sigma into errors in h_1,h_2
        self.cons_err = _numpy.column_stack((
            _numpy.zeros(self.num_aper) + tolerance,  # fractional error on aperture mass
            ghm_err[:,0:2] / sigma[:,None] / 2**0.5,  # errors on h_1, h_2
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
            # 0th column is the fractional contribution of each orbit to aperture masses, normalized to unity
            kinem_matrix.reshape(num_orbits, self.num_aper, self.num_bsplines).dot(mod_bsint) / self.aperture_mass,
            # remaining columns are GH moments h_1..h_M computed in the observed GH basis
            # after rescaling the model velocity grid by sqrt(Upsilon) and
            # multiplying the values of coefficients by 1/sqrt(Upsilon)
            _agama.ghMoments(
            degree=self.mod_degree, gridv=self.mod_gridv * Upsilon**0.5,
            matrix=kinem_matrix * Upsilon**-0.5, ghorder=self.num_cons, ghbasis=self.ghbasis).
            reshape(num_orbits, self.num_aper ,  self.num_cons+1)[:,:,1:]
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
        ghm_mod   = (_agama.ghMoments(
            degree=self.mod_degree, gridv=self.mod_gridv * Upsilon**0.5,
            matrix=model_losvd * Upsilon**-0.5, ghorder=self.num_cons).
            reshape(self.num_aper,  self.num_cons+4))
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

    def getGHMoments(self, LOSVD=None, Upsilon=None):
        '''
        Return v,sigma,h3..hM of the observational dataset (if LOSVD is None) or of the model LOSVD
        '''
        if LOSVD is None:
            return self.ghm_val, self.ghm_err
        else:
            return _agama.ghMoments(matrix=LOSVD * Upsilon**-0.5,
                gridv=self.mod_gridv * Upsilon**0.5, degree=self.mod_degree, ghorder=6)[:,(1,2,6,7,8,9)]

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
        gridx = None
        for k in target_params:
            if k.upper() == 'GRIDX': gridx = target_params[k]
        if gridx is None:
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
        return (kinem_matrix.
            reshape(num_orbits, self.num_aper, self.num_bsplines).
            dot(self._getConvMatrix(Upsilon)).
            reshape(num_orbits, self.num_aper * (self.num_cons+1)))

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
        cons_mod = (model_losvd.reshape(self.num_aper, self.num_bsplines).
            dot(self._getConvMatrix(Upsilon)).
            reshape(-1))
        use   = self.cons_err != 0
        error = _numpy.zeros(self.num_aper * (self.num_cons+1))
        error[use] = ( (cons_mod[use] - self.cons_val[use]) / self.cons_err[use] )**2
        return _numpy.sum(error.reshape(self.num_aper, self.num_cons+1), axis=0)

    def getGHMoments(self, LOSVD=None, Upsilon=None):
        '''
        Compute GH moments from the observed LOSVD (if LOSVD is None) or from the model LOSVD
        Return: array of size num_aper x ghorder containing the values  v,sigma,h3..hM  in each aperture
        '''
        ghorder = 6
        ind = tuple([1,2]+range(6,ghorder+4))  # indices of columns containing v,sigma,h3...hM in the matrix returned by ghMoments
        if LOSVD is None:
            ghm_val, ghm_err = ghMomentsErrors(degree=self.obs_degree, gridv=self.obs_gridv,
                values=self.obs_val, errors=self.obs_err, ghorder=ghorder)
            return ghm_val[:,ind], ghm_err[:,ind]
        else:
            return _agama.ghMoments(matrix=LOSVD * Upsilon**-0.5,
                gridv=self.mod_gridv * Upsilon**0.5, degree=self.mod_degree, ghorder=ghorder)[:,ind]

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
    Upsilon=1.0, regul=1.0, multstep=2**(1./10), deltaChi2=100.0,
    filePrefix='', linePrefix='', fileResult='results.dat', nbody=False, nbodyFormat='text'):
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
      filePrefix: filename (w/o extension) for storing the model LOSVDs and orbit weights; solutions for all values of Upsilon are stored in one .npz archive
      linePrefix: data written at the beginning of each line in the results file, before the values of Upsilon and chi2 for each dataset are appended
      fileResult: the name of the text file storing the summary information
      nbody:      if provided, create an N-body representation of the best-fit orbit superposition model in this series, sampled by the given number of particles and written into {filePrefix}_Y{best-fit Upsilon}.nbody
      nbodyformat: format for saving the N-body model (text, nemo or gadget)
    Returns: chi2 of the best-fit model in this series (also stores all models in the result file)
    '''

    try: len(datasets)   # ok if it is a tuple/list
    except: datasets = (datasets,)  # if not, make it a tuple
    if len(datasets)==0:
        raise RuntimeError('Need at least one dataset for the model')
    numKinemDatasets = sum([hasattr(ds, 'mod_gridv') for ds in datasets])
    if numKinemDatasets == 0:
        raise RuntimeError('Running Forstand without any kinematic datasets makes no sense')

    # build orbit library
    ic = ic.astype(_numpy.float32)
    inttime = (potential.Tcirc(ic) * intTime).astype(_numpy.float32)
    inttime[_numpy.isnan(inttime)] = _numpy.nanmax(inttime)
    matrices = _agama.orbit(potential=potential, ic=ic, time=inttime, Omega=Omega,
        targets=[d.target for d in datasets], dtype=object)
    trajs    = matrices[-1]   # list of orbit trajectories
    matrices = matrices[:-1]  # matrices corresponding to datasets
    assert(len(matrices) == len(datasets))

    # record various structural properties of orbits to be stored in the npz archive
    Rm, Lz, L2 = _numpy.zeros((3, len(trajs)))
    for iorb, orb in enumerate(trajs):
        # construct regularly-spaced trajectory from the interpolator
        t = orb(_numpy.linspace(0.0005, 0.9995, 1000) * (orb[-1]-orb[0]) + orb[0])
        Rm[iorb] = _numpy.mean(_numpy.sum(t[:,0:3]**2,axis=1)**0.5)   # mean radius of each orbit
        Lz[iorb] = _numpy.mean( t[:,0]*t[:,4]-t[:,1]*t[:,3])
        L2[iorb] = _numpy.mean((t[:,0]*t[:,4]-t[:,1]*t[:,3])**2 + (t[:,1]*t[:,5]-t[:,2]*t[:,4])**2 + (t[:,2]*t[:,3]-t[:,0]*t[:,5])**2)
    E  = potential.potential(ic[:,0:3]) + _numpy.sum(ic[:,3:6]**2, axis=1) * 0.5  # total energy of each orbit
    Rc = potential.Rcirc(E=E)                      # radius of a circular orbit with the given energy
    Lc = 2*_numpy.pi * Rc**2 / potential.Tcirc(E)  # angular momentum of a circular orbit with this energy
    Ci = Lz / L2**0.5  # Lz/L = cos(incl)
    Lr = L2 / Lc**2    # (L/Lc)^2 = 1-ecc^2 (at least for a Kepler orbit)

    archive = dict( ic=ic, inttime=inttime,
        Rmean=Rm.astype(_numpy.float32),
        circ2=Lr.astype(_numpy.float32),
        cosincl=Ci.astype(_numpy.float32),
        Upsilon=[],  # will be populated by the values of M/L explored during the fit
        weights=[],  # will contain corresponding orbit weights for each M/L
        LOSVD=[list() for i in range(numKinemDatasets)]  # will contain model LOSVDs for each dataset and each M/L
    )

    # placeholder for data shared between subroutines
    class this:
        bestweights = None
        bestchi = _numpy.inf

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
        matrix    = [ ds.getOrbitMatrix(mat, Upsilon).T for ds,mat in zip(datasets, matrices) ]
        rhs       = [ ds.cons_val/mult for ds in datasets ]
        with _numpy.errstate(all='ignore'):  # mute the warning about division by zero if cons_err==0
            pen_cons  = [ 2*ds.cons_err**-2 for ds in datasets ]
        # regularization penalty
        numOrbits = len(ic)
        totalMass = 1.0  # weights are normalized to total mass of unity
        pen_reg   = 2. * regul * _numpy.ones(numOrbits) * numOrbits / totalMass**2

        print('%s, Upsilon=%5.3f: solving...' % (filePrefix, Upsilon))
        # solve the matrix equation
        try:
            weights = _agama.solveOpt(matrix=matrix, rhs=rhs, rpenq=pen_cons, xpenq=pen_reg) * mult
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print('Error! %s' % str(e))
            # arbitrarily set uniform weights
            weights = _numpy.ones(numOrbits) * totalMass / numOrbits

        # analyze the differences in rhs and print out the results
        superpositions = [ weights.dot(mat) for mat in matrices ]
        penalties = [ ds.getPenalty(sup, Upsilon) for ds,sup in zip(datasets, superpositions) ]
        penReg  = 0.5 * _numpy.sum(weights**2 * pen_reg)
        entropy = -sum(weights * _numpy.log(weights+1e-100)) / totalMass + _numpy.log(totalMass / numOrbits)
        # number of orbits contributing 0.999 of total mass
        numUsed =  sum(_numpy.cumsum(_numpy.sort(weights)) > 0.001 * totalMass)
        penalty = []  # penalties for all datasets and for regularization
        print('%s, Upsilon=%5.3f: results:' % (filePrefix, Upsilon))
        indexKinemDataset = 0   # running index enumerating kinematic datasets
        for ds, pen, sup in zip(datasets, penalties, superpositions):
            print('Penalty for %i %s constraints: %s %g' %
                (len(ds.cons_val), str(ds.target), str(pen), _numpy.sum(pen)))
            penalty.append(_numpy.sum(pen))
            if hasattr(ds, 'mod_gridv'):  # kinematic dataset
                archive['LOSVD'][indexKinemDataset].append(
                    sup.reshape(ds.num_aper, ds.num_bsplines).astype(_numpy.float32) )
                indexKinemDataset += 1
        archive['Upsilon'].append(Upsilon)
        archive['weights'].append(weights.astype(_numpy.float32))
        print('Penalty for regularization: %7.2f;  entropy: %.3g;  # of useful orbits: %i / %i' %
            (penReg, entropy, numUsed, numOrbits))

        # workaround for newer versions of numpy, which refuse to save a "ragged" (non-rectangular) array
        # unless its dtype is explicitly set to "object"
        LOSVD_list = archive['LOSVD']
        archive['LOSVD'] = _numpy.array(LOSVD_list, dtype=object)

        # write out the data collected for all values of Upsilon
        _numpy.savez_compressed(filePrefix, **archive)

        archive['LOSVD'] = LOSVD_list

        # append results to the summary file
        with open(fileResult, 'a') as fileout:
            fileout.write('\t'.join(
                [linePrefix, '%5.3f' % Upsilon] +
                ['%7.2f' % _numpy.sum(p) for p in penalty+[penReg] ] +
                ['%s.npz|%i' % (filePrefix, len(archive['Upsilon'])-1)] ) + '\n')

        if sum(penalty) < this.bestchi:
            this.bestchi = sum(penalty)
            this.bestweights = weights * datasets[0].totalMass * _agama.G
            this.Upsilon = Upsilon

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

    if nbody:
        status, particles = _agama.sampleOrbitLibrary(nbody, trajs, this.bestweights)
        if status:
            _agama.writeSnapshot(filePrefix+'_Y%.3f.nbody' % this.Upsilon,
                (_numpy.hstack((particles[0][:,0:3], particles[0][:,3:6]*this.Upsilon**0.5)), particles[1]*this.Upsilon),
                nbodyFormat)
        else:
            print("Failed to produce an N-body model: %s" % particles)

    return best


def runPlot(datasets,                           # list of [kinematic] datasets to be plotted
    aval=[], bval=[], chi2=[], filenames=[],    # data for the chi2(a,b) plot, and the list of associated LOSVD files
    interp='linear', alabel='', blabel='',      # parameters  for the chi2(a,b) plot
    alim=None, blim=None, deltaChi2lim=100,     # axes ranges for the chi2(a,b) plot
    xlim=None, ylim=None,                       # ranges for x,y coordinates on kinematic maps
    v0lim=None, sigmalim=None, hlim=(-0.1,0.1), # ranges for color axes on kinematic maps for v0,sigma,h3..h6
    v0err=None, sigmaerr=None, herr=0.1,        # ranges for color axes on maps of errors in  v0,sigma,h3..h6
    vlim=None,                                  # range of the velocity axis for the LOSVD plot
    potential=None                              # gravitational potential of the model (used to re-integrate the orbit selected for visualization)
    ):
    '''
    Show an interactive plot with several panels:
    - kinematic maps (v,sigma and Gauss-Hermite moments up to h6) with the choice between
      measured values (data), errors (data err), model values (model) and
      differences between model and data, normalized by data errors (model err);
      the measurements are provided in spatial regions (apertures) which can be examined
      interactively.
    - line-of-sight velocity distribution of the data and the model (if loaded)
      in the selected aperture on the map (when clicking on the map), or alternatively
      the 3d shape of the selected orbit (when clicking on the orbit weight plots).
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

    import re, scipy.interpolate, matplotlib, matplotlib.pyplot, traceback
    try: from mpl_toolkits import mplot3d
    except: pass

    # placeholder for data shared between subroutines
    class this: pass

    def loadModel(modelIndex):
        '''
        load the model LOSVD file, compute GH moments
        '''
        # read the model LOSVDs and weights
        try:
            archiveFilename, archiveIndex = filenames[modelIndex].split('|')
            archiveIndex = int(archiveIndex)
            try: archive = _numpy.load(archiveFilename, allow_pickle=True, encoding='latin1')
            except TypeError: archive = _numpy.load(archiveFilename)  # older version of numpy
            Upsilon = archive['Upsilon'][archiveIndex]
            this.modellabel.set_text(archiveFilename.replace('.npz','') + '_Y%.3f' % Upsilon)
            this.selected.set_data([aval[modelIndex]], [bval[modelIndex]])
            LOSVD = archive['LOSVD']
            los_mod = []
            ghm_mod = []
            indexKinemDataset = 0
            num_aper = [0]  # cumulative number of apertures in each kinematic dataset
            for ds in datasets:
                if hasattr(ds, 'mod_gridv'):
                    # assume that all kinematic datasets have the same degree of B-splines and the same gridv!
                    this.mod_degree = ds.mod_degree
                    # rescale the default velocity grid and the B-spline amplitudes to the current Upsilon
                    this.mod_gridv = ds.mod_gridv * Upsilon**0.5
                    los_mod.append(LOSVD[indexKinemDataset][archiveIndex] * Upsilon**-0.5)
                    if hasattr(ds, 'getGHMoments'):
                        ghm_mod.append(ds.getGHMoments(LOSVD[indexKinemDataset][archiveIndex], Upsilon))
                    else:
                        ghm_mod.append(_agama.ghMoments(matrix=LOSVD[indexKinemDataset][archiveIndex] * Upsilon**-0.5,
                            gridv=ds.mod_gridv * Upsilon**0.5, degree=ds.mod_degree, ghorder=6)[:,(1,2,6,7,8,9)])
                    indexKinemDataset += 1
                    num_aper.append(num_aper[-1] + len(ds.target_params['apertures']))
            this.los_mod = _numpy.vstack(los_mod)
            this.ghm_mod = _numpy.vstack(ghm_mod)
            chi2_aper = _numpy.nan_to_num((this.ghm_mod - ghm_val) / ghm_err)**2
            chi2_ds = _numpy.zeros((indexKinemDataset, 6))
            for indexKinemDataset in range(len(chi2_ds)):
                chi2_ds[indexKinemDataset] = _numpy.sum(
                    chi2_aper[num_aper[indexKinemDataset]:num_aper[indexKinemDataset+1]], axis=0)
            text = 'Loaded %s, chi2 for ' % filenames[modelIndex]
            nameGH = ['v', 'sigma', 'h3', 'h4', 'h5', 'h6']
            for indexGH in range(6):
                chi2str = '+'.join(['%.2f' % c for c in chi2_ds[:,indexGH]])
                chi2labels[indexGH].set_text(r'$\chi^2=%s$' % chi2str)
                text += '%s=%s, ' % (nameGH[indexGH], chi2str)
            text += 'total=%.2f, in file=%.2f' % (_numpy.sum(chi2_ds), chi2[modelIndex])
            print(text)
            plotWeights(archive, archiveIndex)
        except:
            traceback.print_exc()
            print("Can't read %s" % filenames[modelIndex])
            if hasattr(this, 'los_mod'): del this.los_mod
            if hasattr(this, 'ghm_mod'): del this.ghm_mod

    def plotWeights(archive, archiveIndex):
        '''
        plot orbit weight distributions
        '''
        R=archive['Rmean']
        L=archive['circ2']
        I=archive['cosincl']
        weights=archive['weights'][archiveIndex]
        if this.axo is None:  # create axes on the first call
            this.axo=fig.add_axes([0.83, 0.09, 0.165, 0.40])
            axa=fig.add_axes([0.995, 0.14, 0.005, 0.30])  # colorbar showing the third variable
            axa.imshow(_numpy.linspace(0,1,256).reshape(-1,1), extent=[0,1,-1,1], origin='lower', interpolation='nearest', aspect='auto', cmap='mist')
            axa.set_xticks([])
            axa.set_ylabel('$L_z/L$', fontsize=10, labelpad=-10)
            this.axp=fig.add_axes([0.83, 0.59, 0.165, 0.40], sharex=this.axo)
            axa=fig.add_axes([0.995, 0.64, 0.005, 0.30])  # colorbar
            axa.imshow(_numpy.linspace(0,1,256).reshape(-1,1), extent=[0,1,0,1], origin='lower', interpolation='nearest', aspect='auto', cmap='mist')
            axa.set_xticks([])
            axa.set_ylabel(r'$[L/L_\mathrm{circ}(E)]^2$', fontsize=10, labelpad=-2)
        this.axo.set_xscale('linear')
        this.axo.cla()
        this.axp.cla()
        # plot orbits which have significant weight in the model in color with larger points, and all other orbits in gray
        use = _numpy.isfinite(L+I) * (weights > 0.1 * _numpy.mean(weights))
        this.axo.scatter(R[~use], L[~use], s=2, marker='o', color='#E0E0E0', edgecolors='none')
        this.sco = this.axo.scatter(R[use], L[use], s=2*weights[use]*len(weights),
            marker='o', picker=3, c=I[use], cmap='mist', vmin=-1, vmax=1, alpha=0.5, edgecolors='none')
        this.axp.scatter(R[~use], I[~use], s=2, marker='o', color='#E0E0E0', edgecolors='none')
        this.scp = this.axp.scatter(R[use], I[use], s=2*weights[use]*len(weights),
            marker='o', picker=3, c=L[use], cmap='mist', vmin= 0, vmax=1, alpha=0.5, edgecolors='none')
        # selected orbit (none initially)
        this.oro = this.axo.plot(_numpy.nan, _numpy.nan, 'xk')[0]
        this.orp = this.axp.plot(_numpy.nan, _numpy.nan, 'xk')[0]
        xlim = _numpy.percentile(R[_numpy.isfinite(R)], [0.1,99.9])
        this.axo.set_xscale('log')
        this.axo.set_xlim(xlim)
        this.axo.set_ylim(0, 1)
        this.axo.set_yticks(_numpy.linspace(0, 1, 6))
        this.axo.set_xlabel(r'$R_\mathrm{circ}(E)$', labelpad=-2, fontsize=12)
        this.axo.set_ylabel(r'$[L/L_\mathrm{circ}(E)]^2$', labelpad=0, fontsize=12)
        this.axp.set_xscale('log')
        this.axp.set_xlim(xlim)
        this.axp.set_ylim(-1, 1)
        this.axp.set_yticks(_numpy.linspace(-1, 1, 5))
        this.axp.set_xlabel(r'$R_\mathrm{circ}(E)$', labelpad=-2, fontsize=12)
        this.axp.set_ylabel(r'$L_z/L$', labelpad=-5, fontsize=12)
        this.ic = archive['ic'][use]
        this.inttime = archive['inttime'][use]
        this.weights = weights[use]

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
                    if hasattr(this, 'ghm_mod'): data = this.ghm_mod[:,p]
                    else: data = ghm_val[:,0]*0
                patch.set_array(data)
                patch.set_clim(panel_params[p]['data_range'])
            elif this.mode == buttons[1]:
                patch.set_cmap('PuBu')
                patch.set_array(ghm_err[:,p])
                patch.set_clim(0, panel_params[p]['error_range'])
            elif this.mode == buttons[3]:
                patch.set_cmap('RdBu_r')
                if hasattr(this, 'ghm_mod'):
                    patch.set_array((this.ghm_mod[:,p] - ghm_val[:,p]) / ghm_err[:,p])
                else:
                    patch.set_array(ghm_val[:,0]*0)
                patch.set_clim([-3,3])
            else: raise ValueError('Unknown mode')

    def plotLOSVD():
        '''
        plot the data and model LOSVDs in the currently selected aperture
        '''
        if this.ind_aper is None: return
        ind = this.ind_aper
        # orbit plot and LOSVD plot are mutually exclusive - if the orbit plot is shown, first remove it
        if this.orbitplot is not None:
            fig.delaxes(this.orbitplot)
            this.orbitplot = None
        if this.losvdplot is None:
            this.losvdplot = fig.add_axes([0.03, 0.07, 0.165, 0.36])
            this.losvdplot.set_yticklabels([])
            this.losvdplot.set_xlim(min(plot_gridv), max(plot_gridv))
            this.losvdplot.set_xlabel('v')
            this.losvdplot.set_ylabel('f(v)')
        this.losvdplot.cla()
        this.losvdplot.fill_between(plot_gridv, obs_losvd[0,ind], obs_losvd[2,ind], facecolor='r', alpha=0.33, lw=0)
        # plot the model LOSVD
        if hasattr(this, 'los_mod'):
            this.losvdplot.plot(plot_gridv, _agama.bsplineInterp(this.mod_degree, this.mod_gridv, this.los_mod[ind], plot_gridv), 'k')[0].set_dashes([5,2])
        this.losvdplot.set_xlim(min(plot_gridv), max(plot_gridv))
        this.losvdplot.set_yticklabels([])
        this.losvdplot.set_xlabel('v')
        this.losvdplot.set_ylabel('f(v)')
        # print some useful info
        coefs = ['v0', 'sigma', 'h3', 'h4', 'h5', 'h6']
        text  = 'Aperture #%i centered at x=%.3f, y=%.3f: ' % (ind, _numpy.mean(apertures[ind][:,0]), _numpy.mean(apertures[ind][:,1]))
        for i in range(6):
            text += '%s=%.3f +- %.3f ' % (coefs[i], ghm_val[ind,i], ghm_err[ind,i])
            if hasattr(this, 'ghm_mod'):
                err = (this.ghm_mod[ind,i]-ghm_val[ind,i]) / ghm_err[ind,i]
                if err < -1.:  text += '[\033[1;31m %.3f \033[0m] ' % this.ghm_mod[ind,i]
                elif err> 1.:  text += '[\033[1;34m %.3f \033[0m] ' % this.ghm_mod[ind,i]
                else: text += '[ %.3f ] ' % this.ghm_mod[ind,i]
        print(text)
        # highlight the selected polygon in all panels (make its boundary thicker)
        lw = _numpy.zeros(len(apertures))
        lw[ind] = 3.
        for p in patchcoll: p.set_linewidths(lw)

    def plotOrbit(indOrbit):
        '''
        show the 3d plot of the selected orbit, re-integrating it in the provided potential
        (the orbits computed during the model construction are not stored to save space).
        note that the provided potential might not be the same as the one originally used for this orbit
        (for instance, the BH mass can be different); at the moment, there is no simple way to ensure
        that orbits are recreated exactly, so this plot serves only as a rough illustration.
        '''
        # in case of more than one point under cursor, pick a random one with probability proportional to orbit weight
        prob = _numpy.cumsum(this.weights[indOrbit])
        prob /= prob[-1]
        indSel = indOrbit[_numpy.searchsorted(prob, _numpy.random.random())]
        this.oro.set_data(_numpy.atleast_2d(this.sco.get_offsets()[indSel]).T)
        this.orp.set_data(_numpy.atleast_2d(this.scp.get_offsets()[indSel]).T)
        print('Selected orbit #%i' % indSel +
            (' from a list of %i orbits' % len(indOrbit) if len(indOrbit)>1 else ''))
        if potential is None: return
        # LOSVD plot and orbit plot are mutually exclusive - if the LOSVD plot is shown, first remove it
        if this.losvdplot is not None:
            fig.delaxes(this.losvdplot)
            this.losvdplot = None
            this.ind_aper  = None
        if this.orbitplot is None:
            try:
                this.orbitplot = fig.add_axes([0.03, 0.06, 0.165, 0.37], projection='3d')
            except:
                traceback.print_exc()
                return
        this.orbitplot.cla()
        this.orbitplot.set_xlabel('x')
        this.orbitplot.set_ylabel('y')
        this.orbitplot.set_zlabel('z')
        try:
            orb = _agama.orbit(potential=potential, ic=this.ic[indSel], time=this.inttime[indSel], trajsize=0)[1]
            this.orbitplot.plot(orb[:,0], orb[:,1], orb[:,2], color='k', lw=0.5)
            #if hasattr(this.orbitplot, 'set_box_aspect'):  # matplotlib>=3.3
            #    this.orbitplot.set_box_aspect((1, 1, 1))   # doesn't help...
        except Exception as e: print(e)
        return

    def onclick(event):
        '''
        handle interactive user input
        '''
        if event.artist == modelgrid:
            loadModel(event.ind[0])
            plotMaps()
            plotLOSVD()
            fig.canvas.draw_idle()
            return
        if event.artist == this.sco or event.artist == this.scp:
            plotOrbit(event.ind)
            fig.canvas.draw_idle()
            return
        if event.artist in buttons:
            this.mode = event.artist
            plotMaps()
            fig.canvas.draw_idle()
            return
        has_poly, ind_poly = event.artist.contains(event.mouseevent)
        if has_poly:
            this.ind_aper = ind_poly['ind'][-1]
            plotLOSVD()
            fig.canvas.draw_idle()


    # main section of the runPlot routine
    fig = matplotlib.pyplot.figure(figsize=(17, 7), dpi=75)
    fig.canvas.mpl_connect('pick_event', onclick)

    # parse and combine all kinematic datasets
    apertures = []
    obs_degree= []
    obs_gridv = []
    ghm_val   = []
    ghm_err   = []
    for d in datasets:  # loop over kinematic datasets only
        if hasattr(d, 'getGHMoments'):
            sing = _numpy.sin(d.target_params['gamma'])
            cosg = _numpy.cos(d.target_params['gamma'])
            for i,a in enumerate(d.target_params['apertures']):
                apertures.append(_numpy.column_stack(( a[:,0] * cosg - a[:,1] * sing, a[:,1] * cosg + a[:,0] * sing )))
            gv, ge = d.getGHMoments()
            # we use exactly 6 GH moments, even if the data have fewer or more
            if gv.shape[1]<6:
                gv = _numpy.column_stack((gv, _numpy.zeros((gv.shape[0], 6-gv.shape[1]))*_numpy.nan ))
                ge = _numpy.column_stack((ge, _numpy.zeros((ge.shape[0], 6-ge.shape[1]))*_numpy.nan ))
            elif gv.shape[1]>6:
                gv = gv[:,0:6]
                ge = ge[:,0:6]
            ghm_val.extend(gv)
            ghm_err.extend(ge)
    ghm_val = _numpy.vstack(ghm_val)
    ghm_err = _numpy.vstack(ghm_err)
    if xlim is None: xlim = (min([_numpy.amin(p[:,0]) for p in apertures]), max([_numpy.amax(p[:,0]) for p in apertures]))
    if ylim is None: ylim = (min([_numpy.amin(p[:,1]) for p in apertures]), max([_numpy.amax(p[:,1]) for p in apertures]))
    if vlim is None: vlim = (min(ghm_val[:,0] - ghm_val[:,1] * 3.0), max(ghm_val[:,0] + ghm_val[:,1] * 3.0))
    plot_gridv = _numpy.linspace(vlim[0], vlim[1], 201)

    # collect the data for plotting the observed LOSVDs in each aperture
    obs_losvd = []
    for d in datasets:
        if hasattr(d, 'getLOSVD'):
            obs_losvd.append(d.getLOSVD(plot_gridv))
    obs_losvd = _numpy.hstack(obs_losvd)

    # compute projected density on a square but non-uniform grid (denser towards the center),
    # and only within the extent of the kinematic datasets
    gridmax = max(abs(xlim[0]), abs(xlim[1]), abs(ylim[0]), abs(ylim[1]))
    gridmin = gridmax * 1e-3
    gridpix = _agama.symmetricGrid(101, gridmin, gridmax)
    plot_gridx = gridpix[(gridpix>=xlim[0]) * (gridpix<=xlim[1])]
    plot_gridy = gridpix[(gridpix>=ylim[0]) * (gridpix<=ylim[1])]
    projectedDensityMag = _numpy.zeros((len(plot_gridx), len(plot_gridy)))
    for d in datasets:
        if hasattr(d, 'projectedDensity'):
            projectedDensity = d.projectedDensity(plot_gridx, plot_gridy)
            # convert to stellar magnitudes per unit surface
            projectedDensityMag = -2.5 * _numpy.log10(projectedDensity / _numpy.max(projectedDensity)) - 0.01

    # parameters of kinematic maps; the default ranges are taken from the input data,
    # but the user may adjust the range of data values and errors
    if v0lim    is None: v0lim   = max(abs(ghm_val[:,0])+ghm_err[:,0]) * _numpy.array([-1,1])  # symmetric range
    if sigmalim is None: sigmalim=(min(ghm_val[:,1]-ghm_err[:,1]), max(ghm_val[:,1]+ghm_err[:,1]))
    if v0err    is None: v0err   = max(ghm_err[:,0])
    if sigmaerr is None: sigmaerr= max(ghm_err[:,1])
    panel_params = [
        dict(title=r'$v_0$',   data_range=v0lim,   error_range=v0err,   extent=[0.24, 0.59, 0.165, 0.40]),
        dict(title=r'$\sigma$',data_range=sigmalim,error_range=sigmaerr,extent=[0.24, 0.09, 0.165, 0.40]),
        dict(title=r'$h_3$',   data_range=hlim,    error_range=herr,    extent=[0.43, 0.59, 0.165, 0.40]),
        dict(title=r'$h_4$',   data_range=hlim,    error_range=herr,    extent=[0.43, 0.09, 0.165, 0.40]),
        dict(title=r'$h_5$',   data_range=hlim,    error_range=herr,    extent=[0.62, 0.59, 0.165, 0.40]),
        dict(title=r'$h_6$',   data_range=hlim,    error_range=herr,    extent=[0.62, 0.09, 0.165, 0.40]),
    ]

    ##### four buttons determining which map to display #####
    radioplot = fig.add_axes([0.03, 0.43, 0.165, 0.1])
    radioplot.set_axis_off()
    buttons = [
        matplotlib.patches.Rectangle((-0.50, 0.05), 0.49,0.4, color='#60f080', picker=True, ec='k'),
        matplotlib.patches.Rectangle(( 0.01, 0.05), 0.49,0.4, color='#ffe000', picker=True, ec='k'),
        matplotlib.patches.Rectangle((-0.50,-0.45), 0.49,0.4, color='#80a0ff', picker=True, ec='k'),
        matplotlib.patches.Rectangle(( 0.01,-0.45), 0.49,0.4, color='#ff80a0', picker=True, ec='k') ]
    for b in buttons:
        radioplot.add_artist(b)
    radioplot.text(-0.255, 0.25, 'data',      ha='center', va='center')
    radioplot.text( 0.255, 0.25, 'data err',  ha='center', va='center')
    radioplot.text(-0.255,-0.25, 'model',     ha='center', va='center')
    radioplot.text( 0.255,-0.25, 'model err', ha='center', va='center')
    radioplot.set_xlim(-0.5,0.5)
    radioplot.set_ylim(-0.5,0.5)
    this.mode = buttons[0]

    ##### LOSVD in the selected aperture #####
    this.losvdplot = None
    this.ind_aper  = None

    ##### 3d plot of the selected orbit #####
    this.orbitplot = None

    ##### two panels with orbit weights #####
    this.axo = this.axp = this.sco = this.scp = None

    ##### maps of v,sigma and higher Gauss-Hermite moments #####
    patchcoll = []
    panels = []
    chi2labels = []
    for param in panel_params:
        patches = matplotlib.collections.PatchCollection(
            [matplotlib.patches.Polygon(p, closed=True) for p in apertures],
            picker=0.0, edgecolor=(0.5,0.5,0.5,0.5), linewidths=0)
        patchcoll.append(patches)
        # make sure that pan/zoom is synchronized between kinematic maps
        if panels:
            kwargs = dict(sharex=panels[0], sharey=panels[0])
        else:
            kwargs = {}
        # enforce a correct aspect ratio for kinematic maps
        kwargs['aspect'] = 'equal'
        ax = fig.add_axes(param['extent'], **kwargs)
        ax.add_collection(patches)
        ax.text(0.02, 0.98, param['title'], fontsize=16, transform=ax.transAxes, ha='left', va='top')
        chi2labels.append(ax.text(0.98, 0.98, '', fontsize=12, transform=ax.transAxes, ha='right', va='top'))
        # plot the surface density profile by contours spaced by a factor 2.5 (i.e. one stellar magnitude)
        # and a total dynamical range of 10 magnitudes, i.e. a factor of 10^4
        ax.contour(plot_gridx, plot_gridy, projectedDensityMag.T,
            levels=_numpy.linspace(0, 10, 11), cmap='Greys_r', vmin=0, vmax=11)
        ax.set_xlim(xlim[1], xlim[0])  # note the inverted X axis!
        ax.set_ylim(ylim[0], ylim[1])
        panels.append(ax)

    plotMaps()
    # add colorbars to data maps
    for patch, param in zip(patchcoll, panel_params):
        cax = fig.add_axes([param['extent'][0], param['extent'][1]-0.05, param['extent'][2], 0.01])
        fig.colorbar(patch, cax=cax, orientation='horizontal', ticks=matplotlib.ticker.MaxNLocator(6))

    ##### likelihood surface #####
    if len(aval)>0:
        print('%i models available' % len(aval))
        ax = fig.add_axes([0.03, 0.59, 0.165, 0.40])
        modelgrid = ax.plot(aval, bval, 'o', c='g', ms=5, picker=5, mew=0, alpha=0.75)[0]
        this.selected  = ax.plot([_numpy.nan], [_numpy.nan], marker='o', c='r', ms=8, mew=0)[0]
        this.modellabel= ax.text(0.01, 0.01, '', color='r', ha='left', va='bottom', transform=ax.transAxes, fontsize=10)
        ax.text(0.5, 0.99, r'$\mathrm{min}\,\chi^2=%.2f$' % min(chi2), color='r', ha='center', va='top', transform=ax.transAxes)
        ax.set_xlabel(alabel, labelpad=0)
        ax.set_ylabel(blabel, labelpad=0)
        if alim is None: alim = (min(aval), max(aval))
        if blim is None: blim = (min(bval), max(bval))
        anorm = alim[1]-alim[0]
        bnorm = blim[1]-blim[0]
        ax.set_xlim(alim[0], alim[1])
        ax.set_ylim(blim[0]-0.05*bnorm, blim[1]+0.05*bnorm)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
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
            # marginalized chi^2 as a function of the variable on the horizontal axis
            with _numpy.errstate(all='ignore'):
                cmarg = -2*_numpy.log(_numpy.sum( _numpy.exp(-0.5*_numpy.where(c==c, c, _numpy.inf)), axis=0))
            ax.plot(a[0], (cmarg-_numpy.nanmin(cmarg))/deltaChi2lim, color='r', transform=ax.get_xaxis_transform())
            for itick in range(5):
                ax.text(1.01, itick/5.0, '%.0f' % (itick/5.0*deltaChi2lim), color='r',
                    clip_on=False, ha='left', va='center', transform=ax.transAxes)
            ax.text(1.0, 0.95, r'$\Delta\chi^2$', color='r', clip_on=False, ha='left', va='center', transform=ax.transAxes)
        except:
            traceback.print_exc()
    else:
        modelgrid = None

    # start the interactive plot
    matplotlib.pyplot.show()
