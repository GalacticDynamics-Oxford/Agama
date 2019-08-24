"""
This is a collection of various routines that complement the Python interface to the Agama library.
The __init__.py file in the root directory of Agama package imports both the C++ extension (agama.so)
and this python module (py/pygama.py) and merges them into a single namespace; hence one may write
>>> import agama                                  # import both C++ and Python modules simultaneously
>>> par = agama.getDensityParams(1., 2., 3., 4.)  # use routines from this file...
>>> den = agama.Density(**par)                    # and classes and routines defined in the C++ library
"""

### ------------------------------------------------------------------------------- ###
### routines for dealing with 2d IFU data, represented on a regular 2d grid of spaxels,
### several spaxels could be Voronoi-binned together into apertures.
### Agama uses arbitrary polygons that define the boundaries of each aperture,
### whether it is just a square pixel or a more complicated region.

def getRegularApertures(xcoords, ycoords):
    """
    Construct boundary polygons corresponding to a regular separable grid in x,y:
    each aperture is a rectangular region spanning x[i]..x[i+1], y[j]..y[j+1].
    Input:  grid nodes in x and y; the number of pixels is (len(xcoords)-1) * (len(ycoords)-1)
    Output: list of pixel boundaries (each element is a 4x2 array of four corner points)
    """
    from numpy import repeat, tile
    ii = range(len(xcoords)-1)
    jj = range(len(ycoords)-1)
    return [ ((xcoords[i],ycoords[j]), (xcoords[i+1],ycoords[j]), \
        (xcoords[i+1],ycoords[j+1]), (xcoords[i],ycoords[j+1])) \
        for i,j in zip(tile(ii, len(jj)), repeat(jj, len(ii))) ]

def getBinnedApertures(xcoords, ycoords, bintags):
    """
    Convert the data for Voronoi binned pixels into the polygons
    describing the boundary of each connected Voronoi region.
    Input: three 1d arrays of equal length: coord x and y of the center of each pixel,
    and bin index for each pixel.
    Output contains a list of length Nbins, each element is the list of vertices
    of the boundary polygon for each bin (a 2d array with 2 columns (x,y) and Nvert rows).
    """
    import numpy
    binTags = numpy.unique(bintags)    # list of bin tags (indices)
    xCoords = numpy.unique(numpy.round(xcoords*1e6)*1e-6)      # list of possible x coordinates
    yCoords = numpy.unique(numpy.round(ycoords*1e6)*1e-6)      # same for y; rounded appropriately
    xpixel  = xCoords[1]-xCoords[0]    # size of a single pixel (assuming they all are equal)
    ypixel  = yCoords[1]-yCoords[0]    # same in y direction
    xcount  = int(round((xCoords[-1]-xCoords[0]) / xpixel)+1)  # total number of pixels in x direction
    ycount  = int(round((yCoords[-1]-yCoords[0]) / ypixel)+1)  # same for y
    if xcount > 10000 or ycount > 10000:
        raise ValueError("Can't determine pixel size: "+str(xpixel)+" * "+str(ypixel)+" doesn't seem right")
    polygons= []        # list of initially empty polygons
    matrix  = numpy.ones((xcount,ycount), dtype=numpy.int32) * -1  # 2d array of binTag for each pixel
    for (p,x,y) in zip(bintags, xcoords, ycoords):  # assign bin tags to each pixel in the 2d array
        matrix[int(round((x-xCoords[0]) / xpixel)), int(round((y-yCoords[0]) / ypixel)) ] = int(p)
    for b in binTags:
        # obtain the table of x- and y-indices of all elements with the same binTag
        ix, iy = (matrix==b).nonzero()
        if len(ix)==0:
            print ("Empty bin %i" % b)
            continue
        minx   = min(ix)-1
        maxx   = max(ix)+2
        miny   = min(iy)-1
        maxy   = max(iy)+2
        pixels = numpy.zeros((maxx-minx, maxy-miny), dtype=numpy.int32)
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
            raise ValueError("Lost in the way for bin "+str(b))
        polygons.append( numpy.array(vertices) )
    return polygons

def writeApertures(filename, polygons):
    """
    Write the list of polygons serving as aperture boundaries to a text file
    """
    with open(filename, 'w') as dfile:
        for i,polygon in enumerate(polygons):
            for vertex in polygon:  dfile.write('%f %f\n' % (vertex[0], vertex[1]))
            if i<len(polygons)-1:   dfile.write('\n')

def readApertures(filename):
    """
    Read the list of polygons from a text file
    """
    from numpy import array
    with open(filename) as dfile:
        return [array([float(a) for a in b.split()]).reshape(-1,2) for b in dfile.read().split("\n\n")]



### ------------------------------------------------------------------ ###
### routines for coordinate transformation, projection and deprojection
### of ellipsoidally stratified profiles (e.g. a Multi-Gaussian expansion)

def makeRotationMatrix(alpha, beta, gamma):
    """
    Construct the matrix for transforming the coordinates (x,y,z) in the intrinsic system
    into the coordinates (X,Y,Z) in the rotated coordinate system.
    Usually the intrinsic coordinates are associated with the stellar system,
    and the rotated ones -- with the observer; in this case X,Y are the coordinates in
    the image plane: Y axis points up/north, X axis points left(!)/east, and Z axis
    points along the line of sight away from the observer.
    alpha, beta, gamma are the three Euler rotation angles:
    alpha is the rotation in the equatorial plane of the stellar system,
    beta is the inclination angle,
    gamma is the rotation in the sky plane (if gamma=0, the z axis projects onto the Y axis).
    """
    from math import sin,cos; from numpy import array
    return array([
        [ cos(alpha) * cos(gamma) - sin(alpha) * cos(beta) * sin(gamma),
          sin(alpha) * cos(gamma) + cos(alpha) * cos(beta) * sin(gamma),
          sin(beta)  * sin(gamma) ],
        [-cos(alpha) * sin(gamma) - sin(alpha) * cos(beta) * cos(gamma),
         -sin(alpha) * sin(gamma) + cos(alpha) * cos(beta) * cos(gamma),
          sin(beta)  * cos(gamma) ],
        [ sin(alpha) * sin(beta),
         -cos(alpha) * sin(beta),
          cos(beta) ] ])

def getProjectedEllipse(Sx, Sy, Sz, alpha, beta, gamma):
    """
    Project a triaxial ellipsoid with intrinsic axes Sx, Sy, Sz onto the image plane XY,
    whose orientation w.r.t. the intrinsic coordinate system is defined by three Euler angles
    alpha, beta, gamma (Y axis points up and X axis points left!).
    The projection is an ellipse in the image plane.
    return: SXp, SYp  are the major and minor axes of the ellipse;
    eta  is the position angle of the major axis of this ellipse in the image plane
    (measured counter-clockwise from the Y axis towards the X axis)
    """
    from math import sin,cos,atan2,sqrt,pi
    if abs(sin(beta)) < 1e-12:  # shortcut for a face-on orientation, avoiding roundoff errors
        return Sx, Sy, ((alpha - pi/2) * cos(beta) + gamma) % pi % pi
    # axis ratios
    p = Sy / Sx
    q = Sz / Sx
    # parameters defined in Binney(1985)
    f = (p * q * sin(beta) * sin(alpha))**2 + (q * sin(beta) * cos(alpha))**2 + (p * cos(beta))**2
    A = (p * cos(beta) * cos(alpha))**2 + (q * sin(beta))**2 + (cos(beta) * sin(alpha))**2
    B = (p*p-1) * cos(beta) * 2 * sin(alpha) * cos(alpha)
    C = cos(alpha)**2 + (p * sin(alpha))**2
    D = sqrt( (A-C)**2 + B**2 )
    # axes of the projected ellipsoid
    SXp = Sx * sqrt(2*f / (A+C-D))
    SYp = Sx * sqrt(2*f / (A+C+D))
    # position angle of the projected major axis ccw from projected z axis
    psi = 0.5 * atan2(B, A-C)
    # position angle of the projected major axis ccw from Y (north) direction in the image plane
    eta = psi + gamma
    return max(Sy, min(Sx, SXp)), max(Sz, min(Sy, SYp)), eta % pi % pi

def getIntrinsicShape(SXp, SYp, eta, alpha, beta, gamma):
    """
    Deproject the ellipse in the image plane into a triaxial ellipsoid,
    for an assumed orientation of the intrinsic coord.sys. of that ellipsoid.
    SXp and SYp are lengths of the major and minor axes of the ellipse,
    eta is the position angle of the major axis of the ellipse, measured counter-clockwise
    from the Y (vertical) axis on the image plane (towards the X axis, which points left).
    alpha, beta, gamma are the assumed angles of rotation of the coordinate frame.
    return: three intrinsic axes (Sx,Sy,Sz) of the triaxial ellipsoid,
    or throw an exception if the deprojection is impossible for these angles.
    The deprojection is not unique in the following special cases:
    - if beta==0 or beta==pi (face-on view down the z axis) - cannot determine q=Sz/Sx, assume q=p;
    - if psi==pi/2 (angle between projected z axis and major axis) - cannot determine p=Sy/Sx, assume p=1;
    """
    from math import pi,sin,cos,tan,sqrt
    Q = SYp / SXp
    if Q>1:
        raise ValueError('Projected axis ratio must be <=1')
    psi = eta - gamma
    if abs(sin(beta)) < 1e-12:  # face-on view (inclination is zero) - assume prolate axisymmetric shape
        return (SXp, SYp, SYp)  # q=Sz/Sx is not constrained, assume it to be equal to p=Sy/Sx
    if abs(cos(psi)) < 1e-12:   # no misalignment - assume oblate axisymmetric shape
        return (SXp, SXp, SXp * sqrt(1 - (1-Q**2) / sin(beta)**2))
    num = (1-Q*Q) * (cos(psi) * cos(beta) / tan(alpha) + sin(psi))
    den = (num * cos(psi) - cos(beta) / tan(alpha))
    q   = sqrt(1 - num * (cos(psi) - sin(psi) * cos(beta) / tan(alpha)) / sin(beta)**2 / den)
    p   = sqrt(1 - (1-Q*Q) * sin(psi) * cos(psi) / sin(alpha)**2 / den)
    f   = (p*q * sin(beta) * sin(alpha))**2 + (q * sin(beta) * cos(alpha))**2 + (p * cos(beta))**2
    Sx  = SXp * sqrt(Q / sqrt(f))
    return (Sx, Sx*p, Sx*q)

def getViewingAngles(SXp, SYp, eta, Sx, Sy, Sz):
    """
    Find the viewing angles that would project a triaxial ellipsoid with axes Sx >= Sy >= Sz
    into the ellipse in the image plane with axes SXp >= SYp and position angle of the major axis eta.
    SXp and SYp are the lengths of the major and minor axes of the ellipse in the image plane,
    eta is the position angle of the major axis of this ellipse, measured counter-clockwise
    from the Y (vertical) axis on the image plane (towards the X axis, which points left).
    Sx, Sy, Sz are the assumed intrinsic axis lengths.
    return: a 4x3 array, where each row represents a triplet of viewing angles alpha, beta, gamma
    (there are four possible deprojections in general, although some of them may be identical),
    or throw an exception if the deprojection is impossible for the given parameters.
    """
    from math import pi,asin,acos,atan,sqrt
    p  = Sy  / Sx
    q  = Sz  / Sx
    u  = SXp / Sx
    v  = SYp / Sx
    if not (0<=q and q<=v and v<=p and p<=u and u<=1):
        raise ValueError('Projected and assumed axis lengths are inconsistent (should be '+
        "0 <= Sz=%.16g <= SYp=%.16g <= Sy=%.16g <= SXp=%.16g <= Sx=%.16g)" % (Sz, SYp, Sy, SXp, Sx))
    if p==q:
        if p==1:
            if abs(u-1) + abs(v-1) > 1e-12:
                raise ValueError('All axes must be equal in the spherical case')
            beta = 0  # assume face-on orientation for a spherical system
        else:
            if abs(v-p) > 1e-12:
                raise ValueError('Prolate axisymmetric system must have SYp = Sy')
            beta = asin(sqrt( (1-u*u) / (1-q*q) ) )
            # this case is not yet supported ?
    else:
        beta  = acos(sqrt( (u*u-q*q) / (1-q*q) * (v*v-q*q) / (p*p-q*q) ) )
    if 1-u < 1e-12:   # u=1 <=> SXp=Sx - assume an oblate axisymmetric system
        alpha = 0
    elif u-p < 1e-12 or p-v < 1e-12:
        alpha = pi/2
    else:
        alpha = atan(sqrt( (1-v*v) * (1-u*u) / (p*p-v*v) / (u*u-p*p) * (p*p-q*q) / (1-q*q) ) )
    if p-v < 1e-12:
        psi   = 0
    elif u-p < 1e-12 or v-q < 1e-12 or 1-u < 1e-12:
        psi   = pi/2
    else:
        psi   = atan(sqrt( (1-v*v) / (1-u*u) * (p*p-v*v) / (v*v-q*q) * (u*u-q*q) / (u*u-p*p) ) )
    # two possible choices of gamma, normalized at first to the range [0..2pi)
    gamma1 = (eta+psi) % (2*pi) % (2*pi)
    gamma2 = (eta-psi) % (2*pi) % (2*pi)
    # normalize alpha and gamma to the range [0..pi); beta is in the range [0..pi] by construction
    def norm(alpha, beta, gamma):
        if gamma < pi:
            return (    alpha %pi%pi, beta,    gamma   )
        else:
            return ((pi-alpha)%pi%pi, pi-beta, gamma-pi)
    # return four possible combinations of viewing angles
    return (
        norm( alpha,    beta, gamma1),
        norm(-alpha, pi-beta, gamma1),
        norm(-alpha,    beta, gamma2),
        norm( alpha, pi-beta, gamma2) )

### ---------------------------------------- ###
### Specific tools for Multi-Gaussian expansions

def getDensityParams(Mass, Sx, Sy, Sz):
    """
    return a dictionary containing parameters for creating agama.Density object
    corresponding to a single Gaussian component of an MGE
    """
    from numpy import pi
    return dict( \
        density = "Spheroid",
        axisRatioY  = Sy/Sx,
        axisRatioZ  = Sz/Sx,
        scaleRadius = 1,
        gamma       = 0,
        beta        = 0,
        alpha       = 1,
        outerCutoffRadius = 2**0.5 * Sx,
        cutoffStrength = 2,
        densityNorm    = Mass / ((2*pi)**1.5 * Sx * Sy * Sz) )

def makeDensityFromMGE(tab, distance, theta, phi, chi):
    """
    Construct an agama.Density object corresponding to a MGE read from a text file
    and deprojected assuming the given viewing angles theta, phi, chi.
    Input:
    tab - array with 3 columns, as read from a text file produced by MGE fitting routines;
    each row contains data for one Gaussian components, columns are:
    central luminosity (Lsun/pc^2), width of the major axis (arcsec), flattening (q<=1).
    distance - assumed distance to the object in pc, needed to convert arcseconds to parsecs.
    theta, phi, chi - three assumed viewing angles
    """
    from numpy import pi,array
    from agama import Density
    conv = distance * pi / 648000   # conversion factor from arcseconds to parsecs
    intrshape = array([
        getIntrinsicShape(conv * m[1], conv * m[1] * m[2], pi/2, theta, phi, chi) for m in tab])
    masses = 2*pi * (conv * tab[:,1])**2 * tab[:,0] * tab[:,2]
    return Density(*[Density( **getDensityParams(mass, *axes)) for mass,axes in zip(masses, intrshape)])

def surfaceDensityFromMGE(tab, xp, yp):
    """
    Evaluate the surface density specified by a MGE at a given set of points xp,yp in the image plane
    Input:
    tab - array with 3 columns, as read from a text file produced by MGE fitting routines
    each row contains data for one Gaussian components, columns are:
    central luminosity (Lsun/pc^2), width of the major axis (arcsec), flattening (q<=1).
    xp, yp - two arrays of equal length, specifying the image plane coordinates of points
    where the surface density should be computed
    """
    from numpy import sum,exp
    return sum([comp[0] * exp( -0.5 * (xp**2 + (yp/comp[2])**2) / comp[1]**2) for comp in tab], axis=0)


### -------------------------------------------------------------------------- ###
### routines for representing a function specified in terms of its coefficients of
### B-spline or Gauss-Hermite expansions

def bsplines(N, grid, x):
    """
    Compute B-splines of degree N over the given grid, for the input point x
    """
    from bisect import bisect
    from numpy import zeros
    npoints = len(grid)
    result = zeros(npoints+N-1)
    if(x<grid[0] or x>grid[npoints-1]):
        return result
    def linInt(x, grid, i1, i2):
        x1 = grid[max(0, min(npoints-1, i1))]
        x2 = grid[max(0, min(npoints-1, i2))]
        if(x1==x2):
            return 1 if x==x1 else 0
        else:
            return (x-x1) / (x2-x1)
    ind = bisect(grid, x)-1
    if(ind == npoints-1): ind-=1
    B = [0.] * (N+1)
    B[N] = 1.
    for l in range(N):
        Bip1=0
        for j in range(ind,ind-l-2,-1):
            i  = j-ind+N
            Bi = B[i] * linInt(x, grid, j, j+l+1) + Bip1 * linInt(x, grid, j+l+2, j+1)
            Bip1 = B[i]
            B[i] = Bi
    for i in range(N+1):
        result[i+ind]=B[i]
    return result

def bsplInt(N, grid, ampl, x):
    """
    Compute interpolated values of B-spline expansion of degree N over the given grid and ampitudes,
    for the input point or an array of points x
    """
    from numpy import array, dot
    if x is float: return dot(bsplines(N, grid, x), ampl)
    return array([ dot(bsplines(N, grid, xx), ampl) for xx in x])

def bsplIntegrals(N, grid, power=0):
    """
    Compute the vector of integrals of B-spline basis functions, optionally multiplied by x^power.
    To obtain the integral of a function represented by amplitudes of its a B-spline expansion,
    multiply this vector by the array of amplitudes.
    """
    from numpy import zeros
    # use a 3-point Gauss-Legendre quadrature on each grid segment, sufficient for polynomials up to
    # degree 5 (e.g. a B-spline of degree 3 times x^2)
    glnode = 0.11270166537926
    grid1  = grid[1:]  * glnode + grid[:-1] * (1-glnode)
    grid2  = grid[:-1] * glnode + grid[1:]  * (1-glnode)
    grid3  = 0.5 * (grid[1:] + grid[:-1])
    result = zeros(len(grid)+N-1)
    for i in range(len(grid)-1):
        result += (grid[i+1] - grid[i]) * (
            0.277777777777778 * bsplines(N, grid, grid1[i]) * grid1[i]**power + \
            0.277777777777778 * bsplines(N, grid, grid2[i]) * grid2[i]**power + \
            0.444444444444444 * bsplines(N, grid, grid3[i]) * grid3[i]**power )
    return result

def GaussHermite(gamma, center, sigma, coefs, xarr):
    """
    Compute the function specified by a Gauss-Hermite expansion with the given
    overall amplitude (gamma), central point (center), width (sigma),
    and a list of N>=2 GH coefficients, at point(s) xarr
    """
    from numpy import pi, exp
    xscaled = (xarr - center) / sigma
    norm    = (0.5/pi)**0.5 * gamma / sigma * exp(-0.5 * xscaled**2)
    if not coefs is None and len(coefs) >= 2:
        hpp = 1.0
        hp  = 2**0.5 * xscaled
        result = hpp * coefs[0] + hp * coefs[1]
        for n in range(2, len(coefs)):
            hn = (2**0.5 * xscaled * hp - (n-1)**0.5 * hpp) / n**0.5
            result += hn * coefs[n]
            hpp= hp
            hp = hn
    else: result = 1.
    return result * norm


### module initialization: add some custom colormaps to matplotlib ###
try:
    import agamacolormaps
except:
    pass
