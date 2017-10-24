"""
This is a collection of various routines that complement the Python interface to the Agama library
"""
import numpy, bisect
from agama import *

### -------------------------------------------------------------------------------------------- ###
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
    ii = range(len(xcoords)-1)
    jj = range(len(ycoords)-1)
    return [ ((xcoords[i],ycoords[j]), (xcoords[i+1],ycoords[j]), \
        (xcoords[i+1],ycoords[j+1]), (xcoords[i],ycoords[j+1])) \
        for i,j in zip(numpy.tile(ii, len(jj)), numpy.repeat(jj, len(ii))) ]

def getBinnedApertures(xcoords, ycoords, bintags):
    """
    Convert the data for Voronoi binned pixels into the polygons
    describing the boundary of each connected Voronoi region.
    Input: three 1d arrays of equal length: coord x and y of the center of each pixel,
    and bin index for each pixel.
    Output contains a list of length Nbins, each element is the list of vertices
    of the boundary polygon for each bin (a 2d array with 2 columns (x,y) and Nvert rows).
    """
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
            print "Empty bin", b
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
    with open(filename, "w") as dfile:
        for i,polygon in enumerate(polygons):
            for vertex in polygon:  print >>dfile, vertex[0], vertex[1]
            if i<len(polygons)-1:   print >>dfile, ""

def readApertures(filename):
    """
    Read the list of polygons from a text file
    """
    with open(filename) as dfile:
        return [numpy.array([float(a) for a in b.split()]).reshape(-1,2) for b in dfile.read().split("\n\n")]



### -------------------------------------------------------------------------------------------- ###
### routines for coordinate transformation, projection and deprojection of a Multi-Gaussian expansion

def makeProjectionMatrix(theta, phi, chi):
    """
    Construct the matrix for transforming the intrinsic coordinates (x,y,z)
    into projected ones (x',y',z') in the rotated coordinate system.
    Projected coords are such that x', y' is the image plane,
    and z' is directed towards the observer.
    theta, phi  are the spherical polar coordinates of the line-of-sight (z' axis)
    in the intrinsic coordinate system;
    chi  is the angle of rotation of the intrinsic coordinate system about the line of sight
    (if chi=0, the z axis is projected onto y' axis, and for chi>0 the z-axis projection
    in the x'y' plane has a positition angle chi w.r.t. y' axis, measured counterclockwise).
    """
    from math import sin,cos
    return numpy.array([
        [ -cos(chi) * sin(phi) + sin(chi) * cos(theta) * cos(phi),
           cos(chi) * cos(phi) + sin(chi) * cos(theta) * sin(phi),
          -sin(chi) * sin(theta) ],
        [ -sin(chi) * sin(phi) - cos(chi) * cos(theta) * cos(phi),
           sin(chi) * cos(phi) - cos(chi) * cos(theta) * sin(phi),
           cos(chi) * sin(theta) ],
        [  sin(theta) * cos(phi),
           sin(theta) * sin(phi),
           cos(theta) ] ])

def getProjectedEllipsoid(Sx, Sy, Sz, theta, phi, chi):
    """
    Project a triaxial ellipsoid with intrinsic axes Sx, Sy, Sz
    onto the image plane defined by three viewing angles theta, phi, chi
    return: Sxp, Syp  are the projected major and minor axes,
    eta  is the position angle of projected major axis ccw from the y' axis in the image plane
    """
    from math import sin,cos,atan2,sqrt
    # axis ratios
    p = Sy / Sx
    q = Sz / Sx
    # parameters defined in Binney(1985)
    f = (p*q*sin(theta)*cos(phi))**2 + (q*sin(theta)*sin(phi))**2 + (p*cos(theta))**2
    A = (p*cos(theta)*sin(phi))**2 + (q*sin(theta))**2 + (cos(theta)*cos(phi))**2
    B = (p*p-1)*cos(theta)*sin(2*phi)
    C = sin(phi)**2 + (p*cos(phi))**2
    D = sqrt( (A-C)**2 + B**2 )
    # axes of the projected ellipsoid
    Sxp = Sx * sqrt(2*f / (A+C-D))
    Syp = Sx * sqrt(2*f / (A+C+D))
    # position angle of the projected major axis ccw from projected z axis
    psi = 0.5 * atan2(B, A-C)
    # position angle of the projected major axis ccw from y' (fixed direction in the image plane)
    eta = psi + chi
    return (Sxp, Syp, eta)

def getIntrinsicShape(Sxp, Syp, eta, theta, phi, chi):
    """
    Deproject the density distribution for an assumed orientation of the intrinsic coord.sys.:
    Sxp and Syp are lengths of projected major and minor axes,
    eta is the position angle of the major axis ccw from the y' axis on the image plane.
    theta, psi, chi are the assumed angles of rotation of the coordinate frame.
    return: three intrinsic axes (Sx,Sy,Sz) of the density profile,
    or throw an exception if the deprojection is impossible for these angles.
    The deprojection is not unique in the following special cases:
    - if theta==0 or theta==pi (face-on view down the z axis) - cannot constrain q=Sz/Sx, assume Sz=Sy;
    - if theta==pi/2 (edge-on view in the x-y plane) - cannot constrain p=Sy/Sx, assume Sy=Sx;
    - if phi==0 or phi==pi/2 (view in the x-z or y-z planes) - same as above;
    """
    from math import pi,sin,cos,tan,sqrt
    qp  = Syp / Sxp
    psi = eta - chi
    if theta==0 or theta==pi:   # face-on view
        if abs(sin(psi+phi)) > 1e-15: raise ValueError('Face-on view must have  eta + phi - chi = 0')
        return (Sxp, Syp, Syp)  # q=Sz/Sx is not constrained, assume it to be equal to p=Sy/Sx
    if theta==pi/2 or phi==0 or phi==pi/2 or phi==pi:  # assume axisymmetric shape with p=1
        if abs(cos(psi)) > 1e-15: raise ValueError('Axisymmetric view must have eta - chi = +-pi/2')
        return (Sxp, Sxp, Sxp * sqrt(qp*qp - cos(theta)**2) / sin(theta))
    num = (1-qp*qp) * (cos(psi) * cos(theta) * tan(phi) + sin(psi))
    den = (num * cos(psi) - cos(theta) * tan(phi))
    q   = sqrt(1 - num * (cos(psi) - sin(psi) * cos(theta) * tan(phi)) / sin(theta)**2 / den)
    p   = sqrt(1 - (1-qp*qp) * sin(psi) * cos(psi) / cos(phi)**2 / den)
    f   = (p*q * sin(theta) * cos(phi))**2 + (q * sin(theta) * sin(phi))**2 + (p * cos(theta))**2
    Sx  = Sxp * sqrt(qp / sqrt(f))
    return (Sx, Sx*p, Sx*q)

def getViewingAngles(Sxp, Syp, eta, Sx, Sy, Sz):
    """
    Find viewing angles that would project the 3d ellipsoid with axes Sx >= Sy >= Sz
    into the projected ellipsoid with axes Sxp >= Syp and misalignment angle psi
    (the position angle of the projected major axis ccw from the projected z axis)
    Sxp and Syp are the lengths of projected major and minor axes,
    eta is the position angle of the major axis ccw from the y' axis on the image plane.
    Sx, Sy, Sz are the assumed intrinsic axis lengths.
    return: viewing angles theta, phi, chi,
    or throw an exception if the deprojection is impossible for the given parameters.
    """
    from math import pi,asin,acos,atan,sqrt
    p     = Sy  / Sx
    q     = Sz  / Sx
    qp    = Syp / Sxp
    u     = Sxp / Sx
    if 0>q or q>p or p>1 or 0>qp or qp>1:
        raise ValueError('Axes must be sorted in order of increase')
    if p==q:
        if p==1:
            if abs(u-1)>1e-15 or abs(qp-1)>1e-15:
                raise ValueError('All axes must be equal in the spherical case')
            return (0, 0, eta)
        if abs(qp*u-q)>1e-15:
            raise ValueError('Prolate axisymmetric system must have Syp = Sy')
        theta = asin(sqrt( (1-u*u) / (1-q*q) ) )
        # this case is not yet supported
    else:
        theta = acos(sqrt( (u*u-q*q) * (qp*qp*u*u-q*q) / (1-q*q) / (p*p-q*q) ) )
    if p==1:
        if abs(u-1)>1e-15:
            raise ValueError('Oblate axisymmetric system must have Sxp = Sx')
        phi = 0   # angle phi does not matter in this case, return any reasonable value
        psi = pi/2
    else:
        phi = atan(sqrt( (u*u-p*p) * (p*p-qp*qp*u*u) / (1-u*u) / (p*p-q*q) * (1-q*q) / (1-qp*qp*u*u) ) )
        psi =-atan(sqrt( (u*u-q*q) * (p*p-qp*qp*u*u) / (1-u*u) / (u*u-p*p) * (1-qp*qp*u*u) / (qp*qp*u*u-q*q) ) )
    chi   = eta - psi
    return (theta, phi, chi)

def getDensityParams(Mass, Sx, Sy, Sz):
    """
    return a dictionary containing parameters for creating agama.Density object
    corresponding to a single Gaussian component of an MGE
    """
    return dict( \
        density = "SpheroidDensity",
        axisRatioY  = Sy/Sx,
        axisRatioZ  = Sz/Sx,
        scaleRadius = 1,
        gamma       = 0,
        beta        = 0,
        alpha       = 1,
        outerCutoffRadius = 2**0.5 * Sx,
        cutoffStrength = 2,
        densityNorm    = Mass / ((2*numpy.pi)**1.5 * Sx * Sy * Sz) )

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
    conv = distance * numpy.pi / 648000   # conversion factor from arcseconds to parsecs
    intrshape = numpy.array([
        getIntrinsicShape(conv * m[1], conv * m[1] * m[2], numpy.pi/2, theta, phi, chi) for m in tab])
    masses = 2*numpy.pi * (conv * tab[:,1])**2 * tab[:,0] * tab[:,2]
    return Density(*[Density( **getDensityParams(mass, *axes)) for mass,axes in zip(masses, intrshape)])


### -------------------------------------------------------------------------------------------- ###
### routines for representing a function specified in terms of its coefficients of
### B-spline or Gauss-Hermite expansions

def bsplines(N, grid, x):
    """
    Compute B-splines of degree N over the given grid, for the input point x
    """
    npoints = len(grid)
    result = numpy.zeros(npoints+N-1)
    if(x<grid[0] or x>grid[npoints-1]):
        return result
    def linInt(x, grid, i1, i2):
        x1 = grid[max(0, min(npoints-1, i1))]
        x2 = grid[max(0, min(npoints-1, i2))]
        if(x1==x2):
            return 1 if x==x1 else 0
        else:
            return (x-x1) / (x2-x1)
    ind = bisect.bisect(grid, x)-1
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
    if x is float: return numpy.dot(bsplines(N, grid, x), ampl)
    return numpy.array([ numpy.dot(bsplines(N, grid, xx), ampl) for xx in x])

def bsplIntegrals(N, grid, power=0):
    """
    Compute the vector of integrals of B-spline basis functions, optionally multiplied by x^power.
    To obtain the integral of a function represented by amplitudes of its a B-spline expansion,
    multiply this vector by the array of amplitudes.
    """
    # use a 3-point Gauss-Legendre quadrature on each grid segment, sufficient for polynomials up to
    # degree 5 (e.g. a B-spline of degree 3 times x^2)
    glnode = 0.11270166537926
    grid1  = grid[1:]  * glnode + grid[:-1] * (1-glnode)
    grid2  = grid[:-1] * glnode + grid[1:]  * (1-glnode)
    grid3  = 0.5 * (grid[1:] + grid[:-1])
    result = numpy.zeros(len(grid)+N-1)
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
    xscaled = (xarr - center) / sigma
    norm    = (0.5/numpy.pi)**0.5 * gamma / sigma * numpy.exp(-0.5 * xscaled**2)
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

### --------------------- ###
### module initialization ###
try:
    # register two new colormaps for matplotlib: "sauron", "sauron_r" by Michele Cappellari & Eric Emsellem
    import matplotlib, matplotlib.pyplot
    f=[0.0, 0.17, 0.336, 0.414, 0.463, 0.502, 0.541, 0.590, 0.668, 0.834, 1.0]
    r=[0.01, 0.0, 0.4, 0.5, 0.3, 0.0, 0.7, 1.0, 1.0, 1.0, 0.9]
    g=[0.01, 0.0, 0.85,1.0, 1.0, 0.9, 1.0, 1.0, 0.85,0.0, 0.9]
    b=[0.01, 1.0, 1.0, 1.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9]
    matplotlib.pyplot.register_cmap(cmap=matplotlib.colors.LinearSegmentedColormap('sauron', \
        { 'red': zip(f,r,r), 'green': zip(f,g,g), 'blue': zip(f,b,b) } ))
    matplotlib.pyplot.register_cmap(cmap=matplotlib.colors.LinearSegmentedColormap('sauron_r', \
        { 'red': zip(f,r[::-1],r[::-1]), 'green': zip(f,g[::-1],g[::-1]), 'blue': zip(f,b[::-1],b[::-1]) } ))
except ImportError: pass
