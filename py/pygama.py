'''
This is a collection of various routines that complement the Python interface to the Agama library.
The __init__.py file in the root directory of Agama package imports both the C++ extension (agama.so)
and this python module (py/pygama.py) and merges them into a single namespace; hence one may write
>>> import agama                                # import both C++ and Python modules simultaneously
>>> pot = agama.GalpyPotential(type='Plummer')  # use classes and routines from this file...
>>> af  = agama.ActionFinder(pot)               # ...or those defined in the C++ library
'''
import numpy as _numpy, agama as _agama

### --------------------------------------------------------
### two routines for constructing non-uniformly spaced grids

def nonuniformGrid(nnodes, xmin, xmax=None):
    '''
    Create a grid with unequally spaced nodes:
    x[k] = (exp(Z k) - 1) / (exp(Z) - 1), i.e., coordinates of nodes increase
    nearly linearly at the beginning and then nearly exponentially towards the end;
    the value of Z is computed so the the 1st element is at xmin and last at xmax
    (0th element is always placed at 0).
    Arguments:
      nnodes:   the total number of grid points (>=3);
      xmin:     the location of the innermost nonzero node (>0);
      xmax:     the location of the last node (optional, if not provided, means uniform grid).
    Returns:    the array of grid nodes.
    '''
    if xmax is None or _numpy.isclose(xmax, (nnodes-1)*xmin):
        return _numpy.linspace(0, xmin*(nnodes-1), nnodes)
    if xmin<=0 or xmax<=xmin or nnodes<=2:
        raise ValueError('invalid parameters for nonuniformGrid')
    import scipy.optimize
    ratio = 1.*xmax/xmin
    def fnc(A):
        if abs(A)<1e-8: return nnodes-ratio + 0.5*A*nnodes*(nnodes-1)  # Taylor expansion
        else: return (_numpy.exp(A*(nnodes-1))-1) / (_numpy.exp(A)-1) - ratio
    A = scipy.optimize.brentq(fnc, _numpy.log(1-1./ratio), _numpy.log(ratio)/(nnodes-2))
    return xmin * (_numpy.exp(A * _numpy.linspace(0, nnodes-1, nnodes))-1) / (_numpy.exp(A)-1)


def symmetricGrid(nnodes, xmin, xmax=None):
    '''
    Create a possibly non-uniform grid, similar to 'nonuniformGrid()', but symmetric about origin.
    Arguments:
      nnodes:  the total number of grid points (>=4);
      xmin:    the width of the central grid segment;
      xmax:    the outer edge of the grid (endpoints are at +-xmax);
      if it is provided, the grid segments are gradually stretched as needed,
      otherwise this implies uniform segments and hence xmax = 0.5 * (nnodes-1) * xmin.
    Returns:   the array of grid nodes.
    '''
    if xmax is None:
        return _numpy.linspace(-1, 1, nnodes) * 0.5*xmin*(nnodes-1)
    if nnodes%2 == 1:
        grid = nonuniformGrid(nnodes//2+1, xmin, xmax)
        return _numpy.hstack((-grid[:0:-1], grid))
    else:
        grid = nonuniformGrid(nnodes, xmin*0.5, xmax)[1::2]
        return _numpy.hstack((-grid[::-1], grid))


### -------------------------------------------------------------------
### routines for coordinate transformation, projection and deprojection
### of ellipsoidally stratified profiles (e.g. a Multi-Gaussian expansion)

def makeRotationMatrix(alpha, beta, gamma):
    '''
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
    '''
    sinalpha, sinbeta, singamma = _numpy.sin([alpha, beta, gamma])
    cosalpha, cosbeta, cosgamma = _numpy.cos([alpha, beta, gamma])
    return _numpy.array([
        [ cosalpha * cosgamma - sinalpha * cosbeta * singamma,
          sinalpha * cosgamma + cosalpha * cosbeta * singamma,
          sinbeta  * singamma ],
        [-cosalpha * singamma - sinalpha * cosbeta * cosgamma,
         -sinalpha * singamma + cosalpha * cosbeta * cosgamma,
          sinbeta  * cosgamma ],
        [ sinalpha * sinbeta,
         -cosalpha * sinbeta,
          cosbeta ] ])


def makeCelestialRotationMatrix(lon, lat, psi):
    '''
    Construct the matrix for transforming celestial coordinates (longitude = RA, latitude = DEC)
    into another coordinate system 'centered' on the given sky position (lon, lat)
    and rotated CCW by angle psi.
    The rotation matrix has the same meaning as the one produced by 'makeRotationMatrix',
    only the choice of rotation angles is different. Angles must be in radians, not degrees!
    '''
    sina, sind, sinp = _numpy.sin([lon, lat, psi])
    cosa, cosd, cosp = _numpy.cos([lon, lat, psi])
    return _numpy.array([ [    cosa * cosd,         sina * cosd,                sind        ],
        [-sina * cosp - cosa * sind * sinp,  cosa * cosp - sina * sind * sinp,  cosd * sinp ],
        [ sina * sinp - cosa * sind * cosp, -cosa * sinp - sina * sind * cosp,  cosd * cosp ] ])


# the matrix for transforming sky coordinates in the International Celestial Reference System
# into Galactic coordinates, defined by the RA,DEC coordinates of the Galactic center
# and the tilt angle of the Galactic equator at that location.
fromICRStoGalactic = makeCelestialRotationMatrix(
    266.4049882865447 *_numpy.pi/180,
    -28.93617776179147*_numpy.pi/180,
    58.59866213832327 *_numpy.pi/180)

# the inverse transformation is defined by a transposed matrix
fromGalactictoICRS = fromICRStoGalactic.T


def transformCelestialCoords(rotationMatrix, lon, lat,
    pmlon=None, pmlat=None, sigmalon=None, sigmalat=None, sigmacorr=None):
    '''
    Transform celestial coordinates and optionally proper motions from one coordinate system
    (latitude, longitude - usually RA, DEC) into another one specified by a rotation matrix.
    Arguments:
      rotationMatrix - a matrix describing the transformation, which may be constructed by routines
      'makeRotationMatrix', 'makeCelestialRotationMatrix', etc. - any orthogonal 3x3 matrix is fine.
      The inverse transformation is given by a transposed matrix.
      lon, lat - coordinates in the input celestial frame (e.g., RA, DEC);
      either single numbers or arrays of equal length. Must be in radians!
      pmlon, pmlat - optionally the components of proper motion in the input frame.
      sigmalon, sigmalat, sigmacorr - optionally the components of proper motion dispersion tensor
      in the input frame (standard deviations in both coordinates, and the correlation coefficient
      between -1 and 1).
    Returns:
      coordinates in the rotated frame, and optionally the corresponding components of proper motion
      and its dispersion tensor, if these were provided. Longitude is normalized to the range [-pi,pi)
    '''
    sina, cosa, sind, cosd = _numpy.sin(lon), _numpy.cos(lon), _numpy.sin(lat), _numpy.cos(lat)
    x = cosa * cosd
    y = sina * cosd
    z = sind
    M = rotationMatrix  # shorthand
    sinB = M[2,0]*x + M[2,1]*y + M[2,2]*z
    B =  _numpy.arcsin(sinB)
    L = (_numpy.arctan2( M[1,0]*x + M[1,1]*y + M[1,2]*z, M[0,0]*x + M[0,1]*y + M[0,2]*z ) + _numpy.pi) % (2*_numpy.pi) - _numpy.pi
    if pmlon is None:
        return L, B
    T = _numpy.arctan2( M[2,0]*y - M[2,1]*x, M[2,2] - sinB * sind )
    sinT, cosT = _numpy.sin(T), _numpy.cos(T)
    if sigmalon is None:
        return L, B, cosT * pmlon + sinT * pmlat, -sinT * pmlon + cosT * pmlat
    sigmaL = (sigmalon**2 * cosT**2 + sigmalat**2 * sinT**2 + 2*sigmacorr * sigmalon * sigmalat * cosT * sinT)**0.5
    sigmaB = (sigmalon**2 * sinT**2 + sigmalat**2 * cosT**2 - 2*sigmacorr * sigmalon * sigmalat * cosT * sinT)**0.5
    corrLB = ((sigmalat**2 - sigmalon**2) * sinT*cosT + sigmacorr * sigmalon * sigmalat * (cosT**2 - sinT**2)) / sigmaL / sigmaB
    return L, B, cosT * pmlon + sinT * pmlat, -sinT * pmlon + cosT * pmlat, sigmaL, sigmaB, corrLB


def getCelestialCoords(x, y, z, vx=None, vy=None, vz=None):
    '''
    Convert Cartesian coordinates and velocities into longitude, latitude, distance,
    two components of proper motion, and line-of-sight velocity.
    The celestial coordinates are oriented so that the X axis points towards (lon,lat) = (0,0),
    the Y axis - towards (lon,lat) = (pi/2,0), and the Z axis - towards (lat=pi/2).
    This is a standard spherical coordinate system, except that latitude = pi/2-theta is used
    instead of the polar angle theta.
    Arguments:
      x, y, z -- Cartesian positions (in arbitrary units of length, e.g. kpc);
      vx, vy, vz -- (optional) corresponding Cartesian velocities (in arbitrary velocity units, e.g. km/s);
    Returns:
      longitude, latitude (e.g. RA, DEC or l, b) in radians (not degrees!),
      distance in the input units of length (e.g. kpc);
      optionally (if velocities were provided):
      components of proper motion  pmlon = d(lon)/dt * cos(lat), pmlat = d(lat)/dt
      in the input units of velocity over distance (e.g. km/s/kpc),
      and line-of-sight velocity  vlos = d(dist)/dt  in the input velocity units (e.g. km/s).
    '''
    R2   = x*x + y*y
    dist = (R2 + z*z)**0.5
    lon  = _numpy.arctan2(y, x)
    lat  = _numpy.arcsin (z / dist)
    if vx is None:
        return lon, lat, dist
    R    = R2**0.5
    vlos = (vx * x + vy * y + vz * z) / dist
    pmlon= (vy * x - vx * y) / R / dist
    pmlat= (vz - z / dist * vlos) / R
    return lon, lat, dist, pmlon, pmlat, vlos


def getCartesianCoords(lon, lat, dist, pmlon=None, pmlat=None, vlos=None):
    '''
    Convert celestial coordinates and velocities into Cartesian ones.
    Arguments:
      lon, lat - spherical coordinates (e.g. RA, DEC) in radians! lat=0 is the equator;
      dist - distance (in arbitrary length units);
      pmlon, pmlat - (optional) components of proper motion
      pmlon = d(lon)/dt * cos(lat), pmlat = d(lat)/dt
      (velocity over distance in the same units as vlos/dist, e.g. km/s/kpc);
      vlos - line-of-sight velocity d(dist)/dt (in arbitrary velocity units, e.g. km/s)
    Returns:
      x, y, z - Cartesian coordinates in the input length units;
      vx, vy, vz - corresponding velocities (if pmlon,pmlat,vlos were provided) in input velocity units.
    '''
    sina, cosa, sind, cosd = _numpy.sin(lon), _numpy.cos(lon), _numpy.sin(lat), _numpy.cos(lat)
    x = dist * cosd * cosa
    y = dist * cosd * sina
    z = dist * sind
    if pmlon is None:
        return x, y, z
    vx = -dist * pmlon * sina - dist * pmlat * cosa * sind + vlos * cosa * cosd
    vy =  dist * pmlon * cosa - dist * pmlat * sina * sind + vlos * sina * cosd
    vz =  dist * pmlat * cosd + vlos * sind
    return x, y, z, vx, vy, vz


def getGalacticFromGalactocentric(x, y, z, vx=None, vy=None, vz=None,
    galcen_distance=8.122, galcen_v_sun=(12.9, 245.6, 7.78), z_sun=0.0208):
    '''
    Convert position (and optionally velocity) in the Milky Way Galactocentric Cartesian coordinates
    to Galactic celestial coordinates. This is similar to getCelestialCoords but additionally
    accounts for solar velocity and position offsets, and only works with a specific choice of units.
    Arguments:
      x, y, z - Galactocentric coordinates (in kpc);
      vx, vy, vz - (optional) velocity in the same coordinates (in km/s);
      optional parameters of the reference frame:
      galcen_distance - distance from the observer to the Galactic center (in kpc);
      z_sun - offset from the XY plane (in kpc);
      the observer is located at x=-sqrt(galcen_distance**2-z_sun**2), y=0, z=z_sun;
      galcen_v_sun - three components of solar velocity (in km/s).
    Returns:
      l, b - Galactic longitude and latitude (in radians);
      heliocentric distance (in kpc);
      optionally (if velocities are provided):
      pm_l, pm_b - proper motions in longitude and latitude (in units of km/s/kpc = 0.211 mas/yr);
      vlos - line-of-sight velocity (in km/s).
    '''
    sintheta = z_sun/galcen_distance
    costheta = (1-sintheta**2)**0.5
    xp = x * costheta - z * sintheta + galcen_distance
    yp = y
    zp = x * sintheta + z * costheta
    if vx is None:
        return getCelestialCoords(xp, yp, zp)
    else:
        vxp = vx * costheta + vz * sintheta - galcen_v_sun[0]
        vyp = vy - galcen_v_sun[1]
        vzp = vz * costheta - vx * sintheta - galcen_v_sun[2]
        vxp = (vx - galcen_v_sun[0]) * costheta - (vz - galcen_v_sun[2]) * sintheta
        vyp = (vy - galcen_v_sun[1])
        vzp = (vx - galcen_v_sun[0]) * sintheta + (vz - galcen_v_sun[2]) * costheta
        return getCelestialCoords(xp, yp, zp, vxp, vyp, vzp)


def getGalactocentricFromGalactic(lon, lat, dist, pmlon=None, pmlat=None, vlos=None,
    galcen_distance=8.122, galcen_v_sun=(12.9, 245.6, 7.78), z_sun=0.0208):
    '''
    Convert Galactic celestial coordinates into Galactocentric Cartesian coordinates
    (optionally with velocities), taking into account Solar position and velocity;
    the inverse of getGalacticFromGalactocentric.
    Arguments:
      lon, lat - Galactic longitude (l) and latitude (b) (in radians);
      dist - heliocentric distance (in kpc);
      optionally:
      pmlon, pmlat - components of proper motion (in km/s/kpc = 0.211 mas/yr);
      vlos - heliocentric line-of-sight velocity (in km/s);
      optional parameters of the reference frame (same as in the inverse routine).
    Returns:
      x, y, z - Galactocentric Cartesian coordinates (in kpc);
      optionally (if pmlon,pmlat,vlos are provided):
      vx, vy, vz - corresponding velocities (in km/s).
    '''
    result   = getCartesianCoords(lon, lat, dist, pmlon, pmlat, vlos)
    sintheta = z_sun/galcen_distance
    costheta = (1-sintheta**2)**0.5
    x =  (result[0] - galcen_distance) * costheta + result[2] * sintheta
    y =   result[1]
    z = -(result[0] - galcen_distance) * sintheta + result[2] * costheta
    if pmlon is None:
        return x, y, z
    else:
        vx =  result[3] * costheta + result[5] * sintheta + galcen_v_sun[0]
        vy =  result[4] + galcen_v_sun[1]
        vz = -result[3] * sintheta + result[5] * costheta + galcen_v_sun[2]
        return x, y, z, vx, vy, vz


def getProjectedEllipse(Sx, Sy, Sz, alpha, beta, gamma):
    '''
    Project a triaxial ellipsoid with intrinsic axes Sx, Sy, Sz onto the image plane XY,
    whose orientation w.r.t. the intrinsic coordinate system is defined by three Euler angles
    alpha, beta, gamma (Y axis points up and X axis points left!).
    The projection is an ellipse in the image plane.
    Arguments:
      Sx, Sy, Sz - intrinsic axes of the ellipsoid;
      alpha, beta, gamma - Euler angles;
    Returns:
      SXp, SYp - major and minor axes of the projected ellipse;
      eta - position angle of the major axis of this ellipse in the image plane
      (measured counter-clockwise from the Y axis towards the X axis).
    '''
    pi=_numpy.pi; sin=_numpy.sin; cos=_numpy.cos
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
    D = ( (A-C)**2 + B**2 )**0.5
    # axes of the projected ellipsoid
    SXp = Sx * (2*f / (A+C-D))**0.5
    SYp = Sx * (2*f / (A+C+D))**0.5
    # position angle of the projected major axis ccw from projected z axis
    psi = 0.5 * _numpy.arctan2(B, A-C)
    # position angle of the projected major axis ccw from Y (north) direction in the image plane
    eta = psi + gamma
    return max(Sy, min(Sx, SXp)), max(Sz, min(Sy, SYp)), eta % pi % pi


def getIntrinsicShape(SXp, SYp, eta, alpha, beta, gamma):
    '''
    Deproject the ellipse in the image plane into a triaxial ellipsoid,
    for an assumed orientation of the intrinsic coord.sys. of that ellipsoid.
    Arguments:
      SXp and SYp - lengths of the major and minor axes of the ellipse;
      eta - position angle of the major axis of the ellipse, measured counter-clockwise
      from the Y (vertical) axis on the image plane (towards the X axis, which points left);
      alpha, beta, gamma - assumed angles of rotation of the coordinate frame.
    Returns:
      three intrinsic axes (Sx,Sy,Sz) of the triaxial ellipsoid.
    Throws an exception if the deprojection is impossible for these angles.
    The deprojection is not unique in the following special cases:
    - if beta==0 or beta==pi (face-on view down the z axis) - cannot determine q=Sz/Sx, assume q=p;
    - if eta-gamma==pi/2 (angle between projected z axis and major axis) - cannot determine p=Sy/Sx, assume p=1;
    '''
    sin=_numpy.sin; cos=_numpy.cos; tan=_numpy.tan
    Q = SYp / SXp
    if Q>1:
        raise ValueError('Projected axis ratio must be <=1')
    psi = eta - gamma
    if abs(sin(beta)) < 1e-12:  # face-on view (inclination is zero) - assume prolate axisymmetric shape
        return (SXp, SYp, SYp)  # q=Sz/Sx is not constrained, assume it to be equal to p=Sy/Sx
    if abs(cos(psi)) < 1e-12:   # no misalignment - assume oblate axisymmetric shape
        if 1-Q**2 > sin(beta)**2:
            raise ValueError('Deprojection is impossible for the given inclination')
        return (SXp, SXp, SXp * (1 - (1-Q**2) / sin(beta)**2)**0.5)
    num = (1-Q*Q) * (cos(psi) * cos(beta) / tan(alpha) + sin(psi))
    den = (num * cos(psi) - cos(beta) / tan(alpha))
    q   = (1 - num * (cos(psi) - sin(psi) * cos(beta) / tan(alpha)) / sin(beta)**2 / den)**0.5
    p   = (1 - (1-Q*Q) * sin(psi) * cos(psi) / sin(alpha)**2 / den)**0.5
    f   = (p*q * sin(beta) * sin(alpha))**2 + (q * sin(beta) * cos(alpha))**2 + (p * cos(beta))**2
    Sx  = SXp * (Q / f**0.5)**0.5
    if _numpy.isnan(q+p+Sx):
        raise ValueError('Deprojection is impossible for the given orientation')
    return (Sx, Sx*p, Sx*q)


def getViewingAngles(SXp, SYp, eta, Sx, Sy, Sz):
    '''
    Find the viewing angles that would project a triaxial ellipsoid with axes Sx >= Sy >= Sz
    into the ellipse in the image plane with axes SXp >= SYp and position angle of the major axis eta.
    SXp and SYp are the lengths of the major and minor axes of the ellipse in the image plane,
    eta is the position angle of the major axis of this ellipse, measured counter-clockwise
    from the Y (vertical) axis on the image plane (towards the X axis, which points left).
    Sx, Sy, Sz are the assumed intrinsic axis lengths.
    return: a 4x3 array, where each row represents a triplet of viewing angles alpha, beta, gamma
    (there are four possible deprojections in general, although some of them may be identical),
    or throw an exception if the deprojection is impossible for the given parameters.
    '''
    pi = _numpy.pi
    p  = Sy  / Sx
    q  = Sz  / Sx
    u  = SXp / Sx
    v  = SYp / Sx
    if not (0<=q and q<=v and v<=p and p<=u and u<=1):
        raise ValueError('Projected and assumed axis lengths are inconsistent (should be '+
        '0 <= Sz=%.16g <= SYp=%.16g <= Sy=%.16g <= SXp=%.16g <= Sx=%.16g)' % (Sz, SYp, Sy, SXp, Sx))
    if p==q:
        if p==1:
            if abs(u-1) + abs(v-1) > 1e-12:
                raise ValueError('All axes must be equal in the spherical case')
            beta = 0  # assume face-on orientation for a spherical system
        else:
            if abs(v-p) > 1e-12:
                raise ValueError('Prolate axisymmetric system must have SYp = Sy')
            beta = _numpy.arcsin( ( (1-u*u) / (1-q*q) )**0.5 )
            # this case is not yet supported ?
    else:
        beta  = _numpy.arccos( ( (u*u-q*q) / (1-q*q) * (v*v-q*q) / (p*p-q*q) )**0.5 )
    if 1-u < 1e-12:   # u=1 <=> SXp=Sx - assume an oblate axisymmetric system
        alpha = 0
    elif u-p < 1e-12 or p-v < 1e-12:
        alpha = pi/2
    else:
        alpha = _numpy.arctan( ( (1-v*v) * (1-u*u) / (p*p-v*v) / (u*u-p*p) * (p*p-q*q) / (1-q*q) )**0.5 )
    if p-v < 1e-12:
        psi   = 0
    elif u-p < 1e-12 or v-q < 1e-12 or 1-u < 1e-12:
        psi   = pi/2
    else:
        psi   = _numpy.arctan( ( (1-v*v) / (1-u*u) * (p*p-v*v) / (v*v-q*q) * (u*u-q*q) / (u*u-p*p) )**0.5 )
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


### ------------------------------------------------------------------------------
### routines for representing a function specified in terms of its coefficients of
### B-spline or Gauss-Hermite expansions

def _bsplines(degree, grid, x):
    '''
    Compute B-splines of given degree over the given grid, for the input point x
    '''
    from bisect import bisect
    npoints = len(grid)
    result = _numpy.zeros(npoints+degree-1)
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
    B = [0.] * (degree+1)
    B[degree] = 1.
    for l in range(degree):
        Bip1=0
        for j in range(ind,ind-l-2,-1):
            i  = j-ind+degree
            Bi = B[i] * linInt(x, grid, j, j+l+1) + Bip1 * linInt(x, grid, j+l+2, j+1)
            Bip1 = B[i]
            B[i] = Bi
    for i in range(degree+1):
        result[i+ind]=B[i]
    return result


def _bsplineGaussLegendre(grid):
    '''
    return nodes and weights of a Gauss-Legendre quadrature for integration of piecewise polynomials
    (e.g. products of B-splines or monomials). Use a 4-point rule, sufficient for polynomials up to
    degree 7 (e.g. a B-spline of degree 3 times x^4, or a product of two B-splines of degrees <=3)
    '''
    nodes = _numpy.hstack((
        grid[1:] * 0.0694318442029737 + grid[:-1] * 0.9305681557970263,
        grid[1:] * 0.3300094782075719 + grid[:-1] * 0.6699905217924281,
        grid[1:] * 0.6699905217924281 + grid[:-1] * 0.3300094782075719,
        grid[1:] * 0.9305681557970263 + grid[:-1] * 0.0694318442029737 ))
    weights = ( (grid[1:] - grid[:-1]) *
        _numpy.hstack((0.173927422568727, 0.326072577431273, 0.326072577431273, 0.173927422568727))[:,None]
        ).reshape(-1)
    return nodes, weights


def bsplineInterp(degree, grid, ampl, x):
    '''
    Compute interpolated values of B-spline expansion of given degree over the given grid and ampitudes,
    for the input point or an array of points x
    '''
    if isinstance(x, (int,float,_numpy.float)):
        return            _numpy.dot(_bsplines(degree, grid, x), ampl)
    return _numpy.array([ _numpy.dot(_bsplines(degree, grid, X), ampl) for X in x])


def bsplineIntegrals(degree, grid, power=0):
    '''
    Compute the vector of integrals of B-spline basis functions, optionally multiplied by x^power.
    To obtain the integral of a function represented by amplitudes of its a B-spline expansion,
    multiply this vector by the array of amplitudes and sum the result
    '''
    return _numpy.sum(_numpy.vstack([ _bsplines(degree, grid, x) * x**power * w
        for x, w in zip(*_bsplineGaussLegendre(grid)) ]), axis=0)


def bsplineMatrix(degree1, grid1, degree2=None, grid2=None):
    '''
    Compute the matrix of inner products of B-spline functions:
    M_{ij} = \int_{xmin}^{xmax} B1_i(x) B2_j(x) dx,
    where B1(x) and B2(x) are two (possibly identical) basis sets of B-splines of degree(s),
    defined by the grid nodes (or two separate grids).
    If degree2 and grid2 are not provided, this means that they are identical to degree1, grid1.
    '''
    # since B-splines are piecewise-polynomial function, the product of two B-splines
    # is a polynomial of degree degree1+degree2, which can be integrated exactly using
    # a Gauss-Legendre quadrature with (degree1+degree2)//2+1;
    # as we expect both degree1,degree2 to be <= 3, just use a 4-point quadrature for all cases.
    if (degree2 is None and not grid2 is None) or (not degree2 is None and grid2 is None):
        raise ValueError('Must provide both degree2 and grid2, or neither')
    if degree2 is None:
        grid=grid1
    else:
        grid=_numpy.unique(_numpy.hstack((grid1, grid2)))   # sorted array of all nodes of both grids
    # integration is carried over each segment of the combined grid separately;
    # set up nodes and weights of a suitable quadrature rule for all segments
    nodes, weights = _bsplineGaussLegendre(grid)
    # number of basis functions in each set
    M1 = len(grid1)+degree1-1
    M2 = len(grid2)+degree2-1 if not degree2 is None else M1
    # result matrix
    result = _numpy.zeros((M1, M2))
    # iterate over all points of the integration grid,
    # compute the values of all B-spline basis functions in both sets for each point,
    # and add the outer product of these two vectors, multiplied by GL weight, to the result.
    # this is not optimized, because most of the basis functions will be zero at any point
    # (in fact, only N+1 of them are nonzero), but is sufficient for our purposes
    for i in range(len(nodes)):
        B1 = _bsplines(degree1, grid1, nodes[i])
        B2 = _bsplines(degree2, grid2, nodes[i]) if not degree2 is None else B1
        result += weights[i] * _numpy.outer(B1, B2)
    return result


def ghInterp(ampl, center, width, coefs, x):
    '''
    Compute the function specified by a Gauss-Hermite expansion with the given
    amplitude, center and width, and a list of N>=2 GH coefficients, at point(s) xarr.
    '''
    xscaled = (x[:,None] - center) / width
    norm    = (0.5/_numpy.pi)**0.5 * ampl / width * _numpy.exp(-0.5 * xscaled**2)
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
    return _numpy.squeeze(result * norm)


def sampleOrbitLibrary(nbody, orbits, weights):
    '''
    Construct an N-body snapshot from an orbit library
    Arguments:
      nbody:    the required number of particles in the output snapshot.
      orbits:   an array of trajectories returned by the `orbit()` routine.
      weights:  an array of orbit weights, returned by the `solveOpt()` routine.
    Returns: a tuple of two elements: the flag indicating success or failure, and the result.
    In case of success, the result is a tuple of two arrays: particle coordinates/velocities
    (2d Nx6 array) and particle masses (1d array of length N).
    In case of failure (when some of the orbits, usually with high weights, had fewer points
    recorded from their trajectories during orbit integration than is needed to represent them
    in the N-body snapshot), the result is a different tuple of two arrays:
    list of orbit indices which did not have enough trajectory samples (length is anywhere
    from 1 to N), and corresponding required numbers of samples for each orbit from this list.
    '''
    nbody = int(nbody)
    if nbody <= 0:
        raise ValueError("Argument 'nbody' must be a positive integer")

    # check that orbit weights are correct and their sum is positive
    numOrbits = len(weights)
    if numOrbits <= 0:
        raise ValueError("Argument 'weights' must be a non-empty array of floats")
    cumulMass = _numpy.cumsum(weights)  # sum of orbit weights up to and including each orbit in the list
    totalMass = cumulMass[-1]
    if not (totalMass>0):
        raise ValueError("The sum of weights must be positive")
    if not (_numpy.min(weights) >= 0):
        raise ValueError("Weights must be non-negative")

    # check that the trajectories are provided: it should be an array with PyObjects as elements
    # (they must be arrays themselves, which will be checked later); the shape of this array should be
    # either numOrbits or numOrbits x 2 (the latter variant is returned by the `orbit()` routine).
    if len(orbits.shape) == 2:
        orbits = orbits[:,1]  # take only the trajectories (1st column), not timestamps (0th column)
    if orbits.shape != (numOrbits,):
        raise ValueError("'orbits' must be an array of numpy arrays with the same length as 'weights'")
    orbitLengths = _numpy.array([len(o) for o in orbits])
    # index of first output point for each orbit
    outPointIndex = _numpy.hstack((0, cumulMass / totalMass * nbody )).astype(int)
    # number of output points to sample from each orbit
    pointsToSample = outPointIndex[1:] - outPointIndex[:-1]
    # indices of orbits whose length of recorded trajectory is less than the required number of samples
    badOrbitIndices = _numpy.where(pointsToSample > orbitLengths)[0]
    # if this list is not empty, the procedure failed and we return the indices of orbits for reintegration
    if len(badOrbitIndices) > 0:
        return False, (badOrbitIndices, pointsToSample[badOrbitIndices])
    # otherwise scan the array of orbits and sample appropriate number of points from each trajectory
    posvel = _numpy.zeros((nbody, 6), dtype=orbits[0].dtype)
    for orb in range(numOrbits):
        # copy a random [non-repeating] selection of points from this orbit into the output array
        posvel[outPointIndex[orb] : outPointIndex[orb+1]] = \
            orbits[orb][_numpy.random.choice(orbitLengths[orb], pointsToSample[orb], replace=False)]
    # return a success flag and a tuple of posvel, mass
    return True, (posvel, _numpy.ones(nbody, dtype=orbits[0].dtype) * totalMass / nbody)


### -------------------
### interface for galpy

class GalpyPotential(_agama.Potential):
    '''
    Class that implements a Galpy interface to Agama potentials.
    It can be used as a regular galpy potential class, although for the orbit integration
    or action computation, the native Agama counterparts are preferred.
    '''
    def __init__(self,*args,**kwargs):
        '''
        Initialize a potential from parameters provided in an INI file
        or as named arguments to the constructor.
        Arguments are the same as for regular agama.Potential (see below);
        an extra keyword 'normalize=...' has the same meaning as in Galpy:
        if True, normalize such that vc(1.,0.)=1., or,
        if given as a number, such that the force is this fraction of the force
        necessary to make vc(1.,0.)=1.
        '''
        # importing galpy takes a lot of time (when first called in a script), so we only perform this
        # when the constructor of this class is called, and add the inheritance from
        # galpy.potential.Potential at runtime.
        from galpy.potential import Potential
        GalpyPotential.__bases__ = (Potential, _agama.Potential)
        Potential.__init__(self, amp=1.)
        normalize=False
        for key, value in kwargs.items():
            if key=='normalize':
                normalize=value
                del kwargs[key]
        _agama.Potential.__init__(self, *args, **kwargs)   # construct a regular Agama potential
        if normalize or (isinstance(normalize,(int,float)) and not isinstance(normalize,bool)):
            self.normalize(normalize)
        self.hasC= False
        self.hasC_dxdv=False

    __init__.__doc__ += '\n' + _agama.Potential.__doc__  # extend the docstring of the constructor

    def _coord(self,R,z,phi):
        '''convert input cylindrical coordinates to a Nx3 array in cartesian coords'''
        if phi is None: phi=0.
        return _numpy.array((R*_numpy.cos(phi), R*_numpy.sin(phi), z)).T

    def _evaluate(self,R,z,phi=0.,t=0.):
        '''evaluate the potential at cylindrical coordinates R,z,phi'''
        return self.potential(self._coord(R,z,phi))

    def _Rforce(self,R,z,phi=0.,t=0.):
        '''evaluate the radial force for this potential: -dPhi/dR'''
        coord=self._coord(R,z,phi)
        force=_numpy.array(self.force(coord))
        return (force.T[0]*coord.T[0] + force.T[1]*coord.T[1]) / R

    def _zforce(self,R,z,phi=0.,t=0.):
        '''evaluate the vertical force for this potential: -dPhi/dz'''
        return _numpy.array(self.force(self._coord(R,z,phi))).T[2]

    def _phiforce(self,R,z,phi=0.,t=0.):
        '''evaluate the azimuthal force for this potential: -dPhi/dphi'''
        coord=self._coord(R,z,phi)
        force=_numpy.array(self.force(coord))
        return force.T[1]*coord.T[0] - force.T[0]*coord.T[1]

    def _dens(self,R,z,phi=0.,t=0.):
        '''evaluate the density for this potential'''
        return self.density(self._coord(R,z,phi))

    def _2deriv(self,R,z,phi):
        '''evaluate the potential derivatives in cartesian coordinates'''
        coord=self._coord(R,z,phi)
        force,deriv=self.forceDeriv(coord)
        return coord.T, _numpy.array(force).T, _numpy.array(deriv).T

    def _R2deriv(self,R,z,phi=0.,t=0.):
        '''evaluate the second radial derivative for this potential: d2Phi / dR^2'''
        coord,force,deriv=self._2deriv(R,z,phi)
        return -(deriv[0]*coord[0]**2 + deriv[1]*coord[1]**2 +
               2*deriv[3]*coord[0]*coord[1]) / R**2

    def _z2deriv(self,R,z,phi=0.,t=0.):
        '''evaluate the second vertical derivative for this potential: d2Phi / dz^2'''
        return -_numpy.array(self.forceDeriv(self._coord(R,z,phi))[1]).T[2]

    def _phi2deriv(self,R,z,phi=0.,t=0.):
        '''evaluate the second azimuthal derivative for this potential: d2Phi / dphi^2'''
        coord,force,deriv=self._2deriv(R,z,phi)
        return -(deriv[0]*coord[1]**2 + deriv[1]*coord[0]**2 -
               2*deriv[3]*coord[0]*coord[1] - force[0]*coord[0] - force[1]*coord[1])

    def _Rzderiv(self,R,z,phi=0.,t=0.):
        '''evaluate the mixed R,z derivative for this potential: d2Phi / dR dz'''
        coord,force,deriv=self._2deriv(R,z,phi)
        return -(deriv[5]*coord[0] + deriv[4]*coord[1]) / R

    def _Rphideriv(self,R,z,phi=0.,t=0.):
        '''evaluate the mixed R,phi derivative for this potential: d2Phi / dR dphi'''
        coord,force,deriv=self._2deriv(R,z,phi)
        return -((deriv[1]-deriv[0])*coord[1]*coord[0] + deriv[3]*(coord[0]**2-coord[1]**2)
            - force[0]*coord[1] + force[1]*coord[0]) / R

    def _zphideriv(self,R,z,phi=0.,t=0.):
        '''evaluate the mixed z,phi derivative for this potential: d2Phi / dz dphi'''
        coord,force,deriv=self._2deriv(R,z,phi)
        return -(deriv[4]*coord[0] - deriv[5]*coord[1])
