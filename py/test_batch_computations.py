#!/usr/bin/python

'''
This script tests various classes and routines from the Agama Python interface,
focusing not on numerical accuracy, but rather on the correctness of the usage
with different combinations of input parameters (single or multiple points
evaluated in a single call) and flags specifying the variants of output.
This is mostly a technical consistency check, but it also provides many examples
of usage scenarios and calling conventions.
'''
import numpy
# if the module has been installed to the globally known directory, just import it
try: import agama
except:  # otherwise load the shared library from the parent folder
    import sys
    sys.path += ['../']
    import agama

allok = True

# test if the given condition (a string to be executed) is true
def testCond(condition):
    global allok
    try:
        result = eval(condition)
        if bool(result) is True:
            return  # just as planned
        print('%s \033[1;31m is %s\033[0m' % (condition, result))
        allok = False
    except Exception as ex:
        print('%s \033[1;31m failed:\033[0m %s' % (condition, ex))
        allok = False

# test if the given piece of code (a string to be executed) produces an exception
def testFail(statement):
    global allok
    try:
        exec(statement)
        print('%s \033[1;31m did not fail\033[0m' % statement)
        allok = False
    except Exception: pass  # just as planned

# aux functions for determining the nature of objects
def isFloat(obj):
    return isinstance(obj, float)

def isTuple(obj, length):
    return isinstance(obj, tuple) and len(obj) == length

def isArray(obj, shape):
    return isinstance(obj, numpy.ndarray) and obj.shape == shape


# set up some non-trivial dimensional units
agama.setUnits(length=2, velocity=3, mass=4e6)

dens = agama.Density(type='plummer')
pots = agama.Potential(type='dehnen', gamma=0)  # spherical potential
potf = agama.Potential(type='plummer', q=0.75)  # flattened potential
pott = agama.Potential(type='plummer', p=0.75, q=0.5)  # triaxial potential
actf = agama.ActionFinder(potf)
actm = agama.ActionMapper(potf, [1,1,1])
df0  = agama.DistributionFunction(type='quasispherical', density=pots, potential=pots, r_a=2.0)
df1  = agama.DistributionFunction(type='quasispherical', density=dens, potential=pots, beta0=-0.2)
df2  = agama.DistributionFunction(df1, df0)     # composite DF with two components
gms1 = agama.GalaxyModel(pots, df1)  # simple DF, spherical potential
gms2 = agama.GalaxyModel(pots, df2)  # composite DF (2 components), spherical
gmf1 = agama.GalaxyModel(potf, df1)  # simple, flattened
Phi0 = pots.potential(0,0,0)         # value of the potential at origin

### test the shapes of output arrays for various inputs ###
# Density class methods
testCond('isFloat(dens.density( 1,0,0 ))')  # single point (3 coordinates) => output is a scalar value
testCond('isFloat(dens.density((1,0,0)))')  # equivalent to previous
testCond('isFloat(dens.density([1,0,0]))')  # again equivalent
testCond('isArray(dens.density([[1,0,0]]), (1,))')  # not equivalent: an array of 1 point is not the same as 1 point, but instead produces an 1d array of length 1
testCond('isArray(dens.density( [1,0,0],[2,0,0] ), (2,))')  # more generally, N input points => 1d array of length N (here N=2)
testCond('isArray(dens.density([[1,0,0],[2,0,0]]), (2,))')  # equivalent: an array of 2 points is the same as a tuple of 2 points
testFail('dens.density([[[1,0,0],[2,0,0]]])')  # too many dimensions
testFail('dens.density(1,0)')                  # too few numbers
testFail('dens.density(1,0,0,0)')              # too many numbers
testFail('dens.density([1,0,0],0')             # not an array at all

testCond('isFloat(dens.surfaceDensity( 1,0 ))')             # may give either two numbers...
testCond('isFloat(dens.surfaceDensity((1,0)))')             # ...or an array (tuple, list) of two numbers...
testCond('isFloat(dens.surfaceDensity([1,0]))')             # ...for a single input point
testCond('dens.surfaceDensity([1,0],0,0,0) > 0')            # first argument is an array, remaining are angles
testFail('dens.surfaceDensity( 1,0, 0,0,0) > 0')            # in this case, cannot replace the array with just two numbers
testCond('dens.surfaceDensity( 1,0, alpha=0,beta=0) > 0')   # but can do when using named arguments for angles
testCond('dens.surfaceDensity([1,0],gamma=0,beta=0) > 0')   # or when using an array for the point plus named args for angles
testCond('dens.surfaceDensity([[1,0]],0,0).shape == (1,)')  # if the input is a 2d array (even with one row), output is a 1d array
testCond('dens.surfaceDensity([[1,0],[2,0]]).shape == (2,)')  # N input points (Nx2 array) => 1d array of length N

# Potential class methods
testCond('isFloat(pots.potential(1,0,0))')
testCond('isArray(pots.force(  1,0,0  ), (3,))')
testCond('isArray(pots.force( [1,0,0] ), (3,))')   # equivalent
testCond('isArray(pots.force([[1,0,0]]), (1,3))')  # not equivalent: an array of 1 point is not the same as 1 point
testCond('isArray(pots.force( [1,0,0],[2,0,0] ), (2,3))')
testCond('isArray(pots.force([[1,0,0],[2,0,0]]), (2,3))')  # equivalent
testCond('isTuple(pots.forceDeriv(1,0,0), 2)')
testCond('isArray(pots.forceDeriv(1,0,0)[0], (3,))')
testCond('isArray(pots.forceDeriv(1,0,0)[1], (6,))')

testFail('pots.Rcirc(2)')  # need a named argument: E=... or L=...
testCond('pots.Rcirc(L=0) == 0')  # radius is zero for L==0,
testCond('pots.Rcirc(L= 2) > 0')  # and positive for either L>0
testCond('pots.Rcirc(L=-2) > 0')  # or L<0
testFail('bool(pots.Rcirc(L=[2,-2]) > 0)')  # output is an array, not a single value
testCond('all (pots.Rcirc(L=[2,-2]) > 0)')  # this way it works
testCond('pots.Rcirc(E=Phi0/2) > 0')                     # radius is positive for Phi(0) < E < 0,
testCond('pots.Rcirc(E=0) == numpy.inf')                 # infinite for E == 0,
testCond('all(numpy.isnan(pots.Rcirc(E=[Phi0*2, 1])))')  # and NAN for E < Phi(0) or E > 0
testCond('numpy.isnan(pots.Rcirc(E=numpy.nan)+pots.Rcirc(L=numpy.nan))')  # NAN in => NAN out, no errors
testCond('isArray(pots.Rcirc(E=[-1,1,numpy.nan]), (3,))')  # output shape is the same as input regardless of values
testFail('pots.Rcirc(E=[[-1,1]])')  # too many dimensions

testCond('isFloat(pots.Tcirc(1))')                  # a single input value is treated as E and produces a single output value
testCond('isArray(pots.Tcirc([1,2,3,4,5]), (5,))')  # an input array - as an array of E, producing an output array
testCond('isFloat(pots.Tcirc([1,2,3,4,5,6]))')      # but an array with 6 elements - as x,v (a single point)
testCond('isArray(pots.Tcirc([[1,2,3,4,5,6],[7,8,9,10,11,numpy.nan]]), (2,))')  # and a 2d array Nx6 - as N points x,v
testFail('pots.Tcirc(1,2,3,4,5)')  # this function expects a single argument (possibly an array), not a tuple

testCond('abs(pots.Rmax(pots.potential(1.5,0,0))-1.5) < 1e-10')  # Rmax(E=Phi(r)) = r  with high accuracy,
testCond('pots.Rmax(0) == numpy.inf')                            # infinite for E == 0,
testCond('all(numpy.isnan(pots.Rmax([Phi0*2, 1])))')             # and NAN for E < Phi(0) or E > 0
testCond('isArray(pots.Rmax([0,1,numpy.nan]), (3,))')   # output array has the same length as input array / list
testFail('pots.Rmax( 0,1,2 )')                          # this function also expects only one argument, not a tuple

testCond('isArray(pots.Rperiapo( -1,1 ), (2,))')        # two numbers are a single point (E,L) => array of two values (Rperi,Rapo)
testCond('isArray(pots.Rperiapo([-1,1]), (2,))')        # same two numbers as a 1d array of length 2, same output
testCond('isArray(pots.Rperiapo([1,2,3,4,5,6]), (2,))') # 1d array of length 6 is a single point (x,v), same output
testFail('isArray(pots.Rperiapo( 1,2,3,4,5,6 ), (2,))') # but not when given as 6 separate arguments, this is not allowed
testCond('isArray(pots.Rperiapo(numpy.random.random(size=(5,2))), (5,2))')  # a 2d array of (E,L) values with shape Nx2 => Nx2
testCond('isArray(pots.Rperiapo(numpy.random.random(size=(5,6))), (5,2))')  # or a 2d array of (x,v) with shape Nx6 => Nx2
testFail('isArray(pots.Rperiapo(numpy.random.random(size=(5,3))), (5,2))')  # invalid shape of input
testCond('numpy.all(numpy.isnan(pots.Rperiapo([[Phi0*2, 1],[0, 1]])))')     # bad (but valid shape) input => NAN
testFail('pott.Rperiapo(1,1)')  # this method should only work for axisymmetric potentials

# ActionFinder class methods
testCond('isArray(actf([1,2,3,4,5,6]), (3,))')  # 1d array of length 6 is a single point (x,v) => output array of 3 actions
testFail('actf( 1,2,3,4,5,6 )')                 # must be an array and not just 6 numbers
testCond('isTuple(actf([1,2,3,4,5,6], angles=True), 3)')        # extra argument angles=True produces a tuple of length 3
testCond('isArray(actf([1,2,3,4,5,6], angles=True)[0], (3,))')  # with actions, angles and frequencies,
testCond('isArray(actf([1,2,3,4,5,6], angles=True)[1], (3,))')  # each of them being an array of length 3 (for a single point)
testCond('isArray(actf([1,2,3,4,5,6], angles=True)[2], (3,))')
testCond('isArray(actf(numpy.random.random(size=(5,6))), (5,3))')  # when the input has N points (Nx6 array), output is Nx3,
testCond('isArray(actf(numpy.random.random(size=(5,6)), angles=True)[2], (5,3))')  # or three arrays of shape Nx3 if angles=True
testFail('agama.ActionFinder(pott)')  # should not work for non-axisymmetric potentials

# ActionMapper class methods
testCond('isArray(actm(1,2,3), (6,))')             # action mapper takes 3 angles as input, and produces 1d array of length 6 (x,v)
testCond('isArray(actm([1,2,3],[4,5,6]), (2,6))')  # N input points => 2d output array (Nx6)
testCond('isArray(actm(numpy.random.random(size=(100,3))), (100,6))')
testFail('agama.ActionMapper(pott, [1,2,3])')     # should not work for non-axisymmetric potentials

# standalone action finder routine
testCond('isArray(agama.actions(pots, [1,2,3,4,5,6]), (3,))')   # input: potential, one or more points (x,v) => 3 actions per point
testCond('isArray(agama.actions(potf, numpy.random.random(size=(42,6)), angles=True, fd=0.5)[2], (42,3))')  # optionally angles and frequencies of the same shape
testFail('agama.actions(pott, [1,2,3,4,5,6])')    # should not work for non-axisymmetric potentials 

# DistributionFunction class methods
testCond('isFloat(df1( 1,2,3 ))')          # input: 3 actions => output is a scalar value
testCond('isFloat(df2([1,2,3]))')          # equivalent input
testCond('isArray(df1([[1,2,3]]), (1,))')  # input: Nx3 array (here N=1) => output is a 1d array of length N
testCond('isArray(df2(numpy.random.random(size=(17,3))), (17,))')  # same but N>1
testCond('len(df2)==2 and df2[0](1,2,3) == df1(1,2,3)')  # df2 is a composite DF with two components, and the first one is df1

# GalaxyModel class members
testCond('isTuple(gms1.moments([1,2,3], dens=True, vel=True, vel2=True), 3)')    # three outputs per point, as requested
testCond('isTuple(gms2.moments([1,2,3], dens=True, vel=False,vel2=True), 2)')    # only two requested here
testCond('isFloat(gmf1.moments([1,2,3], dens=True, vel=False,vel2=False))')      # only one output (density), and it is a scalar value
testCond('isFloat(gmf1.moments([1,2,3], dens=False,vel=True, vel2=False))')      # same for velocity
testCond('isArray(gms1.moments([1,2,3], dens=False,vel=False,vel2=True), (6,))') # one output (elements of dispersion tensor) - 1d array of length 6
testCond('isArray(gms1.moments([[1,2,3]],dens=True,vel=False,vel2=False),(1,))') # input is an array of points with length 1, and so is the output
testCond('isArray(gms2.moments([[1,2,3],[4,5,6],[7,8,9]], dens=True, vel=False, vel2=True)[1], (3,6))')  # N input points => dispersion tensor has shape Nx6
testCond('isArray(gms2.moments([[1,2,3],[4,5,6],[7,8,9]], dens=False,vel=False, vel2=True, separate=True), (3,2,6))')  # DF with C components treated separately => NxCx6
testCond('isArray(gms1.moments([[1,2,3],[4,5,6],[7,8,9]], dens=False,vel=False, vel2=True, separate=True), (3,1,6))')  # same here, but C=1 - output array is still 3d
testCond('isArray(gms1.moments([1,2,3], dens=True, vel=False,vel2=False, separate=True), (1,))')  # single point, single output, but composite DF with C=1 => 1d array of length C
# check that separately computed moments of a composite DF (df2) agree with the moments of its 0th component (df1)
testCond('numpy.isclose(gms1.moments([1,2,3],dens=1,vel=0,vel2=0), gms2.moments([1,2,3],dens=1,vel=0,vel2=0,separate=True)[0], 1e-4)')

testCond('isTuple(gms1.projectedMoments(1), 3)')  # input: one number per point => output: a tuple of three numbers
testCond('isArray(gms1.projectedMoments([1,2,3,4])[0], (4,))')  # input: array of N points => output: a tuple of three arrays of length N
testCond('isArray(gms2.projectedMoments([1,2,3,4], separate=True)[1], (4,2))')  # when separate=True, output arrays have one extra dimension of size C

testCond('isFloat(gms1.projectedDF([1,2,0]))')  # input: a triplet of numbers per point (x,y,v_z) => output: one number per point
testCond('isArray(gms2.projectedDF([[1,2,3],[4,5,6],[7,8,9]], separate=True), (3,2))')  # when separate=True, output for N points and C components is a 2d array (NxC)

testCond('isTuple(gms1.vdf([1,2,3]), 3)')             # input: one point in 3d(xyz) => output 3 spline objects
testCond('isArray(gms1.vdf([[1,2],[3,4]])[0], (2,))') # input: two points in 2d(xy) => output a tuple of 3 arrays with splines
testCond('isTuple(gms1.vdf([1,2,3], dens=True), 4)')  # if a flag dens=True is set, output is a tuple of length 4: 3 splines and density
testCond('isArray(gms1.vdf([1,2,3])[0]([1,2,3,4,5]), (5,))')  # spline objects are functions that convert input arrays to output arrays of the same length
testCond('isArray(gms2.vdf([[1,2,3],[4,5,6],[7,8,9]], separate=True)[0], (3,2))')  # when invoked for N points with separate=True, output arrays are NxC
# when vdf() is invoked with dens=True, the computed 3d density should agree with the one produced by moments()
testCond('numpy.isclose(gms1.vdf([1,2,3], dens=True)[3], gms1.moments([1,2,3])[0], 1e-3)')
# and similarly for the case of 2d input (projected VDF): projected density should agree with projectedMoments()
testCond('numpy.isclose(gms1.vdf([1,0], dens=True)[3], gms1.projectedMoments([1])[0], 1e-3)')
# finally check that the projected VDF in v_z agrees with the output of projectedDF() after multiplying the former by the projected density
vz = numpy.linspace(0, 0.5*(-2*pots.potential(1,0,0))**0.5, 4)  # grid in v_z from 0 to half the escape velocity
xyvz = numpy.column_stack((vz*0+1, vz*0, vz))  # input points for projectedDF()
testCond('numpy.allclose(gms1.vdf([1,0])[1](vz) * gms1.projectedMoments(1)[0], gms1.projectedDF(xyvz), 1e-2)')

if allok:
    print("\033[1;32mALL TESTS PASSED\033[0m")
else:
    print("\033[1;31mSOME TESTS FAILED\033[0m")
