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
except ImportError:  # otherwise load the shared library from the parent folder
    import sys
    sys.path += ['../']
    try: import agama
    except ImportError as ex: sys.exit("\033[1;31mFAILED TO IMPORT AGAMA: %s\033[0m" % ex)

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
potf = agama.Potential(type='spheroid', alpha=2, beta=5, gamma=0, q=0.75)  # flattened Plummer potential
pott = agama.Potential(type='spheroid', alpha=2, beta=5, gamma=0, p=0.75, q=0.5)  # triaxial Plummer potential
actf = agama.ActionFinder(potf)
actm = agama.ActionMapper(potf)
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

testCond('isFloat(dens.projectedDensity( 1,0 ))')             # may give either two numbers...
testCond('isFloat(dens.projectedDensity((1,0)))')             # ...or an array (tuple, list) of two numbers...
testCond('isFloat(dens.projectedDensity([1,0]))')             # ...for a single input point
testCond('dens.projectedDensity( 1,0, alpha=0,beta=0) > 0')   # can provide an input point as a pair of numbers, plus optional named arguments (angles)
testCond('dens.projectedDensity([1,0],beta=0,gamma=0) > 0')   # first argument is an array, optional named arguments are angles
testFail('dens.projectedDensity([1,0],0,0)')                  # angles must be named, not positional
testFail('dens.projectedDensity([1,0],zeta=0)')               # and of course, named arguments must match the function signature
testCond('dens.projectedDensity([[1,0]]).shape == (1,)')      # if the input is a 2d array (even with one row), output is a 1d array
testCond('dens.projectedDensity([[1,0],[2,0]],t=0).shape == (2,)')  # N input points (Nx2 array) => 1d array of length N
testCond('dens.projectedDensity([[1,0],[2,0]],beta=0).shape == (2,)')  # one may specify a single value of angle for all points...
testCond('dens.projectedDensity([[1,0],[2,0]],beta=(0,0),gamma=0,t=(0,0)).shape == (2,)')  # or one angle or time value per point (each angle can be a single point or an array independent of others)
testFail('dens.projectedDensity([[1,0],[2,0]],beta=(0,0,0))') # should fail when the size of the angles array does not match the number of points

testCond('isArray(dens.density([[1,2,3],[4,5,6]], t=1), (2,))')     # optional time argument as a single number
testCond('isArray(dens.density([[1,2,3],[4,5,6]], t=[1,2]), (2,))') # optional time argument as an array of the same length as points
testFail('dens.density([[1,2,3],[4,5,6],[7,8,9]], t=[1,2]))')       # wrong size of the optional time argument

# Potential class methods
testCond('isFloat(pots.potential(1,0,0))')
testCond('isArray(pots.force(  1,0,0  ), (3,))')
testCond('isArray(pots.force( [1,0,0] ), (3,))')   # equivalent
testCond('isArray(pots.force([[1,0,0]]), (1,3))')  # not equivalent: an array of 1 point is not the same as 1 point
testCond('isArray(pots.force( [1,0,0],[2,0,0] ), (2,3))')
testCond('isArray(pots.force([[1,0,0],[2,0,0]]), (2,3))')  # equivalent
testCond('isTuple(pots.eval(1,0,0,pot=True,acc=True,der=True), 3)')  # evaluate three quantities at once: potential, acceleration and its derivatives
testCond('isArray(pots.eval(1,0,0,acc=True,t=0), (3,))')
testCond('isArray(pots.eval([[1,2,3],[2,3,4]],der=True,t=[0,1]), (2,6))')  # one can specify times separately for each point
testCond('isArray(pots.projectedEval(1,0,acc=True), (2,))')  # similar to projectedDensity, but produces two values for each input point consisting of two numbers
testCond('isTuple(pots.projectedEval(1,1,pot=True,acc=True,der=True), 3)')  # evaluate three projected quantities at once: potential, acceleration and its derivatives
testCond('isArray(pots.projectedEval([[1,0],[2,0],[3,2]],acc=True,beta=(0,0,0),gamma=0,t=(1,2,3)), (3,2))')  # one can specify angles and time separately for each input point or one for all (or not specify them at all)
testCond('isArray(pots.projectedEval([[1,0],[1,2]],pot=True), (2,))')
testCond('isArray(pots.projectedEval([[1,1],[2,2]],der=True), (2,3))')

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
testCond('isArray(pots.Tcirc( 1,2,3,4,5 ), (5,))')  # same when the points are given as a tuple of arguments
testCond('isFloat(pots.Tcirc([1,2,3,4,5,6]))')      # but an array with 6 elements is treated as x,v (a single point)
testCond('isArray(pots.Tcirc([[1,2,3,4,5,6],[7,8,9,10,11,numpy.nan]]), (2,))')  # and a 2d array Nx6 - as N points x,v

testCond('abs(pots.Rmax(pots.potential(1.5,0,0))-1.5) < 1e-10')  # Rmax(E=Phi(r)) = r  with high accuracy,
testCond('pots.Rmax(0) == numpy.inf')                            # infinite for E == 0,
testCond('all(numpy.isnan(pots.Rmax([Phi0*2, 1])))')             # and NAN for E < Phi(0) or E > 0
testCond('isArray(pots.Rmax([0,1,numpy.nan]), (3,))')            # output array has the same length as input array / list
testCond('isArray(pots.Rmax( 0,1,2,3 ), (4,))')                  # and several arguments are treated in the same way as a 1d array

testCond('isArray(pots.Rperiapo( -1,1 ), (2,))')        # two numbers are a single point (E,L) => array of two values (Rperi,Rapo)
testCond('isArray(pots.Rperiapo([-1,1]), (2,))')        # same two numbers as a 1d array of length 2, same output
testCond('isArray(pots.Rperiapo([1,2,3,4,5,6]), (2,))') # 1d array of length 6 is a single point (x,v), same output
testCond('isArray(pots.Rperiapo( 1,2,3,4,5,6 ), (2,))') # same when given as 6 separate arguments
testCond('isArray(pots.Rperiapo(numpy.random.random(size=(5,2))), (5,2))')  # a 2d array of (E,L) values with shape Nx2 => Nx2
testCond('isArray(pots.Rperiapo(numpy.random.random(size=(5,6))), (5,2))')  # or a 2d array of (x,v) with shape Nx6 => Nx2
testFail('isArray(pots.Rperiapo(numpy.random.random(size=(5,3))), (5,2))')  # invalid shape of input
testCond('numpy.all(numpy.isnan(pots.Rperiapo([[Phi0*2, 1],[0, 1]])))')     # bad (but valid shape) input => NAN

# ActionFinder class methods
testCond('isArray(actf([1,2,3,4,5,6]), (3,))')  # 1d array of length 6 is a single point (x,v) => output array of 3 actions
testFail('actf( 1,2,3,4,5,6 )')                 # must be an array and not just 6 numbers
testCond('isTuple(actf([1,2,3,4,5,6], frequencies=True), 2)')   # extra argument frequencies=True produces a tuple of two arrays (actions and frequencies)
testCond('isTuple(actf([1,2,3,4,5,6], angles=True), 3)')        # extra argument angles=True implies frequencies=True and produces a tuple of length 3
testCond('isArray(actf([1,2,3,4,5,6], angles=True)[0], (3,))')  # with actions, angles and frequencies,
testCond('isArray(actf([1,2,3,4,5,6], angles=True)[1], (3,))')  # each of them being an array of length 3 (for a single point)
testCond('isArray(actf([1,2,3,4,5,6], angles=True)[2], (3,))')
testCond('isArray(actf([1,2,3,4,5,6], actions=False, frequencies=True), (3,))')  # when only frequencies but no actions are requested, the output is just an array of length 3 freqs (for a single point)
testCond('isArray(actf(numpy.random.random(size=(5,6))), (5,3))')  # when the input has N points (Nx6 array), output is Nx3,
testCond('isArray(actf(numpy.random.random(size=(5,6)), angles=True)[2], (5,3))')  # or three arrays of shape Nx3 if angles=True
testFail('agama.ActionFinder(pott)')  # should not work for non-axisymmetric potentials

# ActionMapper class methods
testCond('isArray(actm([1,2,3,1,2,3]), (6,))')  # 1d array of length 6 (3 actions + 3 angles for a single point) => output 1d array of length 6 (x,v)
testFail('actm( 1,2,3,4,5,6 )')                 # must be an array and not just 6 numbers
testCond('isArray(actm(numpy.random.random(size=(10,6))), (10,6))')  # N input points => 2d output array (Nx6)
testFail('agama.ActionMapper(pott)')  # should not work for non-axisymmetric potentials

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
testCond('len(df1([1,2,3], der=True))==2')  # DF derivatives w.r.t. actions are reported in the second element of the output tuple
testCond('isArray(df1([1,2,3], der=True)[1], (3,)) and isArray(df1(numpy.random.random(size=(7,3)), der=True)[1], (7,3))')  # derivs array has the same shape as input actions

# GalaxyModel class members
testCond('isTuple(gms1.moments([1,2,3], dens=True, vel=True, vel2=True), 3)')    # three outputs per point, as requested
testCond('isTuple(gms2.moments([1,2,3], dens=True, vel=False,vel2=True), 2)')    # only two requested here
testCond('isFloat(gmf1.moments([1,2,3], dens=True, vel=False,vel2=False))')      # only one output (density), and it is a scalar value
testCond('isFloat(gmf1.moments( 1,2,3 , dens=True, vel=False,vel2=False))')      # a single input point may be given as three numbers, not an array of length three
testCond('isArray(gmf1.moments([1,2,3], dens=False,vel=True, vel2=False),(3,))') # one output (3 components of velocity)
testCond('isArray(gms1.moments([1,2,3], dens=False,vel=False,vel2=True), (6,))') # one output (elements of dispersion tensor) - 1d array of length 6
testCond('isArray(gms1.moments([[1,2,3]],dens=True,vel=False,vel2=False),(1,))') # input is an array of points with length 1, and so is the output
testCond('isArray(gms2.moments([[1,2,3],[4,5,6],[7,8,9]], dens=True, vel=False, vel2=True)[1], (3,6))')  # N input points => dispersion tensor has shape Nx6
testCond('isArray(gms2.moments([[1,2,3],[4,5,6],[7,8,9]], dens=False,vel=False, vel2=True, separate=True), (3,2,6))')  # DF with C components treated separately => NxCx6
testCond('isArray(gms1.moments([[1,2,3],[4,5,6],[7,8,9]], dens=False,vel=False, vel2=True, separate=True), (3,1,6))')  # same here, but C=1 - output array is still 3d
testCond('isArray(gms1.moments([1,2,3], dens=True, vel=False,vel2=False, separate=True), (1,))')  # single point, single output, but composite DF with C=1 => 1d array of length C
# check that separately computed moments of a composite DF (df2) agree with the moments of its 0th component (df1)
testCond('numpy.isclose(gms1.moments([1,2,3],dens=1,vel=0,vel2=0), gms2.moments([1,2,3],dens=1,vel=0,vel2=0,separate=True)[0], 1e-4)')
# projected moments (input point(s) have only X,Y coordinates)
testCond('isTuple(gms1.moments([1,0], dens=True, vel=True, vel2=True), 3)')  # same as before, but for the projected case  => output: a tuple of three numbers
testCond('isArray(gms1.moments([[1,0],[2,0],[3,0],[4,0]], dens=True, vel=False, vel2=False), (4,))')  # input: array of N points, only one output array (density) with N elements
testCond('isArray(gms2.moments([[1,0],[2,0],[3,0],[4,0]], dens=True, vel=False, vel2=False, separate=True), (4,2))')  # when separate=True, output arrays have one extra dimension of size C
# optional angle arguments
testCond('isArray(gms1.moments([[1,2,3],[4,5,6]], beta=1, gamma=[1,2], dens=True, vel=False, vel2=False), (2,))')  # each of the orientation angles can be specified as single value for all points or one value per point
testFail('gms1.moments([1,2,3], beta=[1,2,3])')   # length of the angles array must match the number of input points

testCond('isFloat(gms1.projectedDF([1,2,0,0,0,0,0,0]))')  # input: 8 numbers per point (x,y,vx,vy,vz,vxerr,vyerr,vzerr) => output: one number per point
testCond('isArray(gms2.projectedDF(numpy.random.random(size=(4,8)), separate=True), (4,2))')  # when separate=True, output for N points and C components is a 2d array (NxC)

testCond('isTuple(gms1.vdf([1,2,3]), 3)')             # input: one point in 3d(xyz) => output 3 spline objects
testCond('isArray(gms1.vdf([[1,2],[3,4]])[0], (2,))') # input: two points in 2d(xy) => output a tuple of 3 arrays with splines
testCond('isTuple(gms1.vdf([1,2,3], dens=True), 4)')  # if a flag dens=True is set, output is a tuple of length 4: 3 splines and density
testCond('isArray(gms1.vdf([1,2,3])[0]([1,2,3,4,5]), (5,))')  # spline objects are functions that convert input arrays to output arrays of the same length
testCond('isArray(gms2.vdf([[1,2,3],[4,5,6],[7,8,9]], separate=True)[0], (3,2))')  # when invoked for N points with separate=True, output arrays are NxC
# when vdf() is invoked with dens=True, the computed 3d density should agree with the one produced by moments()
testCond('numpy.isclose(gms1.vdf([1,2,3], dens=True)[3], gms1.moments([1,2,3])[0], 1e-3)')
# and similarly for the case of 2d input (projected VDF): projected density should agree with projected moments()
testCond('numpy.isclose(gms1.vdf([1,0], dens=True)[3], gms1.moments([1,0])[0], 1e-3)')
# likewise, projectedDF with infinite uncertainties on all three velocity components should give projected density
testCond('numpy.isclose(gms1.projectedDF([1,0,0,0,0,numpy.inf,numpy.inf,numpy.inf]), gms1.moments([1,0],vel2=False), 1e-3)')
# finally check that the projected VDF in v_z agrees with the output of projectedDF() after multiplying the former by the projected density
vz = numpy.linspace(0, 0.5*(-2*pots.potential(1,0,0))**0.5, 4)  # grid in v_z from 0 to half the escape velocity
xyv = numpy.column_stack((vz*0+1, vz*0, vz*0, vz*0, vz, vz+numpy.inf, vz+numpy.inf, vz*0))  # input points for projectedDF()
testCond('numpy.allclose(gms1.vdf([1,0])[2](vz) * gms1.moments([1,0])[0], gms1.projectedDF(xyv), 1e-2)')

# test that zero-length inputs are acceptable and produce zero-length outputs of appropriate shape
arr0x3 = numpy.zeros(shape=(0,3))
arr0x8 = numpy.zeros(shape=(0,8))
arr0x1 = []
testCond('isArray(pots.density(arr0x3), (0,))')
testCond('isArray(pots.enclosedMass(arr0x1), (0,))')
testCond('isArray(gms2.moments(arr0x3, separate=False)[0], (0,)) and isArray(gms2.moments(arr0x3, separate=False)[1], (0,6))')
testCond('isArray(gms2.moments(arr0x3, separate=True)[0], (0,2)) and isArray(gms2.moments(arr0x3, separate=True)[1], (0,2,6))')
testCond('isArray(gms2.projectedDF(arr0x8, separate=False), (0,))')
testCond('isArray(gms2.projectedDF(arr0x8, separate=True), (0,2))')
testCond('isArray(gms2.vdf(arr0x3, separate=False)[0], (0,))')
testCond('isArray(gms2.vdf(arr0x3, separate=True)[0], (0,2))')

if allok:
    print("\033[1;32mALL TESTS PASSED\033[0m")
else:
    print("\033[1;31mSOME TESTS FAILED\033[0m")
