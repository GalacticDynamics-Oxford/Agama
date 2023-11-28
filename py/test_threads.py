'''
Test the vectorized and OpenMP-parallelized execution of various tasks involving user-defined Python callback functions.
'''
import numpy, sys
# if the module has been installed to the globally known directory, just import it
try: import agama
except ImportError:  # otherwise load the shared library from the parent folder
    import sys
    sys.path += ['../']
    try: import agama
    except ImportError as ex: sys.exit("\033[1;31mFAILED TO IMPORT AGAMA: %s\033[0m" % ex)

if len(sys.argv)>1:
    print('Adding nested calls to parallelized routines from Agama (test runs much slower)')
    NESTED = True
else:
    NESTED = False

# manually implemented plummer density profile (which also doubles as a distribution function and a selection function)
def myfnc(x):
    # print(len(x), end=' ')  # uncomment to see a lot of numbers... showing how many points per call are processed
    if NESTED:
        # simplest/fastest way of invoking another level of parallelization for a user-defined python callback function
        agama.sampleNdim(lambda x: numpy.ones(len(x)), 2000, [0,0,0], [1,1,1])
    return 0.75/numpy.pi * (numpy.sum(x**2, axis=1) + 1)**-2.5

# manually implemented plummer potential
def mypotfnc(x):
    # print(len(x), end=' ')  # uncomment to see even more numbers...
    if NESTED:
        agama.sampleNdim(lambda x: numpy.ones(len(x)), 2000, [0,0,0], [1,1,1])
    return -(numpy.sum(x**2, axis=1) + 1)**-0.5

# it is sufficient to use only 2 threads to test the correctness of running in a multi-threaded context;
# more will only slow things down (in fact, when using user-defined Python callbacks, it is advisable
# to call setNumThreads(1), possibly in a "with" statement around the code block, as shown below)
with agama.setNumThreads(2):
    plumpot = agama.Potential(type='Plummer')
    plumden = plumpot
    userpot = agama.Potential(mypotfnc, symmetry='s')
    userden = agama.Density(myfnc, symmetry='s')
    targetd = agama.Target(type='DensityClassicLinear', gridr=[0,1,2])
    targetl = agama.Target(type='LOSVD', degree=0, gridx=[0,0.5,1], gridv=[-1,1], apertures=([[0,0], [0.5,1], [1,0]],))
    numpy.random.seed(42)

    print('integrateNdim for a user-defined function')
    agama.integrateNdim(myfnc, [0,0,0], [2,2,2])  # does not actually use multithreading
    print('sampleNdim for a user-defined function')
    agama.sampleNdim(myfnc, 10000, [0,0,0], [2,2,2])  # parallel evaluation of user function

    print('vectorized methods of a user-defined density')
    userden.density(numpy.random.normal(size=(10000,3)))  # parallel loop in BatchFunction with vectorized calls to density
    userden.projectedDensity(numpy.random.normal(size=(100,2)))  # another parallel loop
    print('sampling a user-defined density')
    userden.sample(10000)  # parallel loop in sampleNdim

    print('constructing CylSpline potential from a user-defined density')
    agama.Potential(type='CylSpline', density=userden, gridsizer=5, gridsizez=5, rmin=0.1, rmax=10)  # a lot of calls!

    print('sampling velocity in a user-defined potential')
    plumden.sample(2000, potential=userpot)  # uses threads in constructing the spherical DF and in sampling

    print('action finder for a user-defined potential')
    agama.ActionFinder(userpot)  # uses threads in creating the action interpolator - by far the largest number of calls

    print('spherical df constructed in a user-defined potential')
    agama.DistributionFunction(type='QuasiSpherical', potential=userpot)  # creates action finder for the user potential

    print('vectorized methods of a user-defined potential')
    userpot.Rperiapo(numpy.random.normal(size=(1000,2))*numpy.array([-1,1]))  # parallel loop in BatchFunction
    print('sampling density from a user-defined potential')
    userpot.sample(2000)  # parallelization in sampleNdim

    print('orbit integration in a user-defined potential')
    agama.orbit(ic=numpy.random.normal(size=(4,6)), potential=userpot, time=10, dtype=object)  # parallel loop over orbits

    print('applying density target to user density')
    targetd(userden)  # currently not using parallelization
    print('applying losvd target to user density')
    targetl(userden)  # same here

    print('user-defined distribution function')
    gm = agama.GalaxyModel(plumpot, myfnc)
    gm.moments([[1,2,3],[4,5,6]], vel=False, vel2=False)  # parallel loop over points
    gm.sample(2000)  # uses threads in sampling

    print('user-defined selection function')
    df = agama.DistributionFunction(type='QuasiSpherical', potential=plumpot)
    gm = agama.GalaxyModel(plumpot, df, sf=myfnc)
    gm.moments([[1,2,3],[4,5,6]], vel=False, vel2=False)  # parallel loop over points
    gm.sample(2000)  # uses threads in sampling

    print('self-consistent modelling with user-defined potential and df')
    scm = agama.SelfConsistentModel(lmaxAngularSph=0, sizeRadialSph=25, rminSph=0.01, rmaxSph=100)
    scm.components.append(agama.Component(df=myfnc, disklike=False, lmaxAngularSph=0, sizeRadialSph=15, rminSph=0.01, rmaxSph=100))
    scm.potential = userpot
    scm.iterate()  # parallel recomputation of density from df, parallelization in the action finder constructor

    # if we got here without deadlocking, everything is fine
    print("\033[1;32mALL TESTS PASSED\033[0m")
