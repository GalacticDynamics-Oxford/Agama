#!/usr/bin/python
'''
This test compares two trajectory representations in orbit integration:
array of position/velocity points at regularly sampled times,
and interpolated solution provided by agama.Orbit class,
which gives a typical relative accuracy of ~1e-6 for position and ~1e-5
for velocity and energy compared to the finely sampled array representation.
It also illustrates how this spline interpolation works, using a pure Python
implementation of the agama.Orbit class.
A second example demonstrates the computation of orbit derivatives.
'''
import numpy, sys, time, warnings
# if the module has been installed to the globally known directory, just import it
try: import agama
except ImportError:  # otherwise load the shared library from the parent folder
    import sys
    sys.path += ['../']
    try: import agama
    except ImportError as ex: sys.exit("\033[1;31mFAILED TO IMPORT AGAMA: %s\033[0m" % ex)

class Orbit(object):
    """
    A pure Python analogue of agama.Orbit class providing interpolated trajectory
    (of course, much less efficient than the native C++ implementation)
    """
    def __init__(self, t, xv, Omega=0):
        t = numpy.array(t)
        xv= numpy.array(xv)
        if len(t.shape) != 1 or len(xv.shape) != 2 or xv.shape[1] != 6 or t.shape[0] != xv.shape[0]:
            raise RuntimeError("Invalid size of input arrays")
        # if the orbit was integrated backward in time, need to reverse the time axis passed to interpolators
        self.reversed = len(t)>1 and t[-1]<t[0]
        step = -1 if self.reversed else 1
        # when the orbit was integrated in the rotating frame, v != dx/dt:
        # in order to create the quintic spline, we transform the orbit to the non-rotating frame,
        # and then transform the interpolated result back to the rotated frame on evaluation
        self.Omega = Omega
        ca, sa = numpy.cos(Omega*t), numpy.sin(Omega*t)
        xi  = xv[:,0] * ca - xv[:,1] * sa
        yi  = xv[:,1] * ca + xv[:,0] * sa
        vxi = xv[:,3] * ca - xv[:,4] * sa
        vyi = xv[:,4] * ca + xv[:,3] * sa
        self.x = agama.Spline(t[::step], xi[::step],  vxi[::step])
        self.y = agama.Spline(t[::step], yi[::step],  vyi[::step])
        self.z = agama.Spline(t[::step], xv[::step,2], xv[::step,5])

    def __call__(self, t):
        """
        Compute the interpolated orbit at time(s) t.
        Argument:   a single number or an array of timestamps of length N.
        Return:     an array of position and velocity points at corresponding times (shape Nx6);
                    when the time is outside the range spanned by orbit, the result is NaN.
        """
        t = numpy.array(t)
        # transform the interpolated trajectory back to rotated frame
        xi  = self.x(t)
        yi  = self.y(t)
        vxi = self.x(t, der=1)
        vyi = self.y(t, der=1)
        ca, sa = numpy.cos(self.Omega*t), numpy.sin(self.Omega*t)
        result = numpy.column_stack([xi * ca + yi * sa,  yi * ca - xi * sa,  self.z(t),
            vxi * ca + vyi * sa,  vyi * ca - vxi * sa,  self.z(t, der=1)])
        if t.shape == ():
            result = result[0]  # if the input is one point, the output is a 1d array of length 6
        if len(self.x) == 0:    # an empty orbit is also permitted, though not very useful
            tmin, tmax = numpy.nan, numpy.nan
        else:
            tmin, tmax = self.x[0], self.x[-1]
        # no interpolation outside the time range spanned by orbit
        result[numpy.where(numpy.logical_not(numpy.logical_and(t >= tmin, t <= tmax)))] *= numpy.nan
        return result

    def __len__(self):
        """
        Return the number of points in the orbit
        """
        return len(self.x)

    def __getitem__(self, i):
        """
        Return the timestamp with the given index
        """
        n = len(self.x)
        if i<0: i=n+i
        if i<0 or i>=n:
            raise IndexError("Orbit timestamp index out of range")
        if self.reversed:
            return self.x[n-1-i]
        else:
            return self.x[i]

class Check(object):
    def __init__(self):
        self.ok = True
    def __call__(self, value, limit):
        if value > limit:
            self.ok = False
            return '%g \033[1;31m**\033[0m' % value
        else:
            return '%g' % value
    def __nonzero__(self):  # Python 2
        return self.ok
    def __bool__(self):     # Python 3
        return self.ok

def testHermite(pot, ic, time_in_periods, accuracy):
    check = Check()
    inttime = pot.Tcirc(ic) * time_in_periods
    t0 = time.time()
    orb_dop = agama.orbit(potential=pot, ic=ic, time=inttime, trajsize=0, dtype=float, accuracy=accuracy, method='dop853')[1]
    t1 = time.time()
    orb_her = agama.orbit(potential=pot, ic=ic, time=inttime, trajsize=0, dtype=float, accuracy=accuracy, method='hermite')[1]
    t2 = time.time()
    E_dop = pot.potential(orb_dop[:,0:3]) + 0.5 * numpy.sum(orb_dop[:,3:6]**2, axis=1)
    E_her = pot.potential(orb_her[:,0:3]) + 0.5 * numpy.sum(orb_her[:,3:6]**2, axis=1)
    errE_dop = max(abs(E_dop/E_dop[0]-1))
    errE_her = max(abs(E_her/E_her[0]-1))
    print("Orbit integration with dop853,  accuracy %g: %6i steps, time=%.3f s, energy error=%s" %
        (accuracy, len(orb_dop), t1-t0, check(errE_dop, accuracy**0.5)))
    print("Orbit integration with hermite, accuracy %g: %6i steps, time=%.3f s, energy error=%s" %
        (accuracy, len(orb_her), t2-t1, check(errE_her, accuracy**0.5)))
    return bool(check)

def testAccuracy(pot, ic, time_in_periods, accuracy, Omega=0, ax=None):
    check = Check()
    time = pot.Tcirc(ic) * time_in_periods
    def JacobiEnergy(orb):
        return ( pot.potential(orb[:,0:3]) + 0.5 * numpy.sum(orb[:,3:6]**2, axis=1)
            -Omega * (orb[:,0] * orb[:,4] - orb[:,1] * orb[:,3]) )

    # "exact" orbit computed with very high accuracy and returned as an agama.Orbit instance
    orb_exact = agama.orbit(potential=pot, ic=ic, time=time, Omega=Omega, dtype=object, accuracy=1e-15)
    traj_exact = orb_exact(orb_exact)  # convert it to a trajectory array
    E_exact = JacobiEnergy(traj_exact)
    errE_exact = abs(E_exact/E_exact[0]-1)
    print("'exact' orbit of length %i, energy error=%g" % (len(orb_exact), numpy.max(errE_exact)))

    # orbit sampled at regular intervals, not related to the actual timestep of the ODE integrator
    # (thus the trajectory is produced by a 7th order interpolating polynomial within each timestep)
    time_reg, traj_reg = agama.orbit(potential=pot, ic=ic, time=time, Omega=Omega,
        dtype=float, trajsize=abs(time_in_periods)*100, accuracy=accuracy)
    errX_reg = numpy.sum((traj_reg - orb_exact(time_reg))[:,0:3]**2, axis=1)**0.5
    E_reg = JacobiEnergy(traj_reg)
    errE_reg = abs(E_reg/E_reg[0]-1)
    errE_max = max(2.0 * accuracy, 2e-4 * accuracy**0.5) * abs(time_in_periods)
    print("regularly sampled orbit of length %i, energy error=%s" % (len(time_reg),
        check(numpy.max(errE_reg), errE_max) ))

    # orbit stored at each timestep of the ODE integrator and then converted to a 5th order spline
    orb_spl = agama.orbit(potential=pot, ic=ic, time=time, Omega=Omega, dtype=object, accuracy=accuracy)
    time_spl = numpy.array(orb_spl)  # unequally-spaced timestamps
    traj_spl = orb_spl(time_spl)     # convert it to the array of trajectory points at each spline node
    traj_spl_reg = orb_spl(time_reg) # intepolated trajectory at regularly spaced intervals
    errX_spl_reg = numpy.sum((traj_reg - traj_spl_reg)[:,0:3]**2, axis=1)**0.5

    # construct a pure Python equivalent of agama.Orbit class from the same trajectory
    Orb_spl = Orbit(time_spl, traj_spl, Omega)
    # it should be identical to the native C++ interpolator at all times
    # (up to floating-point error when Omega=0, or up to differences in implementation of trig functions otherwise)
    errX_Orb_spl = numpy.sum((Orb_spl(time_reg) - traj_spl_reg)[:,0:3]**2, axis=1)**0.5

    # compare the interpolated trajectory provided by Orbit with the regularly sampled trajectory
    E_spl = JacobiEnergy(traj_spl)
    E_spl_reg = JacobiEnergy(traj_spl_reg)
    errE_spl = abs(E_spl/E_spl[0]-1)
    errE_spl_reg = abs(E_spl_reg/E_reg-1)
    print("naturally sampled orbit of length %i, energy error at spline nodes=%s, at interpolated times=%s" %
        (len(orb_spl),
        check(numpy.max(errE_spl), errE_max),
        check(numpy.max(errE_spl_reg), 1.0 * accuracy**0.5) ))
    print("difference in position between interpolated and regularly sampled orbits=%s" %
        check(numpy.max(errX_spl_reg), 0.1 * accuracy**0.5) )
    print("difference in position between Python and C++ versions of Orbit interpolator: %s" %
        check(numpy.max(errX_Orb_spl), 1e-15 * abs(time_in_periods)) )
    if ax is not None:
        ax.plot(time_reg, errX_reg      , '.', ms=3, mew=0, color='b')
        ax.plot([numpy.nan], [numpy.nan], '.', ms=8, mew=0, color='b', label=r'$|{\bf x}_{reg} - {\bf x}_{true}|$')
        ax.plot(time_reg, errX_spl_reg,   '.', ms=3, mew=0, color='r')
        ax.plot([numpy.nan], [numpy.nan], '.', ms=8, mew=0, color='r', label=r'$|{\bf x}_{spl} - {\bf x}_{reg}|$')
        ax.plot(time_reg, errE_reg,       '.', ms=3, mew=0, color='g')
        ax.plot([numpy.nan], [numpy.nan], '.', ms=8, mew=0, color='g', label=r'$|E_{reg}/E_{true} - 1|$')
        ax.plot(time_reg, errE_spl_reg,   '.', ms=3, mew=0, color='c')
        ax.plot([numpy.nan], [numpy.nan], '.', ms=8, mew=0, color='c', label=r'$|E_{spl}/E_{reg} - 1|$')
        ax.set_yscale('log')
        ax.set_ylim(1e-10, 1e-4)
        ax.set_xlim(0, time)
        ax.set_xlabel('time')
        ax.text(0.5, 0.98, 'accuracy=%g' % accuracy, ha='center', va='top', transform=ax.transAxes)
    return bool(check)

def testDerivatives(pot, ic, time_in_periods, accuracy, Omega, ax=None):
    check = Check()
    time = pot.Tcirc(ic) * time_in_periods
    delta_ic = 1e-8 * numpy.random.normal(size=6)  # perturbation to the initial conditions
    times, trajs, der = agama.orbit(potential=pot, ic=[ic, ic+delta_ic], time=time, Omega=Omega,
        trajsize=1000, dtype=float, accuracy=accuracy, der=True, verbose=False, separateTime=True)
    jac = numpy.dstack(der[0])  # size: (trajsize,6,6); jac[i] is the Jacobian matrix at time t[i]
    linear_dif = jac.dot(delta_ic)
    actual_dif = trajs[1] - trajs[0]
    max_offset = numpy.max(abs(actual_dif[:,0:3]))
    max_error  = numpy.max(abs(actual_dif - linear_dif)[:,0:3])
    print('Comparison of a slightly perturbed orbit with the deviation vectors from the variational equation:')
    print('difference in position between original and perturbed orbit: %s' % check(max_offset, 1e-4))
    print('difference between the perturbed orbit and the linear prediction: %s' % check(max_error, 1e-8))
    if ax is not None:
        ax.plot(times[0], actual_dif[:,0:3], lw=1)
        ax.plot(times[0], linear_dif[:,0:3], dashes=[2,4], lw=2)
        ax.set_xlim(0, time)
        ax.set_xlabel('time')
        ax.set_ylabel('deviations from initial orbit')
        ax.text(0.02, 0.98, 'solid - difference between nearby orbits\ndotted - deviation vectors',
            ha='left', va='top', transform=ax.transAxes)
    return bool(check)

# test if the given condition (a string to be executed) is true
def testCond(condition):
    global ok
    try:
        result = eval(condition)
        if bool(result) is True:
            return  # just as planned
        print('%s \033[1;31m is %s\033[0m' % (condition, result))
        ok = False
    except Exception as ex:
        print('%s \033[1;31m failed:\033[0m %s' % (condition, ex))
        ok = False


if __name__ == '__main__':
    plot = len(sys.argv)>1
    if plot:
        import matplotlib.pyplot as plt
        ax = plt.subplots(1, 3, figsize=(18,6), dpi=75)[1]
    else:
        ax = None, None, None
    agama.setUnits(length=1, velocity=1, mass=1)
    numpy.random.seed(42)
    ok = True
    pot = agama.Potential(type='spheroid', mass=1e10, scaleradius=1, gamma=1, beta=4, alpha=1, p=0.7, q=0.4)

    # first part: run a bunch of tests verifying the correctness of agama.orbit API for various combinations of arguments
    ic = numpy.random.random(size=(4,6))
    N = len(ic)
    S = 5  # size of output trajectory, when fixed to the same value for all orbits
    times = numpy.linspace(0, 1, N+1)[1:] * 4
    target0 = agama.Target(type='DensityClassicTopHat', gridr=[0.1,1,10])
    target1 = agama.Target(type='KinemShell', degree=1, gridr=[0.1,1,10])
    warnings.simplefilter("ignore")  # suppress the warnings issued by agama.orbit when using non-optimal output storage
    ranf = agama.orbit(potential=pot, verbose=False, ic=ic, time=times, trajsize=0)               # dtype='float32' by default
    rand = agama.orbit(potential=pot, verbose=False, ic=ic, time=times, trajsize=0, dtype=float)  # same as dtype='float64'
    ranc = agama.orbit(potential=pot, verbose=False, ic=ic, time=times, trajsize=0, dtype='complex64')
    ranz = agama.orbit(potential=pot, verbose=False, ic=ic, time=times, trajsize=0, dtype='complex128')
    rano = agama.orbit(potential=pot, verbose=False, ic=ic, time=times, dtype=object)  # implies trajsize=0; output is stored as instances of agama.Orbit
    raff = agama.orbit(potential=pot, verbose=False, ic=ic, time=times, trajsize=S)
    roff = agama.orbit(potential=pot, ic=ic[0:1], time=times[0], trajsize=S)  # one orbit, but IC is still a 2d array (even if with one row), so the output format is the same as for many orbits
    rofo = agama.orbit(potential=pot, ic=ic[0:1], time=times[0], trajsize=S, dtype=object)
    rsff = agama.orbit(potential=pot, ic=ic[0  ], time=times[0], trajsize=S)  # single orbit, and IC is a 1d array - outputs arrays have the 0th dimension removed
    rsfo = agama.orbit(potential=pot, ic=ic[0  ], time=times[0], trajsize=S, dtype=object)
    sanf = agama.orbit(potential=pot, verbose=False, ic=ic, time=times, trajsize=0, separateTime=True)  # new flag for storing timestamps in a separate output array
    saff = agama.orbit(potential=pot, verbose=False, ic=ic, time=times, trajsize=S, separateTime=True)
    sano = agama.orbit(potential=pot, verbose=False, ic=ic, time=times, dtype=object, separateTime=True)
    sonf = agama.orbit(potential=pot, ic=ic[0:1], time=times[0], trajsize=0, separateTime=True)
    soff = agama.orbit(potential=pot, ic=ic[0:1], time=times[0], trajsize=S, separateTime=True)
    sono = agama.orbit(potential=pot, ic=ic[0:1], time=times[0], dtype=object, separateTime=True)
    ssnf = agama.orbit(potential=pot, ic=ic[0  ], time=times[0], trajsize=0, separateTime=True)
    ssff = agama.orbit(potential=pot, ic=ic[0  ], time=times[0], trajsize=S, separateTime=True)
    ssno = agama.orbit(potential=pot, ic=ic[0  ], time=times[0], dtype=object, separateTime=True)
    ranfv= agama.orbit(potential=pot, verbose=False, ic=ic, time=times, trajsize=0, der=True)  # output 6 deviation vectors per orbit in addition to the trajectory itself
    raffv= agama.orbit(potential=pot, verbose=False, ic=ic, time=times, trajsize=S, der=True)
    rafov= agama.orbit(potential=pot, verbose=False, ic=ic, time=times, trajsize=S, der=True, dtype=object)
    saffv= agama.orbit(potential=pot, verbose=False, ic=ic, time=times, trajsize=S, der=True, separateTime=True)
    ssffv= agama.orbit(potential=pot, ic=ic[0  ], time=times[0], trajsize=S, der=True, separateTime=True)
    sanot= agama.orbit(potential=pot, verbose=False, ic=ic, time=times, dtype=object, separateTime=True, der=True, lyapunov=True, targets=(target0,target1))  # output everything!
    rnnf = agama.orbit(potential=pot, ic=ic[0:0], time=times[0], trajsize=0)  # degenerate case of zero-length IC - still has to produce all required output, but array(s) have zero rows
    snff = agama.orbit(potential=pot, ic=ic[0:0], time=times[0], trajsize=S, separateTime=True)
    rnnot= agama.orbit(potential=pot, ic=ic[0:0], time=times[0], dtype=object, lyapunov=True, der=True, targets=(target0, target1))  # everything should have zero length
    snfft= agama.orbit(potential=pot, ic=ic[0:0], time=times[0], trajsize=S, lyapunov=True, der=True, targets=(target0, target1), separateTime=True)
    warnings.resetwarnings()
    agamaOrbit = type(rano[0])  # agama.Orbit class, but it cannot be accessed directly, so we take a circuitous route
    testCond('isinstance(ranf, numpy.ndarray) and ranf.shape==(N,2) and ranf.dtype==object')  # result is a numpy array containing objects (other numpy arrays)
    testCond('ranf[0,0].dtype==numpy.float64 and ranf[0,1].dtype==numpy.float32')    # dtype of timestamps is always float64, whereas dtype of trajectories matches the requested one
    testCond('rand[0,0].dtype==numpy.float64 and rand[0,1].dtype==numpy.float64')
    testCond('ranc[0,0].dtype==numpy.float64 and ranc[0,1].dtype==numpy.complex64')
    testCond('ranz[0,0].dtype==numpy.float64 and ranz[0,1].dtype==numpy.complex128')
    testCond('numpy.all(ranc[0,1].real==ranf[0,1][:,0:3]) and numpy.all(ranc[0,1].imag==ranf[0,1][:,3:6])')  # test that real and imag parts of complex outputs correspond to position and velocity components of 'normal' outputs
    testCond('numpy.all(ranz[0,1].real==rand[0,1][:,0:3]) and numpy.all(ranz[0,1].imag==rand[0,1][:,3:6])')
    testCond('numpy.all(rand[0,1].astype(numpy.float32)==ranf[0,1])')  # test rounding down to float32
    testCond('all([numpy.allclose(rand[n,1][0], ic[n], rtol=2.3e-16, atol=0) for n in range(N)])')  # 0th row of each trajectory matches IC to within roundoff error caused by back-and-forth unit conversion
    testCond('isinstance(rano, numpy.ndarray) and rano.shape==(N,) and rano.dtype==object')  # orbits stored as agama.Orbit instances
    testCond('all([len(rano[n])==len(rand[n,1]) for n in range(N)])')  # all agama.Orbit lengths match those of corresponding trajectories stored as arrays
    testCond('all([numpy.all(rano[n](rand[n,0])==rand[n,1]) for n in range(N)])')  # interpolated orbits produce exactly the same results as (float64) arrays when evaluated at the same timestamps
    testCond('all([raff[n,0].dtype==float and raff[n,1].dtype==numpy.float32 and raff[n,0].shape==(S,) and raff[n,1].shape==(S,6) for n in range(N)])')  # test that all orbits have the same length S when requested
    testCond('isinstance(roff, numpy.ndarray) and roff.shape==(1,2) and roff.dtype==object and roff[0,0].shape==(S,) and roff[0,1].shape==(S,6)')  # when IC is a 2d array of shape (1,6), the result is still a 2d array with one row only (timestamps and trajectories)
    testCond('isinstance(rsff, numpy.ndarray) and rsff.shape==(2, ) and rsff.dtype==object and rsff[  0].shape==(S,) and rsff[  1].shape==(S,6)')  # but when IC is a 1d array of length 6 (a single point), the result is a 1d array of length 2 (timestamps and trajectories)
    testCond('isinstance(rofo, numpy.ndarray) and rofo.shape==(1, ) and rofo.dtype==object and isinstance(rofo[0], agamaOrbit) and len(rofo[0])==S')  # when output is represented by agama.Orbit interpolators, the array has the same length as the number of rows in IC
    testCond('isinstance(rsfo, agamaOrbit) and len(rsfo)==S and all(rsfo(0)==rand[0,1][0])')  # when IC is a 1d array (a single point) and dtype=object, the output is a single instance of agama.Orbit
    testCond('isinstance(sanf, tuple) and len(sanf)==2 and sanf[0].shape==(N,) and sanf[0].dtype==object and sanf[1].shape==(N,) and sanf[1].dtype==object')  # when separateTime=True, the output is a tuple of two arrays of length N each: timestamps and trajectories, rather than a single array of shape (N,2)
    testCond('isinstance(saff, tuple) and len(saff)==2 and saff[0].shape==(N,S) and saff[0].dtype==numpy.float64 and saff[1].shape==(N,S,6) and saff[1].dtype==numpy.float32')  # but when in addition all trajsize's are the same (S), this turns on the optimized storage, where all timestamps are combined into one 2d array of shape (N,S) and all trajectories - into a 3d array of shape (N,S,6)
    testCond('all([numpy.all(sanf[0][n]==ranf[n,0]) and numpy.all(sanf[1][n]==ranf[n,1]) for n in range(N)])')  # with separateTime=True, the output arrays are the same, just the access order is transposed
    testCond('all([numpy.all(saff[0][n]==raff[n,0]) and numpy.all(saff[1][n]==raff[n,1]) for n in range(N)])')  # remarkably, this remains true even in the case of optimized storage
    testCond('numpy.all(saff[0]==numpy.vstack(raff[:,0])) and numpy.all(saff[1]==numpy.vstack(raff[:,1]).reshape(N,S,6))')  # but avoids the need to stack individual trajectories returned when separateTime=False
    testCond('isinstance(sonf, tuple) and len(sonf)==2 and sonf[0].shape==(1,) and sonf[0][0].shape==sanf[0][0].shape and sonf[1].shape==(1,) and sonf[1][0].shape==sanf[1][0].shape')  # when IC is a 2d array even with one row, the output is still a tuple of two 1d arrays (timestamps and trajectories), even if each has only one element (itself an array)
    testCond('isinstance(ssnf, tuple) and len(ssnf)==2 and                         ssnf[0]   .shape==sanf[0][0].shape and                         ssnf[1]   .shape==sanf[1][0].shape')  # however, when IC is a single point (1d array of length 6), the output is a tuple of two arrays: timestamps with shape (len(orbit),) and trajectory with shape (len(orbit),6)
    testCond('isinstance(soff, tuple) and len(soff)==2 and soff[0].shape==(1,S) and soff[0].dtype==numpy.float64 and soff[1].shape==(1,S,6) and soff[1].dtype==numpy.float32')  # when the trajectory size is fixed beforehand, the outputs are optimized to be just arrays of floats instead of arrays of arrays
    testCond('isinstance(ssff, tuple) and len(ssff)==2 and ssff[0].shape==(S, ) and ssff[1].shape==(S,6) and numpy.all(ssff[0]==soff[0][0]) and numpy.all(ssff[1]==soff[1][0])')  # and the first dimension is squeezed out if IC is a single point
    testCond('isinstance(sano, numpy.ndarray) and sano.shape==(N,) and all([isinstance(sano[n], agamaOrbit) and len(sano[n])==len(rano[n]) for n in range(N)])')  # when dtype==object, separateTime has no effect, the output is still a 1d array of objects (instances of agama.Orbit)
    testCond('isinstance(sono, numpy.ndarray) and sono.shape==(1,) and isinstance(sono[0], agamaOrbit)')  # even if the length of this array is 1 (when IC is a 2d array with 1 row)
    testCond('isinstance(ssno, agamaOrbit) and len(ssno)==len(sono[0])')  # but when IC is a 1d array (single point), the output is a single instance of agama.Orbit, not an array
    testCond('isinstance(ranfv, tuple) and len(ranfv)==2 and ranfv[0].shape==(N,2) and all([numpy.all(ranfv[0][n,1]==ranf[n,1]) for n in range(N)])')  # output is now a 2-tuple: (timestamps_and_trajectories, dev_vectors), the first element is identical to the earlier run w/o dev.vectors
    testCond('isinstance(ranfv[1], numpy.ndarray) and ranfv[1].shape==(N,6) and all([ranfv[1][n,d].shape==ranfv[0][n,1].shape for d in range(6) for n in range(N)])')  # the shape of each dev.vector is the same as the corresponding trajectory
    testCond('isinstance(raffv, tuple) and len(raffv)==2 and raffv[0].shape==(N,2) and isinstance(raffv[1], numpy.ndarray) and raffv[1].dtype==numpy.float32 and raffv[1].shape==(N,6,S,6)')  # when all orbits have the same length, dev.vectors are stored in a single 4d array (N,6,S,6_or_3)
    testCond('isinstance(rafov, tuple) and len(rafov)==2 and rafov[0].shape==(N, ) and rafov[1].shape==(N,6) and isinstance(rafov[0][0], agamaOrbit) and isinstance(rafov[1][0,0], agamaOrbit)')  # however, if dtype=object, this falls back to storing dev.vectors as a (N,6) array of agama.Orbit instances
    testCond('isinstance(saffv, tuple) and len(saffv)==3 and numpy.all(saffv[0]==saff[0]) and numpy.all(saffv[1]==saff[1]) and numpy.all(saffv[2]==raffv[1])')  # when separateTime is on, the output tuple has 3 elements (timestamps, trajectories, dev_vectors), the latter still a single 4d array
    testCond('isinstance(saffv, tuple) and len(saffv)==3 and numpy.all(ssffv[0]==ssff[0]) and numpy.all(ssffv[1]==ssff[1]) and numpy.all(ssffv[2]==saffv[2][0])')  # when IC is a 1d array (single orbit), all outputs have one fewer dimension
    testCond('isinstance(sanot, tuple) and len(sanot)==5 and sanot[0].shape==(N,len(target0)) and sanot[1].shape==(N,len(target1)) and sanot[2].shape==(N,) and isinstance(sanot[2][0], agamaOrbit) and sanot[3].shape==(N,6) and sanot[4].shape==(N,2)')  # first two output items are matrices produced by Target objects, then comes the trajectory (represented by agama.Orbit, so timestamps are not given separately, despite the flag), then the array of dev.vectors, and finally the array of Lyapunov indicators (two per orbit)
    testCond('rnnf.shape==(0,2) and isinstance(snff, tuple) and len(snff)==2 and snff[0].shape==(0,S) and snff[1].shape==(0,S,6)')  # zero-length IC should still produce valid (but zero-length) outputs
    testCond('isinstance(rnnot, tuple) and len(rnnot)==5 and rnnot[0].shape==(0,len(target0)) and rnnot[1].shape==(0,len(target1)) and rnnot[2].shape==(0,) and rnnot[3].shape==(0,6) and rnnot[4].shape==(0,2)')
    testCond('isinstance(snfft, tuple) and len(snfft)==6 and snfft[0].shape==(0,len(target0)) and snfft[1].shape==(0,len(target1)) and snfft[2].shape==(0,S) and snfft[3].shape==(0,S,6) and snfft[4].shape==(0,6,S,6) and snfft[5].shape==(0,2)')

    # second part: run various accuracy tests
    ic  = numpy.array([1,0,0.5,0.,50.,0.])
    ok &= testAccuracy(pot, ic, -100, 1e-8,  50., ax[0])  # default accuracy
    ok &= testAccuracy(pot, ic, -100, 1e-10, 50., ax[1])  # higher  accuracy
    ok &= testDerivatives(pot, ic, 100, 1e-8, 50., ax[2])
    ok &= testHermite(pot, ic, 1000, 1e-4)
    ok &= testHermite(pot, ic, 1000, 1e-6)
    if plot:
        print('The numerically computed orbit accumulates energy error (green curve) and '
              'gradually deviates from the true orbit due to phase drift(blue curve).\n'
              'The difference between interpolated position provided by agama.Orbit and '
              'the same trajectory sampled at regular intervals of time (red points) '
              'stays at the same level (~10^-6 for the standard accuracy parameter), '
              'although the difference in velocity and hence in energy is about an order '
              'of magnitude higher (cyan points), because the velocity is computed as '
              'the derivative of the spline interpolator and carries a larger error.')
        print('In the right panel, we show the time evolution of the offset between '
              'two nearby orbits (solid lines), and the deviation vectors from the '
              'variational equation (dashed), which should match in the linear regime.')
        ax[0].legend(loc='lower right', ncol=2, numpoints=1)
        plt.tight_layout(h_pad=0.1)
        plt.show()
    else:
        print('Run the script with a non-empty command-line argument to show a plot')
    if ok:
        print("\033[1;32mALL TESTS PASSED\033[0m")
    else:
        print("\033[1;31mSOME TESTS FAILED\033[0m")
