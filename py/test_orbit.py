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
import numpy, sys, time
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
    orb, der = agama.orbit(potential=pot, ic=[ic, ic+delta_ic], time=time, Omega=Omega,
        trajsize=1000, dtype=float, accuracy=accuracy, der=True, verbose=False)
    times = orb[0,0]
    jac = numpy.dstack(der[0])  # size: (trajsize,6,6); jac[i] is the Jacobian matrix at time t[i]
    linear_dif = jac.dot(delta_ic)
    actual_dif = orb[1,1] - orb[0,1]
    max_offset = numpy.max(abs(actual_dif[:,0:3]))
    max_error  = numpy.max(abs(actual_dif - linear_dif)[:,0:3])
    print('Comparison of a slightly perturbed orbit with the deviation vectors from the variational equation:')
    print('difference in position between original and perturbed orbit: %s' % check(max_offset, 1e-4))
    print('difference between the perturbed orbit and the linear prediction: %s' % check(max_error, 1e-8))
    if ax is not None:
        ax.plot(times, actual_dif[:,0:3])
        ax.plot(times, linear_dif[:,0:3], dashes=[2,4], lw=2)
        ax.set_xlim(0, time)
        ax.set_xlabel('time')
        ax.set_ylabel('deviations from initial orbit')
        ax.text(0.02, 0.98, 'solid - difference between nearby orbits\ndotted - deviation vectors',
            ha='left', va='top', transform=ax.transAxes)
    return bool(check)

if __name__ == '__main__':
    plot = len(sys.argv)>1
    if plot:
        import matplotlib.pyplot as plt
        ax = plt.subplots(1, 3, figsize=(18,6))[1]
    else:
        ax = None, None, None
    agama.setUnits(length=1, velocity=1, mass=1)
    numpy.random.seed(42)
    pot = agama.Potential(type='spheroid', mass=1e10, scaleradius=1, gamma=1, beta=4, alpha=1, p=0.7, q=0.4)
    ic  = numpy.array([1,0,0.5,0.,50.,0.])
    ok  = testAccuracy(pot, ic, -100, 1e-8,  50., ax[0])  # default accuracy
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
        plt.tight_layout()
        plt.show()
    else:
        print('Run the script with a non-empty command-line argument to show a plot')
    if ok:
        print("\033[1;32mALL TESTS PASSED\033[0m")
    else:
        print("\033[1;31mSOME TESTS FAILED\033[0m")
