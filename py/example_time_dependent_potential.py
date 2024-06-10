#!/usr/bin/python

"""
Example of simple time-dependent potentials -- two point masses on a circular orbit,
representing the star and the planet. We compute orbits of test particles in this system,
i.e., solve the restricted three-body problem; in this example, we pick the initial conditions
corresponding to a horseshoe orbit (a coorbital motion with the planet with low-frequency
libration in angle).
We illustrate four alternative ways of creating time-dependent potentials:
1) fix the two point masses in space, but integrate the orbit in a uniformly rotating frame
   (of course, this is only possible if the orbit of the planet is circular);
2) same as above, but make the potential itself rotating, and integrate the orbit
   in the original inertial frame;
3) let the two point masses move around on a pre-computed trajectory, and integrate the motion
   of the test particle in an inertial frame, but with a time-dependent potential;
4) same as above, but use a built-in potential of two moving point masses -- KeplerOrbit.
The second approach is identical to the first one in this case, but can be used more generally
with a non-uniformly rotating potential.
The fourth approach is more accurate than the third, because the orbit of the two massive
bodies is computed analytically, and is more general than the first one for this case,
because it can be used even for an eccentric planet.
However, the third approach is even more general, and can be used with an arbitrary number
of potentials (not necessarily point masses) moving on different trajectories.
On the other hand, the second approach can create non-uniformly rotating arbitrary potentials.
"""
import agama, numpy, matplotlib.pyplot as plt

plt.rc('font', size=10)
ax = plt.subplots(1, 3, figsize=(12,4), dpi=100)[1]

au = 4.84814e-9  # 1 astronomical unit in kpc
# work in solar system units: 1 AU, 1 Msun, 1 km/s, 4.74 yr
agama.setUnits(length=au, mass=1, velocity=1)

r0 = 1.            # AU
v0 = agama.G**0.5  # ~30 km/s - orbital velocity of Earth
om = v0/r0         # Omega - angular frequency of rotation
tu = om/2/numpy.pi # years in one time unit
ic = [r0, 0, 0, 0, v0, 0]  # initial conditions for the orbit - at the other side of the Sun from Earth
tmax = 20

# in cases the orbit is integrated in an inertial frame with a rotating potential,
# we need to transform it back to the corotating frame, in which the potential is stationary
def convertToCorotatingFrame(t, o):
    return numpy.column_stack((
    o[:,0] * numpy.cos(om*t) + o[:,1] * numpy.sin(om*t),
    o[:,1] * numpy.cos(om*t) - o[:,0] * numpy.sin(om*t),
    o[:,2],
    o[:,3] * numpy.cos(om*t) + o[:,4] * numpy.sin(om*t),
    o[:,4] * numpy.cos(om*t) - o[:,3] * numpy.sin(om*t),
    o[:,5] ))

# compute the Jacobi energy, which should be conserved along the orbit
# (note that here we evaluate the potential at t=0 for all points in the trajectory,
# because the orbit is represented in a rotating frame where the potential is stationary)
def JacobiEnergy(pot, orb):
    return pot.potential(orb[:,0:3]) + 0.5 * numpy.sum(orb[:,3:6]**2, axis=1) - (orb[:,0]*orb[:,4]-orb[:,1]*orb[:,3])*om

# variant 1: set up two fixed point masses (Sun and super-Earth of mass 0.001 Msun),
# and integrate the test-particle orbit in the rotating frame, in which Sun&Earth are stationary.
# the potential is initialized as two components with different 'modifier' parameters
# (in this case, center=... creates a Shifted modifier on top of the potential,
# and we provide just a triplet of numbers to establish a constant shift;
# it could be a string or a list/array).
p1 = agama.Potential(
    dict(type='plummer', mass=0.999, scaleradius=0, center="0.001,0,0"),  # Sun
    dict(type='plummer', mass=0.001, scaleradius=0, center=[-.999,0,0]))  # Earth (a very massive one)
t1, o1 = agama.orbit(potential=p1, ic=[r0,0,0,0,v0,0], time=tmax, trajsize=501, Omega=om, dtype=float)
E1 = JacobiEnergy(p1, o1)
ax[0].plot(o1[:,0], o1[:,1], c='b')
ax[1].plot(t1*tu, o1[:,0], c='b')
ax[1].plot(t1*tu, o1[:,1], c='b', dashes=[3,2])
ax[2].plot(t1*tu, E1, c='b', label='Two fixed point masses in rotating frame')

# variant 2: same setup, but now make the potential of the two point masses rotate
# in the inertial frame, and integrate the orbit of the test particle in this frame.
# If the original potential is a simple built-in model (e.g., type='Plummer'),
# one could construct both the potential and the modifier in one expression:
# p2 = agama.Potential(type=..., mass=..., rotation=...)
# if the potential consists of several components with the same modifier parameters,
# these components will be merged into a single Composite potential and then
# a common modifier created on top of it.
# However, in our case this doesn't apply because
# (a) the offsets are different, and
# (b) if multiple modifiers are applied to a single potential, their implied order is
# first rotation, then tilt, then shift - in other words, two point masses will be
# made rotating individually (which doesn't make physical sense for them) and then
# shifted by a constant offset each, but this does not create a two-body orbit.
# Therefore, we use the most general approach: first create a composite potential
# of two fixed point masses, both shifted from origin (reuse the potential from variant 1),
# and then wrap it into a 'Rotating' modifier by providing an array with the rotation angle
# as a function of time (changing linearly in this case, so we only need two timestamps,
# and it is extrapolated linearly beyond the end of the specified time interval).
p2 = agama.Potential(potential=p1, rotation=[[0, 0], [1, om]])
# integrate the orbit of a test particle in the inertial frame, but using a rotating potential
t2, o2 = agama.orbit(potential=p2, ic=[r0,0,0,0,v0,0], time=tmax, trajsize=501, dtype=float)
o2 = convertToCorotatingFrame(t2, o2)   # transform the orbit back into the corotating frame
E2 = JacobiEnergy(p2, o2)
ax[0].plot(o2[:,0], o2[:,1], c='g')
ax[1].plot(t2*tu, o2[:,0], c='g')
ax[1].plot(t2*tu, o2[:,1], c='g', dashes=[3,2])
ax[2].plot(t2*tu, E2, c='g', label='Two rotating point masses in inertial frame')

# variant 3: same setup, but now make the two point masses move on pre-computed circular orbits,
# and integrate the orbit of the test particle in the inertial frame
tt = numpy.linspace(0, tmax*1.01, 6000)
sinomt, cosomt = numpy.sin(om*tt), numpy.cos(om*tt)
center1 = numpy.column_stack((tt, cosomt*0.001, sinomt*0.001, 0*tt, -om*sinomt*0.001, om*cosomt*0.001, 0*tt))
center2 = numpy.column_stack((tt, cosomt*-.999, sinomt*-.999, 0*tt, -om*sinomt*-.999, om*cosomt*-.999, 0*tt))
# this time we create a composite potential in which each component has a different time-dependent shift
p3 = agama.Potential(
    dict(type='plummer', mass=0.999, scaleradius=0, center=center1),
    dict(type='plummer', mass=0.001, scaleradius=0, center=center2) )
t3,o3 = agama.orbit(potential=p3, ic=[r0,0,0,0,v0,0], time=tmax, trajsize=501, dtype=float)
o3 = convertToCorotatingFrame(t3, o3)   # transform the orbit back into the corotating frame
E3 = JacobiEnergy(p3, o3)
ax[0].plot(o3[:,0], o3[:,1], c='r')
ax[1].plot(t3*tu, o3[:,0], c='r')
ax[1].plot(t3*tu, o3[:,1], c='r', dashes=[3,2])
ax[2].plot(t3*tu, E3, c='r', label='Two moving point masses in inertial frame')

# variant 4: instead of manually creating two point masses and putting them on a Kepler orbit,
# use a special type of potential, which does just that without any modifiers: KeplerBinary
p4 = agama.Potential(type='KeplerBinary', mass=1, binary_sma=1.0, binary_q=1./999, binary_ecc=0)
t4,o4 = agama.orbit(potential=p4, ic=[r0,0,0,0,v0,0], time=tmax, trajsize=501, dtype=float)
o4 = convertToCorotatingFrame(t4, o4)   # transform the orbit back into the corotating frame
E4 = JacobiEnergy(p4, o4)
ax[0].plot(o4[:,0], o4[:,1], c='y')
ax[1].plot(t4*tu, o4[:,0], c='y')
ax[1].plot(t4*tu, o4[:,1], c='y', dashes=[3,2])
ax[2].plot(t4*tu, E4, c='y', label='KeplerOrbit potential in inertial frame')
# adorn the plot with the symbols representing the Sun and the Planet
ax[0].plot(0.001, 0, 'o', ms=12, c='k', markerfacecolor='none')
ax[0].plot(0.001, 0, 'o', ms=2, c='k')
ax[0].plot(-.999, 0, 'o', ms=10, c='k', markerfacecolor='none')
ax[0].plot(-.999, 0, '+', ms=9, c='k')
ax[0].set_xlim(-1.1, 1.1)
ax[0].set_ylim(-1.1, 1.1)
ax[0].set_xlabel('X [au]')
ax[0].set_ylabel('Y [au]')
ax[1].set_xlabel('time [yr]')
ax[1].set_ylabel('X (solid), Y (dashed) [au]')
ax[2].set_xlabel('time [yr]')
ax[2].set_ylabel('Jacobi energy [km/s]^2')
ax[2].legend(loc='upper left', frameon=False, fontsize=8)

print('Potentials used in the orbit integration:\n%s\n%s\n%s\n%s' % (p1, p2, p3, p4))
plt.tight_layout()
plt.show()
