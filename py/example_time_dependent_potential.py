#!/usr/bin/python

"""
Example of simple time-dependent potentials -- two point masses on a circular orbit,
representing the star and the planet. We compute orbits of test particles in this system,
i.e., solve the restricted three-body problem; in this example, we pick the initial conditions
corresponding to a horseshoe orbit (a coorbital motion with the planet with low-frequency
libration in angle).
We illustrate three alternative ways of creating time-dependent potentials:
1) fix the two point masses in space, but integrate the orbit in a uniformly rotating frame
   (of course, this is only possible if the orbit of the planet is circular);
2) let the two point masses move around on a pre-computed trajectory, and integrate the motion
   of the test particle in an inertial frame, but with a time-dependent potential;
3) same as above, but use a built-in potential of two moving point masses -- KeplerOrbit.
The third approach is more accurate than the second, because the orbit of the two massive
bodies is computed analytically, and is more general than the first one, because it can be
used even for an eccentric planet. However, the second approach is still more general
and can be used with an arbitrary number of moving potentials.
"""
import agama, numpy, os, matplotlib.pyplot as plt

ax = plt.subplots(1, 3, figsize=(15,5))[1]

au = 4.84814e-9  # 1 astronomical unit in kpc
# work in solar system units: 1 AU, 1 Msun, 1 km/s, 4.74 yr
print agama.setUnits(length=au, mass=1, velocity=1)

r0 = 1.         # AU
v0 = 29.784694  # km/s - orbital velocity of Earth
om = v0/r0      # Omega - angular frequency of rotation
tu = 4.7405     # years in one time unit
ic = [r0, 0, 0, 0, v0, 0]  # initial conditions for the orbit - at the other side of the Sun from Earth
tmax = 20

def convertToCorotatingFrame(t, o):
    return numpy.column_stack((
    o[:,0] * numpy.cos(om*t) + o[:,1] * numpy.sin(om*t),
    o[:,1] * numpy.cos(om*t) - o[:,0] * numpy.sin(om*t),
    o[:,2],
    o[:,3] * numpy.cos(om*t) + o[:,4] * numpy.sin(om*t),
    o[:,4] * numpy.cos(om*t) - o[:,3] * numpy.sin(om*t),
    o[:,5] ))

# variant 1: set up two fixed point masses (Sun and super-Earth of mass 0.001 Msun),
# and integrate the test-particle orbit in the rotating frame, in which Sun&Earth are stationary
p1 = agama.Potential(
    dict(type='plummer', mass=0.999, scaleradius=0, center="0.001,0,0"),  # Sun
    dict(type='plummer', mass=0.001, scaleradius=0, center="-.999,0,0"))  # Earth (very massive one)
t1,o1 = agama.orbit(potential=p1, ic=[r0,0,0,0,v0,0], time=tmax, trajsize=501, Omega=om)
E1 = p1.potential(o1[:,0:3]) + 0.5 * numpy.sum(o1[:,3:6]**2, axis=1) - (o1[:,0]*o1[:,4]-o1[:,1]*o1[:,3])*om
ax[0].plot(o1[:,0], o1[:,1])
ax[1].plot(t1*tu, o1[:,0], c='b')
ax[1].plot(t1*tu, o1[:,1], c='b', dashes=[3,2])
ax[2].plot(t1*tu, E1, label='Two fixed point masses in rotating frame')

# variant 2: same setup, but now make the two point masses move on circular orbits,
# and integrate the orbit of the test particle in the inertial frame
tt = numpy.linspace(0, tmax, 2001)
numpy.savetxt('tmp_center1', numpy.column_stack((tt, numpy.cos(om*tt)*0.001, numpy.sin(om*tt)*0.001, 0*tt)), '%g')
numpy.savetxt('tmp_center2', numpy.column_stack((tt, numpy.cos(om*tt)*-.999, numpy.sin(om*tt)*-.999, 0*tt)), '%g')
p2 = agama.Potential(
    dict(type='plummer', mass=0.999, scaleradius=0, center="tmp_center1"),
    dict(type='plummer', mass=0.001, scaleradius=0, center="tmp_center2"))
t2,o2 = agama.orbit(potential=p2, ic=[r0,0,0,0,v0,0], time=tmax, trajsize=501)
os.remove('tmp_center1')  # clean up
os.remove('tmp_center2')
o2 = convertToCorotatingFrame(t2, o2)   # transform the orbit back into the corotating frame
E2 = p2.potential(o2[:,0:3]) + 0.5 * numpy.sum(o2[:,3:6]**2, axis=1) - (o2[:,0]*o2[:,4]-o2[:,1]*o2[:,3])*om
ax[0].plot(o2[:,0], o2[:,1])
ax[1].plot(t2*tu, o2[:,0], c='g')
ax[1].plot(t2*tu, o2[:,1], c='g', dashes=[3,2])
ax[2].plot(t2*tu, E2, label='Two moving point masses in inertial frame')

# variant 3: instead of manually creating two point masses and putting them on a Kepler orbit,
# use a special type of potential: KeplerBinary
p3 = agama.Potential(type='KeplerBinary', mass=1, binary_sma=1.0, binary_q=1./999, binary_ecc=0)
t3,o3 = agama.orbit(potential=p3, ic=[r0,0,0,0,v0,0], time=tmax, trajsize=501)
o3 = convertToCorotatingFrame(t3, o3)   # transform the orbit back into the corotating frame
E3 = p3.potential(o3[:,0:3]) + 0.5 * numpy.sum(o3[:,3:6]**2, axis=1) - (o3[:,0]*o3[:,4]-o3[:,1]*o3[:,3])*om
ax[0].plot(o3[:,0], o3[:,1])
ax[1].plot(t3*tu, o3[:,0], c='r')
ax[1].plot(t3*tu, o3[:,1], c='r', dashes=[3,2])
ax[2].plot(t3*tu, E3, label='KeplerOrbit potential in inertial frame')

#print p2.potential(1,0,0,t=tmax), p3.potential(1,0,0,t=tmax)
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
ax[2].legend(loc='upper left', frameon=False, fontsize=10)

plt.tight_layout()
plt.show()
