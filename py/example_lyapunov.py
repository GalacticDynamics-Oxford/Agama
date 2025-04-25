#!/usr/bin/python
"""
Illustration of the method for determining the (largest) Lyapunov characteristic exponent
and the timescale for the onset of chaos.
We first compute 10^3 orbits in a cuspy triaxial Dehnen potential, of which ~1/5 are chaotic.
These are plotted on the 2d plane of the normalized Lyapunov exponent (multiplied by
the orbital period Torb to make it dimensionless) vs. the timescale at which the chaos
(i.e. exponential growth of deviation vectors) starts to manifest itself, expressed
in units of orbital period (thus also dimensionless).
Upon clicking on any point in the left panel, the corresponding orbit will be shown
in the right panel, and the time evolution of ln(|w|/t), where |w| is the magnitude
of the largest deviation vector, will be shown in the central panel, together with
the fitting curve, from which the Lyapunov exponent and the chaos onset time
are determined.
"""
import numpy, agama, scipy.optimize, matplotlib.pyplot as plt, mpl_toolkits.mplot3d

def logexpm1overx(x):
    # compute F(x) = ln((exp(x)-1)/x) while avoiding over/underflows
    return numpy.where(abs(x)<1e-2, x * (0.5 + x * (1./24 - (1./2880) * x**2)),
           numpy.where(x<33, numpy.log(numpy.expm1(x)/x), x - numpy.log(x)))

def derlogexpm1overx(x):
    # compute the derivative of the above function while avoiding over/underflows
    return numpy.where(abs(x)<1e-2, 0.5 + x * (1./12 - (1./720) * x**2), -1/x - 1/numpy.expm1(-x))

# fit ln(w(t)) by the following function:
# A + ln(t)                                      if t< tch
# A + ln(t) + F(lambda * t) - F(lambda * tch)    if t>=tch
def curve(t, *params):
    Lyap = params[0]
    offset = params[1]
    tch = params[2]
    fnc = offset + numpy.log(t) + numpy.where(t<tch, 0, logexpm1overx(Lyap*t) - logexpm1overx(Lyap*tch))
    return fnc

def jac(t, *params):
    # jacobian of the above function w.r.t. its parameters
    Lyap = params[0]
    offset = params[1]
    tch = params[2]
    return numpy.column_stack((
        numpy.where(t<tch, tch*0, (t-tch) * derlogexpm1overx(Lyap*t)),
        numpy.ones(len(t)),
        numpy.where(t<tch, tch*0, -Lyap * derlogexpm1overx(Lyap*tch))
    ))

def getLyapunovExponent(times, logDeviationVector, orbitalPeriod, ax, color):
    maxTime = times[-1]
    MIN_BIN_SIZE = 5     #// minimum bin size for coarse-graining the input values
    minBinTime = max(maxTime * 0.01, orbitalPeriod)
    # step 1: coarse-grain the array of times & log(w)'s, replacing them with averages in bins
    t = []     # mean timestamp of each bin
    lnw = []   # ln(mean|w|) in each bin
    binStart = numpy.nan
    size = len(times)-1
    for i in range(size):
        ti = times[i];   vi=logDeviationVector[i]
        tp = times[i+1]; vp=logDeviationVector[i+1]
        if(binStart != binStart):
            binStart = ti
            sumDevVecBin = 0
            binSize = 0
        sumDevVecBin += (tp-ti) * numpy.exp(0.5 * (vi+vp))
        binSize += 1
        if(binSize >= MIN_BIN_SIZE and (tp >= binStart + minBinTime or i>=size-1)):
            # finish the current bin
            t.append(0.5 * (binStart + tp))
            lnw.append(numpy.log(sumDevVecBin / (tp - binStart)))
            binStart = numpy.nan
    t = numpy.array(t)
    lnw = numpy.array(lnw)
    ax.plot(t/orbitalPeriod, lnw - numpy.log(t), color, lw=1, zorder=4, label='smoothed data')
    # estimate the Lyapunov exponent neglecting the linear growth: ln w(t) ~= Lyap * t + const
    Lyap, offset = numpy.linalg.lstsq(numpy.column_stack((t, t*0+1)), lnw, rcond=0)[0]
    # arbitrarily set the starting guess of tch to half the total time
    tch = maxTime * 0.5
    # run nonlinear least-square fit (Levenberg-Marquardt)
    try:
        Lyap, offset, tch = scipy.optimize.curve_fit(curve, t, lnw, [Lyap, offset, tch], jac=jac)[0]
    except TypeError:  # 'jac' argument is not available in older versions of scipy
        Lyap, offset, tch = scipy.optimize.curve_fit(curve, t, lnw, [Lyap, offset, tch])[0]
    if tch < 0 and Lyap > 0:
        offset -= logexpm1overx(Lyap * tch)
        tch = 0
    # check how significant is the evidence for nonzero lambda:
    # compare the increase in ln(w) on this interval (tch..end) due to exponential growth
    # with the increase in ln(w) on the same interval expected from the linear growth alone
    delta_lnw = logexpm1overx(Lyap * t[-1]) - logexpm1overx(Lyap * tch)
    if Lyap < 0 or tch >= t[-2] or delta_lnw < numpy.log(2):
        tch = numpy.inf
        Lyap = 0  # no strong evidence for Lambda>0, assume linear growth of w(t)
    return Lyap, offset, tch

def onclick(event):
    which = numpy.where(use)[0][event.ind[0]]
    (time,traj), devvec, (lyap,tch) = agama.orbit(potential=pot, ic=ic[which], time=numPeriods*Torb[which],
        trajsize=0, der=True, lyapunov=True, dtype=float)
    time = time[1:]  # exclude 0th point (initial conditions)
    devvec = numpy.dstack(devvec)[1:]  # shape: len(times) * 6 (phase-space dimension) *  6 (num of vectors)
    # construct a linear superposition of six dev.vectors, equivalent to a fiducial vector
    # initialized with specific values given below (same as in agama c++ core)
    dvinit = numpy.array([(7./90)**0.5, (11./90)**0.5, (13./90)**0.5, (17./90)**0.5, (19./90)**0.5, (23./90)**0.5])
    lnw = numpy.log(numpy.linalg.norm(numpy.einsum('ijk,k->ij', devvec, dvinit), axis=1))
    ax1.cla()
    ax2.cla()
    Lyap, intercept, Tch = getLyapunovExponent(time, lnw, Torb[which], ax1, 'r')
    print('Normalized Lyapunov exponent (C++/Python implementations): %.10g/%.10g; '
        'normalized chaos onset time: %.10g/%.10g' % (lyap, Lyap*Torb[which], tch, Tch/Torb[which]) )
    ax1.plot(time/Torb[which], lnw - numpy.log(time), c='g', lw=0.5, label='original data', alpha=0.5)
    ax1.plot(time/Torb[which], curve(time, Lyap, intercept, Tch) - numpy.log(time), c='b', lw=1.5, dashes=[2,1], label='fitted curve')
    ax1.legend(loc='upper left', frameon=False)
    ax1.set_xlabel('time (orbital periods)')
    ax1.set_ylabel('log(deviation vector) - log(t)')
    ax1.set_xlim(0, numPeriods)
    ax2.plot(traj[:,0], traj[:,1], traj[:,2], color='b')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    fig.canvas.draw_idle()

pot = agama.Potential(type='spheroid', gamma=1.5, axisRatioY=0.8, axisRatioZ=0.6)
ic = pot.sample(1000, potential=pot)[0]
Torb = pot.Tcirc(ic)
numPeriods = 100
# compute an ensemble of orbits and their Lyapunov exponents using one dev.vector only
lyap = agama.orbit(potential=pot, ic=ic, time=numPeriods*Torb, lyapunov=True)
use = lyap[:,0] > 0
print('%i/%i orbits are chaotic\n' % (sum(use), len(ic)) +
'Left panel shows the chaotic orbits in the plane of Lyapunov exponent vs. chaos onset time Tch.\n'
'Click on a point to display the orbit and the evolution of the magnitude of deviation vectors |w|.\n'
'For a regular orbit, |w| grows linearly with time, while for a chaotic orbit it eventually starts '
'to grow exponentially (after some time Tch). The values of |w| recorded at every timestep are first '
'smoothed by binning, then fitted by a function that transitions from linear to exponential growth.\n'
'There are two variants of fits: one performed for the largest of six independent deviation vectors, '
'the other for just one fiducial vector (which is a linear combination of all six principal vectors); '
'the second approach is faster and is used when only the Lyapunov exponent is requested during orbit '
'integration, while the first approach is used when the orbit derivatives are requested in addition.\n'
'The two fits usually produce similar values of Lyapunov exponent and somewhat similar values of Tch, '
'but may sometimes differ considerably.')
fig = plt.figure(figsize=(18,6), dpi=75)
fig.canvas.mpl_connect('pick_event', onclick)
ax0 = plt.axes([0.04, 0.08, 0.30, 0.90])
ax1 = plt.axes([0.38, 0.08, 0.30, 0.90])
ax2 = plt.axes([0.69, 0.08, 0.30, 0.90], projection='3d')
ax0.scatter(lyap[use,0], lyap[use,1], s=3, picker=3, clip_on=False)
ax0.set_xlabel('normalized Lyapunov exponent')
ax0.set_ylabel('normalized chaos onset time (periods)')
ax0.set_xlim(0, 0.5)
ax0.set_ylim(0, numPeriods)
plt.show()
