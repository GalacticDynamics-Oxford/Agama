/** \file    orbit_variational.h
    \brief   Integration of the variational equation along the trajectory,
             and analysis of chaotic properties of orbits using the Lyapunov exponent
    \author  Eugene Vasiliev
    \date    2009-2024
*/
#pragma once
#include "orbit.h"
#include "math_linalg.h"

namespace orbit {

/** Estimate the (largest) Lyapunov exponent from the time series of the (log) of deviation 
    vector 'w' computed along the orbit.
    If the orbit is regular, |w| grows at most linearly with time, whereas if it is chaotic,
    then eventually |w| starts to grow exponentially, after some initial 'hidden' period.
    Thus for a regular orbit, the value of ln(|w|/t) should fluctuate around some constant,
    and for a chaotic orbit, ln(|w|) ~ const + ln(t) + Lambda * t.
    We determine whether the linear fit to ln(|w|/t) requires a positive value of Lambda,
    and return this value, normalized to the characteristic orbital time;
    if the fit appears to be good without the need for a linear term, the returned value is zero.
    \param[in]  logDeviationVector  is the array of pairs of (t, ln(|w|)) collected during
    orbit integration.
    \param[in]  orbitalPeriod  is the characteristic time of the orbit, used for normalizing
    the output value of the Lyapunov exponent to the characteristic orbital time
    (it may then be taken as a direct measure of chaoticity - values around unity indicate
    strongly chaotic orbits, and much less than unity - weakly chaotic).
    \return  the estimate of the normalized Lyapunov exponent, zero if the orbit seems to be
    regular, NAN if the length of the time series is insufficient to estimate the value.
*/
double calcLyapunovExponent(
    const std::vector< std::pair<double, double> >& logDeviationVector,
    const double orbitalPeriod);


/** Runtime function that integrates the variational equation describing the evolution of
    the deviation vector along the orbit, and estimates the [largest] Lyapunov exponent,
    which is an indicator of chaos.

    The 6d deviation vector w = {Dx, Dv} measures the distance between the orbit and another
    infinitesimally close neighbor orbit, and its evolution is described by the variational equation:
    \f$  d[Dx(t)] / dt =       Dv(t) + Omega \times Dx(t)  \f$,
    \f$  d[Dv(t)] / dt = -H(t) Dx(t) + Omega \times Dv(t)  \f$,
    where Omega is the pattern speed of the potential, and
    \f$  H_{ik}(t) = d^2\Phi[ x(t) ] / dx_i dx_k  \f$  is the Hessian of the potential,
    evaluated at the time-dependent point along the orbit.
    Hence the evolution of the deviation vector is described by a linear ODE with time-dependent
    coefficients, which are themselves functions of the original orbit.

    There are two alternative approaches for numerical solution of the deviation equation:
    - incorporate it together with the equations for the orbit itself into one ODE system with
    12 variables, and evolve it using the standard ODE solver;
    - follow the 6d orbit itself with the standard ODE solver, and use another specialized
    ODE solver for the variational equation, feeding it with the potential derivatives computed
    along the original orbit.
    The first approach is straightforward, but has a disadvantage that the original ODE system
    is modified (hence orbits computed with or without the variational equation will be different).
    In particular, it typically leads to shorter timesteps and hence longer integration time.
    The second approach, by contrast, is transparent for the orbit integrator, but requires
    a few more potential evaluations per timestep to compute the Hessian.
    This class implements the second approach, with the 6d variational equation transformed into
    three coupled second-order linear ODEs with variable coefficients.
*/
class RuntimeVariational: public BaseRuntimeFnc, public math::IOde2System {
    /// number of independent deviation vectors
    const unsigned int numVectors;

    /// the internal solver for the variational equation (different from the ODE solver for the orbit)
    math::Ode2SolverGL4<3> varEqSolver;  ///< - use the 8th-order method with 4 points per timestep

    /// time interval for storing the entire set of deviation vectors if they are requested
    const double samplingInterval;

    /// pointer to the external variable that will store the entire set of deviation vectors,
    /// if needed (setting it to NULL disables the storage, and only one vector will be followed)
    Trajectory* outputDeviationVectors;

    /// pointer to the external variable that will store the Lyapunov exponent
    /// when the orbit integration is finished (if NULL, it will not be computed)
    double* outputLyapunovExponent;

    /// storage for the logarithm of the norm of the deviation vector and corresponding timestamps
    std::vector< std::pair<double, double> > logDeviationVector;

    /// offset added to the logarithm of the deviation vector (if the latter grows exponentially,
    /// it is periodically renormalized to avoid overflow, and the offset in log is accumulated here)
    double addLogDevVec;

    /// estimate of the orbital period, used to normalize the [relative] Lyapunov exponent
    double orbitalPeriod;

    /// initial time (recorded at the beginning of the first timestep)
    double t0;

    /// temporary storage space for the full set of deviation vectors including orthogonalization
    math::Matrix<double> matQ, matR, matS;

    /// assemble all deviation vectors that are being followed (one or six) at the given time;
    /// if ind>=0, write them to the ind'th element of the output arrays (outputDeviationVectors).
    /// \return the logarithm of the norm of the largest deviation vector
    double storeSolution(double time, ptrdiff_t ind);

    /// when the deviation vector(s) become too large, renormalize their magnitude,
    /// and possibly orthogonalize the entire set of vectors
    void renormalize(double time);

public:
    /** construct the runtime function:
        \param[in]  orbint  is the orbit integrator that this function is attached to.
        \param[in]  samplingInterval  determines the frequency of recording the entire set of
        deviation vectors if they are requested (i.e. outputDeviationVectors != NULL);
        same convention as for the trajectory recording: 0 means store at every timestep,
        INFINITY means store only the last point, any other number - store at this regular interval.
        \param[in,out] outputDeviationVectors  is the optional pointer to an external variable
        that will store the evolution of all six deviation vectors; if NULL, they will not be stored.
        \param[in,out] outputLyapunovExponent  is the optional pointer to an external variable
        that will store the estimated largest Lyapunov exponent when the orbit integration
        is finished and this object is destroyed; if NULL, this estimate will be skipped.
    */
    RuntimeVariational(BaseOrbitIntegrator& orbint,
        double samplingInterval,
        Trajectory outputDeviationVectors[6] = NULL,
        double* outputLyapunovExponent = NULL);

    /** estimate the Lyapunov exponent from the data stored during orbit integration,
        and store it in the external variable 'outputLyapunovExponent' */
    ~RuntimeVariational() {
        if(outputLyapunovExponent)
            *outputLyapunovExponent = calcLyapunovExponent(logDeviationVector, orbitalPeriod);
    }

    /** follow the evolution of the deviation vector during one timestep and record its magnitude */
    virtual bool processTimestep(double tbegin, double tend);

    /// implements the IOde2System interface (needed for the internal variational equation solver) 
    virtual void eval(const double t, double a[], double b[]) const;

    /// size of the deviation vector (3 positions and 3 velocities)
    virtual unsigned int size() const { return 6; }
};

}  // namespace