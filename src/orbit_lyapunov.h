/** \file    orbit_lyapunov.h
    \brief   Analysis of chaotic properties of orbits using the Lyapunov exponent
    \author  Eugene Vasiliev
    \date    2009-2018
*/
#pragma once
#include "orbit.h"

namespace orbit {

/** Estimate the (largest) Lyapunov exponent from the time series of the (log) of deviation vector 'w'
    computed along the orbit.
    If the orbit is regular, |w| grows at most linearly with time, whereas if it is chaotic,
    then eventually |w| starts to grow exponentially, after some initial 'hidden' period.
    We keep track of the running average of |w|/t, and when it exceeds some threshold,
    this is taken to be the transition to the exponential growth regime.
    From that point on, we collect the average of ln(|w|)/t as a proxy for the Lyapunov exponent,
    and return the value normalized to the characteristic time (orbital period).
    If no such transition was detected, then the return value is zero.
    \param[in]  logDeviationVector  is the array of collected values of ln(|w|) during orbit integration;
    \param[in]  samplingInterval  is the time between consecutive values;
    \param[in]  orbitalPeriod  is the characteristic time of the orbit, needed both to define
    a suitable interval for the running average of ln(|w|), because it exhibits large fluctuations
    over the timescale of orbital period, and for normalizing the output value of the Lyapunov exponent
    to the characteristic orbital time (it may then be taken as a direct measure of chaoticity -
    values around unity indicate strongly chaotic orbits, and much less than unity - weakly chaotic).
    \return  the estimate of the normalized Lyapunov exponent, or zero if the orbit seems to be regular.
*/
double calcLyapunovExponent(const std::vector<double>& logDeviationVector,
    const double samplingInterval, const double orbitalPeriod);


/** Runtime function that monitors the evolution of the deviation vector along the orbit,
    and estimates the [largest] Lyapunov exponent, which is an indicator of chaos.

    The 6d deviation vector w = {Dx, Dv} measures the distance between the orbit and another
    infinitesimally close neighbor orbit, and its evolution is described by the variational equation:
    \f$  d[Dx(t)] / dt =       Dv(t) + Omega \times Dx(t)  \f$,
    \f$  d[Dv(t)] / dt = -H(t) Dx(t) + Omega \times Dv(t)  \f$,
    where Omega is the pattern speed of the potential, and
    \f$  H_{ik}(t) = d^2\Phi[ x(t) ] / dx_i dx_k  \f$  is the Hessian of the potential,
    evaluated at the time-dependent point along the orbit.
    Hence the evolution of the deviation vector is described by a linear ODE with time-dependent
    coefficients, which are themselves functions of the original orbit.

    We have two alternative approaches for numerical solution of the deviation equation:
    - incorporate it together with the equations for the orbit itself into one ODE system with
    12 variables, and evolve it using the standard ODE solver;
    - follow the 6d orbit itself with the standard ODE solver, and use another specialized ODE solver
    for the variational equation, feeding it with the potential derivatives computed along the original
    orbit.
    The first approach is straightforward, but has a disadvantage that the original ODE system
    is modified (hence orbits computed with or without the variational equation will be different).
    In particular, it typically leads to shorter timesteps and hence longer integration time.
    The second approach, by contrast, is transparent for the orbit integrator, but requires a few more
    potential evaluations per timestep to compute the Hessian; in the present implementation,
    it transforms the 6d variational equation into three second-order ODEs, but this is not possible
    in a rotating frame.
    Both approaches are implemented in this class, the choice is controlled by the template parameter.
    When using the first approach, the orbit integrator must also be changed to `OrbitIntegratorVarEq`
    or a similar one providing the variational equation for the second half of the 12d solution vector.

    \tparam  UseInternalVarEqSolver determines the approach for dealing with variational equation:
    if false, it must be solved by the orbit integrator (providing a 12d solution vector),
    if true,  an internal ODE integrator for the var.eq. is used with any orbit integrator.
*/
template<bool UseInternalVarEqSolver>
class RuntimeLyapunov: public BaseRuntimeFnc, public math::IOde2System {
    /// the internal solver for the variational equation (different from the ODE solver for the orbit)
    math::Ode2SolverGL4<3> varEqSolver;  ///< - use the 8th-order method with 4 points per timestep

    /// the ODE solver for the orbit, providing the position as a function of time
    const math::BaseOdeSolver* orbitSolver;

    /// the potential in which the orbit is computed (its Hessian is used in the variational equation)
    const potential::BasePotential& potential;

    /// time interval between recorded samples of the log-magnitude of the deviation vector
    const double samplingInterval;

    // reference to the external variable that will store the Lyapunov exponent
    // when the orbit integration is finished
    double& outputLyapunovExponent;

    /// storage for the logarithm of the magnitude of the deviation vector,
    /// sampled at regular intervals of time (used if an external storage array is not provided)
    std::vector<double>  logDeviationVectorInternal;

    /// reference to either the internal or the external storage array
    std::vector<double>& logDeviationVector;

    /// offset added to the logarithm of the deviation vector (if the latter grows exponentially,
    /// it is periodically renormalized to avoid overflow, and the offset in log is accumulated here)
    double addLogDevVec;

    /// estimate of the orbital period, used to normalize the [relative] Lyapunov exponent
    double orbitalPeriod;

    /// initial time (recorded at the beginning of the first timestep)
    double t0;

public:
    /** construct the runtime function:
        \param[in]  potential  is the instance of potential (same as used in the orbit integration);
        \param[in]  samplingInterval  determines the frequency of recording the magnitude
        of the deviation vector (should be ~ 0.1 times the orbital period);
        \param[in,out] outputLyapunovExponent  is the reference to an external variable that will
        store the estimated largest Lyapunov exponent when the orbit integration is finished and
        this object is destroyed;
        \param[in,out] outputLogDeviationVector  is the optional pointer to an external variable
        that will store the logarithm of the magnitude of the recorded deviation vector, sampled
        at equal intervals; if not provided, this information is stored and used internally and
        is not accessible from outside.
    */
    RuntimeLyapunov(const potential::BasePotential& potential, double samplingInterval,
        double& outputLyapunovExponent, std::vector<double>* outputLogDeviationVector = NULL);

    /** estimate the Lyapunov exponent from the data stored during orbit integration,
        and store it in the external variable 'outputLyapunovExponent' */
    ~RuntimeLyapunov() {
        outputLyapunovExponent = calcLyapunovExponent(logDeviationVector, samplingInterval, orbitalPeriod);
    }

    /** follow the evolution of the deviation vector during one timestep and record its magnitude:
        if UseInternalVarEqSolver = true, this evolution is performed by this routine,
        otherwise it was done by the orbit integrator, and this routine only occasionally rescales
        the dev.vector to avoid overflows (returning SR_REINIT), otherwise returns SR_CONTINUE as usual. */
    virtual StepResult processTimestep(
        const math::BaseOdeSolver& sol, const double tbegin, const double tend, double vars[]);

    /// implements the IOde2System interface (needed for the internal variational equation solver) 
    virtual void eval(const double t, double mat[]) const;
    virtual unsigned int size() const { return 6; }
};


/** The function providing the RHS of the differential equation dX/dt = f(X)
    for the orbit computed in the given potential, optionally rotating about the z axis
    with a constant pattern speed Omega, and one deviation vector computed along with the orbit.
    X is a 12-dimensional vector:
    first 6 components are the ordinary position and velocity in cartesian coordinates,
    and the rest are the corresponding components of the deviation vector,
    which measures the distance between the orbit and its infinitesimally close counterpart.
    The evolution of the deviation vector is described by the variational equation,
    which contains second derivatives of the potential computed along the original orbit.
    This class may be provided to the `orbit::integrate()` routine along with an instance of
    a RuntimeLyapunov runtime function that analyzes the evolution of the deviation vector.
*/
class OrbitIntegratorVarEq: public math::IOdeSystem {
    /// gravitational potential in which the orbit is computed
    const potential::BasePotential& potential;
    /// angular frequency (pattern speed) of the rotating frame
    const double Omega;
public:
    OrbitIntegratorVarEq(const potential::BasePotential& _potential, double _Omega=0) :
        potential(_potential), Omega(_Omega) {}

    /// compute the time derivative of the position/velocity vector and the deviation vector at time t
    virtual void eval(const double t, const double x[], double dxdt[]) const;

    /** The size of the position/velocity vector plus the deviation vector */
    virtual unsigned int size() const { return 12; }
};

}  // namespace