/** \file    orbit.h
    \brief   Orbit integration
    \author  Eugene Vasiliev
    \date    2009-2017

    This module provides classes and routines for numerical computation of orbits in a given potential.
    This task is divided between three classes:
    - a generic ODE integrator;
    - the function that provides the r.h.s. of the ODE (time derivatives of position and velocity 
    in a particular coordinate system) for the given potential, optionally with some figure rotation;
    - functions that perform useful data collection tasks on the orbit while it is being computed.

    The first part is implemented by one of the available methods from math_ode.h,
    and the instance of an appropriate class derived from `math::BaseOdeSolver` is
    constructed internally for each orbit.
    The second part is implemented by any class derived from `math::IOdeSystem`, and this module
    provides such classes (`orbit::OrbitIntegrator`) for the three standard coordinate systems
    with time-independent potentials, which however may have a nonzero pattern speed.
    The third part is realized through a generic system of 'runtime functions', which are attached
    to the orbit and called after each timestep of the ODE integrator, so that they can access
    the trajectory at any point within the current timestep (obtain the interpolated solution)
    and use it to collect and store any relevant data. For instance, the `orbit::RuntimeTrajectory`
    class stores the trajectory at regular intervals in the appropriate coordinate system.
    The `orbit::integrate()` function binds together these constituents and computes the orbit
    starting with given initial conditions for a given interval of time, taking an arbitrary number
    of runtime functions; another convenience function `orbit::integrateTraj()` performs a simplified
    task of just recording the trajectory.
*/
#pragma once
#include "coord.h"
#include "math_ode.h"
#include "smart.h"
#include <vector>
#include <utility>

/** Orbit integration routines and classes */
namespace orbit {

/** Return value of a runtime function called during orbit integration */
enum StepResult {
    SR_CONTINUE,  ///< nothing happened, may continue orbit integration
    SR_TERMINATE, ///< orbit integration should be terminated
    SR_REINIT     ///< position/velocity has been changed, need to reinit
                  ///< the ODE solver before continuing the orbit integration
};

/** Interface for a runtime function that is called during orbit integration.
    This function is called after each internal timestep of the ODE solver;
    its return value determines what to do next (terminate or continue the integration,
    and in the latter case, whether the position/velocity has been modified by the function).
    There may be several such functions that are invoked sequentially;
    they may maintain their own internal state or accumulate some information
    during the orbit integration (the relevant method is non-const),
    either in a member variable or in a reference to an external variable.
*/
class BaseRuntimeFnc {
public:
    virtual ~BaseRuntimeFnc() {}

    /** Prototype of the runtime function.
        \param[in]  sol  is the instance of the ODE solver, which provides the method
        to compute the position/velocity at any time within the current timestep;
        \param[in]  tbegin  is the start of the current timestep;
        \param[in]  tend    is the end of the current timestep;
        \param[in,out]  vars  is the vector of 6 phase-space coordinates (position/velocity)
        at the end of the timestep;
        the function may modify this vector and return SR_REINIT, in which case
        the internal state of the ODE solver will be updated before the next timestep begins.
        \return  the status code determining how the orbit integration should proceed.
    */
    virtual StepResult processTimestep(
        const math::BaseOdeSolver& sol, const double tbegin, const double tend, double vars[]) = 0;
};

/** Shared pointer to a runtime function */
typedef shared_ptr<BaseRuntimeFnc> PtrRuntimeFnc;

/** Array of runtime functions attached to the given orbit */
typedef std::vector<PtrRuntimeFnc> RuntimeFncArray;


/** Runtime function that records the orbit trajectory either at every integration timestep,
    or at regular intervals of time (unrelated to the internal timestep of the orbit integrator).
    \tparam  CoordT  is the type of coordinate system used in orbit integration (Car, Cyl or Sph)
*/
template<typename CoordT>
class RuntimeTrajectory: public BaseRuntimeFnc {
    /// if positive, indicates the time interval between trajectory samples,
    /// otherwise samples are stored every timestep
    const double samplingInterval;

    /// sampled trajectory (position/velocity and time), stored in an external array
    /// referenced by this variable.
    /// last element of this array always contains the last recorded point in trajectory;
    /// if samplingInterval is zero, then the trajectory is stored at the end of every timestep,
    /// otherwise at regular intervals of time, and if samplingInterval is infinity, then only
    /// a single (last point) is recorded, otherwise 0th point contains the initial conditions.
    std::vector< std::pair<coord::PosVelT<CoordT>, double> >& trajectory;

    double t0;  ///< initial time (recorded at the beginning of the first timestep)

public:
    RuntimeTrajectory(double _samplingInterval,
        std::vector<std::pair<coord::PosVelT<CoordT>, double> >& _trajectory) :
        samplingInterval(_samplingInterval), trajectory(_trajectory), t0(NAN) {}

    virtual StepResult processTimestep(
        const math::BaseOdeSolver& sol, const double tbegin, const double tend, double vars[]);
};


/** The function that provides the RHS of the differential equation, i.e., the derivatives of
    position and velocity in different coordinate systems.
    \tparam  CoordT  is the type of coordinate system (Car, Cyl or Sph).
*/
template<typename CoordT>
class OrbitIntegrator: public math::IOdeSystem {
    /// gravitational potential in which the orbit is computed
    const potential::BasePotential& potential;
public:
    /// initialize the object for the given potential
    OrbitIntegrator(const potential::BasePotential& _potential) :
        potential(_potential) {};

    /// apply the equations of motion
    virtual void eval(const double t, const double x[], double dxdt[]) const;

    /// provide a tighter accuracy tolerance when |Epot| >> |Ekin+Epot|
    /// to improve the total energy conservation
    virtual double getAccuracyFactor(const double t, const double x[]) const;

    /// return the size of ODE system - three coordinates and three velocities
    virtual unsigned int size() const { return 6; }
};

/** The function providing the RHS of the differential equation in the cartesian coordinate system
    rotating about the z axis with a constant pattern speed Omega.
    The equations of motion assume that the position is given in the rotating frame,
    and the momentum (velocity) corresponds to the inertial frame. For instance, if the potential
    is z-axisymmetric, pattern rotation has no effect on the actual shape of the orbit,
    and numerically identical initial conditions would result in physically the same orbit
    regardless of Omega, but the recorded trajectory will read differently.
*/
class OrbitIntegratorRot: public math::IOdeSystem {
    /// gravitational potential in which the orbit is computed
    const potential::BasePotential& potential;
    /// angular frequency (pattern speed) of the rotating frame
    const double Omega;
public:
    /// initialize the object for the given potential and pattern speed
    OrbitIntegratorRot(const potential::BasePotential& _potential, double _Omega=0) :
        potential(_potential), Omega(_Omega) {};

    /// apply the equations of motion
    virtual void eval(const double t, const double x[], double dxdt[]) const;

    /// provide a tighter accuracy tolerance when |Epot| >> |Ekin+Epot|
    /// to improve the total energy conservation
    virtual double getAccuracyFactor(const double t, const double x[]) const;

    /// return the size of ODE system - three coordinates and three velocities
    virtual unsigned int size() const { return 6; }
};


/** Assorted parameters of orbit integration */
struct OrbitIntParams {
    //math::OdeSolverType solver;  ///< choice of the ODE integrator
    double accuracy;             ///< accuracy parameter for the ODE integrator
    size_t maxNumSteps;          ///< upper limit on the number of steps of the ODE integrator

    /// assign default values
    OrbitIntParams(double _accuracy=1e-8, size_t _maxNumSteps=1e8) :
        accuracy(_accuracy), maxNumSteps(_maxNumSteps) {}
};

/** Numerically compute an orbit in the specific coordinate system.
    This routine binds together the ODE solver (created internally),
    the orbit integrator (constructed for the specific potential and coordinate system),
    and optionally the runtime functions that collect information during orbit integration.
    \tparam     CoordT  is the type of coordinate system (Car, Cyl or Sph);
    \param[in]  initialConditions  is the initial position/velocity pair in the given 
                coordinate system (the same c.s. is used for orbit integration and for output);
    \param[in]  totalTime  is the maximum duration of orbit integration;
    \param[in]  orbitIntegrator  is the function that provides the r.h.s. of the differential equation;
                normally this would be an instance of `OrbitIntegrator<CoordT>`;
    \param[in]  runtimeFncs  is the list of runtime functions that are called after each
                internal timestep of the ODE integrator (for instance, to store the trajectory);
    \param[in]  params  are the extra parameters for the integration (default values may be used);
    \param[in]  startTime  is the initial time of the integration (default 0).
    \return     the end state of the orbit integration at the time when it is terminated
                (either startTime+totalTime or earlier -- the latter may occur if any of the runtime
                functions requested the integration to be terminated by returning SR_TERMINATE,
                or if the ODE solver returned an error (e.g. a zero or too small timestep),
                or the number of steps exceeded the limit).
    \throw      any possible exceptions from the ODE solver or the runtime functions.
*/
template<typename CoordT>
coord::PosVelT<CoordT> integrate(
    const coord::PosVelT<CoordT>& initialConditions,
    const double totalTime,
    const math::IOdeSystem& orbitIntegrator,
    const RuntimeFncArray&  runtimeFncs = RuntimeFncArray(),
    const OrbitIntParams&   params = OrbitIntParams(),
    const double startTime = 0);


/** A convenience function to compute the trajectory for the given initial conditions and potential
    in a specific coordinate system.
    \tparam     CoordT  is the coordinate system;
    \param[in]  initialConditions  is the initial position and velocity in the given coordinate system;
    \param[in]  totalTime  is the maximum duration of orbit integration;
    \param[in]  samplingInterval  is the time interval between successive point recorded
                from the trajectory; if zero, then the trajectory is recorded at every timestep
                of the orbit integrator, otherwise it is stored at these regular intervals;
    \param[in]  potential  is the gravitational potential in which the orbit is computed;
    \param[in]  params  are the extra parameters for the integration (default values may be used);
    \param[in]  startTime  is the initial time of the integration (default 0).
    \return     the recorded trajectory (0th point is the initial conditions) -
                an array of pairs of position/velocity points and associated moments of time.
    \throw      an exception if something goes wrong.
*/
template<typename CoordT>
inline std::vector<std::pair<coord::PosVelT<CoordT>, double> > integrateTraj(
    const coord::PosVelT<CoordT>& initialConditions,
    const double totalTime,
    const double samplingInterval,
    const potential::BasePotential& potential,
    const OrbitIntParams& params = OrbitIntParams(),
    const double startTime = 0)
{
    std::vector<std::pair<coord::PosVelT<CoordT>, double> > output;
    if(samplingInterval > 0)
        // reserve space for the trajectory, including one extra point for the final state
        output.reserve(totalTime * (1+1e-15) / samplingInterval + 1);
    integrate(initialConditions, totalTime,
        OrbitIntegrator<CoordT>(potential),
        RuntimeFncArray(1, PtrRuntimeFnc(new RuntimeTrajectory<CoordT>(samplingInterval, output))),
        params, startTime);
    return output;
}

}  // namespace