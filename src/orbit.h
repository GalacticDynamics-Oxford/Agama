/** \file    orbit.h
    \brief   Orbit integration
    \author  Eugene Vasiliev
    \date    2009-2025

    This module provides classes and routines for numerical computation of orbits in a given potential.
    This task is divided between three classes:
    - a general-purpose ODE integrator;
    - the function that provides the r.h.s. of the ODE (time derivatives of position and velocity 
    in a particular coordinate system) for the given potential, optionally with some figure rotation;
    - functions that perform useful data collection tasks on the orbit while it is being computed.

    The first part is implemented by one of the available methods from math_ode.h,
    and the instance of an appropriate class derived from `math::BaseOdeStepper` is
    constructed internally for each orbit.
    The second part is implemented by any class derived from `math::IOdeSystem`.
    The third part is realized through a generic system of 'runtime functions', which are attached
    to the orbit and called after each timestep of the ODE integrator, so that they can access
    the trajectory at any point within the current timestep (obtain the interpolated solution)
    and use it to collect and store any relevant data. For instance, the `orbit::RuntimeTrajectory`
    class stores the trajectory at regular intervals. These functions can also influence the
    orbit integration by rewriting the orbit state with the new position/velocity.
    To isolate the data collection from the implementation of the orbit integrator (in particular,
    the choice of its internal coordinate system), the runtime functions do not interact directly with
    the ODE solver, but rather with an interface provided by the class `orbit::BaseOrbitIntegrator`,
    which is the base class for the three variants of the class `orbit::OrbitIntegrator`
    for different coordinate systems. This class, and in particular its method `run()`,
    is the actual entry point for computing an orbit in the given potential, optionally rotating
    about the z axis with some pattern speed Omega, while any number of attached runtime functions
    performing data collection tasks. Another convenience function `orbit::integrateTraj()`
    performs a simplified task of just recording the trajectory.
*/
#pragma once
#include "smart.h"
#include "potential_base.h"
#include "math_ode.h"
#include <vector>
#include <utility>

/** Orbit integration routines and classes */
namespace orbit {

// forward declaration
class BaseOrbitIntegrator;

/** Interface for a runtime function that is called during orbit integration.
    Any number of such functions can be attached to an instance of the BaseOrbitIntegrator class
    as a "plugin system", and they are called sequentially after each internal timestep of the 
    integrator; the return value of the `processTimestep()` method determines what to do next
    (terminate or continue the integration), and it may update the current orbit state if needed
    by calling the `init()` method of the host orbit integrator.
    A runtime function may maintain its own internal state or accumulate some information
    during the orbit integration (the relevant method is non-const),
    either in a member variable or in a reference to an external variable,
    and perform some finalization duties in the destructor.
*/
class BaseRuntimeFnc {
protected:
    /// instance of the orbit integrator to which this function is attached
    BaseOrbitIntegrator& orbint;
public:
    /// the constructor stores the reference to the orbit integrator that this function
    /// is attached to, but the calling code must ensure that the reciprocal method
    /// `BaseOrbitIntegrator.addRuntimeFunction()` is also called with this function as an argument
    explicit BaseRuntimeFnc(BaseOrbitIntegrator& _orbint) : orbint(_orbint) {}

    /// some finalization work may be performed in the destructor after the orbit is completed
    virtual ~BaseRuntimeFnc() {}

    /** Prototype of the runtime function called after each timestep of the orbit integrator.
        \param[in]  tbegin  is the start time of the current timestep.
        \param[in]  timestep  is the length of the current timestep; can have either sign.
        \return  false if the integration must be terminated, or true if it may continue.
    */
    virtual bool processTimestep(double tbegin, double timestep) = 0;
};

/** Shared pointer to a runtime function */
typedef shared_ptr<BaseRuntimeFnc> PtrRuntimeFnc;

/** Array of pairs: position/velocity in cartesian coordinates and timestamp */
typedef std::vector< std::pair<coord::PosVelCar, double> > Trajectory;

/** Runtime function that records the orbit trajectory either at every integration timestep,
    or at regular intervals of time (unrelated to the internal timestep of the orbit integrator).
*/
class RuntimeTrajectory: public BaseRuntimeFnc {
    /// if positive, indicates the time interval between trajectory samples,
    /// otherwise samples are stored every timestep
    const double samplingInterval;

    /// sampled trajectory (cartesian position/velocity and time), stored in an external array
    /// referenced by this variable.
    /// Last element of this array always contains the last recorded point in trajectory;
    /// if samplingInterval is zero, then the trajectory is stored at the end of every timestep,
    /// otherwise at regular intervals of time, and if samplingInterval is infinity, then only
    /// a single (last point) is recorded, otherwise 0th point contains the initial conditions.
    Trajectory& trajectory;

    double t0;  ///< initial time (recorded at the beginning of the first timestep)

public:
    RuntimeTrajectory(BaseOrbitIntegrator& orbint,
        double _samplingInterval, Trajectory& _trajectory)
    :
        BaseRuntimeFnc(orbint), samplingInterval(_samplingInterval), trajectory(_trajectory), t0(NAN)
    {
        trajectory.clear();
    }

    virtual bool processTimestep(double tbegin, double timestep);
};


/** Assorted parameters of orbit integration */
struct OrbitIntParams {
    enum Method {
        DOP853,
        DPRKN8,
        HERMITE
    };
    double accuracy;     ///< accuracy parameter for the ODE integrator
    size_t maxNumSteps;  ///< upper limit on the number of steps of the ODE integrator
    Method method;       ///< choice of the ODE integrator

    /// assign default values
    OrbitIntParams(double _accuracy=1e-8, size_t _maxNumSteps=1e8, Method _method=DOP853) :
        accuracy(_accuracy), maxNumSteps(_maxNumSteps), method(_method) {}
};

/** Interface for the orbit integrator in the given potential, optionally in a reference frame
    rotating about the z axis with angular frequency Omega.
    This class serves as a bridge between the actual ODE integrator, which performs its job in
    some internal coordinate system, and the runtime functions, which retrieve or update 
    the position/velocity in cartesian coordinates.
    The actual implementations in different internal coordinate systems are provided by
    the derived classes OrbitIntegrator<CoordT>.
    The connection between the orbit integrator interface and the associated runtime functions is
    bi-directional: the integrator keeps track of the runtime functions attached to the given orbit,
    and each runtime function stores the reference to the orbit that it is attached to.
    Since the functions can change their internal state during orbit integration, and also may
    change the state of the orbit itself (its current 6d phase-space coordinates), these pointers
    and references are non-const, as well as the relevant methods of both class interfaces.
    The workflow for orbit computation is the following:
    1) Construct the orbit integrator instance.
    2) Optionally create some runtime functions and add them to the list of functions associated
    with the integrator object with the `addRuntimeFnc()` method;
    it is the responsibility of the caller to provide the reference to the same integrator object
    to the constructor of each runtime function (otherwise the workflow will be disrupted).
    3) Set up orbit initial conditions and time by calling `init()` method of the integrator.
    4) Evolve the orbit for the desired amount of time by calling `run()` method of the integrator.
    It will perform zero or more steps of the internal ODE solver, and call each runtime function's
    `processTimestep()` method after each completed timestep. These methods may, in turn, change
    the current state of the orbit by calling the `init()` method of the integrator.
    The `run()` method of a given orbit instance may be called more than once; the orbit state
    is preserved between runs, unless reset by `init()`, but this should be used only to continue
    the integration of the same orbit, not to start a new one (there is no mechanism to inform
    the runtime functions that a new orbit is now computed - they are single-use objects).
    5) The runtime functions may perform some finalization tasks once the orbit is completed,
    but they would not know that this is the case until their destructor is called.
    So the standard approach should be to construct the orbit integrator in a nested scope block,
    populate it with newly constructed runtime functions that do not have any external references,
    run the orbit, and then close the nested scope block to ensure correct finalization.
*/
class BaseOrbitIntegrator: public math::IOdeSystem2ndOrder {
    const size_t maxNumSteps;        ///< maximum allowed number of integration steps
    std::vector<PtrRuntimeFnc> fncs; ///< list of runtime functions attached to the given orbit
protected:
    shared_ptr<math::BaseOdeStepper> stepper;  ///< the actual ODE integrator
    double timeBegin;  ///< time at the beginning of the current timestep
public:
    /// gravitational potential in which the orbit is computed (accessible to runtime functions)
    const potential::BasePotential& potential;

    /// angular frequency (pattern speed) of the rotating frame (accessible to runtime functions)
    const double Omega;

    /// initialize the object for the given potential, pattern speed, and other integration params
    BaseOrbitIntegrator(const potential::BasePotential& _potential, double _Omega,
        const OrbitIntParams& params)
    :
        maxNumSteps(params.maxNumSteps),
        timeBegin(0),
        potential(_potential), Omega(_Omega)
    {
        switch(params.method) {
            case OrbitIntParams::DOP853:
                stepper.reset(new math::OdeStepperDOP853 (*this, params.accuracy)); break;
            case OrbitIntParams::DPRKN8:
                stepper.reset(new math::OdeStepperDPRKN8 (*this, params.accuracy)); break;
            case OrbitIntParams::HERMITE:
                stepper.reset(new math::OdeStepperHermite(*this, params.accuracy)); break;
        }
    }

    virtual ~BaseOrbitIntegrator() {}

    /** Numerically compute an orbit in the specific coordinate system, the main method of this class.
        The initial state should be assigned beforehand by calling `init()`;
        this function may be called more than once if need to continue integrating the same orbit.
        \param[in]  totalTime  is the maximum duration of orbit integration (positive or negative);
        \return     the end state of the orbit integration at the time when it is terminated --
                    after totalTime has elapsed under normal circumstances, or earlier if any of the
                    runtime functions requested the integration to be terminated by returning false,
                    or if the ODE solver returned an error (e.g. a zero or too small timestep),
                    or the number of steps exceeded the limit).
        \throw      any possible exceptions from the ODE solver or the runtime functions.
    */
    coord::PosVelCar run(double totalTime);

    /// add a new item to the list of runtime functions called at each step of the integrator.
    /// Typically this function will be newly constructed and passed directly to this method,
    /// without storing it elsewhere - in this case the orbit integrator retains exclusive
    /// ownership of the function and will properly dispose of it at the end of its own lifetime.
    void addRuntimeFnc(PtrRuntimeFnc fnc) { fncs.push_back(fnc); }

    /// initialize or reset the current orbit state, and optionally set new time (if not NAN);
    /// this function should be called before run().
    virtual void init(const coord::PosVelCar& ic, double newTime=NAN) = 0;

    /// obtain the interpolated solution within the last completed timestep;
    /// the time offset is specified relative to the beginning of the last timestep
    virtual coord::PosVelCar getSol(double timeOffset) const = 0;

    /// return the size of ODE system - three coordinates and three velocities
    virtual unsigned int size() const { return 6; }
};


/** The class that performs the actual orbit integration in the given coordinate system.
    It implements the IOdeSystem interface for the ODE integrator, providing the RHS of
    the differential equation, i.e., the derivatives of position and velocity
    in the internal coordinate system, optionally taking into account
    rotation about the z axis with a constant pattern speed Omega.
    Note that the choice of coordinate system is an internal implementation detail,
    and does not affect the way the initial conditions are fed in or the orbit is presented
    to the outside world, which is always in cartesian coordinates.
    The conversion between the internal representation and the externally accessible cartesian
    phase-space point is performed by `init()` and `getSol()` methods.
    In the case of a nonzero rotation frequency Omega, the position is given in the rotating
    frame, and the momentum (velocity) corresponds to the inertial frame that instantaneously
    coincides with the rotating frame. For instance, if the potential is z-axisymmetric,
    pattern rotation has no effect on the actual shape of the orbit, and numerically identical
    initial conditions would result in physically the same orbit regardless of Omega,
    but the recorded trajectory will read differently.
    \tparam  CoordT  is the type of coordinate system (Car, Cyl or Sph).
*/
template<typename CoordT>
class OrbitIntegrator: public BaseOrbitIntegrator {
public:
    /** Initialize the orbit integrator for the given potential and pattern speed */
    OrbitIntegrator(const potential::BasePotential& potential, double Omega=0,
        const OrbitIntParams& params = OrbitIntParams()) :
        BaseOrbitIntegrator(potential, Omega, params) {}

    /// IOdeSystem interface: equations of motion for all coordinate systems and the DOP853 integrator;
    /// optional output argument accFac will request tighter accuracy when |Epot| >> |Epot+Ekin|
    virtual void eval(const double t, const double x[], double dxdt[], double* accFac=NULL) const;

    /// IOdeSystem2ndOrder interface: equations of motion for the Hermite and DPRKN8 integrators
    /// (implemented only in Cartesian coordinates)
    virtual void eval2(const double t, const double x[], double d2xdt2[], double d3xdt3[]=NULL,
        double* accFac=NULL) const;

    /// (re)initialize the orbit state; if time is non NAN, also update the current time
    virtual void init(const coord::PosVelCar& ic, double time=NAN);

    /// obtain the solution (position and velocity in cartesian coordinates) within the last
    /// completed timestep; time is specified as an offset from the beginning of the last timestep.
    /// Typically invoked from within `processTimestep()` methods of associated runtime functions
    virtual coord::PosVelCar getSol(double timeOffset) const
    { return toPosVelCar(getSolNative(timeOffset)); }

    /// obtain the solution in the native coordinate system, in which the integration is performed;
    /// time offset is specified relative to the beginning of the last completed timestep
    coord::PosVelT<CoordT> getSolNative(double timeOffset) const;
};

// explicitly specified template declaration is needed for integrateTraj
template<> void OrbitIntegrator<coord::Car>::init(const coord::PosVelCar& ic, double time);


/** A convenience function to compute the trajectory for the given initial conditions and potential.
    \param[in]  initialConditions  is the initial position and velocity in cartesian coordinates;
    \param[in]  totalTime  is the maximum duration of orbit integration;
    \param[in]  samplingInterval  is the time interval between successive point recorded
                from the trajectory; if zero, then the trajectory is recorded at every timestep
                of the orbit integrator, otherwise it is stored at these regular intervals;
    \param[in]  potential  is the gravitational potential in which the orbit is computed;
    \param[in]  Omega  is the angular frequency of the rotation of the reference frame (default 0);
    \param[in]  params  are the extra parameters for the integration (default values may be used);
    \param[in]  startTime  is the initial time of the integration (default 0).
    \return     the recorded trajectory (0th point is the initial conditions) -
                an array of pairs of position/velocity points and associated moments of time.
    \throw      an exception if something goes wrong.
*/
inline Trajectory integrateTraj(
    const coord::PosVelCar& initialConditions,
    const double totalTime,
    const double samplingInterval,
    const potential::BasePotential& potential,
    const double Omega = 0,
    const OrbitIntParams& params = OrbitIntParams(),
    const double startTime = 0)
{
    Trajectory output;
    if(samplingInterval > 0)
        // reserve space for the trajectory, including one extra point for the final state
        output.reserve((totalTime>=0 ? totalTime : -totalTime) * (1+1e-15) / samplingInterval + 1);
    OrbitIntegrator<coord::Car> orbint(potential, Omega, params);
    orbint.addRuntimeFnc(PtrRuntimeFnc(new RuntimeTrajectory(orbint, samplingInterval, output)));
    orbint.init(initialConditions, startTime);
    orbint.run(totalTime);
    return output;
}

}  // namespace