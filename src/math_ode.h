/** \file    math_ode.h
    \brief   ODE integration classes
    \author  Eugene Vasiliev
    \date    2008-2025

    This module implements classes for integration of ordinary differential equation systems.

    OdeStepperDOP853 is a modification of the 8th order Runge-Kutta Solver from
    Hairer, Norsett & Wanner, "Solving ordinary differential equations", 1987, Berlin:Springer.
    Based on the C version (by J.Colinge) of the original Fortran code by E.Hairer & G.Wanner.
*/

#pragma once
#include "math_base.h"
#include <vector>

namespace math{

/** Prototype of a function that is used in integration of ordinary differential equation systems:
    dx/dt = f(t, x), where y is an N-dimensional vector. */
class IOdeSystem {
public:
    IOdeSystem() {};
    virtual ~IOdeSystem() {};

    /** Compute the r.h.s. of the differential equation: 
        \param[in]  t  is the integration variable (time), measured from the beginning of the timestep.
        \param[in]  x  is the vector of values of dependent variables.
        \param[out] dxdt  should return the time derivatives of these variables.
        \param[in,out] af  if not NULL, can store the optional multiplicative accuracy factor;
        its purpose is to increase the accuracy (use a tighter tolerance) when needed,
        for instance, when the potential energy of the system is much larger than the total energy.
        On input, this parameter is set to 1, and if no adjustment is needed, it can be left
        unchanged, otherwise it could be set to a lower value, which is multiplied by accRel
        in the ODE stepper (not every method will use this parameter, though).
    */
    virtual void eval(const double t, const double x[], double dxdt[], double* af=0) const = 0;

    /** Return the size of ODE system (number of variables N) */
    virtual unsigned int size() const = 0;
};

/** Prototype of a function that is used in integration of second-order ODEs of the following form:
    d2x(t) / dt2 = a(t, x),
    where x is an N-dimensional vector, and the acceleration may depend only on position x and time t,
    but not on velocity dx/dt.
    It may also provide the third time derivative:  d3x(t) / dt3 = j(t, x).
    Note: the size of the system reported by size() is 2N, i.e. vectors x and dx/dt.
*/
class IOdeSystem2ndOrder: public IOdeSystem {
public:
    /** Compute 2nd and optionally 3rd time derivatives of x(t): a(t,x)=d2x/dt2 and j(t,x)=d3x/dt3.
        \param[in]  t is the integration variable (time), measured from the beginning of the timestep.
        \param[in]  x is the vector of values of dependent variables: x(t).
        \param[out] d2xdt2 should return the second time derivative of x(t).
        \param[out] d3xdt3 if not NULL, should return the third  time derivative of x(t).
        \param[in,out] af  if not NULL, can store the optional multiplicative accuracy factor.
    */
    virtual void eval2(const double t, const double xv[], double d2xdt2[], double d3xdt3[]=0,
        double* af=0) const = 0;

    // represent the second-order ODE system as a first-order one for the vector w = {x,v=dx/dt}
    virtual void eval(const double t, const double w[], double dwdt[], double* af=0) const;
};

/** Prototype of a function that is used in integration of second-order
    linear ordinary differential equation systems with variable coefficients:
    d2x(t) / dt2 = A(t) x(t) + B(t) dx(t)/dt,
    where x is an N-dimensional vector, and A, B are NxN matrices;
    the acceleration depends on time t and both the position x and velocity dx/dt, but only linearly
    (thus is is not a subclass of IOdeSystem2ndOrder)
    Note: the size of the system reported by size() is 2N, i.e. vectors x and dx/dt.
*/
class IOdeSystem2ndOrderLinear: public IOdeSystem {
public:
    /** Compute the matrices A and B in the r.h.s. of the differential equation at the given time:
        \param[in]  t  is the integration variable (time), measured from the beginning of the timestep.
        \param[out] a  should point to an existing array of length N^2,
        which will be filled with the flattened (row-major) matrix A: mat[i*N+j] = A_{ij}.
        \param[out] b  same for the matrix B.
    */
    virtual void evalMat(const double t, double a[], double b[]) const = 0;

    /** Represent the system as a generic first-order ODE for the vector w = {x,v=dx/dt} */
    virtual void eval(const double t, const double w[], double dwdt[], double* af=0) const;
};


/** Base class for numerical integrators of ODE systems.
    The task of this class is to advance the solution by one timestep at a time,
    evaluating the r.h.s. of the ODE at some intermediate times within the current timestep,
    and possibly adjusting the timestep to satisfy the accuracy requirements.
    The calling code is responsible for the overall process of integrating the ODE
    on a given time interval, keeping track of the total time, etc.
*/
class BaseOdeStepper {
public:
    virtual ~BaseOdeStepper() {};

    /** (re-)initialize the internal state from the given ODE system state */
    virtual void init(const double stateNew[]) = 0;

    /** advance the solution by one timestep.
        \param[in]  maxTimeStep is the upper limit on the length of the timestep (can have any sign);
        the actual timestep is controlled by the accuracy requirements and may be shorter.
        \return the length of the timestep taken, or zero on error
    */
    virtual double doStep(double maxTimeStep) = 0;

    /** return the interpolated solution within the last completed timestep.
        \param[in]  timeOffset  is the time offset from the beginning of the last completed step,
            and it should not exceed the length of this step (taking into account its sign).
        \param[in]  ind  is the index of the component of the solution vector;
        \return  the interpolated solution at the given time.
        \throw  std::out_of_range if the index is not in the range (0 .. N-1),
            or if the requested time offset falls outside the last completed timestep.
    */
    virtual double getSol(double timeOffset, unsigned int ind) const = 0;
};


/** 8th order Runge-Kutta integrator with 7th order interpolation for the dense output
    (modification of the original algorithm from Hairer,Norsett&Wanner, reducing the order of
    interpolation from 8 to 7 and saving 3 function evaluations per timestep) */
class OdeStepperDOP853: public BaseOdeStepper {
public:
    OdeStepperDOP853(const IOdeSystem& _odeSystem, double _accRel=1e-8, double _accAbs=0) :
        odeSystem(_odeSystem), NDIM(odeSystem.size()),
        accRel(_accRel), accAbs(_accAbs),
        prevTimeStep(0), nextTimeStep(0),
        state(NDIM * 10)  // storage for the current values and derivs of x and for 8 interpolation coefs
    {}
    virtual void init(const double stateNew[]);
    virtual double doStep(double maxTimeStep);
    virtual double getSol(double timeOffset, unsigned int ind) const;
private:
    const IOdeSystem& odeSystem; ///< the interface providing the r.h.s. of the ODE
    const int NDIM;              ///< number of equations
    const double accRel, accAbs; ///< relative and absolute tolerance parameters
    double prevTimeStep;         ///< length of the last completed time step
    double nextTimeStep;         ///< predicted length of the next timestep (not the one just completed)
    std::vector<double> state;   ///< 10*NDIM values: x, dx/dt, and 8 interpolation coefs for dense output
};


/** 8th order Runge-Kutta-Nystrom scheme with nine function evaluations per timestep,
    requires a special type of ODE system that provides second time derivatives of x.
    The order of solution is 8, the order of interpolation is 6 for x, 5 for dx/dt.
*/
class OdeStepperDPRKN8: public BaseOdeStepper {
public:
    OdeStepperDPRKN8(const IOdeSystem2ndOrder& _odeSystem, double _accRel=1e-8);
    virtual void init(const double stateNew[]);
    virtual double doStep(double maxTimeStep);
    virtual double getSol(double timeOffset, unsigned int ind) const;
private:
    /// the object providing the r.h.s. of the ODE
    const IOdeSystem2ndOrder& odeSystem; ///< interface for evaluating the acceleration
    const int NDIM;              ///< number of equations
    const double accRel;         ///< relative tolerance parameter
    double prevTimeStep;         ///< length of the last completed time step
    double nextTimeStep;         ///< predicted length of the next timestep (not the one just completed)
    std::vector<double> state;   ///< 3*NDIM values
    double qold;                 ///< adaptive timestep change in the previous step
};


/** 4th order Hermite scheme with two function evaluations per timestep,
    requires a special type of ODE system that provides second and third time derivatives of x */
class OdeStepperHermite: public BaseOdeStepper {
public:
    OdeStepperHermite(const IOdeSystem2ndOrder& _odeSystem, double _accRel=1e-8);
    virtual void init(const double stateNew[]);
    virtual double doStep(double maxTimeStep);
    virtual double getSol(double timeOffset, unsigned int ind) const;
private:
    const IOdeSystem2ndOrder& odeSystem; ///< interface for evaluating the acceleration and jerk
    const int NDIM;              ///< number of equations
    const double accRel;         ///< relative tolerance parameter
    double prevTimeStep;         ///< length of the last completed time step
    double nextTimeStep;         ///< predicted length of the next timestep
    std::vector<double> state;   ///< 5*NDIM values
};


/** Implicit Gauss-Legendre method with 3 collocation points for second-order linear ODE systems:
    d2x(t) / dt2 = A(t) x(t) + B(t) dx(t)/dt,
    where x is a N-dimensional vector, and A, C are NxN matrices.
    It has no built-in error control, i.e. no adaptive timestepping,
    is intended for solving the variational equation during orbit integration,
    and may evolve K >= 1 independent vectors x_k simultaneously.
    The order of solution is 6, the order of interpolation is 5 for x, 4 for dx/dt.
    \tparam NDIM is the size of vector x (hence the size of the entire ODE system is
    2 NDIM * numVectors); only the cases NDIM=1,2,3 are compiled.
*/
template<int NDIM>
class Ode2StepperGL3: public BaseOdeStepper {
public:
    Ode2StepperGL3(const IOdeSystem2ndOrderLinear& _odeSystem, unsigned int numVectors=1);
    virtual void init(const double stateNew[]);
    virtual double doStep(double timeStep);
    virtual double getSol(double timeOffset, unsigned int ind) const;
private:
    const IOdeSystem2ndOrderLinear& odeSystem;   ///< interface providing the r.h.s. of the ODE
    const unsigned int numVectors;  ///< number of independent vectors being evolved
    double prevTimeStep;  ///< length of the just completed timestep
    bool newstep;         ///< whether the extrapolation coefs are known from the previous step
    std::vector<double> state;  ///< internal state, including interpolation coefficients
};


/** Implicit Gauss-Legendre method with 4 collocation points for second-order linear ODE systems:
    d2x(t) / dt2 = A(t) x(t) + B(t) dx(t)/dt,
    where x is a N-dimensional vector, and A, C are NxN matrices.
    It has no built-in error control, i.e. no adaptive timestepping,
    is intended for solving the variational equation during orbit integration,
    and may evolve K >= 1 independent vectors x_k simultaneously.
    The order of solution is 8, the order of interpolation is 6 for x, 5 for dx/dt.
    \tparam NDIM is the size of vector x (hence the size of the entire ODE system is
    2 NDIM * numVectors); only the cases NDIM=1,2,3 are compiled.
*/
template<int NDIM>
class Ode2StepperGL4: public BaseOdeStepper {
public:
    Ode2StepperGL4(const IOdeSystem2ndOrderLinear& _odeSystem, unsigned int numVectors=1);
    virtual void init(const double stateNew[]);
    virtual double doStep(double timeStep);
    virtual double getSol(double timeOffset, unsigned int ind) const;

private:
    const IOdeSystem2ndOrderLinear& odeSystem;   ///< interface providing the r.h.s. of the ODE
    const unsigned int numVectors;  ///< number of independent vectors being evolved
    double prevTimeStep;  ///< length of the just completed timestep
    bool newstep;         ///< whether the extrapolation coefs are known from the previous step
    std::vector<double> state;  ///< internal state, including interpolation coefficients
};

}  // namespace
