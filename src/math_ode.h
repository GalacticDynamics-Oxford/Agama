/** \file    math_ode.h
    \brief   ODE integration classes
    \author  Eugene Vasiliev
    \date    2008-2015

    This module implements classes for integration of ordinary differential equation systems.

    OdeSolverDOP853 is a modification of the 8th order Runge-Kutta Solver from
    Hairer, Norsett & Wanner, "Solving ordinary differential equations", 1987, Berlin:Springer.
    Based on the C version (by J.Colinge) of the original Fortran code by E.Hairer & G.Wanner.

    OdeSolverIAS15 is an adapted version of the 15th order Solver from Rebound:
    Rein & Spiegel, 2014, MNRAS.
*/

#pragma once
#include <vector>

namespace math{

typedef std::vector<double> OdeStateType;

/** Prototype of a function that is used in integration of ordinary differential equation systems:
    dy/dt = f(t, y), where y is an N-dimensional vector. */
class IOdeSystem {
public:
    IOdeSystem() {};
    virtual ~IOdeSystem() {};

    /** Compute the r.h.s. of the differential equation: 
        \param[in]  t    is the integration variable (time),
        \param[in]  y    is the vector of values of dependent variables, 
        \param[out] dydt should return the time derivatives of these variables */
    virtual void eval(double t, const OdeStateType& y, OdeStateType& dydt) const = 0;

    /** Return the size of ODE system (number of variables) */
    virtual unsigned int size() const = 0;

    /** Inform whether the ODE is in the 'standard Hamiltonian' form,
        i.e. if the time derivatives of the first half of variables (coordinates)
        are given by the values of the second half of variables (velocity components).
        If true, this allows for simplifications in some ODE solvers */
    virtual bool isStdHamiltonian() const { return false; };
};

/** basic class for numerical integrators of ODE system */
class BaseOdeSolver
{
public:
    BaseOdeSolver(const IOdeSystem& _odeSystem):
        odeSystem(_odeSystem), timePrev(0), timeCurr(0) {};

    virtual ~BaseOdeSolver() {};

    /** (Re-)initialize the internal state from the given ODE system state */
    virtual void init(const OdeStateType& state) = 0;

    /** perform one timestep of variable length, determined by internal accuracy requirements;
        \return the length of timestep taken, or zero on error */
    virtual double doStep() = 0;

    /** return the time to which the integration has proceeded so far */
    double getTime() const { return timeCurr; }

    /** return interpolated solution at time t, which must lie within current timestep interval */
    virtual void getSol(double t, double x[]) const = 0;

protected:
    const IOdeSystem& odeSystem;
    /// values of integration variable (time) at the beginning and the end of the current timestep
    double timePrev, timeCurr;
};

/** 8th order Runge-Kutta integrator from Hairer,Norsett&Wanner */
class OdeSolverDOP853: public BaseOdeSolver
{
public:
    OdeSolverDOP853(const IOdeSystem& _odeSystem, double _accRel, double _accAbs=0);
    virtual void init(const OdeStateType& state);
    virtual double doStep();
    /** dense output with 6th order interpolation (modification of the original algorithm) */
    virtual void getSol(double t, double x[]) const;
    static const char* myName() { return "DOP853"; };
private:
    const double accRel, accAbs; ///< relative and absolute tolerance parameters
    double timeStep;             ///< length of next timestep (not the one just taken)
    OdeStateType statePrev;      ///< variables at the beginning of timestep
    OdeStateType stateCurr;      ///< current (end-of-timestep) values of variables inside the integrator
    /** temporary storage for intermediate Runge-Kutta steps */
    OdeStateType k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, ytemp;
    /** temporary storage for dense output interpolation */
    OdeStateType rcont1, rcont2, rcont3, rcont4, rcont5, rcont6, rcont7, rcont8;

    double initTimeStep();   ///< determine initial timestep
};

#if 0
/** 15-th order implicit Gauss-Radau scheme from Rein & Spiegel, 2015, MNRAS, 446, 1424
    (adapted from Rebound).
    It has its own accuracy parameter, with typical values 10^-4..10^-3 providing relative accuracy 
    in the range 10^-15(almost machine precision)..10^-10. 
    It works only for "Standard Hamiltonian" systems (position+velocity variables). */
class OdeSolverIAS15: public BaseOdeSolver
{
public:
    OdeSolverIAS15(const IOdeSystem& _odeSystem, double _acc):
      BaseOdeSolver(_odeSystem), integrator_epsilon(_acc) {};
    virtual ~OdeSolverIAS15() {};

    virtual void integrateToTime(double timeEnd);

    virtual double getInterpolatedSolution(unsigned int c, double t) const;

    static const char* myName() { return "IAS15"; };

private:
    double integrator_epsilon;  ///< accuracy for the integrator
    unsigned int nfcn,          ///< statistics: number of force evaluations
        nstep, naccpt, nrejct;  ///< statistics: number of timesteps (total, accepted, rejected)
    OdeStateType stateCurr;     ///< ODE system state at the end of timestep
    OdeStateType statePrev;     ///< ODE system state at the beginning of timestep
    OdeStateType deriv;         ///< temp.buffer for computing rhs of equations of motion
    vectord at;                 ///< Temp.buffer for accelerations
    vectord a0;                 ///< Accelerations at the beginning of timestep
    vectord compsum;            ///< Temp.buffer for compensated summation
    vectord g[7],b[7],e[7];     ///< coefficients of approximation in the integrator
    vectord br[7], er[7];       ///< coefficients of approximation at the beginning of timestep
    double dt;                  ///< current timestep length
    double dt_last_success;     ///< Last accepted timestep (corresponding to br and er)
    /// Helper function for resetting the b and e coefficients
    void predict_next_step(double ratio, int N3, vectord _e[7], vectord _b[7]);
    /// Actual integration for one timestep
    int integrator_ias15_step();
};

/** Hermite integrator that uses information about force derivatives (jerk) 
    in a 4th-order predictor-corrector scheme with only two force evaluations per timestep */
class OdeSolverHermite: public BaseOdeSolver
{
public:
    OdeSolverHermite(const IOdeSystem& _odeSystem, double _accur, bool _twoCorrectorSteps=false):
      BaseOdeSolver(_odeSystem), accur(_accur), dt(0), twoCorrectorSteps(_twoCorrectorSteps) {};
    virtual ~OdeSolverHermite() {};

    virtual void integrateToTime(double timeEnd);

    virtual double getInterpolatedSolution(unsigned int c, double t) const;

    static const char* myName() { return "Hermite"; };

private:
    double accur;            ///< accuracy parameter for computing the timestep
    double dt;               ///< current timestep
    bool twoCorrectorSteps;  ///< use two corrector stages instead of one (for rotating potential)
    OdeStateType statePrev, stateCurr;  ///< position and velocity at the beginning and end of timestep
    OdeStateType derivPrev, derivCurr;  ///< jerk and acceleration at the beginning and end of timestep
    OdeStateType snapcrac;              ///< snap and crackle (2nd and 3rd derivatives of acceleration) at the beginning of timestep
    void hermite_step();     ///< perform one predictor-corrector step and readjust the timestep
};

/** A template class for integrators based on the boost::numeric::odeint library.
    The argument of template class is the class name of the stepper algorithm from odeint. */
template<class Stepper >
class OdeSolverOdeint: public BaseOdeSolver
{
public:
    /// category of the integrator (could be fixed-timestep, adaptive-timestep, adaptive with dense output, etc)
    typedef typename boost::numeric::odeint::unwrap_reference< Stepper >::type::stepper_category stepper_category;

    OdeSolverOdeint(const IOdeSystem& _odeSystem, double _accAbs, double _accRel);
    virtual void integrateToTime(double timeEnd);
    virtual double getInterpolatedSolution(unsigned int c, double t) const;
    static const char* myName();
private:
    Stepper stepper;                         ///< the instance of stepper that does actual integration
    OdeStateType stateCurr, statePrev;       ///< ODE system state at the beginning and the end of timestep
    mutable OdeStateType stateIntermediate;  ///< temporary cached storage for interpolated trajectory
    mutable double timeIntermediate;         ///< temporary cache for interpolation is for this particular value of time
    double timeStep;                         ///< current timestep length
    bool isStdHamiltonian;                   ///< whether the ODE system is a "standard Hamiltonian" (positions, then velocities)
    /// driver routine for generic (fixed-timestep) steppers
    void integrateToTimeImpl(double timeEnd, boost::numeric::odeint::stepper_tag);
    /// driver routine for adaptive-timestep steppers
    void integrateToTimeImpl(double timeEnd, boost::numeric::odeint::controlled_stepper_tag);
    /// driver routine for adaptive-timestep steppers with dense output
    void integrateToTimeImpl(double timeEnd, boost::numeric::odeint::dense_output_stepper_tag);
    /// interpolate trajectory for a generic stepper
    double getInterpolatedSolutionImpl(unsigned int c, double t, boost::numeric::odeint::stepper_tag) const
    { return isStdHamiltonian ? getInterpolatedSolutionStdHamiltonian(c, t) : getInterpolatedSolutionGeneric(c, t); }
    /// interpolate trajectory for an adaptive-timestep stepper
    double getInterpolatedSolutionImpl(unsigned int c, double t, boost::numeric::odeint::controlled_stepper_tag) const
    { return isStdHamiltonian ? getInterpolatedSolutionStdHamiltonian(c, t) : getInterpolatedSolutionGeneric(c, t); }
    /// interpolate trajectory for a stepper that provides dense output
    double getInterpolatedSolutionImpl(unsigned int c, double t, boost::numeric::odeint::dense_output_stepper_tag) const;
    /// simple first-order interpolation of trajectory for an arbitrary stepper and ODE system of unknown structure
    double getInterpolatedSolutionGeneric(unsigned int c, double t) const;
    /// interpolation of trajectory for an ODE system that is a "standard Hamiltonian" -- 3rd order in position, 2nd order in velocity
    double getInterpolatedSolutionStdHamiltonian(unsigned int c, double t) const;
};

/// \cond INTERNAL_DOCS
/** Helper class that provides timestep control for odeint steppers */
class my_error_checker
{
public:
    typedef double value_type;
    typedef boost::numeric::odeint::range_algebra algebra_type;
    typedef boost::numeric::odeint::default_operations operations_type;
    my_error_checker(
        const IOdeSystem& _odeSystem,
        value_type eps_abs = static_cast< value_type >( 1.0e-6 ) ,
        value_type eps_rel = static_cast< value_type >( 1.0e-6 ) ,
        value_type a_x = static_cast< value_type >( 1 ) ,
        value_type a_dxdt = static_cast< value_type >( 1 ) )
    : odeSystem(_odeSystem), m_eps_abs(eps_abs) , m_eps_rel(eps_rel) , m_a_x(a_x) , m_a_dxdt(a_dxdt)
    { }

    template< class State , class Deriv , class Err , class Time >
    value_type error( const State &x_old , const Deriv &dxdt_old , Err &x_err , Time dt ) const
    {
        return error( algebra_type() , x_old , dxdt_old , x_err , dt );
    }

    template< class State , class Deriv , class Err , class Time >
    value_type error( algebra_type &algebra , const State &x_old , const Deriv &dxdt_old , Err &x_err , Time dt ) const
    {
        double eps_rel=m_eps_rel * odeSystem->getRelErrorToleranceFactor(x_old);
        algebra.for_each3( x_err , x_old , dxdt_old ,
            typename operations_type::template rel_error< value_type >( m_eps_abs , eps_rel , m_a_x , m_a_dxdt * boost::numeric::odeint::get_unit_value( dt ) ) );
        return algebra.norm_inf( x_err );
    }

private:
    const IOdeSystem& odeSystem;
    value_type m_eps_abs;
    value_type m_eps_rel;
    value_type m_a_x;
    value_type m_a_dxdt;
};
/// \endcond

/// Implementation of Bulirsch-Stoer method with dense output
typedef boost::numeric::odeint::bulirsch_stoer_dense_out< OdeStateType >  StepperBS;

/// Implementation of the Dormand-Prince 5th order Runge-Kutta method with adaptive timestep and dense output
typedef boost::numeric::odeint::dense_output_runge_kutta<
        boost::numeric::odeint::controlled_runge_kutta< 
        boost::numeric::odeint::runge_kutta_dopri5< OdeStateType >, my_error_checker > >  StepperDP5;

/// Implementation of the Cash-Karp 5th order Runge-Kutta method with adaptive timestep
typedef boost::numeric::odeint::controlled_runge_kutta< 
        boost::numeric::odeint::runge_kutta_cash_karp54< OdeStateType >, my_error_checker > StepperCK5;

/// Implementation of the standard fixed-timestep 4th order Runge-Kutta method
typedef boost::numeric::odeint::runge_kutta4< OdeStateType >  StepperRK4;

/// Implementation of the fixed-timestep 4th order symplectic Runge-Kutta method
typedef boost::numeric::odeint::symplectic_rkn_sb3a_mclachlan< OdeStateType >  StepperSympl4;
#endif

};  // namespace
