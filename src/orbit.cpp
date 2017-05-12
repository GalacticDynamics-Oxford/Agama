#include "orbit.h"
#include "potential_base.h"
#include "utils.h"
#include <stdexcept>
#include <cmath>

namespace orbit{

template<typename coordT>
StepResult RuntimeTrajectory<coordT>::processTimestep(
    const math::BaseOdeSolver& solver, const double /*tbegin*/, const double tend, double[])
{
    // store trajectory at regular intervals of time
    while(samplingInterval * trajectory.size() <= tend) {
        double data[6];
        solver.getSol(samplingInterval * trajectory.size(), data);
        trajectory.push_back(coord::PosVelT<coordT>(data));
    }
    return SR_CONTINUE;
}


template<>
void OrbitIntegrator<coord::Car>::eval(const double /*t*/,
    const math::OdeStateType& x, math::OdeStateType& dxdt) const
{
    coord::GradCar grad;
    potential.eval(coord::PosCar(x[0], x[1], x[2]), NULL, &grad);
    // time derivative of position
    dxdt[0] = x[3];
    dxdt[1] = x[4];
    dxdt[2] = x[5];
    // time derivative of velocity
    dxdt[3] = -grad.dx;
    dxdt[4] = -grad.dy;
    dxdt[5] = -grad.dz;
}
template<>
bool OrbitIntegrator<coord::Car>::isStdHamiltonian() const { return true; }

void OrbitIntegratorRot::eval(const double /*t*/,
    const math::OdeStateType& x, math::OdeStateType& dxdt) const
{
    coord::GradCar grad;
    potential.eval(coord::PosCar(x[0], x[1], x[2]), NULL, &grad);
    // time derivative of position
    dxdt[0] = x[3] + Omega * x[1];
    dxdt[1] = x[4] - Omega * x[0];
    dxdt[2] = x[5];
    // time derivative of velocity
    dxdt[3] = -grad.dx + Omega * x[4];
    dxdt[4] = -grad.dy - Omega * x[3];
    dxdt[5] = -grad.dz;
}

template<>
void OrbitIntegrator<coord::Cyl>::eval(const double /*t*/,
    const math::OdeStateType& x, math::OdeStateType& dxdt) const
{
    coord::PosVelCyl p(&x.front());
    if(x[0]<0) {    // R<0
        p.R = -p.R; // apply reflection
        p.phi += M_PI;
    }
    coord::GradCyl grad;
    potential.eval(p, NULL, &grad);
    double Rsafe = p.R!=0 ? p.R : 1e-100;  // avoid NAN in degenerate cases
    dxdt[0] = p.vR;
    dxdt[1] = p.vz;
    dxdt[2] = p.vphi/Rsafe;
    dxdt[3] = (x[0]<0 ? grad.dR : -grad.dR) + pow_2(p.vphi) / Rsafe;
    dxdt[4] = -grad.dz;
    dxdt[5] = -(grad.dphi + p.vR*p.vphi) / Rsafe;
}
template<>
bool OrbitIntegrator<coord::Cyl>::isStdHamiltonian() const { return false; }
        
template<>
void OrbitIntegrator<coord::Sph>::eval(const double /*t*/,
    const math::OdeStateType& x, math::OdeStateType& dxdt) const
{
    const coord::PosVelSph p(&x.front());
    coord::GradSph grad;
    if(x[0]<0) {  // r<0: apply transformation to bring the coordinates into a valid range
        potential.eval(coord::PosSph(-p.r, M_PI-p.theta, p.phi+M_PI), NULL, &grad);
        grad.dr = -grad.dr;
        grad.dtheta = -grad.dtheta;
    } else
        potential.eval(p, NULL, &grad);
    double rsafe = p.r!=0 ? p.r : 1e-100;
    double sintheta = sin(p.theta);
    if(sintheta == 0) sintheta = 1e-100;
    double cottheta = cos(p.theta)/sintheta;
    dxdt[0] = p.vr;
    dxdt[1] = p.vtheta/rsafe;
    dxdt[2] = p.vphi/(rsafe*sintheta);
    dxdt[3] = -grad.dr + (pow_2(p.vtheta) + pow_2(p.vphi)) / rsafe;
    dxdt[4] = (-grad.dtheta + pow_2(p.vphi)*cottheta - p.vr*p.vtheta) / rsafe;
    dxdt[5] = (-grad.dphi/sintheta - (p.vr+p.vtheta*cottheta)*p.vphi) / rsafe;
}
template<>
bool OrbitIntegrator<coord::Sph>::isStdHamiltonian() const { return false; }


template<typename coordT>
coord::PosVelT<coordT> integrate(
    const coord::PosVelT<coordT>& initialConditions,
    const double totalTime,
    const math::IOdeSystem& orbitIntegrator,
    const RuntimeFncArray& runtimeFncs,
    const OrbitIntParams& params)    
{
    // create an internal instance of the solver, will be destroyed automatically
    unique_ptr<math::BaseOdeSolver> solver;
    switch(params.solver) {
        case math::OS_DOP853:
            solver.reset(new math::OdeSolverDOP853(orbitIntegrator, params.accuracy));
            break;
        default:
            throw std::invalid_argument("orbit::integrate(): unknown ODE solver type");
    }
    math::OdeStateType vars(orbitIntegrator.size());
    initialConditions.unpack_to(&vars[0]);
    solver->init(vars);
    unsigned int numSteps = 0;
    double timePrev = 0., timeCurr = 0.;
    while(timeCurr < totalTime) {
        if(solver->doStep() <= 0.) {  // signal of error
            utils::msg(utils::VL_WARNING, FUNCNAME,
                "timestep is zero at t="+utils::toString(timeCurr));
            break;
        }
        timeCurr = fmin(solver->getTime(), totalTime);
        solver->getSol(timeCurr, &vars[0]);
        bool reinit = false, finish = false;
        for(unsigned int i=0; i<runtimeFncs.size(); i++) {
            switch(runtimeFncs[i]->processTimestep(*solver, timePrev, timeCurr, &vars[0]))
            {
                case orbit::SR_TERMINATE: finish = true; break;
                case orbit::SR_REINIT:    reinit = true; break;
                default: ;
            }
        }
        timePrev = timeCurr;
        if(reinit)
            solver->init(vars);
        if(finish || ++numSteps > params.maxNumSteps)
            break;
    }
    return coord::PosVelT<coordT>(&vars[0]);
}
    
// explicit template instantiations to make sure all of them get compiled
template class OrbitIntegrator<coord::Car>;
template class OrbitIntegrator<coord::Cyl>;
template class OrbitIntegrator<coord::Sph>;
template class RuntimeTrajectory<coord::Car>;
template class RuntimeTrajectory<coord::Cyl>;
template class RuntimeTrajectory<coord::Sph>;
template coord::PosVelCar integrate(const coord::PosVelCar&,
    const double, const math::IOdeSystem&, const RuntimeFncArray&, const OrbitIntParams&);
template coord::PosVelCyl integrate(const coord::PosVelCyl&,
    const double, const math::IOdeSystem&, const RuntimeFncArray&, const OrbitIntParams&);
template coord::PosVelSph integrate(const coord::PosVelSph&,
    const double, const math::IOdeSystem&, const RuntimeFncArray&, const OrbitIntParams&);

}  // namespace orbit