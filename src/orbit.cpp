#include "orbit.h"
#include <stdexcept>
#include <cmath>
#include "mathutils.h"

namespace orbit{

/* evaluate r.h.s. of ODE in different coordinate systems */
template<typename coordT>
void apply_ode(const double y[], const coord::GradT<coordT>& grad, double dydt[]);

template<>
void apply_ode(const double y[], const coord::GradCar& grad, double dydt[]) {
    dydt[0]=y[3]; dydt[1]=y[4]; dydt[2]=y[5];  // time deriv. of position
    dydt[3]=-grad.dx;
    dydt[4]=-grad.dy;
    dydt[5]=-grad.dz;
}

template<>
void apply_ode(const double y[], const coord::GradCyl& grad, double dydt[]) {
    const coord::PosVelCyl p(y);
    const coord::PosVelCyl pdot( p.vR, p.vz, p.vphi/p.R,
        -grad.dR+pow_2(p.vphi)/p.R, -grad.dz, -(grad.dphi+p.vR*p.vphi)/p.R);
    pdot.unpack_to(dydt);
}

template<>
void apply_ode(const double y[], const coord::GradSph& grad, double dydt[]) {
    const coord::PosVelSph p(y);
    const double sintheta=sin(p.theta), cottheta=cos(p.theta)/sintheta;
    const coord::PosVelSph pdot( p.vr, p.vtheta/p.r, p.vphi/(p.r*sintheta),
        -grad.dr + (pow_2(p.vtheta)+pow_2(p.vphi))/p.r,
        (-grad.dtheta + pow_2(p.vphi)*cottheta - p.vr*p.vtheta)/p.r,
        (-grad.dphi/sintheta - (p.vr+p.vtheta*cottheta)*p.vphi)/p.r );
    pdot.unpack_to(dydt);
}

/* derivatives for orbit integration, common part */
template<typename coordT>
int derivs(double /*t*/, const double y[], double f[],void *params) {
    potential::BasePotential *Pot = static_cast<potential::BasePotential*>(params);
    coord::PosT <coordT> pos(y[0], y[1], y[2]);
    coord::GradT<coordT> grad;
    Pot->eval(pos, NULL, &grad);
    apply_ode<coordT>(y, grad, f);
    return 0; //GSL_SUCCESS
}

template<typename coordT>
int integrate(const potential::BasePotential& poten,
    const coord::PosVelT<coordT>& initial_conditions,
    const double total_time,
    const double output_timestep,
    std::vector<coord::PosVelT<coordT> >& output_trajectory,
    const double accuracy)
{
    int nsteps = static_cast<int>(total_time/output_timestep);
    output_trajectory.reserve(nsteps+1);
    mathutils::odesolver ODE(derivs<coordT>, const_cast<potential::BasePotential*>(&poten), 6, accuracy);
    double vars[6];
    initial_conditions.unpack_to(vars);
    int numsteps=0;
    for(int i=0; i<=nsteps; i++){
        output_trajectory.push_back(coord::PosVelT<coordT>(vars));
        double time = output_timestep*i;
        numsteps += ODE.advance(time, time+output_timestep, vars);
    }
    return numsteps;
}

// explicit template instantiations to make sure all of them get compiled
template int integrate(const potential::BasePotential&, const coord::PosVelCar&,
    const double, const double, std::vector<coord::PosVelCar>&, const double);
template int integrate(const potential::BasePotential&, const coord::PosVelCyl&,
    const double, const double, std::vector<coord::PosVelCyl>&, const double);
template int integrate(const potential::BasePotential&, const coord::PosVelSph&,
    const double, const double, std::vector<coord::PosVelSph>&, const double);

}  // namespace orbit