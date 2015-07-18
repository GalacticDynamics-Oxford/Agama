#include "orbit.h"
#include <stdexcept>
#include <cmath>
#include "mathutils.h"

namespace orbit{

/* evaluate r.h.s. of ODE in different coordinate systems */
template<typename coordT>
class OrbitIntegrator: public mathutils::IOdeSystem {
public:
    OrbitIntegrator(const potential::BasePotential& p) :
        potential(p) {};
    
    /** apply the equations of motion */
    virtual void eval(const double t, const double y[], double* dydt) const;
    
    /** return the size of ODE system - three coordinates and three velocities */
    virtual int size() const { return 6;} ;
    
private:
    const potential::BasePotential& potential;
};


template<>
void OrbitIntegrator<coord::Car>::eval(const double /*t*/, const double y[], double* dydt) const {
    coord::GradCar grad;
    potential.eval(coord::PosCar(y[0], y[1], y[2]), NULL, &grad);
    dydt[0]=y[3]; dydt[1]=y[4]; dydt[2]=y[5];  // time deriv. of position
    dydt[3]=-grad.dx;
    dydt[4]=-grad.dy;
    dydt[5]=-grad.dz;
}

template<>
void OrbitIntegrator<coord::Cyl>::eval(const double /*t*/, const double y[], double* dydt) const {
    const coord::PosVelCyl p(y);
    coord::GradCyl grad;
    potential.eval(p, NULL, &grad);
    const coord::PosVelCyl pdot( p.vR, p.vz, p.vphi/p.R,
        -grad.dR+pow_2(p.vphi)/p.R, -grad.dz, -(grad.dphi+p.vR*p.vphi)/p.R);
    pdot.unpack_to(dydt);
}

template<>
void OrbitIntegrator<coord::Sph>::eval(const double /*t*/, const double y[], double* dydt) const {
    const coord::PosVelSph p(y);
    coord::GradSph grad;
    potential.eval(p, NULL, &grad);
    const double sintheta=sin(p.theta), cottheta=cos(p.theta)/sintheta;
    const coord::PosVelSph pdot( p.vr, p.vtheta/p.r, p.vphi/(p.r*sintheta),
        -grad.dr + (pow_2(p.vtheta)+pow_2(p.vphi))/p.r,
        (-grad.dtheta + pow_2(p.vphi)*cottheta - p.vr*p.vtheta)/p.r,
        (-grad.dphi/sintheta - (p.vr+p.vtheta*cottheta)*p.vphi)/p.r );
    pdot.unpack_to(dydt);
}

template<typename coordT>
int integrate(const potential::BasePotential& potential,
    const coord::PosVelT<coordT>& initial_conditions,
    const double total_time,
    const double output_timestep,
    std::vector<coord::PosVelT<coordT> >& output_trajectory,
    const double accuracy)
{
    int nsteps = static_cast<int>(total_time/output_timestep);
    output_trajectory.reserve(nsteps+1);
    double vars[6];
    initial_conditions.unpack_to(vars);
    double norm=0;  // magnitude of variables used to compute absolute error tolerance
    for(int i=0; i<6; i++) 
        norm+=fabs(vars[i]);
    OrbitIntegrator<coordT>  orbint(potential);
    mathutils::OdeSolver     ode(orbint, accuracy*norm, accuracy);
    int numsteps=0, result=0;
    for(int i=0; i<=nsteps && result>=0; i++){
        output_trajectory.push_back(coord::PosVelT<coordT>(vars));
        double time = output_timestep*i;
        result = ode.advance(time, time+output_timestep, vars);
        if(result>0) 
            numsteps+=result;
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