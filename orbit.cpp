#include "orbit.h"
#include <stdexcept>
#include "GSLInterface.hpp"

namespace orbit{

    /* evaluate r.h.s. of ODE in different coordinate systems */
    template<typename coordT>
    void apply_ode(const double y[], const coord::GradT<coordT> grad, double dydt[]);

    template<>
    void apply_ode(const double y[], const coord::GradCar grad, double dydt[]) {
        dydt[0]=y[3]; dydt[1]=y[4]; dydt[2]=y[5];  // time deriv. of position
        dydt[3]=-grad.dx;
        dydt[4]=-grad.dy;
        dydt[5]=-grad.dz;
    }

    template<>
    void apply_ode(const double y[], const coord::GradCyl grad, double dydt[]) {
        throw std::runtime_error("Integration in cylindrical coordinates not properly implemented");
        dydt[0]=y[3]; dydt[1]=y[4]; dydt[2]=y[5]/y[0];  // time deriv. of position in cyl.coord.
        dydt[3]=-grad.dR;
        dydt[4]=-grad.dz;
        dydt[5]=-grad.dphi/y[0];
    }

    template<>
    void apply_ode(const double y[], const coord::GradSph grad, double dydt[])
    {
        throw std::runtime_error("Integration in spherical coordinates not implemented");
        dydt[0]=y[3]; dydt[1]=y[4]; dydt[2]=y[5];  // time deriv. of position in sph.coord.
        dydt[3]=-grad.dr;
        dydt[4]=-grad.dtheta;
        dydt[5]=-grad.dphi;
    }

    /* derivatives for orbit integration, common part */
    template<typename coordT>
    int derivs(double t, const double y[], double f[],void *params){
        potential::BasePotential *Pot = static_cast<potential::BasePotential*>(params);
        coord::PosT <coordT> pos(y[0], y[1], y[2]);
        coord::GradT<coordT> grad;
        Pot->eval(pos, NULL, &grad);
        apply_ode<coordT>(y, grad, f);
        return GSL_SUCCESS;
    }

    template<typename coordT>
    void integrate(const potential::BasePotential& poten,
        const coord::PosVelT<coordT>& initial_conditions,
        const double total_time,
        const double output_timestep,
        std::vector<coord::PosVelT<coordT> >& output_trajectory,
        const double accuracy)
    {
        int nsteps = static_cast<int>(total_time/output_timestep);
        output_trajectory.reserve(nsteps+1);
        ode ODE(derivs<coordT>, 6, accuracy, const_cast<potential::BasePotential*>(&poten));
        double vars[6];
        initial_conditions.unpack_to(vars);
        for(int i=0; i<=nsteps; i++){
            output_trajectory.push_back(coord::PosVelT<coordT>(vars[0], vars[1], vars[2], vars[3], vars[4], vars[5]));
            double time = output_timestep*i;
            ODE.step(time, time+output_timestep, vars, output_timestep);
        }
    }

    // explicit template instantiations to make sure all of them get compiled
    template void integrate(const potential::BasePotential&, const coord::PosVelCar&,
        const double, const double, std::vector<coord::PosVelCar>&, const double);
    template void integrate(const potential::BasePotential&, const coord::PosVelCyl&,
        const double, const double, std::vector<coord::PosVelCyl>&, const double);
    template void integrate(const potential::BasePotential&, const coord::PosVelSph&,
        const double, const double, std::vector<coord::PosVelSph>&, const double);

}  // namespace orbit