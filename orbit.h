#pragma once
#include "potential_base.h"
#include <vector>

namespace orbit{

    /** Perform orbit integration in the specific coordinate system.
        Currently is implemented through GSL odeint2 solver with 8th order Runge Kutta method.
        \param[in]  potential  is the reference to the galactic potential;
        \param[in]  initial_conditions  is the initial position/velocity pair in the given 
                    coordinate system (which determines the choice of c.s. for 
                    orbit integration, so make sure you pick up a reasonable one);
        \param[in]  total_time  is the duration of integration interval;
        \param[in]  output_timestep  determines the frequency of trajectory output, 
                    which doesn't influence the integration accuracy;
                    if set to zero, output will be produced after each internal integrator timestep;
        \param[out] output_trajectory  will contain the trajectory in the given c.s.;
        \param[in]  integration_accuracy  is the accuracy parameter for ODE integrator. */
    template<typename coordT>
    void integrate(const potential::BasePotential& potential,
        const coord::PosVelT<coordT>& initial_conditions,
        const double total_time,
        const double output_timestep,
        std::vector<coord::PosVelT<coordT> >& output_trajectory,
        const double accuracy=1e-10
        );
};
