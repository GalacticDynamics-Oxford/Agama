#pragma once
#include "potential_base.h"
#include <vector>

namespace orbit{

    template<typename coordT>
    void integrate(const potential::BasePotential& potential,
        const coord::PosVelT<coordT>& initial_conditions,
        const double total_time,
        const double output_timestep,
        std::vector<coord::PosVelT<coordT> >& output_trajectory,
        const double accuracy=1e-10
        );
};
