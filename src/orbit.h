/** \file    orbit.h
    \brief   Orbit integration
    \author  Eugene Vasiliev
    \date    2009-2015
*/
#pragma once
#include "potential_base.h"
#include <vector>

/** Orbit integration routines and classes */
namespace orbit {

/** Perform orbit integration in the specific coordinate system.
    \param[in]  potential  is the reference to the gravitational potential;
    \param[in]  initialConditions  is the initial position/velocity pair in the given 
                coordinate system (the same c.s. is used for orbit integration and for output);
    \param[in]  totalTime  is the duration of integration interval;
    \param[in]  outputTimestep  determines the frequency of trajectory output, 
                which doesn't influence the integration accuracy;
    \param[out] outputTrajectory  will contain the trajectory in the given c.s.;
    \param[in]  accuracy  is the accuracy parameter for ODE integrator.
    \returns    number of elementary integration steps completed.
*/
template<typename coordT>
unsigned int integrate(const potential::BasePotential& potential,
    const coord::PosVelT<coordT>& initialConditions,
    const double totalTime,
    const double outputTimestep,
    std::vector<coord::PosVelT<coordT> >& outputTrajectory,
    const double accuracy=1e-10);

};
