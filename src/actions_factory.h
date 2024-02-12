/** \file    actions_factory.h
    \brief   Creation of potential-specific action finders/mappers and standalone action evaluation
    \author  Eugene Vasiliev
    \date    2024
*/
#include "actions_base.h"
#include "smart.h"

namespace actions{

/** Evaluate any combination of actions, angles and frequencies in a given potential,
    using the appropriate method (Isochrone, general spherical, or Staeckel Fudge).
    \param[in]  potential is the potential.
    \param[in]  point     is the position/velocity point.
    \param[out] act   if not NULL, will contain computed actions.
    \param[out] ang   if not NULL, will contain corresponding angles.
    \param[out] freq  if not NULL, will contain corresponding frequencies.
    \param[in]  focalDistance (optional) is the geometric parameter of best-fit coordinate system,
    needed only for axisymmetric potentials that employ the Staeckel Fudge approach.
    \throw      std::invalid_argument exception if the potential is not suitable (non-axisymmetric).
*/
void eval(
     const potential::BasePotential& potential,
     const coord::PosVelCyl& point,
     Actions* act=NULL,
     Angles* ang=NULL,
     Frequencies* freq=NULL,
     double focalDistance=0);

/** Create an instance of ActionFinder*** class appropriate for the given potential.
    \param[in]  potential  is a shared pointer to the potential.
    \param[in]  interpolate  (optional, used only for Staeckel Fudge) determines whether to use
    the interpolated implementation (faster but less accurate);
    note that a spherical action finder always uses interpolation regardless of this parameter.
    \return  an instance of action finder.
    \throw   std::invalid_argument exception if the potential is not suitable (non-axisymmetric).
*/
actions::PtrActionFinder createActionFinder(
    const potential::PtrPotential& potential,
    bool interpolate=false);

/** Create an instance of ActionMapper*** class appropriate for the given potential.
    \param[in]  potential  is a shared pointer to the potential.
    \param[in]  tol (optional)  is the accuracy parameter for Torus Mapper.
    \return  an instance of action mapper.
    \throw   std::invalid_argument exception if the potential is not suitable (non-axisymmetric).
*/
actions::PtrActionMapper createActionMapper(const potential::PtrPotential& potential, double tol=NAN);

}  // namespace actions
