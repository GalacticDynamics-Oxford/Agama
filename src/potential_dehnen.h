/** \file    potential_dehnen.h
    \brief   implementation of triaxial Dehnen potential
    \author  Eugene Vasiliev
    \date    2009-2015
**/
#pragma once
#include "potential_base.h"

namespace potential {

/** Dehnen(1993) double power-law model **/
class Dehnen: public BasePotentialCar
{
public:
    Dehnen(double _mass, double _scalerad, double _q, double _p, double _gamma);
    virtual const char* name() const { return myName(); };
    static const char* myName() { return "Dehnen"; };
    virtual SymmetryType symmetry() const { 
        return (q==1 ? (p==1 ? ST_SPHERICAL : ST_AXISYMMETRIC) : ST_TRIAXIAL); };
private:
    const double mass;       ///< total mass of the model
    const double scalerad;   ///< scale radius
    const double q, p;       ///< axis ratio (y/x and z/x) of equidensity surfaces
    const double gamma;      ///< cusp exponent for Dehnen potential
    virtual void eval_car(const coord::PosCar &pos,
        double* potential, coord::GradCar* deriv, coord::HessCar* deriv2) const;
    virtual double density_car(const coord::PosCar &pos) const;
    virtual double density_cyl(const coord::PosCyl &pos) const {
        return density_car(coord::toPosCar(pos)); }
    virtual double density_sph(const coord::PosSph &pos) const {
        return density_car(coord::toPosCar(pos)); }
};

}  // namespace
