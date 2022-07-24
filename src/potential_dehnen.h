/** \file    potential_dehnen.h
    \brief   Triaxial Dehnen potential
    \author  Eugene Vasiliev
    \date    2009-2015
**/
#pragma once
#include "potential_base.h"

namespace potential {

/** Dehnen(1993) double power-law model **/
class Dehnen: public BasePotentialCar {
public:
    Dehnen(double _mass, double _scalerad, double _gamma, double _axisRatioY=1., double _axisRatioZ=1.);
    virtual std::string name() const { return myName(); }
    static std::string myName() { return "Dehnen"; }
    virtual coord::SymmetryType symmetry() const { 
        return (axisRatioY==1 ?
            (axisRatioZ==1 ? coord::ST_SPHERICAL : coord::ST_AXISYMMETRIC) : coord::ST_TRIAXIAL); }
    virtual double totalMass() const { return mass; }
private:
    const double mass;       ///< total mass of the model
    const double scalerad;   ///< scale radius
    const double gamma;      ///< cusp exponent for Dehnen potential
    const double axisRatioY; ///< axis ratio y/x of equidensity surfaces
    const double axisRatioZ; ///< axis ratio z/x of equidensity surfaces

    virtual void evalCar(const coord::PosCar &pos,
        double* potential, coord::GradCar* deriv, coord::HessCar* deriv2, double time) const;
    virtual double densityCar(const coord::PosCar &pos, double time) const;
};

}  // namespace
