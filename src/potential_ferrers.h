/** \file    potential_ferrers.h
    \brief   Triaxial Ferrers potential
    \author  EV
    \date    2014-2015
*/
#pragma once
#include "potential_base.h"

namespace potential {

/** Ferrers density profile (with finite extent R and density (1-(r/R)^2)^2, where r is the triaxial radius).
    The potential is calculated using expressions from Pfenniger(1984) with elliptic integrals, 
    under assumption that p<q<1 strictly (will not work if any of two axes are equal). **/
class Ferrers: public BasePotentialCar {
public:
    /// Construct the potential for the following parameters: mass, radius, axis ratios q=y/x, p=z/x
    Ferrers(double _mass, double _R, double _q, double _p);
    ~Ferrers() {};
    virtual const char* name() const { return myName(); }
    static const char* myName() { return "Ferrers"; }
    virtual SymmetryType symmetry() const { return ST_TRIAXIAL; }
private:
    const double a, b, c;       ///< principal axis of ellipsoidal density 
    const double mass, rho0;    ///< total mass and central density of the model
    double W0[20];              ///< pre-computed coefficients for lambda=0

    virtual void evalCar(const coord::PosCar &pos,
        double* potential, coord::GradCar* deriv, coord::HessCar* deriv2) const;
    virtual double densityCar(const coord::PosCar &pos) const;

    /** compute the array of 20 coefficients W_{ijk} listed in Pfenniger(1984), in order of appearance in that paper;
        \param[in] lambda is zero inside the model and >0 outside;
        \param[out] W is the array of 20 coefs  */
    void computeW(double lambda, double W[20]) const;
};

}  // namespace
