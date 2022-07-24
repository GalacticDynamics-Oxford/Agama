/** \file    potential_ferrers.h
    \brief   Triaxial Ferrers potential
    \author  Eugene Vasiliev
    \date    2014-2015
*/
#pragma once
#include "potential_base.h"

namespace potential {

/** Triaxial Ferrers (n=2) model.
    The density is given by \f$ \rho = \frac{105 M}{32\pi p q Rscale^3} (1-(r/Rscale)^2)^2 \f$,
    where \f$  r = \sqrt{x^2 + (y/p)^2 + (z/q)^2}  \f$ is the triaxial radius;
    the density is zero if r>Rscale.
    The potential is calculated using expressions from Pfenniger(1984) with elliptic integrals, 
    under assumption that q<p<1 strictly (will not work if any of two axes are equal).
*/
class Ferrers: public BasePotentialCar {
public:
    /// Construct the potential for the following parameters: mass, radius, axis ratios p=y/x, q=z/x
    Ferrers(double _mass, double _R, double _axisRatioY, double _axisRatioZ);
    ~Ferrers() {};
    virtual std::string name() const { return myName(); }
    static std::string myName() { return "Ferrers"; }
    virtual coord::SymmetryType symmetry() const { return coord::ST_TRIAXIAL; }
private:
    const double a, b, c;       ///< principal axis of ellipsoidal density 
    const double mass, rho0;    ///< total mass and central density of the model
    double W0[20];              ///< pre-computed coefficients for lambda=0

    virtual void evalCar(const coord::PosCar &pos,
        double* potential, coord::GradCar* deriv, coord::HessCar* deriv2, double time) const;
    virtual double densityCar(const coord::PosCar &pos, double time) const;

    /** compute the array of 20 coefficients W_{ijk} listed in Pfenniger(1984),
        in order of appearance in that paper;
        \param[in] lambda is zero inside the model and >0 outside;
        \param[out] W is the array of 20 coefs  */
    void computeW(double lambda, double W[20]) const;
};

}  // namespace
