/** \file    potential_spheroid.h
    \brief   Density models for spheroidal systems (double-power-law, Nuker, Sersic)
    \author  Eugene Vasiliev
    \date    2015-2020
*/

#pragma once
#include "potential_base.h"
#include "smart.h"
#include <vector>

namespace potential{

/// \name  Spheroidal density profiles
///@{

/** Parameters describing a double-power-law spheroidal density profile.
    The density is given by the Zhao(1996) alpha-beta-gamma model
    multiplied by an optional exponential cutoff, and includes many popular profiles
    (e.g., Dehnen, Einasto, Prugniel-Simien, Gaussian):
    \f$
    \rho = \rho_0  (r/r_0)^{-\gamma} ( 1 + (r/r_0)^\alpha )^{(\gamma-\beta) / \alpha}
    \exp[ -(r/r_{cut})^\xi],
    \f$.
*/
struct SpheroidParam{
    double densityNorm;         ///< density normalization rho_0
    double scaleRadius;         ///< transition radius r_0
    double outerCutoffRadius;   ///< outer cut-off radius r_{cut}
    double cutoffStrength;      ///< steepness of the exponential cutoff xi
    double alpha;               ///< steepness of transition alpha
    double beta;                ///< outer power slope beta
    double gamma;               ///< inner power slope gamma
    double axisRatioY;          ///< axis ratio p (y/x)
    double axisRatioZ;          ///< axis ratio q (z/x)
    /// set up default values for all parameters
    SpheroidParam() :
        densityNorm(0), scaleRadius(1), outerCutoffRadius(INFINITY), cutoffStrength(2),
        alpha(1), beta(4), gamma(1), axisRatioY(1), axisRatioZ(1)
    {}
    double mass() const;        ///< return the total mass of a density profile with these parameters
    static const char* myName() ///< the name of the density model in potential_factory routines
    { static const char* text = "Spheroid"; return text; }
};

/** Parameters describing a Nuker density profile.
    In the spherical case, the projected density is given by
    \f$
    \Sigma(R) = \Sigma_0  (R/R_0)^{-\gamma} ( 1/2 + 1/2 (R/R_0)^\alpha )^{(\gamma-\beta) / \alpha}
    \f$,
    where R_0 is the scale radius, Sigma_0 is the surface density at this radius,
    gamma is the slope of the surface density profile at R --> 0  (0 <= gamma < 2),
    beta  is the slope at R --> infinity  (beta > 2),
    alpha is the steepness of transition between these asymptotic regimes  (alpha > 0).
    The 3d density profile is obtained by deprojecting this expression and interpolating the result.
    Similarly to the Spheroid profile, it has asymptotic power-law behaviour at large and small r:
    the outer asymptotic slope is  beta+1,  i.e.  rho ~ r^{-(beta+1)},
    and the inner slope is gamma+1  when gamma>0, or  1-alpha  when 0<alpha<1,  or 0 when alpha>1.
    In non-spherical cases, the flattening with constant axis ratios is applied to the 3d density,
    so that equidensity surfaces are concentric ellipsoids.
*/
struct NukerParam{
    double surfaceDensity;      ///< surface density at scale radius Sigma_0
    double scaleRadius;         ///< transition radius R_0
    double alpha;               ///< steepness of transition alpha
    double beta;                ///< outer power slope beta
    double gamma;               ///< inner power slope gamma
    double axisRatioY;          ///< axis ratio p (y/x)
    double axisRatioZ;          ///< axis ratio q (z/x)
    /// set up default values for all parameters (NAN means no default)
    NukerParam() :
        surfaceDensity(0), scaleRadius(1),
        alpha(NAN), beta(NAN), gamma(NAN), axisRatioY(1), axisRatioZ(1)
    {}
    double mass() const;        ///< return the total mass of a density profile with these parameters
    static const char* myName() ///< the name of the density model in potential_factory routines
    { static const char* text = "Nuker"; return text; }
};

/** Parameters describing a Sersic density profile.
    In the spherical case, the projected density is given by
    \f$
    \Sigma(R) = \Sigma_0  \exp[ -b (R/R_e)^{1/n} ]
    \f$,
    where Sigma_0 is the central surface density, R_e is the effective radius,
    n is the shape parameter (Sersic index), and b is the internally computed numerical constant,
    approximately equal to 2n - 1/3.
    The 3d density profile is obtained by deprojecting this expression and interpolating the result.
    In non-spherical cases, the flattening with constant axis ratios is applied to the 3d density,
    so that equidensity surfaces are concentric ellipsoids.
    For consistency with other density models, we use the central value of surface density,
    not the one at the effective radius (they are related by \f$  \Sigma_e = \Sigma_0 \exp(-b)  \f$);
    moreover, it is possible to provide the total mass rather than the surface density
    when constructing this model via the `createDensity()` routine.
*/
struct SersicParam{
    double surfaceDensity; ///< central surface density Sigma_0
    double scaleRadius;    ///< effective radius containing 1/2 of the total mass in projection
    double sersicIndex;    ///< shape parameter `n` (Sersic index), should be positive
    double axisRatioY;     ///< axis ratio p (Y/X)
    double axisRatioZ;     ///< axis ratio q (Z/X)
    /// set up default values for all parameters
    SersicParam() :
        surfaceDensity(0), scaleRadius(1), sersicIndex(4), axisRatioY(1), axisRatioZ(1)
    {}
    double b() const;      ///< compute the numerical coefficient b as a function of n
    double mass() const;   ///< return the total mass of a density profile with these parameters
    static const char* myName() ///< the name of the density model in potential_factory routines
    { static const char* text = "Sersic"; return text; }
};

/// helper routine to construct a one-dimensional function describing a double-power-law profile
math::PtrFunction createSpheroidDensity(const SpheroidParam& params);

/// helper routine to construct a one-dimensional function describing a Nuker profile
math::PtrFunction createNukerDensity(const NukerParam& params);

/// helper routine to construct a one-dimensional function describing a Sersic profile
math::PtrFunction createSersicDensity(const SersicParam& params);

/** Density profile described by an arbitrary function f of ellipsoidal radius:
    \f$  \rho(x,y,z) = f(\tilde r),  \tilde r = \sqrt{x^2 + (y/p)^2 + (z/q)^2} \f$,
    where p = y/x and q = z/x  are two axis ratios.
*/
class SpheroidDensity: public BaseDensity{
public:
    /// construct a generic profile with a user-specified one-dimensional function and two axis ratios
    SpheroidDensity(const math::PtrFunction& fnc, const double axisRatioY=1, const double axisRatioZ=1) :
        p2(pow_2(axisRatioY)), q2(pow_2(axisRatioZ)), rho(fnc) {}

    /// construct a model with the provided Spheroid density parameters (convenience overload)
    SpheroidDensity(const SpheroidParam& params) :
        p2(pow_2(params.axisRatioY)), q2(pow_2(params.axisRatioZ)),
        rho(createSpheroidDensity(params)) {}

    /// construct a model with the provided Nuker density parameters (convenience overload)
    SpheroidDensity(const NukerParam& params) :
        p2(pow_2(params.axisRatioY)), q2(pow_2(params.axisRatioZ)),
        rho(createNukerDensity(params)) {}

    /// construct a model with the provided Sersic density parameters (convenience overload)
    SpheroidDensity(const SersicParam& params) :
        p2(pow_2(params.axisRatioY)), q2(pow_2(params.axisRatioZ)),
        rho(createSersicDensity(params)) {}

    virtual coord::SymmetryType symmetry() const {
        return p2==1 ? (q2==1 ? coord::ST_SPHERICAL : coord::ST_AXISYMMETRIC) : coord::ST_TRIAXIAL; }
    virtual const char* name() const { return myName(); }
    /// the name reported by the density model is always 'Spheroid',
    /// regardless of whether it was initialized from SersicParam or NukerParam or SpheroidParam
    static const char* myName() { return SpheroidParam::myName(); }
private:
    const double p2, q2;    ///< squared axis ratios p=y/x, q=z/x
    math::PtrFunction rho;  ///< one-dimensional density as a function of elliptical radius
    virtual double densityCar(const coord::PosCar &pos) const;
    virtual double densityCyl(const coord::PosCyl &pos) const;
    virtual double densitySph(const coord::PosSph &pos) const;
};

///@}

} // namespace potential
