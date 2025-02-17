/** \file    df_spherical.h
    \brief   Spherically-symmetric (an)isotropic distribution functions
    \date    2016-2019
    \author  Eugene Vasiliev
*/
#pragma once
#include "df_base.h"
#include "actions_spherical.h"
#include "potential_utils.h"
#include "math_spline.h"

namespace df{

/** Construct a spherical isotropic distribution function of phase volume h for the given pair
    of density and potential profiles (which need not be related through the Poisson equation),
    using the Eddington inversion formula.
    \param[in]  density   is any one-dimensional function returning rho(r); may be constructed
    from any `BaseDensity` object using the `Sphericalized<BaseDensity>` wrapper class.
    \param[in]  potential  is any one-dimensional function representing the spherically-symmetric
    potential (may be constructed using the `Sphericalized<BasePotential>` wrapper class).
    \param[in,out]  gridh  is the array of phase volumes at which the DF is defined;
    if this array is empty on input, a plausible grid will be created automatically,
    but in any case it may later be modified by this routine, to eliminate negative DF values.
    \param[out]     gridf  is the array of DF values at corresponding nodes of gridh,
    guaranteed to contain only positive values (thus a LogLogSpline may be constructed from it).
    \throw  std::runtime_error if no valid DF can be constructed.
*/
void createSphericalIsotropicDF(
    const math::IFunction& density,
    const math::IFunction& potential,
    /*in/out*/ std::vector<double>& gridh,
    /*output*/ std::vector<double>& gridf);


/** Construct a spherical isotropic distribution function using the Eddington formula.
    This is a convenience overloaded routine that first computes the values of f at a grid in h,
    using the previous function with the same name, and then creates an instance of LogLogSpline
    interpolator to represent log(f) as a function of log(h) in terms of a cubic spline.
    \param[in]  density    is any one-dimensional function returning rho(r);
    \param[in]  potential  is any one-dimensional function representing the potential;
    \return  an instance of math::LogLogSpline interpolator for the distribution function.
*/
inline math::LogLogSpline createSphericalIsotropicDF(
    const math::IFunction& density,
    const math::IFunction& potential)
{
    std::vector<double> gridh, gridf;
    createSphericalIsotropicDF(density, potential, gridh, gridf);
    return math::LogLogSpline(gridh, gridf);
}


/** Construct a spherical isotropic distribution function f(h) from an array of particles.
    \param[in]  hvalues  is the array of values of h (phase volume) for each particle;
    \param[in]  masses   is the array of particle masses;
    \param[in]  gridSize is the number of nodes in the interpolated function
    (20-40 is a reasonable choice); the grid nodes are assigned automatically.
    \return     an instance of LogLogSpline representing the spherical isotropic function f(h).
    \throw  std::invalid_argument exception if the input data is bad (e.g., masses are negative,
    or array sizes are not equal, etc.)
*/
math::LogLogSpline fitSphericalIsotropicDF(
    const std::vector<double>& hvalues,
    const std::vector<double>& masses,
    unsigned int gridSize);


/** A wrapper class for representing an arbitrary spherical isotropic distribution function,
    originally expressed in terms of phase volume `h`, as a regular action-based DF.
    The phase volume `h` in a spherically-symmetric potential is uniquely related to energy,
    and plays the same role as actions in a more general geometry: it is an adiabatic invariant,
    f(h) dh is the mass in the given volume of phase space regardless of the potential,
    and f(h) has the same expression in any potential.
    In order to put these models on the common ground with other action-based DFs,
    we express them in terms of actions, by using two successive mappings:
    (Jr, Jz+|Jphi|) => E  is provided by a spherical action mapper;
    E => h(E)  is provided by the PhaseVolume class,
    and finally h is used as the argument of DF.
    Both these auxiliary transformations are constructed for a given spherically-symmetric
    potential, but then the resulting DF can be viewed as a function of actions only,
    and used in an arbitrary potential just as any other f(J); the intermediate mappings
    become part of its definition. Of course, it may no longer yield isotropic velocity
    distributions -- hence the name 'quasi'.
    If one needs to construct a spherical isotropic DF for a particular potential-density pair
    using the Eddington formula, it is better to use the class `QuasiSphericalCOM`, which performs
    both tasks (constructing the DF and computing its value as a function of actions) with lower
    overhead costs (it only performs the conversion J => E, but not E => h).
    On the other hand, this wrapper class should be used if one already has a DF in the form f(h).
*/
class QuasiSphericalIsotropic: public BaseDistributionFunction{
    const math::LogLogSpline df;              ///< one-dimensional function of spherical phase volume h
    const potential::PhaseVolume pv;          ///< correspondence between E and h
    const actions::ActionFinderSpherical af;  ///< correspondence between (Jr,L) and E
public:
    QuasiSphericalIsotropic(const math::LogLogSpline& _df, const potential::BasePotential& potential) :
        df(_df), pv(potential::Sphericalized<potential::BasePotential>(potential)), af(potential) {}

    virtual void evalDeriv(const actions::Actions &J, double *f,
        DerivByActions *deriv=NULL) const
    {
        if(deriv) {
            // derivatives are obtained by a simple application of the chain rule
            double Omegar, Omegaz, E = af.E(J), L = J.Jz + (J.Jphi>=0 ? J.Jphi : -J.Jphi);
            af.Jr(E, L, &Omegar, &Omegaz);
            double Omegaphi = J.Jphi>=0 ? Omegaz : -Omegaz;
            double h, dhdE, dfdh;
            pv.evalDeriv(E, &h, &dhdE);
            df.evalDeriv(h,  f, &dfdh);
            deriv->dbyJr   = dfdh * dhdE * Omegar;  // Omega_r = dE/dJ_r, etc.
            deriv->dbyJz   = dfdh * dhdE * Omegaz;
            deriv->dbyJphi = dfdh * dhdE * Omegaphi;
        } else
            *f = df(pv(af.E(J)));
    }
};


/// a triplet of classical integrals of motion in a spherical potential:
/// energy E, total angular momentum L (non-negative), and z-component of angular momentum Lz
struct ClassicalIntegrals{
    double E, L, Lz;
};

/// derivatives of the DF expressed in terms of classical integrals
struct DerivByClassicalIntegrals{
    double dbyE, dbyL, dbyLz;
};

/** Parent class for spherical (an)isotropic distribution functions expressed in terms of E,L,Lz.
    These functions are constructed for a given potential-density pair using a suitable inversion
    formula (e.g., Eddington in the isotropic case), and provide two interfaces for computing the
    DF value.
    The 'classical' form `f.value(E,L,Lz)` can only be used in the same potential Phi
    that was used in constructing the DF, but incurs no extra costs.
    The 'universal' action-based form `f.value(J)` can be used in any potential Phi_1,
    and works as follows: first the actions are converted to the triplet E,L,Lz using the
    spherical action finder created together with the DF in the original potential Phi,
    and then the DF value at these arguments is returned. In essense, this assumes that the DF
    was adiabatically transformed into the new potential, keeping its numerical value as a function
    of actions (hence the original potential is used in this context).
    The second form obviously involves extra costs associated with the transformation J => E,
    but allows this DF to be used in the general routines for computing moments, sampling, etc.
    In the future, optimized code paths for this subtype of DFs may be implemented in these routines.
    If the potential Phi_1 is not spherical, the DF will also produce a non-spherical and possibly
    anisotropic density and velocity dispersion profile (even if it was originally isotropic),
    hence the name 'quasi'.
    The descendant classes implement the specific way of constructing a DF and compute its value
    as a function of E,L,Lz.
*/
class QuasiSpherical: public BaseDistributionFunction {
public:
    const actions::ActionFinderSpherical af;  ///< correspondence between (Jr,L) and E
    QuasiSpherical(const math::IFunction& potential) :
        af(potential::FunctionToPotentialWrapper(potential)) {}
    virtual ~QuasiSpherical() {}

    /** convert actions to E,L,Lz, then compute the DF and optionally its derivatives w.r.r. J */
    virtual void evalDeriv(const actions::Actions &J, double *value,
        DerivByActions *deriv=NULL) const
    {
        double signJphi = J.Jphi >=0 ? 1 : -1;
        ClassicalIntegrals ints;
        DerivByClassicalIntegrals d;
        ints.E  = af.E(J);
        ints.L  = J.Jz + J.Jphi * signJphi;
        ints.Lz = J.Jphi;
        evalDeriv(ints, /*output*/ value, deriv ? &d : NULL);
        if(deriv) {
            double Omegar, Omegaz;
            af.Jr(ints.E, ints.L, &Omegar, &Omegaz);
            double Omegaphi = Omegaz * signJphi;
            deriv->dbyJr   = d.dbyE * Omegar;  // Omega_r = dE/dJ_r, etc.
            deriv->dbyJz   = d.dbyE * Omegaz   + d.dbyL;
            deriv->dbyJphi = d.dbyE * Omegaphi + d.dbyL * signJphi + d.dbyLz;
        }
    }

    /** compute the value of distribution function for the given E, L, Lz,
        and optionally its derivatives w.r.t. the classical integrals
        (to be implemented in derived classes)
    */
    virtual void evalDeriv(const ClassicalIntegrals& ints,
        double *value, DerivByClassicalIntegrals *der=NULL) const=0;
};


/** Spherical isotropic (Eddington) or anisotropic (Cuddeford-Osipkov-Merritt) distribution function
    constructed for a given combination of potential and density.
*/
class QuasiSphericalCOM: public QuasiSpherical {
    const double invPhi0, beta0, r_a, rotFrac, Jphi0;
    const math::LogLogSpline df;
public:
    /** construct the DF for the provided density/potential pair and anisotropy parameters:
        \param[in]  density    is the spherically-symmetric density profile specified by a function
        of one variable; one may use `potential::Sphericalized<potential::BaseDensity>(density)`
        to represent an instance of BaseDensity-derived class as a 1d function;
        \param[in]  potential  is the total spherical potential specified by a 1d function
        (doesn't need to be related to density via the Poisson equation); again, one may use
        `potential::Sphericalized<potential::BasePotential>(pot)` to convert a BasePotential-derived
        class into a function of one variable;
        \param[in]  beta0      is the value of anisotropy coefficient at r-->0, should be -1/2<=beta0<=1
        \param[in]  r_a        is the Osipkov-Merritt anisotropy radius (may be infinite).
        \param[in]  rotFrac    optionally introduces some rotation by multiplying the DF by
        [1 + rotFrac * tanh(Jphi/Jphi0)];
        rotFrac is the relative amplitude of the odd-Jphi component (-1 to 1, 0 means no rotation).
        \param[in]  Jphi0      controls the sharpness of transition between positive and negative
        parts of the azimuthal velocity distribution, as defined by the above expression.
    */
    QuasiSphericalCOM(const math::IFunction& density, const math::IFunction& potential,
        double beta0=0, double r_a=INFINITY, double rotFrac=0, double Jphi0=0);

    using QuasiSpherical::evalDeriv;  // bring both overloaded functions into scope

    virtual void evalDeriv(const ClassicalIntegrals& ints,
        double *value, DerivByClassicalIntegrals *deriv=NULL) const;
};

}  // namespace
