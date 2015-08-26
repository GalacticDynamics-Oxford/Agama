/** \file    potential_cylspline.h
    \brief   potential approximation based on 2d spline in cylindrical coordinates
    \author  Eugene Vasiliev
    \date    2014-2015
**/
#pragma once
#include "potential_base.h"
#include "particles_base.h"
#include "math_spline.h"

namespace potential {

/** Angular expansion of potential in azimuthal angle 
    with coefficients being 2d spline functions of R,z.
    This is a very flexible and powerful potential approximation that can be used 
    to represent arbitrarily flattened axisymmetric or non-axisymmetric 
    (if the number of azimuthal terms is >0) density profiles.
*/
class CylSplineExp: public BasePotentialCyl
{
public:
    /// init potential from analytic mass model specified by its density profile
    /// (using CPotentialDirect for intermediate potential computation)
    CylSplineExp(unsigned int _Ncoefs_R, unsigned int _Ncoefs_z, unsigned int _Ncoefs_phi, 
        const BaseDensity& density, 
        double radius_min=0, double radius_max=0, double z_min=0, double z_max=0);

    /// init potential from analytic mass model specified by a potential-density pair
    CylSplineExp(unsigned int _Ncoefs_R, unsigned int _Ncoefs_z, unsigned int _Ncoefs_phi, 
        const BasePotential& potential, 
        double radius_min=0, double radius_max=0, double z_min=0, double z_max=0);

    /// init potential from stored coefficients
    CylSplineExp(const std::vector<double>& gridR, const std::vector<double>& gridz, 
        const std::vector< std::vector<double> >& coefs);

    /// init potential from N-body snapshot
    CylSplineExp(unsigned int _Ncoefs_R, unsigned int _Ncoefs_z, unsigned int _Ncoefs_phi, 
        const particles::PointMassArray<coord::PosCyl>& points, SymmetryType _sym=ST_TRIAXIAL, 
        double radius_min=0, double radius_max=0, double z_min=0, double z_max=0);

    ~CylSplineExp() {};
    virtual const char* name() const { return myName(); };
    static const char* myName() { return "CylSplineExp"; };
    virtual SymmetryType symmetry() const { return mysymmetry; };

    /** retrieve coefficients of potential approximation.
        \param[out] gridR will be filled with the array of R-values of grid nodes
        \param[out] gridz will be filled with the array of z-values of grid nodes
        \param[out] coefs will contain array of sequentially stored 2d arrays 
        (the size of the outer array equals the number of terms in azimuthal expansion,
        inner arrays contain gridR.size()*gridz.size() values). */
    void getCoefs(std::vector<double> &gridR, std::vector<double>& gridz, 
        std::vector< std::vector<double> > &coefs) const;

private:
    SymmetryType mysymmetry;                  ///< may have different type of symmetry
    std::vector<double> grid_R, grid_z;       ///< nodes of the grid in cylindrical radius and vertical direction
    double Rscale;                            ///< scaling coefficient for transforming the interpolated potential; computed as -Phi(0)/Mtotal.
    std::vector<math::CubicSpline2d> splines; ///< array of 2d splines (for each m-component in the expansion in azimuthal angle)
    double C00, C20, C40, C22;                ///< multipole coefficients for extrapolation beyond the grid

    /// compute potential and its derivatives
    virtual void evalCyl(const coord::PosCyl &pos,
        double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const;

    /// create interpolation grid and compute potential at grid nodes
    void initPot(unsigned int _Ncoefs_R, unsigned int _Ncoefs_z, 
        unsigned int _Ncoefs_phi, const BasePotential& potential, 
        double radius_min, double radius_max, double z_min, double z_max);

    /// create 2d interpolation splines in scaled R,z grid
    void initSplines(const std::vector< std::vector<double> > &coefs);

    /// compute m-th azimuthal harmonic of potential 
    /// (either by Fourier transform or calling the corresponding method of DirectPotential)
    double computePhi_m(double R, double z, int m, const BasePotential& potential) const;
};

}  // namespace
