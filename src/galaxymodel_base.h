/** \file    galaxymodel_base.h
    \brief   A complete galaxy model and associated routines (computation of moments, sampling)
    \date    2015-2021
    \author  Eugene Vasiliev, Payel Das, James Binney
*/
#pragma once
#include "potential_base.h"
#include "actions_base.h"
#include "df_base.h"
#include "particles_base.h"
#include "math_sample.h"

/// A complete galaxy model (potential, action finder and distribution function) and associated routines
namespace galaxymodel{

/** Base class for selection functions that depend on (x,v);
    the value of the distribution function is multiplied by the selection function in all
    operations provided by this module.
*/
class BaseSelectionFunction {
public:
    virtual ~BaseSelectionFunction() {}
    /// return a value in the range [0..1]
    virtual double value(const coord::PosVelCar& point) const = 0;
    /// evaluate the function at several input points at once (could be more efficient than one-by-one)
    virtual void evalmany(const size_t npoints, const coord::PosVelCar points[], double values[]) const {
        // default implementation is a simple sequential loop
        for(size_t p=0; p<npoints; p++)
            values[p] = value(points[p]);
    }
};

/** A trivial selection function that is identically unity */
class SelectionFunctionTrivial: public BaseSelectionFunction {
public:
    SelectionFunctionTrivial() {}
    virtual double value(const coord::PosVelCar& /*point*/) const { return 1; }
};

/** A single instance of the trivial selection function used as a default ingredient in GalaxyModel */
extern const SelectionFunctionTrivial selectionFunctionTrivial;

/** A rather simple selection function that depends on the distance from a given point x0:
    S(x) = exp[ - (|x-x0|/R0)^xi ],
    where R0 is the cutoff radius and xi is the cutoff steepness
*/
class SelectionFunctionDistance: public BaseSelectionFunction {
    const coord::PosCar point0;
    const double radius, steepness;
public:
    SelectionFunctionDistance(const coord::PosCar& point0, double radius, double steepness);
    virtual double value(const coord::PosVelCar& point) const;
};


/** Data-only structure defining a galaxy model:
    a combination of potential, action finder, distribution function, and selection function.
    Its purpose is to temporarily bind together the four common ingredients that are passed
    to various functions; however, as it only keeps references and not shared pointers, 
    it should not generally be used for a long-term storage.
*/
struct GalaxyModel{
public:
    const potential::BasePotential&     potential;  ///< gravitational potential
    const actions::BaseActionFinder&    actFinder;  ///< action finder for the given potential
    const df::BaseDistributionFunction& distrFunc;  ///< distribution function expressed in terms of actions
    const BaseSelectionFunction&        selFunc;    ///< selection function sf(x,v)

    /** Create an instance of the galaxy model from the four ingredients
        (with a trivial selection function by default, if nothing else is provided) */
    GalaxyModel(
        const potential::BasePotential& pot,
        const actions::BaseActionFinder& af,
        const df::BaseDistributionFunction& df,
        const BaseSelectionFunction& sf = selectionFunctionTrivial) :
    potential(pot), actFinder(af), distrFunc(df), selFunc(sf) {}
};


/** Compute density, first-order, and second-order moments of velocity in Cartesian coordinates
    either at the given 3d input point X,Y,Z (when the input point is coord::PosCar)
    or projected (integrated along Z) at the point X,Y (when the input is coord::PosProj).
    The 'observed' coordinate system XYZ may be arbitrarily oriented with respect to the 'intrinsic'
    coordinate system xyz of the model; in the projected case, Z is the line of sight.
    The output consists of the density (or projected density), three components of mean velocity,
    and the tensor of second moments of velocity (intrinsic or projected) in any combination:
    if some of these values are not needed, pass NULL as the corresponding argument, and
    they will not be computed.
    In case of a multicomponent DF and if the flag 'separate' is set to true, the output values
    are provided separately for each component, so the corresponding non-NULL arguments must point
    to arrays of length df.numValues().
    \param[in]  model  is the galaxy model (potential + DF + action finder).
    \param[in]  point  is the position at which the quantities should be computed.
    \param[out] density  will contain the integral of DF over all velocities (and optionally by Z.
    \param[out] velocityFirstMoment  will contain the mean velocity.
    \param[out] velocitySecondMoment will contain the tensor of mean squared velocity components.
    \param[in]  separate    whether to compute moments separately for each element of
    a multicomponent DF; in this case, the output arrays should have length equal to df.numValues().
    \param[in]  orientation  is the triplet of Euler angles specifying the orientation of
    the 'observed' coordinate system XYZ with respect to 'intrinsic' model coordinates xyz;
    in particular, beta is the inclination angle, alpha is irrelevant for axisymmetric systems,
    and when all three angles are zero, XYZ coincides with xyz.
    \param[in]  reqRelError is the required relative error in the integral.
    \param[in]  maxNumEval  is the maximum number of evaluations in integral.
*/
void computeMoments(
    const GalaxyModel& model,
    const coord::PosCar& point,   // variant of the function for intrinsic moments at a 3d point
    // output arrays - one element per DF component
    double density[],
    coord::VelCar velocityFirstMoment[],
    coord::Vel2Car velocitySecondMoment[],
    // optional input parameters
    bool separate=false,
    const coord::Orientation& orientation=coord::Orientation(),
    double reqRelError=1e-3,
    int maxNumEval=1e5);

void computeMoments(
    const GalaxyModel& model,
    const coord::PosProj& point,   // variant of the function for projected moments at a 2d point
    // output arrays - one element per DF component
    double density[],
    coord::VelCar velocityFirstMoment[],
    coord::Vel2Car velocitySecondMoment[],
    // optional input parameters
    bool separate=false,
    const coord::Orientation& orientation=coord::Orientation(),
    double reqRelError=1e-3,
    int maxNumEval=1e5);


/** Compute the value of 'projected distribution function' at the given point
    specified by two coordinates in the sky plane (X,Y) and possibly some velocity components.
    The DF is integrated along the line of sight (Z) and all three velocity components -
    either across the entire range of available velocity, or over the Gaussian uncertainties.
    The orientation of the 'observational' coordinate system XYZ with respect to the 'intrinsic'
    coordinate system xyz of the model is specified by three Euler angles alpha, beta, gamma.
    The input point has precisely known X,Y coordinates, unknown Z (i.e. the DF is integrated over Z),
    and three velocity components vX,vY,vZ that may be known precisely, imprecisely (with some
    uncertainty), or not at all (with infinite uncertainty).
    When a given velocity component is not known precisely, the DF is integrated over the
    available range of this component, weighted with the Gaussian the measurement uncertainty
    (unless it is infinite, in which case the weight factor is identically unity).
    Thus the uncertainties on each velocity component can be anywhere between 0 and INFINITY
    (including endpoints), with the only restriction that vX,vY uncertainties can be either
    both finite or both infinite.
    Note that in the case of a finite but relatively very large uncertainty compared to the
    available range of velocities in the model, the result is almost identical to the case
    of infinite uncertainty, but differs in normalization:  e.g., when vZ_error >> vZ,
    projectedDF(..., vZ, vZ_error) =
    projectedDF(...,  0, INFINITY) * exp(-0.5 * (vZ/vZ_error)^2) / sqrt(2*pi) / vZ_error.
    When all three velocity uncertainties are infinite, the result is the surface density.
    \param[in]  model  is the galaxy model.
    \param[in]  point  specifies the two coordinates in the 'observed' coordinate system.
    \param[in]  vel  are the three velocity components in the 'observed' coordinate system.
    \param[in]  velerr  are their respective uncertainties;
    note that if the uncertainty is INFINITY, the corresponding velocity value is irrelevant.
    \param[out] result  will contain the values of projected DF (a single value if separate==false,
    otherwise an array of length df.numValues() - should be allocated by the caller).
    \param[in]  separate    whether to compute moments separately for each element of
    a multicomponent DF; in this case, the result array should have length equal to df.numValues().
    \param[in]  orientation  is the triplet of Euler angles specifying the orientation of
    the 'observed' coordinate system with respect to 'intrinsic' model coordinates;
    in particular, beta is the inclination angle, and alpha is irrelevant for axisymmetric systems.
    \param[in]  reqRelError is the required relative error in the integral.
    \param[in]  maxNumEval  is the maximum number of evaluations in integral.
    \throws  std::invalid_argument  if the combination of velocity uncertainties is unsupported,
    or any exception that occurs during the integration.
*/
void computeProjectedDF(
    const GalaxyModel& model,
    const coord::PosProj& point,
    const coord::VelCar& vel,
    const coord::VelCar& velerr,
    /*output*/ double result[],
    // optional input arguments
    bool separate=false,
    const coord::Orientation& orientation=coord::Orientation(),
    double reqRelError=1e-3,
    int maxNumEval=1e5);


/** Compute the velocity distribution functions (VDF) in three directions at the given point in space.
    The input point and the orthogonal velocity axes are specified in the 'observed' Cartesian
    coordinate system XYZ, which may be arbitrarily oriented with respect to the 'intrinsic'
    coordinate system xyz of the model.
    The VDF is represented as a weighted sum of B-splines of degree N:
    0 means histogramming, 1 - linear interpolation, 3 - cubic interpolation.
    Higher degrees not only result in a smoother VDF, but also are easier to compute:
    the integration over the velocity space is carried out for all B-splines simultaneously,
    and the smoother are the basis functions, the easier is it for the quadrature to achieve
    the target level of accuracy.
    \tparam     N is the degree of B-splines (0, 1 or 3).
    \param[in]  model  is the galaxy model.
    \param[in]  point  is the position at which the VDF are computed (either a 3d or a 2d point).
    In the first variant (when point is coord::PosCar), the VDF is computed at the given 3d point,
    in the second one (when point is coord::PosProj), the VDF is further integrated over the Z axis
    (line of sight) and represents the distribution of line-of-sight and sky-plane velocities.
    \param[in]  gridvX  is the array of grid nodes for the v_X velocity component,
    which define the set of basis functions for interpolation.
    The nodes must be sorted in increasing order, and typically should range from -V_escape to V_escape.
    A convenience overloaded version of this routine creates a suitable grid automatically.
    \param[in]  gridvY  is the same thing for the VDF of v_Y.
    \param[in]  gridvZ  is the same thing for v_Z (the line-of-sight component).
    \param[out] density  will contain the density at the given point XYZ (if projected==false)
    or the surface density at the point XY (if projected==true): it is used to normalize the
    amplitudes of the interpolator, and comes for free during integration anyway.
    If the flag separate==false, this will be a single value, otherwise an array of densities of
    each element of a multicomponent DF (should be allocated by the caller); same convention is
    used for other output arrays.
    \param[out] amplvX  should point to a vector (if separate==false) or an array of df.numValues()
    vectors (if separate==true), which will be filled with amplitudes for constructing an interpolated
    VDF for the v_X component, namely:
    ~~~~
    math::BsplineInterpolator1d<N> interp(gridvX);
    double f_of_v = interp.interpolate(v, amplvX);  // amplvX was obtained from this routine
    ~~~~
    The number of elements in amplvX is gridvX.size() + N - 1;
    for N=0 they are essentially the heights of the histogram bars between gridvX[i] and gridvX[i+1],
    and for N=1 the values of amplvX at each node of gridvX coincide with the interpolated f(v);
    however for higher N the correspondence is not so obvious.
    The VDF is normalized such that \f$  \int_{-V_{escape}}^{V_{escape}} f(v) dv = 1  \f$
    (if the provided interval specified by gridvX is smaller than V_escape, the integral
    over the extend ot gridvX will be smaller than unity).
    \param[out] amplvY  will contain the amplitudes for the v_Y component.
    \param[out] amplvZ  will contain the amplitudes for the v_Z component.
    \param[in]  separate    whether to compute moments separately for each element of
    a multicomponent DF; in this case, the output arrays should have length equal to df.numValues().
    \param[in]  orientation specifies the orientation of the observed coordinate system XYZ
    with respect to the intrinsic system xyz of the model, parametrized by three Euler angles;
    the default zero values of these angles make XYZ coincide with xyz.
    \param[in]  reqRelError is the required relative error in the integrals.
    \param[in]  maxNumEval  is the maximum number of evaluations for all integral together.
    \throw  std::invalid_argument if the input arrays are incorrect.
*/
template <int N>
void computeVelocityDistribution(
    const GalaxyModel& model,
    const coord::PosCar& point,   // variant of the function for intrinsic vdf at a 3d point
    const std::vector<double>& gridvX,
    const std::vector<double>& gridvY,
    const std::vector<double>& gridvZ,
    // output arrays - one element per DF component (if separate==true), otherwise just one element
    double density[],
    std::vector<double> amplvX[],
    std::vector<double> amplvY[],
    std::vector<double> amplvZ[],
    // optional input parameters
    bool separate=false,
    const coord::Orientation& orientation=coord::Orientation(),
    double reqRelError=1e-2,
    int maxNumEval=1e6);

template <int N>
void computeVelocityDistribution(
    const GalaxyModel& model,
    const coord::PosProj& point,   // variant of the function for projected vdf at a 2d point
    const std::vector<double>& gridvX,
    const std::vector<double>& gridvY,
    const std::vector<double>& gridvZ,
    // output arrays - one element per DF component (if separate==true), otherwise just one element
    double density[],
    std::vector<double> amplvX[],
    std::vector<double> amplvY[],
    std::vector<double> amplvZ[],
    // optional input parameters
    bool separate=false,
    const coord::Orientation& orientation=coord::Orientation(),
    double reqRelError=1e-2,
    int maxNumEval=1e6);


/** A convenience overloaded version of the routine that automatically initializes the velocity
    grids so that they cover the range [-v_escape, +v_escape] with the given number of points.
    Arguments have the same meaning as for the actual computeVelocityDistribution() routine;
    \param[in]  gridsize  is the size of velocity grid (equal for all three components);
    \param[out] gridv  will be initialized by this routine.
*/
template <int N>
void computeVelocityDistribution(
    const GalaxyModel& model,
    const coord::PosCar& point,   // variant of the function for intrinsic vdf at a 3d point
    size_t gridsize,
    // velocity grid will be initialized by this routine
    std::vector<double>& gridv,
    // output arrays - one element per DF component (if separate==true), otherwise just one element
    double density[],
    std::vector<double> amplvX[],
    std::vector<double> amplvY[],
    std::vector<double> amplvZ[],
    // optional input parameters
    bool separate=false,
    const coord::Orientation& orientation=coord::Orientation(),
    double reqRelError=1e-2,
    int maxNumEval=1e6);

template <int N>
void computeVelocityDistribution(
    const GalaxyModel& model,
    const coord::PosProj& point,   // variant of the function for projected vdf at a 2d point
    size_t gridsize,
    // velocity grid will be initialized by this routine
    std::vector<double>& gridv,
    // output arrays - one element per DF component (if separate==true), otherwise just one element
    double density[],
    std::vector<double> amplvX[],
    std::vector<double> amplvY[],
    std::vector<double> amplvZ[],
    // optional input parameters
    bool separate=false,
    const coord::Orientation& orientation=coord::Orientation(),
    double reqRelError=1e-2,
    int maxNumEval=1e6);


/** this will be redesigned */
void computeProjection(
    const GalaxyModel& model,
    const math::IFunctionNdim& spatialSelection,
    const double Xlim[2],
    const double Ylim[2],
    const coord::Orientation& orientation,
    double* result,
    double reqRelError=1e-3,
    int maxNumEval=1e5);


/** Compute the total mass represented by the DF in the region determined by the selection function
    (if the latter is trivial, the mass should be equivalent to df.totalMass()
    up to integration errors, but is much more computationally heavy because this involves
    integration over 6d phase space and action computation for every point allowed by the SF).
    \param[in]  model  is the galaxy model.
    \param[out] result  will contain the integral(s) of DF * SF over the entire 6d phase space.
    \param[in]  separate    whether to compute moments separately for each element of
    a multicomponent DF; in this case, the output arrays should have length equal to df.numValues().
    \param[in]  reqRelError is the required relative error in the integral.
    \param[in]  maxNumEval  is the maximum number of evaluations in integral.
    Note that if the SF is very localized, the integration may terminate early with a zero result,
    if it is not able to locate the region where the SF is nonzero.
*/
void computeTotalMass(
    const GalaxyModel& model,
    // output
    double* result,
    // optional input
    bool separate=false,
    double reqRelError=1e-3,
    int maxNumEval=1e6);


/** Generate N-body samples of the distribution function multiplied by the selection function
    by sampling in position/velocity space:
    use action finder to compute the actions corresponding to the given point,
    and evaluate the value of DF times SF at the given actions.
    The output points have uniform weights.
    \param[in]  model  is the galaxy model;
    \param[in]  numPoints  is the required number of samples;
    \param[in]  method  (optional) is the mode of operation for the sampling routine 
                (see the docstring of math::sampleNdim for details);
    \param[in,out]  state  is the seed for the pseudo-random number generator;
                if not provided (NULL), use the global state.
    \returns    a new array of particles (position/velocity/mass)
                sampled from the distribution function;
*/
particles::ParticleArrayCar samplePosVel(
    const GalaxyModel& model, const size_t numPoints,
    math::SampleMethod method=math::SM_DEFAULT, math::PRNGState* state=NULL);


/** Sample the density profile by discrete points.
    \param[in]  dens  is the density model;
    \param[in]  numPoints  is the required number of sampling points;
    \param[in]  method  (optional) is the mode of operation for the sampling routine 
                (see the docstring of math::sampleNdim for details);
    \param[in,out]  state  is the seed for the pseudo-random number generator;
                if not provided (NULL), use the global state.
    \returns    a new array with the sampled coordinates and masses.
*/
particles::ParticleArray<coord::PosCyl> sampleDensity(
    const potential::BaseDensity& dens, const size_t numPoints,
    math::SampleMethod method=math::SM_DEFAULT, math::PRNGState* state=NULL);


/// Helper class for providing a BaseDensity interface to a density computed via integration over DF
class DensityFromDF: public potential::BaseDensity{
public:
    DensityFromDF(const GalaxyModel& _model, double _relError, unsigned int _maxNumEval) :
        model(_model), relError(_relError), maxNumEval(_maxNumEval) {}

    // in the current version, action finders are only implemented for axisymmetric potentials
    virtual coord::SymmetryType symmetry() const { return coord::ST_AXISYMMETRIC; }
    virtual std::string name() const { return "DensityFromDF"; }
    virtual double enclosedMass(const double) const { return NAN; } /// should never be used -- too slow
private:
    const GalaxyModel model;  ///< aggregate of potential, action finder and DF
    double       relError;    ///< requested relative error of density computation
    unsigned int maxNumEval;  ///< max # of DF evaluations per one density calculation

    /// compute the density as the integral of DF over velocity at a given position
    virtual double densityCar(const coord::PosCar &pos, double /*time*/) const {
        double result;
        computeMoments(model, pos, &result, NULL, NULL, false, coord::Orientation(), relError, maxNumEval);
        return result;
    }

    virtual double densityCyl(const coord::PosCyl &pos, double time) const {
        return densityCar(toPosCar(pos), time); }

    virtual double densitySph(const coord::PosSph &pos, double time) const {
        return densityCar(toPosCar(pos), time); }

    /// functions for computing the density for an array of points in three coordinate systems.
    /// \note OpenMP-parallelized loop over points.
    virtual void evalmanyDensityCar(const size_t npoints, const coord::PosCar pos[],
        /*output*/ double values[], /*input*/ double t=0) const;
    virtual void evalmanyDensityCyl(const size_t npoints, const coord::PosCyl pos[],
        /*output*/ double values[], /*input*/ double t=0) const;
    virtual void evalmanyDensitySph(const size_t npoints, const coord::PosSph pos[],
        /*output*/ double values[], /*input*/ double t=0) const;
};

}  // namespace