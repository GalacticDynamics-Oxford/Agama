/** \file    galaxymodel_base.h
    \brief   A complete galaxy model and associated routines (computation of moments, sampling)
    \date    2015-2020
    \author  Eugene Vasiliev, Payel Das, James Binney
*/
#pragma once
#include "potential_base.h"
#include "actions_base.h"
#include "df_base.h"
#include "particles_base.h"

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


/** Compute density, first-order, and second-order moments of velocity in cylindrical coordinates;
    if some of them are not needed, pass NULL as the corresponding argument, and it will not be 
    computed. In case of a multicomponent DF and if the flag 'separate' is set to true,
    the output values are provided separately for each component, so the corresponding non-NULL
    arguments must point to arrays of length df.numValues().
    \param[in]  model  is the galaxy model (potential + DF + action finder);
    \param[in]  point  is the position at which the quantities should be computed;
    \param[out] density  will contain the integral of DF over all velocities;
    \param[out] velocityFirstMoment  will contain the mean streaming velocity <v_phi>;
    \param[out] velocitySecondMoment will contain the tensor of mean squared velocity components;
    \param[out] densityErr  will contain the error estimate of density;
    \param[out] velocityFirstMomentErr  will contain the error estimate of 1st velocity moment;
    \param[out] velocitySecondMomentErr will contain the error estimate of 2nd velocity moment;
    \param[in]  separate    whether to compute moments separately for each element of
    a multicomponent DF; in this case, the output arrays should have length equal to df.numValues();
    \param[in]  reqRelError is the required relative error in the integral;
    \param[in]  maxNumEval  is the maximum number of evaluations in integral.
*/
void computeMoments(
    const GalaxyModel& model,
    const coord::PosCyl& point,
    double density[],
    double velocityFirstMoment[],
    coord::Vel2Cyl velocitySecondMoment[],
    double densityErr[]=NULL,
    double velocityFirstMomentErr[]=NULL,
    coord::Vel2Cyl velocitySecondMomentErr[]=NULL,
    const bool separate=false,
    const double reqRelError=1e-3,
    const int maxNumEval=1e5);


/** Compute the projected moments of distribution function:
    surface density, scale height, and line-of-sight velocity dispersion at a given projected radius.
    TODO: generalize to allow an arbitrary orientation of the projection.
    \param[in]  model  is the galaxy model;
    \param[in]  R is the cylindrical radius;
    \param[out] surfaceDensity if not NULL will contain the computed surface density;
    \param[out] rmsHeight if not NULL, will contain the rms height;
    \param[out] rmsVel if not NULL, will contain the rms line-of-sight velocity;
    \param[out] surfaceDensityErr if not NULL, will contain the error estimate for density;
    \param[out] rmsHeightErr if not NULL, will contain the error estimate for rms height;
    \param[out] rmsVelErr if not NULL, will contain the error estimate for rms velocity;
    \param[in]  separate    whether to compute moments separately for each element of
    a multicomponent DF; in this case, the output arrays should have length equal to df.numValues();
    \param[in]  reqRelError is the required relative error in the integral;
    \param[in]  maxNumEval  is the maximum number of evaluations in integral;
*/
void computeProjectedMoments(
    const GalaxyModel& model,
    const double R,
    double surfaceDensity[],
    double rmsHeight[],
    double rmsVel[],
    double surfaceDensityErr[]=NULL,
    double rmsHeightErr[]=NULL,
    double rmsVelErr[]=NULL,
    bool separate=false,
    const double reqRelError=1e-3,
    const int maxNumEval=1e5);


/** Compute the value of 'projected distribution function' at the given point
    specified by two coordinates in the sky plane and line-of-sight velocity.
    TODO: generalize to allow an arbitrary orientation of the projection.
    \param[in]  model  is the galaxy model;
    \param[in]  R is the cylindrical radius;
    \param[in]  vz is the line-of-sight velocity;
    \param[out] result  will contain the values of projected DF (a single value if separate==false,
    otherwise an array of length df.numValues() - should be allocated by the caller);
    \param[in]  vz_error is the assumed velocity error (assumed Gaussian):
    if nonzero, then the DF is additionally convolved with the error function;
    \param[in]  separate    whether to compute moments separately for each element of
    a multicomponent DF; in this case, the result array should have length equal to df.numValues();
    \param[in]  reqRelError is the required relative error in the integral;
    \param[in]  maxNumEval  is the maximum number of evaluations in integral;
*/
void computeProjectedDF(
    const GalaxyModel& model,
    const double R, const double vz,
    double result[],
    const double vz_error=0,
    bool separate=false,
    const double reqRelError=1e-3,
    const int maxNumEval=1e5);


/** Compute the velocity distribution functions (VDF) in three directions in cylindrical coordinates
    at the given point in space.
    The VDF is represented as a weighted sum of B-splines of degree N:
    0 means histogramming, 1 - linear interpolation, 3 - cubic interpolation.
    Higher degrees not only result in a smoother VDF, but also are easier to compute:
    the integration over the velocity space is carried out for all B-splines simultaneously,
    and the smoother are the basis functions, the easier is it for the quadrature to achieve
    the target level of accuracy.
    TODO: generalize to allow an arbitrary orientation of the velocity ellipsoid.
    \tparam     N is the degree of B-splines (0, 1 or 3).
    \param[in]  model  is the galaxy model.
    \param[in]  point  is the position at which the VDF are computed.
    \param[in]  projected  determines whether to integrate over z:
    if true, then only two coordinates (R,phi) of the input point is used, and the VDF is averaged
    over z (i.e., represents the distribution of line-of-sight velocities and proper motions),
    otherwise the VDF is computed for the given 3d point.
    \param[in]  gridVR  is the array of grid nodes for the V_R velocity component,
    which define the set of basis functions for interpolation.
    The nodes must be sorted in increasing order, and typically should range from -V_escape to V_escape.
    A suitable array can be created by `math::createUniformGrid(numNodes, -V_escape, V_escape)`.
    \param[in]  gridVz  is the same thing for the VDF of V_z.
    \param[in]  gridVphi  is the same thing for V_phi.
    \param[out] density  will contain the density at the given point (it is used to normalize the
    amplitudes of the interpolator, and comes for free during integration anyway).
    If the flag separate==false, this will be a single value, otherwise an array of densities of
    each element of a multicomponent DF (should be allocated by the caller); same convention is
    used for other output arrays.
    \param[out] amplVR  should point to a vector (if separate==false) or an array of df.numValues()
    vectors (if separate==true), which will be filled with amplitudes for constructing an interpolated
    VDF for the V_R component, namely:
    ~~~~
    math::BsplineInterpolator1d interp(gridVR);
    double f_of_v = interp.interpolate(v, amplVR);  // amplVR was obtained from this routine
    ~~~~
    The number of elements in amplVR is gridVR.size() + N - 1;
    for N=0 they are essentially the heights of the histogram bars between gridVR[i] and gridVR[i+1],
    and for N=1 the values of amplVR at each node of gridVR coincide with the interpolated f(v);
    however for higher N the correspondence is not so obvious.
    The VDF is normalized such that \f$  \int_{-V_{escape}}^{V_{escape}} f(v) dv = 1  \f$
    (if the provided interval specified by gridVR is smaller than V_escape, this integral may
    be lower than unity).
    \param[out] amplVz  will contain the amplitudes for the V_z component.
    \param[out] amplVphi  will contain the amplitudes for the V_phi component.
    \param[in]  separate    whether to compute moments separately for each element of
    a multicomponent DF; in this case, the output arrays should have length equal to df.numValues().
    \param[in]  reqRelError is the required relative error in the integrals.
    \param[in]  maxNumEval  is the maximum number of evaluations for all integral together.
    \throw  std::invalid_argument if the input arrays are incorrect.
*/
template <int N>
void computeVelocityDistribution(
    const GalaxyModel& model,
    const coord::PosCyl& point,
    bool projected,
    const std::vector<double>& gridVR,
    const std::vector<double>& gridVz,
    const std::vector<double>& gridVphi,
    // output arrays - one element per DF component (if separate==true), otherwise just one element
    double density[],
    std::vector<double> amplVR[],
    std::vector<double> amplVz[],
    std::vector<double> amplVphi[],
    // optional input parameters
    const bool separate=false,
    const double reqRelError=1e-2,
    const int maxNumEval=1e6);


/** this will be redesigned */
void computeProjection(
    const GalaxyModel& model,
    const math::IFunctionNdim& spatialSelection,
    const double Xlim[2],
    const double Ylim[2],
    const double transformMatrix[9],
    double* result,
    double* error=NULL,
    const double reqRelError=1e-3,
    const int maxNumEval=1e5);


/** Compute the total mass represented by the DF in the region determined by the selection function
    (if the latter is trivial, the mass should be equivalent to df.totalMass()
    up to integration errors, but is much more computationally heavy because this involves
    integration over 6d phase space and action computation for every point allowed by the SF).
    \param[in]  model  is the galaxy model.
    \param[out] result  will contain the integral(s) of DF * SF over the entire 6d phase space.
    \param[out] error  will contain the error estimate of the mass.
    \param[in]  separate    whether to compute moments separately for each element of
    a multicomponent DF; in this case, the output arrays should have length equal to df.numValues().
    \param[in]  reqRelError is the required relative error in the integral.
    \param[in]  maxNumEval  is the maximum number of evaluations in integral.
    Note that if the SF is very localized, the integration may terminate early with a zero result,
    if it is not able to locate the region where the SF is nonzero.
*/
void computeTotalMass(
    const GalaxyModel& model,
    double* result,
    double* error=NULL,
    bool separate=false,
    const double reqRelError=1e-3,
    const int maxNumEval=1e6);


/** Generate N-body samples of the distribution function multiplied by the selection function
    by sampling in action/angle space:
    sample actions directly from DF and angles uniformly from [0:2pi]^3,
    then use torus machinery to convert from action/angles to position/velocity.
    Note that this routine is not able to sample uniformly weighted particles in the case
    of a non-trivial selection function; instead, it samples uniformly in the action/angle space,
    and then multiplies the weight of each output point by the value of the selection function,
    retaining even points with zero weight.
    \param[in]  model  is the galaxy model;
    \param[in]  numPoints  is the required number of samples;
    \param[out] actions (optional) will be filled with values of actions
    corresponding to each point; if not needed may pass NULL as this argument.
    \returns    a new array of particles (position/velocity/mass)
    sampled from the distribution function.
*/
particles::ParticleArrayCyl sampleActions(
    const GalaxyModel& model, const size_t numPoints,
    std::vector<actions::Actions>* actions=NULL);


/** Generate N-body samples of the distribution function multiplied by the selection function
    by sampling in position/velocity space:
    use action finder to compute the actions corresponding to the given point,
    and evaluate the value of DF times SF at the given actions.
    The output points have uniform weights.
    \param[in]  model  is the galaxy model;
    \param[in]  numPoints  is the required number of samples;
    \returns    a new array of particles (position/velocity/mass)
    sampled from the distribution function;
*/
particles::ParticleArrayCyl samplePosVel(
    const GalaxyModel& model, const size_t numPoints);


/** Sample the density profile by discrete points.
    \param[in]  dens  is the density model;
    \param[in]  numPoints  is the required number of sampling points;
    \returns    a new array with the sampled coordinates and masses
*/
particles::ParticleArray<coord::PosCyl> sampleDensity(
    const potential::BaseDensity& dens, const size_t numPoints);


/// Helper class for providing a BaseDensity interface to a density computed via integration over DF
class DensityFromDF: public potential::BaseDensity{
public:
    DensityFromDF(const GalaxyModel& _model, double _relError, unsigned int _maxNumEval) :
        model(_model), relError(_relError), maxNumEval(_maxNumEval) {}

    virtual coord::SymmetryType symmetry() const { return coord::ST_AXISYMMETRIC; }
    virtual const char* name() const { return myName(); }
    static const char* myName() { return "DensityFromDF"; }
    virtual double enclosedMass(const double) const { return NAN; } /// should never be used -- too slow
private:
    const GalaxyModel model;  ///< aggregate of potential, action finder and DF
    double       relError;    ///< requested relative error of density computation
    unsigned int maxNumEval;  ///< max # of DF evaluations per one density calculation

    virtual double densityCar(const coord::PosCar &pos, double time) const {
        return densityCyl(toPosCyl(pos), time); }

    virtual double densitySph(const coord::PosSph &pos, double time) const {
        return densityCyl(toPosCyl(pos), time); }

    /// compute the density as the integral of DF over velocity at a given position
    virtual double densityCyl(const coord::PosCyl &point, double /*time*/) const {
        double result;
        computeMoments(model, point, &result, NULL, NULL, NULL, NULL, NULL, false, relError, maxNumEval);
        return result;
    }

    /// vectorized computation of density in an OpenMP-parallelized loop
    virtual void evalmanyDensityCar(const size_t npoints, const coord::PosCar pos[],
        /*output*/ double values[], /*input*/ double t=0) const;
    virtual void evalmanyDensityCyl(const size_t npoints, const coord::PosCyl pos[],
        /*output*/ double values[], /*input*/ double t=0) const;
    virtual void evalmanyDensitySph(const size_t npoints, const coord::PosSph pos[],
        /*output*/ double values[], /*input*/ double t=0) const;
};

}  // namespace