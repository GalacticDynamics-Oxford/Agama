#include "galaxymodel_base.h"
#include "math_core.h"
#include "math_random.h"
#include "math_sample.h"
#include "math_specfunc.h"
#include "math_spline.h"
#include "math_linalg.h"
#include "potential_utils.h"
#include "smart.h"
#include "utils.h"
#include <cmath>
#include <algorithm>
#include <utility>
#include <stdexcept>
#include <cassert>
#ifndef _MSC_VER
#include <alloca.h>
#else
#include <malloc.h>
#endif

namespace galaxymodel{

// parallelized loop over input points with precautions against exceptions or keyboard interrupt
template<typename CoordT>
void computeDensityParallel(const potential::BaseDensity& density,
    const size_t npoints, const coord::PosT<CoordT> pos[], /*output*/ double values[])
{
    std::string errorMsg;
    utils::CtrlBreakHandler cbrk;  // catch Ctrl-Break keypress
    bool stop = false;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for(int i=0; i<(int)npoints; i++) {
        if(stop) continue;
        if(cbrk.triggered()) stop = true;
        try{
            values[i] = density.density(pos[i]);
        }
        catch(std::exception& e) {
            errorMsg = e.what();
            stop = true;
        }
    }
    if(cbrk.triggered())
        throw std::runtime_error(cbrk.message());
    if(!errorMsg.empty())
        throw std::runtime_error("Error in DensityFromDF: "+errorMsg);
}

void DensityFromDF::evalmanyDensityCar(const size_t npoints, const coord::PosCar pos[],
    double values[], double) const {
    computeDensityParallel(*this, npoints, pos, values);
}
void DensityFromDF::evalmanyDensityCyl(const size_t npoints, const coord::PosCyl pos[],
    double values[], double) const {
    computeDensityParallel(*this, npoints, pos, values);
}
void DensityFromDF::evalmanyDensitySph(const size_t npoints, const coord::PosSph pos[],
    double values[], double) const {
    computeDensityParallel(*this, npoints, pos, values);
}


/** a singleton instance of a trivial selection function */
const SelectionFunctionTrivial selectionFunctionTrivial;

SelectionFunctionDistance::SelectionFunctionDistance(
    const coord::PosCar& _point0, double _radius, double _steepness) :
    point0(_point0), radius(_radius), steepness(_steepness)
{
    if(!(radius>0))
        throw std::invalid_argument("SelectionFunctionDistance: radius must be positive");
    if(!(steepness>=0))
        throw std::invalid_argument("SelectionFunctionDistance: steepness must be positive or zero");
}

double SelectionFunctionDistance::value(const coord::PosVelCar& point) const
{
    if(radius==INFINITY || steepness==0)
        return 1;
    double d2 = (pow_2(point.x-point0.x) + pow_2(point.y-point0.y) + pow_2(point.z-point0.z)) /
        pow_2(radius);
    if(steepness==INFINITY)
        return d2>1 ? 0 : 1;
    return exp( -math::pow(d2, 0.5*steepness) );
}

namespace{   // internal definitions

//------- HELPER ROUTINES -------//

/** convert from scaled velocity variables to the actual velocity.
    \param[in]  pos  is the position (the orientation of velocity basis matches the radius vector).
    \param[in]  velvars are the scaled velocity-space variables from unit cube: s, u, w
    where the magnitude of the velocity is v = v_esc * g(s, zeta), g is some scaling function,
    and the two angles {psi(u), chi(w)} specify the orientation of velocity vector
    in spherical coordinates centered at a given point.
    \param[in]  vesc is the maximum magnutude of velocity (equal to the escape velocity at pos).
    \param[in]  zeta is the scaling factor for the velocity magnitude -
    the ratio of circular to escape velocity at the given radius.
    The non-trivial transformations are needed to accurately handle distribution functions of
    cold disks at large radii, which are very strongly peaked near {v_R,v_z,v_phi} = {0,0,v_circ},
    or strongly radially anisotropic haloes, concentrated at v_r >> v_theta, v_phi.
    To reduce the chance of missing a dominant part of velocity space during the integration,
    we make sure that a large proportion of unit cube in scaled variables maps onto 
    relatively small regions where these cold DFs could be concentrated:
    1) the scaled velocity magnitude v/v_esc = g(p) is nearly horizontal in a large range of chi
    where its value is close to zeta (i.e. v is close to circular velocity);
    2) the two angles psi, chi specifying the orientation of the velocity vector
    are mapped from the scaled variables u, w using an "undulating" transformation
    psi / pi = u - k * sin(4 pi u) / (4 pi),  and similarly for chi / (2pi) as a function of w,
    with k ~ 0.9 creating nearly horizontal stretches around  psi or xi ~ 0, 0.5 and 1,
    corresponding to the velocity oriented parallel to one of the three principal axes
    of spherical velocity coordinates v_r, v_theta, v_phi.
    \param[out] jac will contain the jacobian of transformation.
    \return  three components of velocity in spherical coordinates.
*/
coord::PosVelCar unscaleVelocity(const coord::PosCar& pos,
    const double velvars[], const double vesc, const double zeta, double &jac)
{
    double s = velvars[0], u = velvars[1], w = velvars[2],
    s0  = 0.6*zeta + 0.2,
    ks  = 0.9,
    vel = vesc * (s + ks * 0.5 / M_PI * (sin(2*M_PI*(s0-s)) - sin(2*M_PI*s0))),
    sinu, cosu, sinw, cosw, sinpsi, cospsi, sinchi, coschi;
    math::sincos(4*M_PI * u, sinu, cosu);
    math::sincos(4*M_PI * w, sinw, cosw);
    double ku = 0.6, kw = 0.3 * s*(2-s),
    psi =   M_PI * (u - ku/(4*M_PI) * sinu),
    chi = 2*M_PI * (w - kw/(4*M_PI) * sinw);
    math::sincos(psi, sinpsi, cospsi);
    math::sincos(chi, sinchi, coschi);
    jac = nan2num(2*M_PI*M_PI *
                  (1 - ku * cosu) * sinpsi *
                  (1 - kw * cosw) *
                  (1 - ks * cos(2*M_PI*(s-s0))) *
                  vesc * pow_2(vel));
    // components of velocity in spherical coordinates
    double vr = vel * sinpsi * coschi, vtheta = vel * sinpsi * sinchi, vphi = vel * cospsi;

    // convert velocity to cartesian coords at the given position
    double r = sqrt(pos.x*pos.x + pos.y*pos.y + pos.z*pos.z), R = sqrt(pos.x*pos.x + pos.y*pos.y),
    sintheta = r>0 ? R/r : 1, costheta = r>0 ? pos.z/r : 0,
    sinphi = R>0 ? pos.y/R : 0, cosphi = R>0 ? pos.x/R : 1,
    vR = vr * sintheta + vtheta * costheta,
    vx = vR * cosphi   - vphi   * sinphi,
    vy = vR * sinphi   + vphi   * cosphi,
    vz = vr * costheta - vtheta * sintheta;
    return coord::PosVelCar(pos, coord::VelCar(vx, vy, vz));
}

/** compute the escape velocity and the ratio of circular to escape velocity
    at a given position in the given ponential */
inline void getVesc(
    const coord::PosCar& pos, const potential::BasePotential& poten, double& vesc, double& zeta)
{
    if(!isFinite(pow_2(pos.x) + pow_2(pos.y) + pow_2(pos.z))) {
        vesc = 0.;
        zeta = 0.5;
        return;
    }
    double Phi;
    coord::GradCar grad;
    poten.eval(pos, &Phi, &grad);
    vesc = sqrt(-2. * Phi);
    zeta = math::clip(sqrt(grad.dx * pos.x + grad.dy * pos.y + grad.dz * pos.z) / vesc, 0.2, 0.8);
    if(!isFinite(vesc)) {
        throw std::invalid_argument("Error in computing moments: escape velocity is undetermined at "
            "("+utils::toString(pos.x)+","+utils::toString(pos.y)+","+utils::toString(pos.z)+
            ") where Phi="+utils::toString(Phi));
    }
}


//------- HELPER CLASSES FOR MULTIDIMENSIONAL INTEGRATION OF DF -------//

/** Base helper class for integrating the distribution function (DF) multiplied by selection
    function (SF) over the position/velocity space.
    Various tasks in this module boil down to computing the integrals or sampling the values of
    DF*SF over the (x,v) space, where the DF is expressed in terms of actions.
    This involves the following steps:
    1) scaled variables in N-dimensional unit cube are transformed to the actual (x,v);
    2) the selection function s(x,v) is evaluated, and if it turns out to be zero,
    no further work is done;
    3) x,v are transformed to actions (J);
    4) the value(s) of DF f(J) is computed (for a multicomponent DF, one may either use the
    sum of all components, or treat them separately, depending on the flag 'separate');
    5) one or more quantities that are products of f(J)*SF(x,v) times something
    (e.g., velocity components) are returned to the integration or sampling routines.
    These tasks differ in the first and the last steps, and also in the number of dimensions
    that the integration/sampling is carried over. This diversity is handled by the class
    hierarchy descending from DFIntegrandNdim, where the base class performs the steps 2 and 3,
    and the derived classes implement virtual methods `unscaleVars()` and `outputValues()`,
    which are responsible for the steps 1 and 5, respectively.
    The derived classes also specify the dimensions of integration space (numVars)
    and the number of simultaneously computed quantities (numValues).
    The selection function at step 2, action finder at the step 3, and DF at the step 4
    are provided as members of the GalaxyModel structure.
*/
class DFIntegrandNdim: public math::IFunctionNdim {
public:
    DFIntegrandNdim(const GalaxyModel& _model, bool separate) :
        model(_model),
        dflen(separate ? model.distrFunc.numValues() : 1)
    {}

    // evaluate a single input point
    virtual void eval(const double vars[], double values[]) const {
        evalmany(1, vars, values);
    }

    // evaluate the integrand for many input points at once
    virtual void evalmany(const size_t npoints, const double vars[], double values[]) const
    {
        // 0. allocate various temporary arrays on the stack - no need to delete them manually
        size_t numvars = numVars(), numvalues = numValues();
        // jacobian of coordinate transformation at each point
        double* jac = static_cast<double*>(alloca(npoints * sizeof(double)));
        // values of selection function at each point
        double* sf  = static_cast<double*>(alloca(npoints * sizeof(double)));
        // values of distribution function (possibly several components) at each point
        double* df  = static_cast<double*>(alloca(npoints * dflen * sizeof(double)));
        // x,v points in cartesian coords (unscaled from the input variables)
        coord::PosVelCar* posvel = static_cast<coord::PosVelCar*>(
            alloca(npoints * sizeof(coord::PosVelCar)));
        // values of actions at all input points where sel.fnc is not zero
        actions::Actions* act = static_cast<actions::Actions*>(
            alloca(npoints * sizeof(actions::Actions)));

        // 1. get the position/velocity components for all input points
        for(size_t p=0; p<npoints; p++) {
            posvel[p] = unscaleVars(/*input point*/ vars + p*numvars, /*output jacobian*/ jac[p]);
        }

        // 2. evaluate the selection function for all input points at once
        try{
            model.selFunc.evalmany(npoints, posvel, /*output*/ sf);
        }
        catch(std::exception& e) {
            FILTERMSG(utils::VL_WARNING, "DFIntegrandNdim", std::string(e.what()));
            std::fill(sf, sf+npoints, 0);
        }

        // 3. evaluate actions at points where sel.fnc. is not zero:
        // store actions contiguously for this subset of points,
        // so that they could be passed to the DF all at once
        size_t nselected = 0;          // number of selected points
        for(size_t p=0; p<npoints; p++) {
            double mult = jac[p] * sf[p];  // overall weight of this point (jacobian * sel.fnc.)
            if(mult > 0 && isFinite(mult)) {
                actions::Actions acts = model.actFinder.actions(toPosVelCyl(posvel[p]));
                // FIXME: in some cases the Fudge action finder may fail and produce
                // zero values of Jr,Jz instead of very large ones, which may lead to
                // unrealistically high DF values. We therefore ignore these points
                // entirely, but the real problem is with the action finder, not here.
                if(isFinite(acts.Jr + acts.Jz + acts.Jphi) && (acts.Jr!=0 || acts.Jz!=0)) {
                    act[nselected] = acts;
                    nselected++;
                } else  // otherwise this output point is ignored and will be overwritten next time
                    sf[p] = 0;
            }
        }
        // check if there are any points selected at all
        if(nselected==0) {
            // fast track, output an array of zeros of appropriate length
            std::fill(values, values + npoints*numvalues, 0);
            return;
        }

        // 4. evaluate the DF for the entire selected subset of points at once
        try{
            model.distrFunc.evalmany(nselected, act, /*separate*/ dflen!=1, /*output*/ df);
        }
        catch(std::exception& e) {
            FILTERMSG(utils::VL_WARNING, "DFIntegrandNdim", std::string(e.what()));
            std::fill(sf, df + npoints * dflen, 0);  // quietly replace output with zeroes
        }

        // 5. perform actual calculation of output values for each [valid and selected] input point
        for(size_t p=0, s=0; p<npoints; p++) {  // p indexes all input points, s - only selected ones
            double mult = jac[p] * sf[p];       // overall weight of this point (jacobian * sel.fnc.)
            if(mult > 0 && isFinite(mult)) {
                double* dfval = df + s * dflen; // array of df values at this selected point
                // multiply by jacobian and selection function,
                // check for possibly invalid DF values and replace them with zeroes
                for(unsigned int i=0; i<dflen; i++) {
                    if(!isFinite(dfval[i])) {
                        dfval[i] = 0;
                    } else
                        dfval[i] *= mult;
                }
                // output the value(s) to the integration routine
                outputValues(posvel[p], dfval, values + p*numvalues);
                s++;  // increment the index of selected points, for which the DF values were computed
            } else {
                // ignore this point and output zeroes
                std::fill(values + p*numvalues, values + (p+1)*numvalues, 0);
            }
        }
    }

    /** convert from scaled variables used in the integration routine
        to the actual position/velocity point.
        \param[in]  vars  is the array of scaled variables;
        \param[out] jac  is the jacobian of transformation;
        \return  the position and velocity in cartesian coordinates.
    */
    virtual coord::PosVelCar unscaleVars(const double vars[], double& jac) const = 0;

    /** output the value(s) computed at a given point to the integration routine.
        \param[in]  point  is the position/velocity point;
        \param[in]  dfval  is the value or array of values of distribution function at this point;
        \param[out] values is the array of one or more values that are computed
    */
    virtual void outputValues(
        const coord::PosVelCar& point, const double dfval[], double values[]) const = 0;

    const GalaxyModel& model;  ///< reference to the galaxy model to work with
    /// number of values in the DF in the case when a composite DF has more than component, and
    /// they are requested to be considered separately, otherwise 1 (a sum of all components)
    unsigned int dflen;
};


/** helper class for computing the projected distribution function at a given point in X,Y space,
    integrated over Z and marginalized over or convolved with uncertainties in some velocity components
*/
class DFIntegrandProjected: public DFIntegrandNdim {
    const double X, Y;           ///< coordinates in the image plane
    const coord::VelCar vel;     ///< components of the velocity
    const coord::VelCar velerr;  ///< uncertainties on these components (0 <= err <= INFINITY)
    const coord::Orientation& orientation;  ///< conversion between intrinsic and observed coords
    double smin, smax;           ///< integration limits in scaled Z coordinate
    const bool halfZ;            ///< whether one may limit the integration to Z>0 half-space
public:
    DFIntegrandProjected(const GalaxyModel& model, bool separate, const coord::PosProj& pos,
        const coord::VelCar& _vel, const coord::VelCar& _velerr, const coord::Orientation& _orientation)
    :
        DFIntegrandNdim(model, separate), X(pos.X), Y(pos.Y),
        vel(_vel), velerr(fabs(_velerr.vx), fabs(_velerr.vy), fabs(_velerr.vz)),
        orientation(_orientation), smin(0), smax(1),
        // if the orientation is face-on AND at least one velocity component is not known
        // (has infinite error), we may save effort by integrating only over the region Z>=0
        halfZ(orientation.isFaceOn() && (velerr.vz == INFINITY || velerr.vx==INFINITY))
    {
        // check that the input uncertainties are coherent
        if((velerr.vx==INFINITY) ^ (velerr.vy==INFINITY))
            throw std::invalid_argument(
               "projectedDF: uncertainties on vx and vy must be either both finite or both infinite");

        // if the uncertainty on any velocity component is zero, this limits the range of Z
        double minvel2 = 0;
        if(velerr.vx == 0)
            minvel2 += pow_2(vel.vx);
        if(velerr.vy == 0)
            minvel2 += pow_2(vel.vy);
        if(velerr.vz == 0)
            minvel2 += pow_2(vel.vz);
        double Z0, Zmin=-INFINITY, Zmax=INFINITY;
        if(minvel2 > 0) {
            // find the range of Z where Phi(X,Y,Z) + 0.5 * minvel^2 <= 0
            findRoots(model.potential, -0.5 * minvel2, X, Y, orientation, Z0, Zmin, Zmax);
            if(Zmin!=Zmin)        // this indicates that the velocity exceeds the escape speed
                Zmin = Zmax = 0;  // even at the minimum of the potential, so the result is zero
        }

        if(halfZ) {
            smax = scale(math::ScalingSemiInf(), Zmax);
            // in this case we assume that Zmin = -Zmax and ignore the lower half-space entirely
        } else {
            math::ScalingDoubleInf scaling(/*R*/ sqrt(X*X+Y*Y));
            smin = scale(scaling, Zmin);
            smax = scale(scaling, Zmax);
        }
    }

    virtual coord::PosVelCar unscaleVars(const double vars[], double& jac) const
    {
        double s = vars[0] * (smax-smin) + smin;  // scaled Z coordinate
        double Z, dZds;  // unscaled Z coordinate
        if(halfZ) {
            // integrating over the half-space Z>=0 using the following transformation,
            // which maps 0..1 to 0..+INFINITY
            Z = unscale(math::ScalingSemiInf(), s, &dZds);
            dZds *= 2;   // factor of 2 compensates that we integrate over half-space only
        } else {
            // integrating over the entire range of Z using a different transformation,
            // which maps 0..1 to -INFINITY..+INFINITY
            Z = unscale(math::ScalingDoubleInf(/*R*/ sqrt(X*X+Y*Y)), s, &dZds);
        }
        jac = dZds * (smax-smin);   // jacobian of all transformations
        // position in intrinsic coords:
        const coord::PosCar pos = orientation.fromRotated(coord::PosCar(X, Y, Z));
        double vX=vel.vx, vY=vel.vy, vZ=vel.vz;      // input velocity in observed coords
        // count up velocity components which have nonzero uncertainties - each one adds a dimension
        int dimvX = (velerr.vx!=0), dimvY = (velerr.vy!=0) + dimvX, dimvZ = (velerr.vz!=0) + dimvY;

        // start off with the velocity component along the Z axis (line of sight in the observed system)
        // escape velocity at the given point:
        double vZmax = sqrt(-2 * model.potential.value(pos));
        if(velerr.vz == INFINITY) {
            // when uncertainty is infinite, this means integrating vZ from -Vescape to +Vescape
            vZ = (2 * vars[dimvZ] - 1) * vZmax;
            jac *= 2 * vZmax;
        } else  if(velerr.vz > 0) {
            double
            uz0 = math::erf( 1/M_SQRT2 * (-vZmax - vZ) / velerr.vz ),
            uz1 = math::erf( 1/M_SQRT2 * (+vZmax - vZ) / velerr.vz );
            vZ += M_SQRT2 * velerr.vz * math::erfinv(vars[dimvZ] * (uz1 - uz0) + uz0);
            jac *= (uz1 - uz0) * 0.5;
        } // else (when velerr.vz==0) don't do anything, since vZ is already set to the exact value

        // next consider the two velocity components in the X,Y plane
        double vXYmax = sqrt(fmax(pow_2(vZmax) - pow_2(vZ), 0));  // remaining velocity budget
        if(velerr.vx == INFINITY && velerr.vy == INFINITY) {
            // when they have infinite uncertainties, this means integrating vX,vY over a circle
            // |vXY| = sqrt(vX^2+vY^2) <= sqrt(Vescape^2 - vZ^2)
            assert(dimvX==1 && dimvY==2);  // the two dimensions of integration in the vX,vY plane
            double sinphi, cosphi, addphi = math::atan2(Y, X);
            math::sincos(2*M_PI*vars[2] + addphi, sinphi, cosphi);
            vX = vXYmax * vars[1] * cosphi;
            vY = vXYmax * vars[1] * sinphi;
            jac *= pow_2(vXYmax) * 2*M_PI * vars[1];
        } else {
            // otherwise the uncertainties on vX,vY are both finite (perhaps even zero)
            if(velerr.vy > 0) {
                double
                uy0 = math::erf( 1/M_SQRT2 * (-vXYmax - vY) / velerr.vy ),
                uy1 = math::erf( 1/M_SQRT2 * (+vXYmax - vY) / velerr.vy );
                vY += M_SQRT2 * velerr.vy * math::erfinv(vars[dimvY] * (uy1 - uy0) + uy0);
                jac *= (uy1 - uy0) * 0.5;
            }  // else (when velerr.vy==0) keep vY assigned to the exact value

            if(velerr.vx > 0) {
                double vXmax = sqrt(fmax(pow_2(vXYmax) - pow_2(vY), 0)),  // remaining velocity
                ux0 = math::erf( 1/M_SQRT2 * (-vXmax - vX) / velerr.vx ),
                ux1 = math::erf( 1/M_SQRT2 * (+vXmax - vX) / velerr.vx );
                vX += M_SQRT2 * velerr.vx * math::erfinv(vars[dimvX] * (ux1 - ux0) + ux0);
                jac *= (ux1 - ux0) * 0.5;
            }  // and again if no error on vx is provided, keep the exact value
        }

        // finally, transform the velocity from observed to intrinsic coordinate system
        coord::VelCar newvel = orientation.fromRotated(coord::VelCar(vX, vY, vZ));
        return coord::PosVelCar(pos, newvel);
    }

    virtual unsigned int numVars()   const {
        // one dimension for integration along Z,
        // and each velocity component with uncertainty adds another dimension (up to 4 in total)
        return 1 + (velerr.vx!=0) + (velerr.vy!=0) + (velerr.vz!=0);
    }
    virtual unsigned int numValues() const { return dflen; }

    /// output array contains the value of DF (or all values for a multicomponent DF)
    virtual void outputValues(const coord::PosVelCar& , const double dfval[], double values[]) const
    {
        std::copy(dfval, dfval+dflen, values);
    }
};


/** helper class for integrating the distribution function over the entire 6d phase space */
class DFIntegrand6dim: public DFIntegrandNdim {
public:
    DFIntegrand6dim(const GalaxyModel& _model, bool separate) :
        DFIntegrandNdim(_model, separate) {}

    /// input variables define 6 components of position and velocity, suitably scaled
    virtual coord::PosVelCar unscaleVars(const double vars[], double& jac) const
    {
        // 1. determine the position from the first three scaled variables
        const coord::PosCar pos = toPosCar(potential::unscaleCoords(vars, &jac));
        // 2. determine the velocity (in spherical coordinates) from the second three scaled vars
        double vesc, zeta, jacvel;
        getVesc(pos, model.potential, vesc, zeta);
        coord::PosVelCar posvel = unscaleVelocity(pos, /*vel*/ &vars[3], vesc, zeta, /*output*/jacvel);
        jac *= jacvel;
        return posvel;
    }

private:
    virtual unsigned int numVars()   const { return 6; }
    virtual unsigned int numValues() const { return 1; }

    /// output contains just one value of DF (even for a multicomponent DF)
    virtual void outputValues(const coord::PosVelCar& , const double dfval[], double values[]) const
    {
        values[0] = dfval[0];
    }
};


/** helper class for conversion of scaled variables (three components of velocity and -
    in the projected case - also the Z-component of position in the observed coord.sys.),
    used in integrating the distribution function over velocity at a fixed position (X,Y,Z)
    or at a point in X,Y space integrated along Z [common functionality for moments() and vdf() ].
    \tparam Projected distinguishes intrinsic (3d point) and projected (2d point) variants.
*/
template<bool Projected> struct Scaling;

// non-projected variant
template<> struct Scaling<false> {
    typedef coord::PosCar  Type;
    const coord::PosCar pos;    ///< 3d point in the intrinsic coordinate system of the model
    const bool symmetrizevRvz;  ///< if the DF is invariant w.r.t {v_R,v_z <-> -v_R,v_z}, use only v_z>=0
    double vesc, zeta;          ///< escape speed and the ratio of circular to escape speed

    Scaling(const potential::BasePotential& pot,
        const coord::PosCar& obspos, const coord::Orientation& orientation)
    :
        pos(orientation.fromRotated(obspos)),
        symmetrizevRvz(true) // this holds for any axisymmetric system and possibly even more generally
    {
        getVesc(pos, pot, vesc, zeta);
    }

    /// non-projected: input variables are three velocity components
    coord::PosVelCar unscale(const double vars[], double &jac) const
    {
        // if symmetrizevRvz, scan only half of the (v_R, v_z) plane
        double scaledvel[3] = {vars[0], vars[1], symmetrizevRvz ? 0.25 + vars[2] * 0.5 : vars[2]};
        return unscaleVelocity(pos, scaledvel, vesc, zeta, jac);
    }
};

// projected variant
template<> struct Scaling<true> {
    typedef coord::PosProj  Type;
    const potential::BasePotential& pot;    ///< potential of the model
    const coord::PosProj obspos;            ///< 2d point in the observed coordinate system
    const coord::Orientation& orientation;  ///< conversion between intrinsic and observed coords
    const bool symmetrizevRvz;  ///< if the DF is invariant w.r.t {v_R,v_z <-> -v_R,v_z}, use only v_z>=0

    Scaling(const potential::BasePotential& _pot,
        const coord::PosProj& _obspos, const coord::Orientation& _orientation)
    :
        pot(_pot), obspos(_obspos), orientation(_orientation),
        symmetrizevRvz(true) // this holds for any axisymmetric system and possibly even more generally
    {}

    /// projected: input variables are scaled Z-coordinate and all three velocity components
    coord::PosVelCar unscale(const double vars[], double &jac) const
    {
        double Z;
        if(orientation.isFaceOn()) {
            // integrating over the half-space Z>=0 using the following transformation
            Z = math::unscale(math::ScalingSemiInf(), vars[0], &jac);
            jac *= 2;   // factor of 2 compensates that we integrate over half-space only
        } else {
            // integrating over the entire range of Z using a different transformation
            Z = math::unscale(math::ScalingDoubleInf(
                /*R*/ sqrt(pow_2(obspos.X) + pow_2(obspos.Y))), vars[0], &jac);
        }
        coord::PosCar pos = orientation.fromRotated(coord::PosCar(obspos.X, obspos.Y, Z));
        double scaledvel[3] = {vars[1], vars[2], symmetrizevRvz ? 0.25 + vars[3] * 0.5 : vars[3]};

        // determine the velocity in the intrinsic coordinate system from the three scaled vars
        double vesc, zeta, jacvel;
        getVesc(pos, pot, vesc, zeta);
        coord::PosVelCar posvel = unscaleVelocity(pos, scaledvel, vesc, zeta, jacvel);
        jac *= jacvel;
        return posvel;
    }
};


/** helper class for integrating the distribution function over velocity
    and optionally along the line of sight (if the template parameter Projected==true) */
template<bool Projected>
class DFIntegrandMoments: public DFIntegrandNdim {
    const coord::Orientation& orientation;
    const Scaling<Projected> scaling;
    const bool needVel, needVel2;
public:
    DFIntegrandMoments(const GalaxyModel& model, bool separate,
        const typename Scaling<Projected>::Type& obspoint, const coord::Orientation& _orientation,
        bool _needVel, bool _needVel2)
    :
        DFIntegrandNdim(model, separate),
        orientation(_orientation),
        scaling(model.potential, obspoint, orientation),
        needVel(_needVel), needVel2(_needVel2)
    {}

    /// dimension of the input array (3 scaled velocity components and optionally scaled Z coordinate)
    virtual unsigned int numVars()   const { return Projected ? 4 : 3; }

    /// dimension of the output array
    virtual unsigned int numValues() const { return dflen * (1 + 4*needVel + 6*needVel2); }

    inline void outerProduct(const coord::VelCar& vel, /*add to*/ coord::Vel2Car& vel2) const
    {
        vel2.vx2  += vel.vx * vel.vx;
        vel2.vy2  += vel.vy * vel.vy;
        vel2.vz2  += vel.vz * vel.vz;
        vel2.vxvy += vel.vx * vel.vy;
        vel2.vxvz += vel.vx * vel.vz;
        vel2.vyvz += vel.vy * vel.vz;
    }

    // input variables are scaled z-coordinate (if projected) and all three velocity components
    virtual coord::PosVelCar unscaleVars(const double vars[], double &jac) const {
        return scaling.unscale(vars, jac);
    }

    // output the value(s) of DF, multiplied by various combinations of velocity components
    virtual void outputValues(const coord::PosVelCar& pv, const double dfval[], double values[]) const
    {
        // transform the velocity back to the observed coordinate system
        coord::VelCar VEL = orientation.toRotated(coord::VelCar(pv));
        coord::Vel2Car VEL2;
        VEL2.vx2 = VEL2.vy2 = VEL2.vz2 = VEL2.vxvy = VEL2.vxvz = VEL2.vyvz = 0;
        outerProduct(VEL, VEL2);  // compute the second moment of velocity
        if(Projected && orientation.isFaceOn()) {
            // add a contribution from {R,-z,vR,-vz}, which projects onto the same X,Y point
            // and has the same vX,vY, but negative sign of vZ, which cancels VEL.vz
            VEL.vz = 0;
            VEL2.vxvz = 0;
            VEL2.vyvz = 0;
        }
        if(scaling.symmetrizevRvz) {
            // add a contribution from {R,z,-vR,-vz}
            double R2 = pow_2(pv.x) + pow_2(pv.y),
            costwophi = R2==0 ? 1 : (pow_2(pv.x) - pow_2(pv.y)) / R2,
            sintwophi = R2==0 ? 0 : 2 * pv.x * pv.y / R2;
            coord::VelCar VELm = orientation.toRotated(coord::VelCar(
                -pv.vx * costwophi - pv.vy * sintwophi, -pv.vx * sintwophi + pv.vy * costwophi, -pv.vz));
            outerProduct(VELm, VEL2);  // add to the second moment
            if(Projected && orientation.isFaceOn()) {
                VELm.vz = 0;    // cancels for the same reason as above
                VEL2.vxvz = 0;
                VEL2.vyvz = 0;
            }
            VEL.vx += VELm.vx;  // add to the first moment
            VEL.vy += VELm.vy;
            VEL.vz += VELm.vz;
        }

        // a trick for optimizing the integration efficiency:
        // since the components of mean velocity and the non-diagonal second moments
        // are often exactly zero or at least (for 2nd moments) subdominant to diagonal moments,
        // we don't want to spend too much effort on computing them to high relative accuracy.
        // The solution is to add a positive number to each of these terms during integration,
        // and then subtract it back when finalizing the output.
        // For the first moments of velocity, we add the total magnitude of the velocity
        // (absolute value) to each component, and for the second moments, we add
        // the squared velocity (which is the sum of the first three terms) to the last three ones.
        double V2 = VEL2.vx2 + VEL2.vy2 + VEL2.vz2,  V = sqrt(V2);

        for(unsigned int ic=0; ic<dflen; ic++) {  // loop over components of DF
            values[ic] = dfval[ic];
            double val = dfval[ic] * (scaling.symmetrizevRvz ? 0.5 : 1);
            unsigned int im=1;  // index of the output moment, increases with each stored value
            if(needVel) {
                values[ic + dflen * (im++)] = val *  V;
                values[ic + dflen * (im++)] = val * (V + VEL.vx);
                values[ic + dflen * (im++)] = val * (V + VEL.vy);
                values[ic + dflen * (im++)] = val * (V + VEL.vz);
            }
            if(needVel2) {
                values[ic + dflen * (im++)] = val * VEL2.vx2;
                values[ic + dflen * (im++)] = val * VEL2.vy2;
                values[ic + dflen * (im++)] = val * VEL2.vz2;
                values[ic + dflen * (im++)] = val * (V2 + VEL2.vxvy);
                values[ic + dflen * (im++)] = val * (V2 + VEL2.vxvz);
                values[ic + dflen * (im++)] = val * (V2 + VEL2.vyvz);
            }
        }
    }

    /// convert the collected datacube to the output arrays
    void finalizeDatacube(const std::vector<double>& result, /*output arrays*/ double density[],
        coord::VelCar velocityFirstMoment[], coord::Vel2Car velocitySecondMoment[]) const
    {
        // store the results
        for(unsigned int ic=0; ic<dflen; ic++) {
            double dens = result[ic];
            if(density!=NULL)
                density[ic] = dens;
            unsigned int im=1;  // index of the computed moment in the results array
            if(velocityFirstMoment!=NULL) {
                double SUB = result[ic + dflen * (im++)];
                velocityFirstMoment[ic].vx = dens? (result[ic + dflen * (im++)]-SUB) / dens : 0;
                velocityFirstMoment[ic].vy = dens? (result[ic + dflen * (im++)]-SUB) / dens : 0;
                velocityFirstMoment[ic].vz = dens? (result[ic + dflen * (im++)]-SUB) / dens : 0;
            }
            if(velocitySecondMoment!=NULL) {
                double SUB = result[ic + dflen * (im+0)] +
                    result[ic + dflen * (im+1)] + result[ic + dflen * (im+2)];
                velocitySecondMoment[ic].vx2  = dens? result[ic + dflen * (im++)] / dens : 0;
                velocitySecondMoment[ic].vy2  = dens? result[ic + dflen * (im++)] / dens : 0;
                velocitySecondMoment[ic].vz2  = dens? result[ic + dflen * (im++)] / dens : 0;
                velocitySecondMoment[ic].vxvy = dens? (result[ic + dflen * (im++)]-SUB) / dens : 0;
                velocitySecondMoment[ic].vxvz = dens? (result[ic + dflen * (im++)]-SUB) / dens : 0;
                velocitySecondMoment[ic].vyvz = dens? (result[ic + dflen * (im++)]-SUB) / dens : 0;
            }
        }
    }
};


/** Helper class for constructing velocity distribution (intrinsic or projected) */
template <int N, bool Projected>
class DFIntegrandVelDist: public DFIntegrandNdim {
    const coord::Orientation& orientation;
    const Scaling<Projected> scaling;
    const math::BsplineInterpolator1d<N> bsplvX, bsplvY, bsplvZ; ///< B-splines for each vel. component
    const unsigned int NX, NY, NZ, Ntotal;  ///< number of B-spline coefficients in each vel. component
public:
    DFIntegrandVelDist(const GalaxyModel& model, bool separate,
        const typename Scaling<Projected>::Type& point, const coord::Orientation& _orientation,
        const std::vector<double>& gridvX,
        const std::vector<double>& gridvY,
        const std::vector<double>& gridvZ)
    :
        DFIntegrandNdim(model, separate),
        orientation(_orientation),
        scaling(model.potential, point, orientation),
        bsplvX(gridvX), bsplvY(gridvY), bsplvZ(gridvZ),
        NX(bsplvX.numValues()), NY(bsplvY.numValues()), NZ(bsplvZ.numValues()),
        Ntotal(1 + NX + NY + NZ)
    {}

    /// dimension of the input array (3 scaled velocity components and optionally scaled Z coordinate)
    virtual unsigned int numVars()   const { return Projected ? 4 : 3; }

    /// total number of B-spline coefficients times the number of DF components
    virtual unsigned int numValues() const { return dflen * Ntotal; }

    // input variables are scaled z-coordinate (if projected) and all three velocity components
    virtual coord::PosVelCar unscaleVars(const double vars[], double &jac) const {
        return scaling.unscale(vars, jac);
    }

    /// output the weighted integrals over basis functions
    virtual void outputValues(const coord::PosVelCar& pv, const double dfval[], double values[]) const
    {
        std::fill(values, values + dflen * Ntotal, 0);
        // transform the velocity back to the observed coordinate system
        coord::VelCar VEL = orientation.toRotated(coord::VelCar(pv));
        if(!isFinite(VEL.vx+VEL.vy+VEL.vz))
            return;
        double valvX[N+1], valvY[N+1], valvZ[N+1];
        unsigned int
            iX = bsplvX.nonzeroComponents(VEL.vx, 0, valvX),
            iY = bsplvY.nonzeroComponents(VEL.vy, 0, valvY),
            iZ = bsplvZ.nonzeroComponents(VEL.vz, 0, valvZ);
        double mult = scaling.symmetrizevRvz ? 0.5 : 1.0;
        for(unsigned int ic=0; ic<dflen; ic++) {  // loop over DF components
            values[ic * Ntotal] = dfval[ic];
            for(int ib=0; ib<=N; ib++) {  // loop over nonzero basis functions
                values[ic * Ntotal + 1 + ib + iX]          += dfval[ic] * valvX[ib] * mult;
                values[ic * Ntotal + 1 + ib + iY + NX]     += dfval[ic] * valvY[ib] * mult;
                values[ic * Ntotal + 1 + ib + iZ + NX + NY]+= dfval[ic] * valvZ[ib] * mult;
            }
        }
        if(!scaling.symmetrizevRvz) return;
        /// we scan only half of the (v_R, v_z) plane, and add the same contributions to (-v_R, -v_z),
        /// since the actions and hence the value of f(J) do not change with this inversion
        double R2 = pow_2(pv.x) + pow_2(pv.y),
        costwophi = R2==0 ? 1 : (pow_2(pv.x) - pow_2(pv.y)) / R2,
        sintwophi = R2==0 ? 0 : 2 * pv.x * pv.y / R2;
        VEL = orientation.toRotated(coord::VelCar(
            -pv.vx * costwophi - pv.vy * sintwophi, -pv.vx * sintwophi + pv.vy * costwophi, -pv.vz));
        if(!isFinite(VEL.vx+VEL.vy+VEL.vz))
            return;
        iX = bsplvX.nonzeroComponents(VEL.vx, 0, valvX),
        iY = bsplvY.nonzeroComponents(VEL.vy, 0, valvY),
        iZ = bsplvZ.nonzeroComponents(VEL.vz, 0, valvZ);
        for(unsigned int ic=0; ic<dflen; ic++) {  // loop over DF components
            for(int ib=0; ib<=N; ib++) {  // loop over nonzero basis functions
                values[ic * Ntotal + 1 + ib + iX]          += dfval[ic] * valvX[ib] * mult;
                values[ic * Ntotal + 1 + ib + iY + NX]     += dfval[ic] * valvY[ib] * mult;
                values[ic * Ntotal + 1 + ib + iZ + NX + NY]+= dfval[ic] * valvZ[ib] * mult;
            }
        }
    }

    /// convert the collected datacube to the amplitudes of B-splines
    void finalizeDatacube(const std::vector<double>& result, /*output arrays*/ double density[],
        std::vector<double> amplvX[], std::vector<double> amplvY[], std::vector<double> amplvZ[]) const
    {
        math::BandMatrix<double>  // matrices for converting collected values into B-spline amplitudes
            matvX = math::FiniteElement1d<N>(bsplvX).computeProjMatrix(),
            matvY = math::FiniteElement1d<N>(bsplvY).computeProjMatrix(),
            matvZ = math::FiniteElement1d<N>(bsplvZ).computeProjMatrix();

        // loop over elements of a multicomponent DF
        for(unsigned int ic=0; ic<dflen; ic++) {
            // beginning of storage for the current element
            const double* begin = &result[Ntotal * ic];
            // compute the amplitudes of un-normalized VDF
            amplvX[ic] = solveBand(matvX, std::vector<double>(begin+1, begin+1+NX));
            amplvY[ic] = solveBand(matvY, std::vector<double>(begin+1+NX, begin+1+NX+NY));
            amplvZ[ic] = solveBand(matvZ, std::vector<double>(begin+1+NX+NY, begin+1+NX+NY+NZ));
            // normalize by the value of density
            density[ic] = *begin;
            if(density[ic]!=0) {
                math::blas_dmul(1/density[ic], amplvX[ic]);
                math::blas_dmul(1/density[ic], amplvY[ic]);
                math::blas_dmul(1/density[ic], amplvZ[ic]);
            }
        }
    }
};


/// this will be redesigned
class DFIntegrandProjection: public math::IFunctionNdim {
    const GalaxyModel& model;
    const math::IFunctionNdim& fnc;  ///< spatial selection function in the observed coords
    const coord::Orientation& orientation;
public:
    DFIntegrandProjection(const GalaxyModel& _model,
        const math::IFunctionNdim& _fnc, const coord::Orientation& _orientation)
    :
        model(_model), fnc(_fnc), orientation(_orientation) {}

    virtual unsigned int numVars()   const { return 6; }
    virtual unsigned int numValues() const { return fnc.numValues(); }

    virtual void eval(const double vars[], double values[]) const
    {
        try{
            Scaling<true> scaling(model.potential,
                coord::PosProj(/*X*/vars[0], /*Y*/vars[1]), orientation);
            double jac;
            coord::PosVelCar posvel = scaling.unscale(/*Z and velocity components*/ &vars[2], jac);
            // process two symmetric velocity orientations and halve their contributions to the integral
            if(scaling.symmetrizevRvz) jac *= 0.5;

            // 2. determine the actions and the DF value times the jacobian
            actions::Actions act = model.actFinder.actions(toPosVelCyl(posvel));
            // FIXME: safety measure for zero actions reported by Fudge
            double dfval = isFinite(act.Jr + act.Jz + act.Jphi) && (act.Jr!=0 || act.Jz!=0) ?
                model.distrFunc.value(act) * jac : 0.;
            if(!isFinite(dfval))
                dfval = 0;

            // transform the position/velocity back to the observed frame
            double posvelobs[6];
            orientation.toRotated(posvel).unpack_to(posvelobs);
            // if the Z coordinate is very large, the inverse transformation might not
            // accurately recover the input X,Y coordinates - put them back manually
            posvelobs[0] = vars[0];
            posvelobs[1] = vars[1];

            // query the spatial selection function (in the observed frame)
            double* fncval = static_cast<double*>(alloca(fnc.numValues() * sizeof(double)));
            fnc.eval(posvelobs, fncval);

            // output the values of df * sf to the integration routine
            for(unsigned int i=0, count=fnc.numValues(); i<count; i++)
                values[i] = dfval * fncval[i];

            if(!scaling.symmetrizevRvz) return;

            // add a contribution of the symmetric point with -vR, -vz (in the intrinsic coordinates)
            double R2 = pow_2(posvel.x) + pow_2(posvel.y),
            costwophi = R2==0 ? 1 : (pow_2(posvel.x) - pow_2(posvel.y)) / R2,
            sintwophi = R2==0 ? 0 : 2 * posvel.x * posvel.y / R2;
            double vel[3] = { -posvel.vx * costwophi - posvel.vy * sintwophi,
                -posvel.vx * sintwophi + posvel.vy * costwophi, -posvel.vz };
            orientation.toRotated(vel, &posvelobs[3]);  // replace the velocity in posvelobs

            fnc.eval(posvelobs, fncval);
            for(unsigned int i=0, count=fnc.numValues(); i<count; i++)
                values[i] += dfval * fncval[i];
        }
        catch(std::exception&) {
            for(unsigned int i=0, count=fnc.numValues(); i<count; i++)
                values[i] = 0;
        }
    }
};

}  // unnamed namespace

//------- DRIVER ROUTINES -------//

// non-projected version
void computeMoments(
    const GalaxyModel& model,
    const coord::PosCar& point,
    // output arrays - one element per DF component
    double density[],
    coord::VelCar velocityFirstMoment[],
    coord::Vel2Car velocitySecondMoment[],
    // optional input parameters
    bool separate,
    const coord::Orientation& orientation,
    double reqRelError,
    int maxNumEval)
{
    DFIntegrandMoments<false> fnc(model, separate, point, orientation,
        velocityFirstMoment!=NULL, velocitySecondMoment!=NULL);
    // the integration region [3 components of scaled velocity]
    double xlower[3] = {0, 0, 0};
    double xupper[3] = {1, 1, 1};
    std::vector<double> result(fnc.numValues());  // temporary storage
    math::integrateNdim(fnc, xlower, xupper, reqRelError, maxNumEval, /*output*/ &result[0]);
    fnc.finalizeDatacube(result, /*output*/ density, velocityFirstMoment, velocitySecondMoment);
}

// projected version
void computeMoments(
    const GalaxyModel& model,
    const coord::PosProj& point,
    // output arrays - one element per DF component
    double density[],
    coord::VelCar velocityFirstMoment[],
    coord::Vel2Car velocitySecondMoment[],
    // optional input parameters
    bool separate,
    const coord::Orientation& orientation,
    double reqRelError,
    int maxNumEval)
{
    DFIntegrandMoments<true> fnc(model, separate, point, orientation,
        velocityFirstMoment!=NULL, velocitySecondMoment!=NULL);
    // the integration region [scaled z, 3 components of scaled velocity]
    double xlower[4] = {0, 0, 0, 0};
    double xupper[4] = {1, 1, 1, 1};
    std::vector<double> result(fnc.numValues());  // temporary storage
    math::integrateNdim(fnc, xlower, xupper, reqRelError, maxNumEval, /*output*/ &result[0]);
    fnc.finalizeDatacube(result, /*output*/ density, velocityFirstMoment, velocitySecondMoment);
}


void computeProjectedDF(const GalaxyModel& model, const coord::PosProj& point,
    const coord::VelCar& vel, const coord::VelCar& velerr, double result[], bool separate,
    const coord::Orientation& orientation, double reqRelError, int maxNumEval)
{
    double xlower[4] = {0, 0, 0, 0};  // integration region in scaled variables
    double xupper[4] = {1, 1, 1, 1};  // (not all 4 dimensions may be used if some of errors are 0)
    math::integrateNdim(DFIntegrandProjected(model, separate, point, vel, velerr, orientation),
        xlower, xupper, reqRelError, maxNumEval, result);
}


// non-projected version
template <int N>
void computeVelocityDistribution(
    const GalaxyModel& model,
    const coord::PosCar& point,
    const std::vector<double>& gridvX,
    const std::vector<double>& gridvY,
    const std::vector<double>& gridvZ,
    // output arrays - one element per DF component (if separate==true), otherwise just one element
    double density[],
    std::vector<double> amplvX[],
    std::vector<double> amplvY[],
    std::vector<double> amplvZ[],
    bool separate,
    const coord::Orientation& orientation,
    double reqRelError,
    int maxNumEval)
{
    DFIntegrandVelDist<N, false> fnc(model, separate, point, orientation, gridvX, gridvY, gridvZ);
    // the integration region [3 components of scaled velocity]
    double xlower[3] = {0, 0, 0};
    double xupper[3] = {1, 1, 1};
    std::vector<double> result(fnc.numValues());  // temporary storage
    math::integrateNdim(fnc, xlower, xupper, reqRelError, maxNumEval, /*output*/ &result[0]);
    fnc.finalizeDatacube(result, /*output*/ density, amplvX, amplvY, amplvZ);
}

// projected version
template <int N>
void computeVelocityDistribution(
    const GalaxyModel& model,
    const coord::PosProj& point,
    const std::vector<double>& gridvX,
    const std::vector<double>& gridvY,
    const std::vector<double>& gridvZ,
    // output arrays - one element per DF component (if separate==true), otherwise just one element
    double density[],
    std::vector<double> amplvX[],
    std::vector<double> amplvY[],
    std::vector<double> amplvZ[],
    bool separate,
    const coord::Orientation& orientation,
    double reqRelError,
    int maxNumEval)
{
    DFIntegrandVelDist<N, true> fnc(model, separate, point, orientation, gridvX, gridvY, gridvZ);
    // the integration region [scaled z if projected, 3 components of scaled velocity]
    double xlower[4] = {0, 0, 0, 0};
    double xupper[4] = {1, 1, 1, 1};
    std::vector<double> result(fnc.numValues());  // temporary storage
    math::integrateNdim(fnc, xlower, xupper, reqRelError, maxNumEval, /*output*/ &result[0]);
    fnc.finalizeDatacube(result, /*output*/ density, amplvX, amplvY, amplvZ);
}

// convenience function, non-projected
template <int N>
void computeVelocityDistribution(
    const GalaxyModel& model,
    const coord::PosCar& point,
    size_t gridsize,
    // velocity grid will be initialized by this routine
    std::vector<double>& gridv,
    // output arrays - one element per DF component (if separate==true), otherwise just one element
    double density[],
    std::vector<double> amplvX[],
    std::vector<double> amplvY[],
    std::vector<double> amplvZ[],
    // optional input parameters
    bool separate,
    const coord::Orientation& orientation,
    double reqRelError,
    int maxNumEval)
{
    double vesc = sqrt(-2 * model.potential.value(point));
    gridv = math::createUniformGrid(gridsize, -vesc, +vesc);
    // now call the actual function
    computeVelocityDistribution<N>(model, point, gridv, gridv, gridv,
        density, amplvX, amplvY, amplvZ, separate, orientation, reqRelError, maxNumEval);
}

// convenience function, projected
template <int N>
void computeVelocityDistribution(
    const GalaxyModel& model,
    const coord::PosProj& point,
    size_t gridsize,
    // velocity grid will be initialized by this routine
    std::vector<double>& gridv,
    // output arrays - one element per DF component (if separate==true), otherwise just one element
    double density[],
    std::vector<double> amplvX[],
    std::vector<double> amplvY[],
    std::vector<double> amplvZ[],
    // optional input parameters
    bool separate,
    const coord::Orientation& orientation,
    double reqRelError,
    int maxNumEval)
{
    // find the minimum of the potential and hence the maximum of the escape velocity
    double Z, dummy;
    findRoots(model.potential, 0, point.X, point.Y, orientation, /*output*/ Z, dummy, dummy);
    double vesc = sqrt(-2 * model.potential.value(coord::PosCar(point.X, point.Y, Z)));
    gridv = math::createUniformGrid(gridsize, -vesc, +vesc);
    // now call the actual function
    computeVelocityDistribution<N>(model, point, gridv, gridv, gridv,
        density, amplvX, amplvY, amplvZ, separate, orientation, reqRelError, maxNumEval);
}

// force compilation of several template instantiations
template void computeVelocityDistribution<0>(const GalaxyModel&, const coord::PosCar&,
    const std::vector<double>&, const std::vector<double>&, const std::vector<double>&,
    double[], std::vector<double>[], std::vector<double>[], std::vector<double>[],
    bool, const coord::Orientation&, double, int);
template void computeVelocityDistribution<1>(const GalaxyModel&, const coord::PosCar&,
    const std::vector<double>&, const std::vector<double>&, const std::vector<double>&,
    double[], std::vector<double>[], std::vector<double>[], std::vector<double>[],
    bool, const coord::Orientation&, double, int);
template void computeVelocityDistribution<2>(const GalaxyModel&, const coord::PosCar&,
    const std::vector<double>&, const std::vector<double>&, const std::vector<double>&,
    double[], std::vector<double>[], std::vector<double>[], std::vector<double>[],
    bool, const coord::Orientation&, double, int);
template void computeVelocityDistribution<3>(const GalaxyModel&, const coord::PosCar&,
    const std::vector<double>&, const std::vector<double>&, const std::vector<double>&,
    double[], std::vector<double>[], std::vector<double>[], std::vector<double>[],
    bool, const coord::Orientation&, double, int);

template void computeVelocityDistribution<0>(const GalaxyModel&, const coord::PosProj&,
    const std::vector<double>&, const std::vector<double>&, const std::vector<double>&,
    double[], std::vector<double>[], std::vector<double>[], std::vector<double>[],
    bool, const coord::Orientation&, double, int);
template void computeVelocityDistribution<1>(const GalaxyModel&, const coord::PosProj&,
    const std::vector<double>&, const std::vector<double>&, const std::vector<double>&,
    double[], std::vector<double>[], std::vector<double>[], std::vector<double>[],
    bool, const coord::Orientation&, double, int);
template void computeVelocityDistribution<2>(const GalaxyModel&, const coord::PosProj&,
    const std::vector<double>&, const std::vector<double>&, const std::vector<double>&,
    double[], std::vector<double>[], std::vector<double>[], std::vector<double>[],
    bool, const coord::Orientation&, double, int);
template void computeVelocityDistribution<3>(const GalaxyModel&, const coord::PosProj&,
    const std::vector<double>&, const std::vector<double>&, const std::vector<double>&,
    double[], std::vector<double>[], std::vector<double>[], std::vector<double>[],
    bool, const coord::Orientation&, double, int);

template void computeVelocityDistribution<0>(const GalaxyModel&, const coord::PosCar&, size_t,
    std::vector<double>&, double[], std::vector<double>[], std::vector<double>[], std::vector<double>[],
    bool, const coord::Orientation&, double, int);
template void computeVelocityDistribution<1>(const GalaxyModel&, const coord::PosCar&, size_t,
    std::vector<double>&, double[], std::vector<double>[], std::vector<double>[], std::vector<double>[],
    bool, const coord::Orientation&, double, int);
template void computeVelocityDistribution<2>(const GalaxyModel&, const coord::PosCar&, size_t,
    std::vector<double>&, double[], std::vector<double>[], std::vector<double>[], std::vector<double>[],
    bool, const coord::Orientation&, double, int);
template void computeVelocityDistribution<3>(const GalaxyModel&, const coord::PosCar&, size_t,
    std::vector<double>&, double[], std::vector<double>[], std::vector<double>[], std::vector<double>[],
    bool, const coord::Orientation&, double, int);

template void computeVelocityDistribution<0>(const GalaxyModel&, const coord::PosProj&, size_t,
    std::vector<double>&, double[], std::vector<double>[], std::vector<double>[], std::vector<double>[],
    bool, const coord::Orientation&, double, int);
template void computeVelocityDistribution<1>(const GalaxyModel&, const coord::PosProj&, size_t,
    std::vector<double>&, double[], std::vector<double>[], std::vector<double>[], std::vector<double>[],
    bool, const coord::Orientation&, double, int);
template void computeVelocityDistribution<2>(const GalaxyModel&, const coord::PosProj&, size_t,
    std::vector<double>&, double[], std::vector<double>[], std::vector<double>[], std::vector<double>[],
    bool, const coord::Orientation&, double, int);
template void computeVelocityDistribution<3>(const GalaxyModel&, const coord::PosProj&, size_t,
    std::vector<double>&, double[], std::vector<double>[], std::vector<double>[], std::vector<double>[],
    bool, const coord::Orientation&, double, int);


void computeProjection(const GalaxyModel& model,
    const math::IFunctionNdim& spatialSelection,
    const double Xlim[2], const double Ylim[2],
    const coord::Orientation& orientation,
    double* result,
    double reqRelError, int maxNumEval)
{
    const double xlower[6] = {Xlim[0], Ylim[0], 0, 0, 0, 0};
    const double xupper[6] = {Xlim[1], Ylim[1], 1, 1, 1, 1};
    DFIntegrandProjection fnc(model, spatialSelection, orientation);
    math::integrateNdim(fnc, xlower, xupper, reqRelError, maxNumEval, result);
}


void computeTotalMass(
    const GalaxyModel& model,
    double* result,
    bool separate,
    double reqRelError,
    int maxNumEval)
{
    DFIntegrand6dim fnc(model, separate);
    double xlower[6] = {0,0,0,0,0,0}; // boundaries of integration region in scaled coordinates
    double xupper[6] = {1,1,1,1,1,1};
    math::integrateNdim(fnc, xlower, xupper, reqRelError, maxNumEval, result);
}


particles::ParticleArrayCar samplePosVel(
    const GalaxyModel& model, const size_t numPoints,
    math::SampleMethod method, math::PRNGState* state)
{
    DFIntegrand6dim fnc(model, /*separate*/ false);
    double xlower[6] = {0,0,0,0,0,0}; // boundaries of sampling region in scaled coordinates
    double xupper[6] = {1,1,1,1,1,1};
    math::Matrix<double> result;      // sampled scaled coordinates/velocities
    std::vector<double> weights;
    math::sampleNdim(fnc, xlower, xupper, numPoints, method, result, weights, NULL, NULL, state);
    particles::ParticleArrayCar points;
    points.data.reserve(result.rows());
    for(size_t i=0; i<result.rows(); i++) {
        double tmp;
        double scaledvars[6] = {
            result(i,0), result(i,1), result(i,2),
            result(i,3), result(i,4), result(i,5)};
        // transform from scaled vars (array of 6 numbers) to real pos/vel
        points.add(fnc.unscaleVars(scaledvars, tmp), weights[i]);
    }
    return points;
}


particles::ParticleArray<coord::PosCyl> sampleDensity(
    const potential::BaseDensity& dens, const size_t numPoints,
    math::SampleMethod method, math::PRNGState* state)
{
    potential::DensityIntegrandNdim fnc(dens, /*require the values of density to be non-negative*/ true);
    double xlower[3] = {0,0,0};       // boundaries of sampling region in scaled coordinates
    double xupper[3] = {1,1,1};
    math::Matrix<double> result;      // sampled scaled coordinates
    std::vector<double> weights;
    math::sampleNdim(fnc, xlower, xupper, numPoints, method, result, weights, NULL, NULL, state);
    particles::ParticleArray<coord::PosCyl> points;
    points.data.reserve(result.rows());
    // if the density profile is axisymmetric, phi is not provided by the sampling routine,
    // so needs to be assigned here; when using QRNG and returning all samples rather than
    // an equally-weighted subset (to be used for high-accuracy integration), it is mandatory
    // to use QRNG for the remaining phi coordinate too; otherwise can use a PRNG.
    bool qrng = method & math::SM_RETURN_ALL_SAMPLES;
    math::QuasiRandomSobol gen2(
        /*dimension: 0 and 1 were already used in sampleNdim, so..*/ 2,
        /*random state*/ state,
        /*ensure enough bits for the given # of samples*/ std::max<int>(ceil(log2(result.rows())), 32));
    for(size_t i=0; i<result.rows(); i++) {
        double scaledvars[3] = {result(i,0), result(i,1), 
            fnc.axisym ? (qrng ? gen2(i) : math::random(state)) : result(i,2)};
        // transform from scaled coordinates to the real ones, and store the point into the array
        points.add(potential::unscaleCoords(scaledvars), weights[i]);
    }
    return points;
}

}  // namespace
