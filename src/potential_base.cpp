#include "potential_base.h"
#include "math_core.h"
#include <cmath>
#include <alloca.h>

namespace potential{
    
namespace{  // internal

/// relative accuracy of density computation by integration
const double EPSREL_DENSITY_INT = 1e-4;

/// max. number of density evaluations for multidimensional integration
const size_t MAX_NUM_EVAL_INT = 10000;

/// at large r, the density computed from potential derivatives is subject to severe cancellation errors;
/// if the result is smaller than this fraction of the absolute value of each term, we return zero
/// (otherwise its relative accuracy is too low and its derivative cannot be reliably estimated)
const double EPSREL_DENSITY_DER = DBL_EPSILON / ROOT3_DBL_EPSILON;

inline double nan2num(double x) { return isFinite(x) ? x : 0; }

/// helper class for finding the radius that encloses the given mass
class RadiusByMassRootFinder: public math::IFunctionNoDeriv {
    const BaseDensity& dens;
    const double m;
public:
    RadiusByMassRootFinder(const BaseDensity& _dens, double _m) :
        dens(_dens), m(_m) {}
    virtual double value(double r) const {
        return dens.enclosedMass(r) - m;
    }
};

/// helper class for integrating the density along the line of sight
class SurfaceDensityIntegrand: public math::IFunctionNoDeriv {
    const BaseDensity& dens;  ///< the density model
    const double X, Y, R;     ///< coordinates in the image plane
    const double* rotmatrix;  ///< rotation matrix for conversion between intrinsic and observed coords
public:
    SurfaceDensityIntegrand(const BaseDensity& _dens, double _X, double _Y, const double* _rotmatrix) :
        dens(_dens), X(_X), Y(_Y), R(sqrt(X*X+Y*Y)), rotmatrix(_rotmatrix) {}
    virtual double value(double s) const
    {
        // unscale the input scaled coordinate, which lies in the range (0..1);
        double t = fabs(s-0.5), u = exp(1/(0.5-t)-1/t);
        double Z = R*(s-0.5) + u*math::sign(s-0.5), dZds = R + u * (1/pow_2(0.5-t) + 1/pow_2(t));
        double XYZ[3] = {X, Y, Z}, xyz[3];
        coord::transformVector(rotmatrix, XYZ, xyz);
        return nan2num(dens.density(coord::PosCar(xyz[0], xyz[1], xyz[2])) * dZds);
    }
};

/// helper class for integrating the force along the line of sight
class ProjectedForceIntegrand: public math::IFunctionNdim {
    const BasePotential& pot; ///< the density model
    const double X, Y, R;     ///< coordinates in the image plane
    const double* rotmatrix;  ///< rotation matrix for conversion between intrinsic and observed coords
public:
    ProjectedForceIntegrand(const BasePotential& _pot, double _X, double _Y, const double* _rotmatrix) :
        pot(_pot), X(_X), Y(_Y), R(sqrt(X*X+Y*Y)), rotmatrix(_rotmatrix) {}

    virtual unsigned int numVars()   const { return 1; }
    virtual unsigned int numValues() const { return 2; }

    virtual void eval(const double vars[], double values[]) const
    {
        // unscale the input scaled coordinate, which lies in the range (0..1);
        double s = vars[0], t = fabs(s-0.5), u = exp(1/(0.5-t)-1/t);
        double Z = R*(s-0.5) + u*math::sign(s-0.5), dZds = R + u * (1/pow_2(0.5-t) + 1/pow_2(t));
        double XYZ[3] = {X, Y, Z}, xyz[3];
        coord::transformVector(rotmatrix, XYZ, xyz);
        coord::GradCar grad;
        pot.eval(coord::PosCar(xyz[0], xyz[1], xyz[2]), NULL, &grad);
        values[0] = nan2num((grad.dx * rotmatrix[0] + grad.dy * rotmatrix[3] + grad.dz * rotmatrix[6]) * dZds);
        values[1] = nan2num((grad.dx * rotmatrix[1] + grad.dy * rotmatrix[4] + grad.dz * rotmatrix[7]) * dZds);
    }
};

}  // internal ns


// -------- Computation of density from Laplacian in various coordinate systems -------- //

double BasePotential::densityCar(const coord::PosCar &pos, double time) const
{
    coord::HessCar deriv2;
    eval(pos, NULL, (coord::GradCar*)NULL, &deriv2, time);
    return (deriv2.dx2 + deriv2.dy2 + deriv2.dz2) * (1. / (4*M_PI));
}

double BasePotential::densityCyl(const coord::PosCyl &pos, double time) const
{
    coord::GradCyl deriv;
    coord::HessCyl deriv2;
    eval(pos, NULL, &deriv, &deriv2, time);
    double derivR_over_R = deriv.dR / pos.R;
    double deriv2phi_over_R2 = deriv2.dphi2 / pow_2(pos.R);
    if(pos.R <= fabs(pos.z) * SQRT_DBL_EPSILON) {  // close to or exactly on the z axis
        derivR_over_R = deriv2.dR2;
        deriv2phi_over_R2 = 0;
    }
    double result = (deriv2.dR2 + derivR_over_R + deriv2.dz2 + deriv2phi_over_R2);
    if(!(fabs(result) > EPSREL_DENSITY_DER * (fabs(deriv2.dR2) + fabs(derivR_over_R) +
        fabs(deriv2.dz2) + fabs(deriv2phi_over_R2))))
        return 0;  // dominated by roundoff errors
    return result / (4*M_PI);
}

double BasePotential::densitySph(const coord::PosSph &pos, double time) const
{
    coord::GradSph deriv;
    coord::HessSph deriv2;
    eval(pos, NULL, &deriv, &deriv2, time);
    double sintheta, costheta;
    math::sincos(pos.theta, sintheta, costheta);
    double derivr_over_r = deriv.dr / pos.r;
    double derivtheta_cottheta = deriv.dtheta * costheta / sintheta;
    if(sintheta==0)
        derivtheta_cottheta = deriv2.dtheta2;
    double angular_part = (deriv2.dtheta2 + derivtheta_cottheta + 
        (sintheta!=0 ? deriv2.dphi2 / pow_2(sintheta) : 0) ) / pow_2(pos.r);
    if(pos.r==0) {
        derivr_over_r = deriv2.dr2;
        angular_part=0;
    }
    double result = deriv2.dr2 + 2*derivr_over_r + angular_part;
    if(!(fabs(result) > EPSREL_DENSITY_DER * 
        (fabs(deriv2.dr2) + fabs(2*derivr_over_r) + fabs(angular_part))))
        return 0;  // dominated by roundoff errors
    return result / (4*M_PI);
}

// ---------- Integration of density by volume ---------- //

// scaling transformation for integration over volume
coord::PosCyl unscaleCoords(const double vars[], double* jac)
{
    double
    scaledr  = vars[0],
    costheta = vars[1] * 2 - 1,
    drds, r  = math::unscale(math::ScalingSemiInf(), scaledr, &drds);
    if(jac)
        *jac = (r<1e-100 || r>1e100) ? 0 :  // if near r=0 or infinity, set jacobian to zero
            4*M_PI * pow_2(r) * drds;
    return r == INFINITY ? coord::PosCyl(INFINITY, 0, 0) :  // avoid possible indeterminacy INF*0
        coord::PosCyl( r * sqrt(1-pow_2(costheta)), r * costheta, vars[2] * 2*M_PI);
}

/// helper class for integrating density over volume
void DensityIntegrandNdim::evalmany(const size_t npoints, const double vars[], double values[]) const
{
    // 0. allocate various temporary arrays on the stack - no need to delete them manually
    // positions in cylindrical coords (unscaled from the input variables)
    coord::PosCyl* pos = static_cast<coord::PosCyl*>(alloca(npoints * sizeof(coord::PosCyl)));
    // jacobian of coordinate transformation at each point
    double* jac = static_cast<double*>(alloca(npoints * sizeof(double)));
    // 1. unscale the input variables
    for(size_t i=0, numvars=numVars(); i<npoints; i++) {
        double scvars[3] = {vars[i*numvars], vars[i*numvars + 1], axisym ? 0. : vars[i*numvars + 2]};
        pos[i] = unscaleCoords(scvars, /*output*/ &jac[i]);
    }
    // 2. compute the density for all these points at once
    dens.evalmanyDensityCyl(npoints, pos, values);
    // 3. multiply by jacobian and post-process if needed
    for(size_t i=0; i<npoints; i++) {
        if(jac[i] == 0 || (nonnegative && values[i]<0))
            values[i] = 0;  // a non-negative result is required sometimes, e.g., for density sampling
        else
            values[i] *= jac[i];
    }
}

double BaseDensity::enclosedMass(const double r) const
{
    if(r==0) return 0;   // this assumes no central point mass! overriden in Plummer density model
    if(r==INFINITY) return totalMass();
    // default implementation is to integrate over density inside given radius;
    // may be replaced by cheaper and more approximate evaluation for derived classes
    double xlower[3] = {0, 0, 0};
    double xupper[3] = {math::scale(math::ScalingSemiInf(), r), 1, 1};
    double result, error;
    math::integrateNdim(DensityIntegrandNdim(*this),
        xlower, xupper, EPSREL_DENSITY_INT, MAX_NUM_EVAL_INT, &result, &error);
    return result;
}

double BasePotential::enclosedMass(const double r) const
{
    if(r==0) return 0;
    if(r==INFINITY)
        return totalMass();
    return sphericalAverage<AV_DRS>(*this, r) * r*r;  // approximate G M(<r) = r^2 dPhi/dr
}

double BaseDensity::totalMass() const
{
    // default implementation attempts to estimate the asymptotic behaviour of density as r -> infinity
    double rad=32;
    double mass1, mass2 = enclosedMass(rad), mass3 = enclosedMass(rad*2);
    double massEst=0, massEstPrev;
    int numIter=0;
    const int maxNumIter=20;
    do{
        rad *= 2;
        mass1 = mass2;
        mass2 = mass3;
        mass3 = enclosedMass(rad*2);
        if(mass3 == 0 || math::fcmp(mass2, mass3, pow_2(EPSREL_DENSITY_INT)) == 0) {
            return mass3;  // mass doesn't seem to grow with radius anymore
        }
        massEstPrev = massEst>0 ? massEst : mass3;
        massEst = (mass2 * mass2 - mass1 * mass3) / (2 * mass2 - mass1 - mass3);
        numIter++;
    } while(numIter<maxNumIter && (massEst<0 || fabs((massEstPrev-massEst)/massEst)>EPSREL_DENSITY_INT));
    if(!isFinite(massEst) || massEst<=0)
        // (negative means that mass is growing at least logarithmically with radius)
        massEst = INFINITY;   // total mass seems to be infinite
    return massEst;
}

double getRadiusByMass(const BaseDensity& dens, const double mass) {
    return math::findRoot(RadiusByMassRootFinder(dens, mass), math::ScalingSemiInf(), EPSREL_DENSITY_INT);
}

double getInnerDensitySlope(const BaseDensity& dens) {
    double mass1, mass2, mass3;
    double rad=1./1024;
    do {
        mass2 = dens.enclosedMass(rad);
        if(mass2<=0) rad*=2;
    } while(rad<1 && mass2==0);
    mass3 = dens.enclosedMass(rad*2);
    if(!isFinite(mass2+mass3))
        return NAN; // apparent error
    double alpha1, alpha2=log(mass3/mass2)/log(2.), gamma1=-1, gamma2=3-alpha2;
    int numIter=0;
    const int maxNumIter=20;
    do{
        rad /= 2;
        mass1 = dens.enclosedMass(rad);
        if(!isFinite(mass1))
            return gamma2;
        alpha1 = log(mass2/mass1)/log(2.);
        gamma2 = gamma1<0 ? 3-alpha1 : gamma1;  // rough estimate
        gamma1 = 3 - (2*alpha1-alpha2);  // extrapolated estimate
        alpha2 = alpha1;
        mass3  = mass2;
        mass2  = mass1;
        numIter++;
    } while(numIter<maxNumIter && fabs(gamma1-gamma2)>1e-3);
    if(fabs(gamma1)<1e-3)
        gamma1=0;
    return gamma1;
}

double surfaceDensity(const BaseDensity& dens, double X, double Y, double alpha, double beta, double gamma)
{
    double mat[9];
    // the rotation matrix corresponds to the transformation from intrinsic to observed coords,
    // but we need the opposite transformation, so we construct a transposed matrix
    // by using negative rotation angles in the inverse order
    coord::makeRotationMatrix(-gamma, -beta, -alpha, mat);
    return math::integrateAdaptive(SurfaceDensityIntegrand(dens, X, Y, mat), 0, 1, EPSREL_DENSITY_INT);
}

void projectedForce(const BasePotential& pot, double X, double Y,
    double alpha, double beta, double gamma, double& fX, double& fY)
{
    double mat[9];
    coord::makeRotationMatrix(-gamma, -beta, -alpha, mat);
    double xlower[1] = {0}, xupper[1] = {1}, result[2];
    math::integrateNdim(ProjectedForceIntegrand(pot, X, Y, mat), xlower, xupper, EPSREL_DENSITY_INT,
        /*maxNumEval*/ MAX_NUM_EVAL_INT, result);
    fX = result[0];
    fY = result[1];
}

}  // namespace potential
