#include "potential_base.h"
#include "math_core.h"
#include "math_spline.h"
#include <cmath>
#include <stdexcept>

namespace potential{

/// relative accuracy of density computation
static const double EPSREL_DENSITY_INT = 1e-4;

// -------- Computation of density from Laplacian in various coordinate systems -------- //

double BasePotential::densityCar(const coord::PosCar &pos) const
{
    coord::HessCar deriv2;
    eval(pos, NULL, (coord::GradCar*)NULL, &deriv2);
    return (deriv2.dx2 + deriv2.dy2 + deriv2.dz2) / (4*M_PI);
}

double BasePotential::densityCyl(const coord::PosCyl &pos) const
{
    coord::GradCyl deriv;
    coord::HessCyl deriv2;
    eval(pos, NULL, &deriv, &deriv2);
    double derivR_over_R = deriv.dR/pos.R;
    double deriv2phi_over_R2 = deriv2.dphi2/pow_2(pos.R);
    if(pos.R==0) {
        if(deriv.dR==0)  // otherwise should remain infinite
            derivR_over_R = deriv2.dR2;
        deriv2phi_over_R2 = 0;
    }
    double result = (deriv2.dR2 + derivR_over_R + deriv2.dz2 + deriv2phi_over_R2);
    if(fabs(result) < 1e-12 * (fabs(deriv2.dR2) + fabs(derivR_over_R) +
        fabs(deriv2.dz2) + fabs(deriv2phi_over_R2)) )
        result = 0;  // dominated by roundoff errors
    return result / (4*M_PI);
}

double BasePotential::densitySph(const coord::PosSph &pos) const
{
    coord::GradSph deriv;
    coord::HessSph deriv2;
    eval(pos, NULL, &deriv, &deriv2);
    double sintheta=sin(pos.theta);
    double derivr_over_r = deriv.dr/pos.r;
    double derivtheta_cottheta = deriv.dtheta*cos(pos.theta)/sintheta;
    if(sintheta==0)
        derivtheta_cottheta = deriv2.dtheta2;
    double angular_part = (deriv2.dtheta2 + derivtheta_cottheta + 
        deriv2.dphi2/pow_2(sintheta))/pow_2(pos.r);
    if(pos.r==0) {
        if(deriv.dr==0)  // otherwise should remain infinite
            derivr_over_r = deriv2.dr2;
        angular_part=0; ///!!! is this correct assumption?
    }
    double result = deriv2.dr2 + 2*derivr_over_r + angular_part;
    if(fabs(result) < 1e-12 * (fabs(deriv2.dr2) + fabs(2*derivr_over_r) + fabs(angular_part)) )
        result = 0;  // dominated by roundoff errors
    return result / (4*M_PI);
}

double BasePotentialSphericallySymmetric::enclosedMass(const double radius) const
{
    if(radius==INFINITY)
        return totalMass();
    double dPhidr;
    evalDeriv(radius, NULL, &dPhidr);
    return pow_2(radius)*dPhidr;
}

// ---------- Integration of density by volume ---------- //

// scaling transformation for integration over volume
coord::PosCyl unscaleCoords(const double vars[], double* jac)
{
    const double scaledr  = vars[0];
    const double costheta = vars[1] * 2 - 1;
    const double r = exp( 1/(1-scaledr) - 1/scaledr );
    if(jac)
        *jac = (r<1e-100 || r>1e100) ? 0 :  // if near r=0 or infinity, set jacobian to zero
            4*M_PI * pow_3(r) * (1/pow_2(1-scaledr) + 1/pow_2(scaledr));
    return coord::PosCyl( r * sqrt(1-pow_2(costheta)), r * costheta, vars[2] * 2*M_PI);
}

// return the scaled radius variable to be used as the integration limit
static double scaledr_from_r(const double r) {
    const double y = log(r);
    return  fabs(y)<1 ? // two cases depending on whether |y| is small or large
        1/(1 + sqrt(1+pow_2(y*0.5)) - y*0.5) :            // y is close to zero
        0.5 + sqrt(0.25+pow_2(1/y))*math::sign(y) - 1/y;  // y is large (or even +-infinity)
}

/// helper class for integrating density over volume
void DensityIntegrandNdim::eval(const double vars[], double values[]) const 
{
    double scvars[3] = {vars[0], vars[1], axisym ? 0. : vars[2]};
    double jac;         // jacobian of coordinate scaling
    const coord::PosCyl pos = unscaleVars(scvars, &jac);
    if(jac!=0)
        values[0] = dens.density(pos) * jac;
    else                // we're almost at infinity or nearly at zero (in both cases,
        values[0] = 0;  // the result is negligibly small, but difficult to compute accurately)
    if(nonnegative && values[0]<0)
        values[0] = 0;  // a non-negative result is required sometimes, e.g., for density sampling
}

double BaseDensity::enclosedMass(const double r) const
{
    if(r<=0) return 0;   // this assumes no central point mass! overriden in Plummer density model
    // default implementation is to integrate over density inside given radius;
    // may be replaced by cheaper and more approximate evaluation for derived classes
    DensityIntegrandNdim fnc(*this);
    double xlower[3] = {0, 0, 0};
    double xupper[3] = {scaledr_from_r(r), 1, 1};
    double result, error;
    int numEval;
    const int maxNumEval = 10000;
    math::integrateNdim(fnc, xlower, xupper, EPSREL_DENSITY_INT, maxNumEval,
        &result, &error, &numEval);
    return result;
}

double BaseDensity::totalMass() const
{
    // default implementation attempts to estimate the asymptotic behaviour of density as r -> infinity
    double rad=32;
    double mass1, mass2 = enclosedMass(rad), mass3 = enclosedMass(rad*2);
    double massEst=0, massEstPrev;
    int numNeg=0, numIter=0;
    const int maxNumNeg=4, maxNumIter=20;
    do{
        rad *= 2;
        mass1 = mass2;
        mass2 = mass3;
        mass3 = enclosedMass(rad*2);
        if(mass2 == mass3) {
            return mass3;  // mass doesn't seem to grow with raduis anymore
        }
        massEstPrev = massEst>0 ? massEst : mass3;
        massEst = (mass2*mass2-mass1*mass3)/(2*mass2-mass1-mass3);
        if(!isFinite(massEst) || massEst<=0)
            numNeg++;  // increase counter of 'bad' attempts (negative means that mass is growing at least logarithmically with radius)
        numIter++;
    } while(numIter<maxNumIter && numNeg<maxNumNeg && mass2!=mass3 &&
        (massEst<0 || fabs((massEstPrev-massEst)/massEst)>EPSREL_DENSITY_INT));
    if(fabs((massEstPrev-massEst)/massEst)>EPSREL_DENSITY_INT)
        massEst = INFINITY;   // total mass seems to be infinite
    return massEst;
}

/// helper class for averaging of density over azimuthal angle
class DensityAzimuthalAverageIntegrand: public math::IFunctionNoDeriv {
public:
    DensityAzimuthalAverageIntegrand(const BaseDensity& _dens, double _R, double _z, int _m) :
    dens(_dens), R(_R), z(_z), m(_m) {};
    virtual double value(double phi) const {
        return dens.density(coord::PosCyl(R, z, phi)) *
        (m==0 ? 1 : m>0 ? cos(m*phi) : sin(-m*phi));
    }
private:
    const BaseDensity& dens;
    double R, z, m;
};

double computeRho_m(const BaseDensity& dens, double R, double z, int m)
{   // compute m-th azimuthal Fourier harmonic coefficient
    // by averaging the input density over phi, if this is necessary at all
    if(isAxisymmetric(dens))
        return (m==0 ? dens.density(coord::PosCyl(R, z, 0)) : 0);
    double phimax = (dens.symmetry() & coord::ST_TRIAXIAL) == coord::ST_TRIAXIAL ? M_PI_2 : 2*M_PI;
    if(m==0)
        return math::integrate(DensityAzimuthalAverageIntegrand(dens, R, z, m),
            0, phimax, EPSREL_DENSITY_INT) / phimax;
    return math::integrateGL(DensityAzimuthalAverageIntegrand(dens, R, z, m),
        0, phimax, std::max<int>(8, math::abs(m))) / phimax;
}

class RadiusByMassRootFinder: public math::IFunctionNoDeriv {
public:
    RadiusByMassRootFinder(const BaseDensity& _dens, double _m) :
        dens(_dens), m(_m), mtot(dens.totalMass()) {};
    virtual double value(double r) const {
        return (r==INFINITY ? mtot : dens.enclosedMass(r)) - m;
    }
private:
    const BaseDensity& dens;
    const double m, mtot;
};

double getRadiusByMass(const BaseDensity& dens, const double mass) {
    return math::findRoot(RadiusByMassRootFinder(dens, mass), 0, INFINITY, EPSREL_DENSITY_INT);
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

}  // namespace potential
