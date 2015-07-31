#include "potential_base.h"
#include "math_core.h"
#include <cmath>
#include <stdexcept>

namespace potential{

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
    return (deriv2.dR2 + derivR_over_R + deriv2.dz2 + deriv2phi_over_R2) / (4*M_PI);
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
    return (deriv2.dr2 + 2*derivr_over_r + angular_part) / (4*M_PI);
}

double BasePotentialSphericallySymmetric::enclosedMass(double radius) const
{
    double dPhidr;
    evalDeriv(radius, NULL, &dPhidr);
    return pow_2(radius)*dPhidr;
}


// convenience function
double v_circ(const BasePotential& potential, double radius)
{
    if((potential.symmetry() & ST_ZROTSYM) != ST_ZROTSYM)
        throw std::invalid_argument("Potential is not axisymmetric, "
            "no meaningful definition of circular velocity is possible");
    coord::GradCyl deriv;
    potential.eval(coord::PosCyl(radius, 0, 0), NULL, &deriv);
    return sqrt(radius*deriv.dR);
}


// routines for integrating density over volume
class DensityAzimuthalIntegrand: public math::IFunctionNoDeriv {
public:
    DensityAzimuthalIntegrand(const BaseDensity& _dens, double r, double costheta) :
        dens(_dens), R(r*sqrt(1-pow_2(costheta))), z(r*costheta) {};
    virtual double value(double phi) const {
        double val = dens.density(coord::PosCyl(R, z, phi));
        if((dens.symmetry() & ST_PLANESYM) == ST_PLANESYM)
            return val;
        // otherwise need to add contributions from other octants
        val +=dens.density(coord::PosCyl(R, z, -phi))
            + dens.density(coord::PosCyl(R, z, M_PI-phi))
            + dens.density(coord::PosCyl(R, z, M_PI+phi))
            + dens.density(coord::PosCyl(R,-z, phi))
            + dens.density(coord::PosCyl(R,-z, -phi))
            + dens.density(coord::PosCyl(R,-z, M_PI-phi))
            + dens.density(coord::PosCyl(R,-z, M_PI+phi));
        return val/8;
    }
private:
    const BaseDensity& dens;
    const double R, z;
};

class DensityPolarIntegrand: public math::IFunctionNoDeriv {
public:
    DensityPolarIntegrand(const BaseDensity& _dens, double _r) :
        dens(_dens), r(_r) {};
    virtual double value(double costheta) const {
        if((dens.symmetry() & ST_AXISYMMETRIC) == ST_AXISYMMETRIC)
            return M_PI_2 * dens.density(coord::PosCyl(r*sqrt(1-pow_2(costheta)), r*costheta, 0));
        else
            return math::integrate(DensityAzimuthalIntegrand(dens, r, costheta), 0, M_PI_2, EPSREL_DENSITY_INT);
    }
private:
    const BaseDensity& dens;
    const double r;
};

class DensityRadialIntegrand: public math::IFunctionNoDeriv {
public:
    DensityRadialIntegrand(const BaseDensity& _dens) :
        dens(_dens) {};
    virtual double value(double rscaled) const {
        const double r = rscaled/(1-rscaled);
        const double mult = 8*r*r/pow_2(1-rscaled);
        if((dens.symmetry() & ST_SPHERICAL) == ST_SPHERICAL)
            return mult * dens.density(coord::PosSph(r, 0, 0));
        else
            return mult * math::integrate(DensityPolarIntegrand(dens, r), 0, 1, EPSREL_DENSITY_INT);
    }
private:
    const BaseDensity& dens;
};

double BaseDensity::enclosedMass(const double r) const
{
    if(r<=0) return 0;
    // default implementation is to integrate over density inside given radius;
    // may be replaced by cheaper and more approximate evaluation for derived classes
    return math::integrate(DensityRadialIntegrand(*this), 0, r/(1+r), EPSREL_DENSITY_INT);
}

double BaseDensity::totalMass() const
{
    // default implementation attempts to estimate the asymptotic behaviour of density as r -> infinity
    double rad=32;
    double mass1, mass2=enclosedMass(rad), mass3=enclosedMass(rad*2);
    double massEst=0, massEstPrev;
    int numNeg=0, numIter=0;
    const int maxNumNeg=4, maxNumIter=20;
    do{
        rad*=2;
        mass1=mass2; mass2=mass3; mass3=enclosedMass(rad*2);
        if(mass2==mass3)
            return mass3;  // mass doesn't seem to grow with raduis anymore
        massEstPrev = massEst>0 ? massEst : mass3;
        massEst = (mass2*mass2-mass1*mass3)/(2*mass2-mass1-mass3);
        if(!math::isFinite(massEst) || massEst<=0)
            numNeg++;  // increase counter of 'bad' attempts (negative means that mass is growing at least logarithmically with radius)
        numIter++;
    } while(numIter<maxNumIter && numNeg<maxNumNeg && mass2!=mass3 &&
        (massEst<0 || fabs((massEstPrev-massEst)/massEst)>EPSREL_DENSITY_INT));
    if(fabs((massEstPrev-massEst)/massEst)>EPSREL_DENSITY_INT)
        return INFINITY;   // total mass seems to be infinite
    else
        return massEst;
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

double getRadiusByMass(const BaseDensity& dens, const double m) {
    return math::findRoot(RadiusByMassRootFinder(dens, m), 0, INFINITY, EPSREL_DENSITY_INT);
}

double getInnerDensitySlope(const BaseDensity& dens) {
    double mass1, mass2, mass3;
    double rad=1./1024;
    do {
        mass2 = dens.enclosedMass(rad);
        if(mass2<=0) rad*=2;
    } while(rad<1 && mass2==0);
    mass3 = dens.enclosedMass(rad*2);
    if(!math::isFinite(mass2+mass3))
        return NAN; // apparent error
    double alpha1, alpha2=log(mass3/mass2)/log(2.), gamma1=-1, gamma2=3-alpha2;
    int numIter=0;
    const int maxNumIter=20;
    do{
        rad /= 2;
        mass1 = dens.enclosedMass(rad);
        if(!math::isFinite(mass1))
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
