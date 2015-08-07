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

double BasePotentialSphericallySymmetric::enclosedMass(const double radius, const double) const
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

class RcircRootFinder: public math::IFunction {
public:
    RcircRootFinder(const BasePotential& _poten, double _E) :
        poten(_poten), E(_E) {};
    virtual void evalDeriv(const double R, double* val=0, double* deriv=0, double* deriv2=0) const {
        double Phi;
        coord::GradCyl grad;
        coord::HessCyl hess;
        poten.eval(coord::PosCyl(R,0,0), &Phi, &grad, &hess);
        if(val) {
            if(R==INFINITY && !math::isFinite(Phi))
                *val = -1-fabs(E);  // safely negative value
            else
                *val = 2*(E-Phi) - (R>0 && R!=INFINITY ? R*grad.dR : 0);
        }
        if(deriv)
            *deriv = -3*grad.dR - R*hess.dR2;
        if(deriv2)
            *deriv2=NAN;
    }
    virtual unsigned int numDerivs() const { return 1; }
private:
    const BasePotential& poten;
    const double E;
};

class RfromLzRootFinder: public math::IFunction {
public:
    RfromLzRootFinder(const BasePotential& _poten, double _Lz) :
        poten(_poten), Lz2(_Lz*_Lz) {};
    virtual void evalDeriv(const double R, double* val=0, double* deriv=0, double* deriv2=0) const {
        coord::GradCyl grad;
        coord::HessCyl hess;
        poten.eval(coord::PosCyl(R,0,0), NULL, &grad, &hess);
        if(val) {
            if(R==INFINITY)
                *val = -1-Lz2;  // safely negative value
            else
                *val = Lz2 - (R>0 ? pow_3(R)*grad.dR : 0);
        }
        if(deriv)
            *deriv = pow_2(R)*( 3*grad.dR - R*hess.dR2);
        if(deriv2)
            *deriv2=NAN;
    }
    virtual unsigned int numDerivs() const { return 1; }
private:
    const BasePotential& poten;
    const double Lz2;
};

double R_circ(const BasePotential& potential, double energy) {
    if((potential.symmetry() & ST_ZROTSYM) != ST_ZROTSYM)
        throw std::invalid_argument("Potential is not axisymmetric, "
            "no meaningful definition of circular orbit is possible");
    return math::findRoot(RcircRootFinder(potential, energy), 0, INFINITY, EPSREL_POTENTIAL_INT);
}

double L_circ(const BasePotential& potential, double energy) {
    double R = R_circ(potential, energy);
    return R * v_circ(potential, R);
}

double R_from_Lz(const BasePotential& potential, double Lz) {
    if(Lz==0)
        return 0;
    if((potential.symmetry() & ST_ZROTSYM) != ST_ZROTSYM)
        throw std::invalid_argument("Potential is not axisymmetric, "
            "no meaningful definition of circular orbit is possible");
    return math::findRoot(RfromLzRootFinder(potential, Lz), 0, INFINITY, EPSREL_POTENTIAL_INT);
}

void epicycleFreqs(const BasePotential& potential, const double R,
    double& kappa, double& nu, double& Omega)
{
    if((potential.symmetry() & ST_ZROTSYM) != ST_ZROTSYM)
        throw std::invalid_argument("Potential is not axisymmetric, "
            "no meaningful definition of circular orbit is possible");
    coord::GradCyl grad;
    coord::HessCyl hess;
    potential.eval(coord::PosCyl(R, 0, 0), NULL, &grad, &hess);
    //!!! no attempt to check if the expressions under sqrt are non-negative, or that R>0
    kappa = sqrt(hess.dR2 + 3*grad.dR/R);
    nu    = sqrt(hess.dz2);
    Omega = sqrt(grad.dR/R);
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
    if((dens.symmetry() & ST_AXISYMMETRIC) == ST_AXISYMMETRIC)
        return (m==0 ? dens.density(coord::PosCyl(R, z, 0)) : 0);
    double phimax = (dens.symmetry() & ST_PLANESYM) == ST_PLANESYM ? M_PI_2 : 2*M_PI;
    if(m==0)
        return math::integrate(DensityAzimuthalAverageIntegrand(dens, R, z, m),
            0, phimax, EPSREL_DENSITY_INT) / phimax;
    return math::integrateGL(DensityAzimuthalAverageIntegrand(dens, R, z, m),
        0, phimax, std::max<int>(8, std::abs(m))) / phimax;
}

/// helper class for integrating density over volume
class DensityNdimIntegrand: public math::IFunctionNdim {
public:
    DensityNdimIntegrand(const BaseDensity& _dens) :
        dens(_dens) {};
    // compute azimuthal integrand of density at a given point in (R,z) plane
    virtual void eval(const double vars[], double values[]) const 
    {   // input array is [scaled coordinate r, cos(theta)]
        const double rscaled = vars[0], costheta = vars[1];
        if(rscaled==1) {
            values[0] = 0;  // we're at infinity
            return;
        }
        const double r = rscaled/(1-rscaled);
        const double R = r*sqrt(1-pow_2(costheta));
        const double z = r*costheta;
        const double mult = 2*M_PI * r*r/pow_2(1-rscaled);
        double val = computeRho_m(dens, R, z, 0);
        if((dens.symmetry() & ST_PLANESYM) == ST_PLANESYM)
            val *= 2;
        else
            val += computeRho_m(dens, R, -z, 0);
        values[0] = val * mult;
    }
    virtual unsigned int numVars() const { return 2; }
    virtual unsigned int numValues() const { return 1; }
private:
    const BaseDensity& dens;
};

double BaseDensity::enclosedMass(const double r, const double relToler) const
{
    if(r<=0) return 0;
    if(relToler<=0)
        throw std::invalid_argument("Invalid relative error tolerance in enclosedMass()");
    // default implementation is to integrate over density inside given radius;
    // may be replaced by cheaper and more approximate evaluation for derived classes
    double xlower[2] = {0, 0};
    double xupper[2] = {r/(1+r), 1};
    double result, error;
    int numEval;
    const int maxNumEval = 10000;
    math::integrateNdim(DensityNdimIntegrand(*this), xlower, xupper, relToler, maxNumEval,
        &result, &error, &numEval);
    return result;
}

double BaseDensity::totalMass(const double rel_toler) const
{
    // default implementation attempts to estimate the asymptotic behaviour of density as r -> infinity
    double rad=32;
    double mass1, mass2 = enclosedMass(rad, rel_toler), mass3 = enclosedMass(rad*2, rel_toler);
    double massEst=0, massEstPrev;
    int numNeg=0, numIter=0;
    const int maxNumNeg=4, maxNumIter=20;
    do{
        rad *= 2;
        mass1 = mass2;
        mass2 = mass3;
        mass3 = enclosedMass(rad*2, rel_toler);
        if(mass2 == mass3) {
            return mass3;  // mass doesn't seem to grow with raduis anymore
        }
        massEstPrev = massEst>0 ? massEst : mass3;
        massEst = (mass2*mass2-mass1*mass3)/(2*mass2-mass1-mass3);
        if(!math::isFinite(massEst) || massEst<=0)
            numNeg++;  // increase counter of 'bad' attempts (negative means that mass is growing at least logarithmically with radius)
        numIter++;
    } while(numIter<maxNumIter && numNeg<maxNumNeg && mass2!=mass3 &&
        (massEst<0 || fabs((massEstPrev-massEst)/massEst)>rel_toler));
    if(fabs((massEstPrev-massEst)/massEst)>rel_toler)
        massEst = INFINITY;   // total mass seems to be infinite
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

double getRadiusByMass(const BaseDensity& dens, const double m, const double rel_toler) {
    return math::findRoot(RadiusByMassRootFinder(dens, m), 0, INFINITY, rel_toler);
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
