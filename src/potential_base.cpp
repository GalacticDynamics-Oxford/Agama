#include "potential_base.h"
#include "math_core.h"
#include "math_linalg.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#ifndef _MSC_VER
#include <alloca.h>
#else
#include <malloc.h>
#endif

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
    if(isUnknown(symmetry()))
        throw std::runtime_error("symmetry is not provided");
    double xlower[3] = {0, 0, 0};
    double xupper[3] = {math::scale(math::ScalingSemiInf(), r), 1, 1};
    double result;
    math::integrateNdim(DensityIntegrandNdim(*this),
        xlower, xupper, EPSREL_DENSITY_INT, MAX_NUM_EVAL_INT, &result);
    return result;
}

double BasePotential::enclosedMass(const double r) const
{
    if(r==0) return 0;
    if(r==INFINITY)
        return totalMass();
    double dPhidr;
    Sphericalized<BasePotential>(*this).evalDeriv(r, NULL, &dPhidr);
    return dPhidr * r*r;  // approximate G M(<r) = r^2 dPhi/dr
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

// averaging over angles

namespace {  // internal

// average the input function over two spherical angles
// \tparam K   is the number of values the function provides at each point
// \param  fnc is the class providing batch evaluation of function values
// \param  r   is the spherical radius at which the values should be computed
// \param  result  will contain K averaged values
template<int K, class F>
void sphericalAverage(const F& fnc, double r, double result[K])
{
    const int ntheta = 8, nphi = 8;
    // nodes and weights of Gauss-Radau quadrature with 8 points in cos(theta)=0..1,
    // which gives exact result for band-limited spherical-harmonic potentials up to lmax=14
    const double
    costh [ntheta] = {0.9775206135612875, 0.8853209468390958, 0.7342101772154106,
        0.5471536263305554, 0.3526247171131696, 0.1802406917368924, 0.0562625605369222, 0},
    sinth [ntheta] = {0.2108398682952634, 0.4649804523718464, 0.6789222456756852,
        0.8370322031996875, 0.9357648256270681, 0.9836225358551961, 0.9984160076249926, 1},
    weight[ntheta] = {0.02862720368606458, 0.06241197533246655, 0.08675369890862478,
        0.09789304186312342, 0.09412938634727970, 0.07603265516169632, 0.04633953870074772, 0.015625};
    bool ysym  = isYReflSymmetric(fnc.src), zsym = isZReflSymmetric(fnc.src);
    int Nphi   = isZRotSymmetric (fnc.src) ? 1 : ysym ? nphi+1 : 2*nphi+1;
    // similarly to the azimuthal average, we only use the z>=0 half-plane
    // and double the weights of corresponding points if the function is z-reflection-symmetric
    int Ntheta = zsym ? ntheta : 2*ntheta-1;
    int npoints= Nphi * Ntheta;
    coord::PosCyl* pos = static_cast<coord::PosCyl*>(alloca(npoints * sizeof(coord::PosCyl)));
    double* values = static_cast<double*>(alloca(npoints * K * sizeof(double)));
    for(int ith=0; ith<Ntheta; ith++) {
        double cth = costh[ith % ntheta] * (ith<ntheta? 1 : -1), sth = sinth[ith % ntheta];
        for(int iph=0; iph<Nphi; iph++)
            pos[ith * Nphi + iph] = coord::PosCyl(r*sth, r*cth, M_PI/(nphi+0.5) * iph);
    }
    fnc.evalSph(npoints, pos, values);
    for(int k=0; k<K; k++)
        result[k] = 0;
    for(int ith=0; ith<Ntheta; ith++) {
        for(int iph=0; iph<Nphi; iph++) {
            double w = weight[ith % ntheta] *
                // if z-symmetric, points with z<0 are not evaluated explicitly,
                // instead the points with z>0 carry 2x weight
                (zsym && ith<ntheta-1 ? 2 : 1) *
                // if y-symmetric, points with y>0 also carry 2x weight
                (ysym && iph>0 ? 2 : 1);
            for(int k=0; k<K; k++)
                result[k] += values[(ith * Nphi + iph) * K + k] * w;
        }
    }
    if(Nphi>1) {
        for(int k=0; k<K; k++)
            result[k] *= 1./(2*nphi+1);
    }
}

template<int K, class F>
void azimuthalAverage(const F& fnc, double R, double z, double result[K])
{
    // averaging is performed by collecting the values of the function
    // at 2*nphi+1 equally-spaced points: phi[i] = i * 2*pi/(2*nphi+1).
    // if the function is symmetric w.r.t. Y-reflection,
    // its values at the last nphi points having y<0, i.e. sin(phi[i])<0,
    // are identical to the symmetric points with y>0, so the function does not need
    // to be evaluated there, but its values from y>0 carry 2x weight in the sum.
    const int nphi = 8;
    bool ysym = isYReflSymmetric(fnc.src);
    int npoints = ysym ? nphi+1 : 2*nphi+1;
    // step 1: assign coordinates
    coord::PosCyl* pos = static_cast<coord::PosCyl*>(alloca(npoints * sizeof(coord::PosCyl)));
    double* values = static_cast<double*>(alloca(npoints * K * sizeof(double)));
    for(int i=0; i<npoints; i++)
        pos[i] = coord::PosCyl(R, z, M_PI/(nphi+0.5) * i);
    // step 2: collect function values
    fnc.evalCyl(npoints, pos, values);
    // step 3: average
    for(int k=0; k<K; k++)
        result[k] = 0;
    for(int i=0; i<npoints; i++)
        for(int k=0; k<K; k++)
            result[k] += values[i * K + k] * (ysym && i>0 ? 2 : 1);
    for(int k=0; k<K; k++)
        result[k] *= 1./(2*nphi+1);
}

template<bool needPhi, bool needDer, bool needDer2>
class GetPotential {
public:
    const BasePotential& src;
    GetPotential(const BasePotential& _src) : src(_src) {}
    void evalSph(int npoints, const coord::PosCyl pos[], double result[]) const
    {
        // radius is supposed to be the same for all points, so precompute its inverse
        double invr = 1 / (sqrt(pow_2(pos[0].R) + pow_2(pos[0].z)) + 1e-100);
        double Phi;
        coord::GradCyl grad;
        coord::HessCyl hess;
        for(int i=0; i<npoints; i++) {
            src.eval(pos[i], needPhi? &Phi : NULL, needDer? &grad : NULL, needDer2? &hess : NULL);
            double dRdr = pos[i].R * invr, dzdr = pos[i].z * invr;
            if(needDer2) {
                double d2Phidr2 =
                    hess.dR2 * pow_2(dRdr) + hess.dz2 * pow_2(dzdr) + 2 * hess.dRdz * dRdr * dzdr;
                if(needDer) {
                    double dPhidr = grad.dR * dRdr + grad.dz * dzdr;
                    if(needPhi) {
                        result[i*3  ] = Phi;
                        result[i*3+1] = dPhidr;
                        result[i*3+2] = d2Phidr2;
                    } else {
                        result[i*2  ] = dPhidr;
                        result[i*2+1] = d2Phidr2;
                    }
                } else {
                    if(needPhi) {
                        result[i*2  ] = Phi;
                        result[i*2+1] = d2Phidr2;
                    } else {
                        result[i    ] = d2Phidr2;
                    }
                }
            } else {
                if(needDer) {
                    double dPhidr = grad.dR * dRdr + grad.dz * dzdr;
                    if(needPhi) {
                        result[i*2  ] = Phi;
                        result[i*2+1] = dPhidr;
                    } else {
                        result[i    ] = dPhidr;
                    }
                } else {
                    if(needPhi) {
                        result[i    ] = Phi;
                    }
                }
            }
        }
    }

    void evalCyl(int npoints, const coord::PosCyl pos[], double result[]) const
    {
        double Phi;
        coord::GradCyl grad;
        coord::HessCyl hess;
        for(int i=0; i<npoints; i++) {
            src.eval(pos[i], needPhi? &Phi : NULL, needDer? &grad : NULL, needDer2? &hess : NULL);
            if(needDer2) {
                if(needDer) {
                    if(needPhi) {
                        result[i*6  ] = Phi;
                        result[i*6+1] = grad.dR;
                        result[i*6+2] = grad.dz;
                        result[i*6+3] = hess.dR2;
                        result[i*6+4] = hess.dz2;
                        result[i*6+5] = hess.dRdz;
                    } else {
                        result[i*5  ] = grad.dR;
                        result[i*5+1] = grad.dz;
                        result[i*5+2] = hess.dR2;
                        result[i*5+3] = hess.dz2;
                        result[i*5+4] = hess.dRdz;
                    }
                } else {  // no grad
                    if(needPhi) {
                        result[i*4  ] = Phi;
                        result[i*4+1] = hess.dR2;
                        result[i*4+2] = hess.dz2;
                        result[i*4+3] = hess.dRdz;
                    } else {
                        result[i*3  ] = hess.dR2;
                        result[i*3+1] = hess.dz2;
                        result[i*3+2] = hess.dRdz;
                    }
                }
            } else {  // no hess
                if(needDer) {
                    if(needPhi) {
                        result[i*3  ] = Phi;
                        result[i*3+1] = grad.dR;
                        result[i*3+2] = grad.dz;
                    } else {
                        result[i*2  ] = grad.dR;
                        result[i*2+1] = grad.dz;
                    }
                } else {  // no grad
                    if(needPhi) {
                        result[i    ] = Phi;
                    }
                }
            }
        }
    }
};

class GetDensity {
public:
    const BaseDensity& src;
    GetDensity(const BaseDensity& _src) : src(_src) {}
    void evalSph(int npoints, const coord::PosCyl pos[], double result[]) const
    {
        src.evalmanyDensityCyl(npoints, pos, result);
    }
    void evalCyl(int npoints, const coord::PosCyl pos[], double result[]) const
    {
        src.evalmanyDensityCyl(npoints, pos, result);
    }
};

}  // internal namespace

Sphericalized<BaseDensity>::Sphericalized(const BaseDensity& _dens) :
    dens(_dens)
{
    if(isUnknown(dens.symmetry()))
        throw std::runtime_error("symmetry of input density is not provided");
}

Sphericalized<BasePotential>::Sphericalized(const BasePotential& _pot) :
    pot(_pot)
{
    if(isUnknown(pot.symmetry()))
        throw std::runtime_error("symmetry of input potential is not provided");
}

double Sphericalized<BaseDensity>::value(double r) const
{
    if(isSpherical(dens) || r==0 || r==INFINITY)  // nothing to average
        return dens.density(coord::PosCyl(r, 0, 0));
    double result;
    sphericalAverage<1>(GetDensity(dens), r, &result);
    return result;
}

double Sphericalized<BasePotential>::densitySph(const coord::PosSph& pos, double /*time*/) const
{
    if(isSpherical(pot) || pos.r==0 || pos.r==INFINITY)  // nothing to average
        return pot.density(pos);
    double result;
    sphericalAverage<1>(GetDensity(pot), pos.r, &result);
    return result;
}

void Sphericalized<BasePotential>::evalDeriv(double r, double* val, double* der, double* der2) const
{
    if(isSpherical(pot) || r==0 || r==INFINITY) {
        // nothing to average - just evaluate the requested values at a single point
        coord::GradCyl grad;
        coord::HessCyl hess;
        pot.eval(coord::PosCyl(r, 0, 0), val, der? &grad : NULL, der2? &hess : NULL);
        if(der)  *der  = grad.dR;
        if(der2) *der2 = hess.dR2;
        return;
    }
    double result[3];
    if(val) {
        if(der) {
            if(der2) {
                sphericalAverage<3>(GetPotential<true,true,true>(pot), r, result);
                *val = result[0];
                *der = result[1];
                *der2= result[2];
            } else {
                sphericalAverage<2>(GetPotential<true,true,false>(pot), r, result);
                *val = result[0];
                *der = result[1];
            }
        } else {
            if(der2) {
                sphericalAverage<2>(GetPotential<true,false,true>(pot), r, result);
                *val = result[0];
                *der2= result[1];
            } else {
                sphericalAverage<1>(GetPotential<true,false,false>(pot), r, result);
                *val = result[0];
            }
        }
    } else {  // no val
        if(der) {
            if(der2) {
                sphericalAverage<2>(GetPotential<false,true,true>(pot), r, result);
                *der = result[0];
                *der2= result[1];
            } else {
                sphericalAverage<1>(GetPotential<false,true,false>(pot), r, result);
                *der = result[0];
            }
        } else {
            if(der2) {
                sphericalAverage<1>(GetPotential<false,false,true>(pot), r, result);
                *der2= result[0];
            }
        }
    }
}
    
Axisymmetrized<BaseDensity>::Axisymmetrized(const BaseDensity& _dens) :
    dens(_dens)
{
    if(isUnknown(dens.symmetry()))
        throw std::runtime_error("symmetry of input density is not provided");
}

Axisymmetrized<BasePotential>::Axisymmetrized(const BasePotential& _pot) :
    pot(_pot)
{
    if(isUnknown(pot.symmetry()))
        throw std::runtime_error("symmetry of input potential is not provided");
}

double Axisymmetrized<BaseDensity>::densityCyl(const coord::PosCyl &pos, double /*time*/) const
{
    if(isZRotSymmetric(dens) || (pos.R==0 && pos.z==0) || pos.R==INFINITY || math::abs(pos.z)==INFINITY)
        // nothing to average
        return dens.density(pos);
    double result;
    azimuthalAverage<1>(GetDensity(dens), pos.R, pos.z, &result);
    return result;
}

double Axisymmetrized<BasePotential>::densityCyl(const coord::PosCyl &pos, double /*time*/) const
{
    if(isZRotSymmetric(pot) || (pos.R==0 && pos.z==0) || pos.R==INFINITY || math::abs(pos.z)==INFINITY)
        // nothing to average
        return pot.density(pos);
    double result;
    azimuthalAverage<1>(GetDensity(pot), pos.R, pos.z, &result);
    return result;
}

void Axisymmetrized<BasePotential>::evalCyl(const coord::PosCyl &pos,
    double* value, coord::GradCyl* grad, coord::HessCyl* hess, double /*time*/) const
{
    if(isZRotSymmetric(pot) || (pos.R==0 && pos.z==0) || pos.R==INFINITY || math::abs(pos.z)==INFINITY) {
        // nothing to average
        pot.eval(pos, value, grad, hess);
        return;
    }
    double result[6];
    if(value) {
        if(grad) {
            if(hess) {
                azimuthalAverage<6>(GetPotential<true,true,true>(pot), pos.R, pos.z, result);
                *value     = result[0];
                grad->dR   = result[1];
                grad->dz   = result[2];
                grad->dphi = 0;
                hess->dR2  = result[3];
                hess->dz2  = result[4];
                hess->dRdz = result[5];
                hess->dRdphi = hess->dzdphi = hess->dphi2 = 0;
            } else {
                azimuthalAverage<3>(GetPotential<true,true,false>(pot), pos.R, pos.z, result);
                *value     = result[0];
                grad->dR   = result[1];
                grad->dz   = result[2];
                grad->dphi = 0;
            }
        } else {  // no grad
            if(hess) {
                azimuthalAverage<4>(GetPotential<true,false,true>(pot), pos.R, pos.z, result);
                *value     = result[0];
                hess->dR2  = result[1];
                hess->dz2  = result[2];
                hess->dRdz = result[3];
                hess->dRdphi = hess->dzdphi = hess->dphi2 = 0;
            } else {
                azimuthalAverage<1>(GetPotential<true,false,false>(pot), pos.R, pos.z, result);
                *value     = result[0];
            }
        }
    } else {  // no value
        if(grad) {
            if(hess) {
                azimuthalAverage<5>(GetPotential<false,true,true>(pot), pos.R, pos.z, result);
                grad->dR   = result[0];
                grad->dz   = result[1];
                grad->dphi = 0;
                hess->dR2  = result[2];
                hess->dz2  = result[3];
                hess->dRdz = result[4];
                hess->dRdphi = hess->dzdphi = hess->dphi2 = 0;
            } else {
                azimuthalAverage<2>(GetPotential<false,true,false>(pot), pos.R, pos.z, result);
                grad->dR   = result[0];
                grad->dz   = result[1];
                grad->dphi = 0;
            }
        } else {  // no grad
            if(hess) {
                azimuthalAverage<3>(GetPotential<false,false,true>(pot), pos.R, pos.z, result);
                hess->dR2  = result[0];
                hess->dz2  = result[1];
                hess->dRdz = result[2];
                hess->dRdphi = hess->dzdphi = hess->dphi2 = 0;
            }
        }
    }
}

}  // namespace potential
