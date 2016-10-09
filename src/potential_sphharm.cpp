#include "potential_sphharm.h"
#include "math_core.h"
#include "math_specfunc.h"
#include "math_sphharm.h"
#include "utils.h"
#include <cmath>
#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <gsl/gsl_sf_legendre.h>

namespace potential {

// internal definitions
namespace{

/// max number of basis-function expansion members (radial and angular).
const unsigned int MAX_NCOEFS_ANGULAR = 101;
const unsigned int MAX_NCOEFS_RADIAL  = 100;

/// minimum number of terms in sph.-harm. expansion used to compute coefficients
/// of a non-spherical density or potential model (it may be larger than
/// the requested number of output terms, to improve the accuracy of integration)    
const unsigned int LMIN_SPHHARM = 16;

/// max number of function evaluations in multidimensional integration
const unsigned int MAX_NUM_EVAL = 4096;

/// relative accuracy of potential computation (integration tolerance parameter)
const double EPSREL_POTENTIAL_INT = 1e-6;

/// relative accuracy in auxiliary root-finding routines
const double ACCURACY_ROOT = 1e-6;

/** helper class for integrating the density weighted with spherical harmonics over 3d volume;
    angular part is shared between BasisSetExp and SplineExp, 
    which further define additional functions for radial multiplication factor. 
    The integration is carried over  scaled r  and  cos(theta). */
class DensitySphHarmIntegrand: public math::IFunctionNdim {
public:
    DensitySphHarmIntegrand(const BaseDensity& _dens, int _l, int _m, 
        const math::IFunction& _radialMultFactor, double _rscale) :
        dens(_dens), l(_l), m(_m), radialMultFactor(_radialMultFactor), rscale(_rscale),
        mult( 0.5*sqrt((2*l+1.) * math::factorial(l-math::abs(m)) / math::factorial(l+math::abs(m))) )
    {};

    /// evaluate the m-th azimuthal harmonic of density at a point in (scaled r, cos(theta)) plane
    virtual void eval(const double vars[], double values[]) const 
    {   // input array is [scaled coordinate r, cos(theta)]
        const double scaled_r = vars[0], costheta = vars[1];
        if(scaled_r == 1) {
            values[0] = 0;  // we're at infinity
            return;
        }
        const double r = rscale * scaled_r / (1-scaled_r);
        const double R = r * sqrt(1-pow_2(costheta));
        const double z = r * costheta;
        const double Plm = gsl_sf_legendre_Plm(l, math::abs(m), costheta);
        double val = computeRho_m(dens, R, z, m) * Plm;
        if((dens.symmetry() & coord::ST_TRIAXIAL) == coord::ST_TRIAXIAL)   // symmetric w.r.t. change of sign in z
            val *= (l%2==0 ? 2 : 0);  // only even-l terms survive
        else
            val += computeRho_m(dens, R, -z, m) * Plm * (l%2==0 ? 1 : -1);
        values[0] = val * mult *
            r*r *                      // jacobian of transformation to spherical coordinates
            rscale/pow_2(1-scaled_r) * // un-scaling the radial coordinate
            radialMultFactor(r);       // additional radius-dependent factor
    }
    /// return the scaled radial variable (useful for determining the integration interval)
    double scaledr(double r) const {
        return r==INFINITY ? 1. : r/(r+rscale); }
    /// dimension of space to integrate over (R,theta)
    virtual unsigned int numVars() const { return 2; }
    /// integrate a single function at a time
    virtual unsigned int numValues() const { return 1; }
protected:
    const BaseDensity& dens;                  ///< input density to integrate
    const int l, m;                           ///< multipole indices
    const math::IFunction& radialMultFactor;  ///< additional radius-dependent multiplier
    const double rscale;                      ///< scaling factor for integration in radius
    const double mult;                        ///< constant multiplicative factor in Y_l^m
};

}  // internal namespace

//----------------------------------------------------------------------------//

// BasePotentialSphericalHarmonic -- parent class for all potentials 
// using angular expansion in spherical harmonics
void SphericalHarmonicCoefSet::setSymmetry(coord::SymmetryType sym)
{
    mysymmetry = sym;
    lmax = (mysymmetry & coord::ST_ROTATION)  ==coord::ST_ROTATION   ? 0 :     // if spherical model, use only l=0,m=0 term
        static_cast<int>(std::min<unsigned int>(Ncoefs_angular, MAX_NCOEFS_ANGULAR-1));
    lstep= (mysymmetry & coord::ST_REFLECTION)==coord::ST_REFLECTION ? 2 : 1;  // if reflection symmetry, use only even l
    mmax = (mysymmetry & coord::ST_ZROTATION) ==coord::ST_ZROTATION  ? 0 : 1;  // if axisymmetric model, use only m=0 terms, otherwise all terms up to l (1 is the multiplying factor)
    mmin = (mysymmetry & coord::ST_TRIAXIAL)  ==coord::ST_TRIAXIAL   ? 0 :-1;  // if triaxial symmetry, do not use sine terms which correspond to m<0
    mstep= (mysymmetry & coord::ST_TRIAXIAL)  ==coord::ST_TRIAXIAL   ? 2 : 1;  // if triaxial symmetry, use only even m
}

void BasePotentialSphericalHarmonic::evalSph(const coord::PosSph &pos,
    double* potential, coord::GradSph* grad, coord::HessSph* hess) const
{
    double result = 0;
    if(grad!=NULL)
        grad->dr = grad->dtheta = grad->dphi = 0;
    if(hess!=NULL)
        hess->dr2 = hess->dtheta2 = hess->dphi2 = hess->drdtheta = hess->dthetadphi = hess->drdphi = 0;
    // arrays where angular expansion coefficients will be accumulated by calling computeSHcoefs() for derived classes
    double coefsF[MAX_NCOEFS_ANGULAR*MAX_NCOEFS_ANGULAR];      // F(theta,phi)
    double coefsdFdr[MAX_NCOEFS_ANGULAR*MAX_NCOEFS_ANGULAR];   // dF(theta,phi)/dr
    double coefsd2Fdr2[MAX_NCOEFS_ANGULAR*MAX_NCOEFS_ANGULAR]; // d2F(theta,phi)/dr2
    computeSHCoefs(pos.r, coefsF, (grad!=NULL || hess!=NULL)? coefsdFdr : NULL, 
        hess!=NULL? coefsd2Fdr2 : NULL);  // implemented in derived classes
    double legendre_array[MAX_NCOEFS_ANGULAR];
    double legendre_deriv_array[MAX_NCOEFS_ANGULAR];
    double legendre_deriv2_array[MAX_NCOEFS_ANGULAR];
    double tau = cos(pos.theta) / (1 + sin(pos.theta));
    for(int m=0; m<=mmax*lmax; m+=mstep) {
        math::sphHarmArray(lmax, m, tau, legendre_array, 
            grad!=NULL||hess!=NULL ? legendre_deriv_array : NULL, 
            hess!=NULL ? legendre_deriv2_array : NULL);
        double cosmphi = (m==0 ? 1 : cos(m*pos.phi)*M_SQRT2) * (2*M_SQRTPI);   // factor \sqrt{4\pi} from the definition of spherical function Y_l^m absorbed into this term
        double sinmphi = (sin(m*pos.phi)*M_SQRT2) * (2*M_SQRTPI);
        int lmin = lstep==2 ? (m+1)/2*2 : m;   // if lstep is even and m is odd, start from next even number greater than m
        for(int l=lmin; l<=lmax; l+=lstep) {
            int indx=l*(l+1)+m;
            result += coefsF[indx] * legendre_array[l-m] * cosmphi;
            if(grad!=NULL) {
                grad->dr +=  coefsdFdr[indx] * legendre_array[l-m] * cosmphi;
                grad->dtheta += coefsF[indx] * legendre_deriv_array[l-m] * cosmphi;
                grad->dphi   += coefsF[indx] * legendre_array[l-m] * (-m)*sinmphi;
            }
            if(hess!=NULL) {
                hess->dr2 +=  coefsd2Fdr2[indx] * legendre_array[l-m] * cosmphi;
                hess->drdtheta+=coefsdFdr[indx] * legendre_deriv_array[l-m] * cosmphi;
                hess->drdphi  +=coefsdFdr[indx] * legendre_array[l-m] * (-m)*sinmphi;
                hess->dtheta2   += coefsF[indx] * legendre_deriv2_array[l-m] * cosmphi;
                hess->dthetadphi+= coefsF[indx] * legendre_deriv_array[l-m] * (-m)*sinmphi;
                hess->dphi2     += coefsF[indx] * legendre_array[l-m] * cosmphi * -m*m;
            }
            if(mmin<0 && m>0) {
                indx=l*(l+1)-m;
                result += coefsF[indx] * legendre_array[l-m] * sinmphi;
                if(grad!=NULL) {
                    grad->dr +=  coefsdFdr[indx] * legendre_array[l-m] * sinmphi;
                    grad->dtheta += coefsF[indx] * legendre_deriv_array[l-m] * sinmphi;
                    grad->dphi   += coefsF[indx] * legendre_array[l-m] * m*cosmphi;
                }
                if(hess!=NULL) {
                    hess->dr2 +=  coefsd2Fdr2[indx] * legendre_array[l-m] * sinmphi;
                    hess->drdtheta+=coefsdFdr[indx] * legendre_deriv_array[l-m] * sinmphi;
                    hess->drdphi  +=coefsdFdr[indx] * legendre_array[l-m] * m*cosmphi;
                    hess->dtheta2   += coefsF[indx] * legendre_deriv2_array[l-m] * sinmphi;
                    hess->dthetadphi+= coefsF[indx] * legendre_deriv_array[l-m] * m*cosmphi;
                    hess->dphi2     += coefsF[indx] * legendre_array[l-m] * sinmphi * -m*m;
                }
            }
        }
    }
    if(potential!=NULL)
        *potential = result;
}

//----------------------------------------------------------------------------//
// Basis-set expansion for arbitrary potential (using Zhao(1995) basis set)

BasisSetExp::BasisSetExp(
    double _Alpha, unsigned int _Ncoefs_radial, unsigned int _Ncoefs_angular, 
                         const particles::ParticleArray<coord::PosSph> &points, coord::SymmetryType _sym):
    BasePotentialSphericalHarmonic(_Ncoefs_angular), 
    Ncoefs_radial(std::min<unsigned int>(MAX_NCOEFS_RADIAL-1, _Ncoefs_radial)),
    Alpha(_Alpha)
{
    setSymmetry(_sym);
    if(points.size()==0)
        throw std::invalid_argument("BasisSetExp: input particle set is empty");
    prepareCoefsDiscrete(points);
    checkSymmetry();
}

BasisSetExp::BasisSetExp(double _Alpha, const std::vector< std::vector<double> > &coefs):
    BasePotentialSphericalHarmonic(coefs.size()>0 ? static_cast<unsigned int>(sqrt(coefs[0].size()*1.0)-1) : 0), 
    Ncoefs_radial(std::min<unsigned int>(MAX_NCOEFS_RADIAL-1, static_cast<unsigned int>(coefs.size()-1))),
    Alpha(_Alpha)  // here Alpha!=0 - no autodetect
{
    if(_Alpha<0.5) 
        throw std::invalid_argument("BasisSetExp: invalid parameter Alpha");
    for(unsigned int n=0; n<coefs.size(); n++)
        if(coefs[n].size()!=pow_2(Ncoefs_angular+1))
            throw std::invalid_argument("BasisSetExp: incorrect size of coefficients array");
    SHcoefs = coefs;
    checkSymmetry();
}

BasisSetExp::BasisSetExp(double _Alpha, unsigned int _Ncoefs_radial, unsigned int _Ncoefs_angular, 
    const BaseDensity& srcdensity):    // init potential from analytic mass model
    BasePotentialSphericalHarmonic(_Ncoefs_angular), 
    Ncoefs_radial(std::min<unsigned int>(MAX_NCOEFS_RADIAL-1, _Ncoefs_radial)),
    Alpha(_Alpha)
{
    setSymmetry(srcdensity.symmetry());
    prepareCoefsAnalytic(srcdensity);
    checkSymmetry();
}

void BasisSetExp::checkSymmetry()
{
    coord::SymmetryType sym=coord::ST_SPHERICAL;  // too optimistic:))
    const double MINCOEF = 1e-8 * fabs(SHcoefs[0][0]);
    for(unsigned int n=0; n<=Ncoefs_radial; n++) {
        for(int l=0; l<=(int)Ncoefs_angular; l++)
            for(int m=-l; m<=l; m++)
                if(fabs(SHcoefs[n][l*(l+1)+m])>MINCOEF) 
                {   // nonzero coef.: check if that breaks any symmetry
                    if(l%2==1)  sym = (coord::SymmetryType)(sym & ~coord::ST_REFLECTION);
                    if(m<0 || m%2==1)  sym = (coord::SymmetryType)(sym & ~coord::ST_TRIAXIAL);
                    if(m!=0) sym = (coord::SymmetryType)(sym & ~coord::ST_ZROTATION);
                    if(l>0) sym = (coord::SymmetryType)(sym & ~coord::ST_ROTATION);
                }
    }
    // now set all coefs excluded by the inferred symmetry  to zero
    for(size_t n=0; n<=Ncoefs_radial; n++) {
        for(int l=0; l<=(int)Ncoefs_angular; l++)
            for(int m=-l; m<=l; m++)
                if( (l>0 && (sym & coord::ST_ROTATION)) ||
                   (m!=0 && (sym & coord::ST_ZROTATION)) ||
                   ((m<0 || m%2==1) && (sym & coord::ST_TRIAXIAL)) ||
                   (l%2==1 && (sym & coord::ST_REFLECTION)) ) 
                        SHcoefs[n][l*(l+1)+m] = 0;
    }
    setSymmetry(sym);
}

/// radius-dependent multiplication factor for density integration in BasisSetExp potential
class BasisSetExpRadialMult: public math::IFunctionNoDeriv {
public:
    BasisSetExpRadialMult(int _n, int _l, double _alpha) :
        n(_n), l(_l), alpha(_alpha), w((2*l+1)*alpha+0.5) {};
    virtual double value(double r) const {
        const double r1alpha = pow(r, 1./alpha);
        const double xi = (r1alpha-1)/(r1alpha+1);
        return math::gegenbauer(n, w, xi) * math::powInt(r, l) * pow(1+r1alpha, -(2*l+1)*alpha) * 4*M_PI;
    }
private:
    const int n, l;
    const double alpha, w;
};

void BasisSetExp::prepareCoefsAnalytic(const BaseDensity& srcdensity)
{
    if(Alpha<0.5)
        Alpha = 1.;
    SHcoefs.resize(Ncoefs_radial+1);
    for(size_t n=0; n<=Ncoefs_radial; n++)
        SHcoefs[n].assign(pow_2(Ncoefs_angular+1), 0);
    const double rscale = 1.0;
    for(unsigned int n=0; n<=Ncoefs_radial; n++)
        for(int l=0; l<=lmax; l+=lstep) {
            double w=(2*l+1)*Alpha+0.5;
            double Knl = (4*pow_2(n+w)-1)/8/pow_2(Alpha);
            double Inl = Knl * 4*M_PI*Alpha * 
                exp( math::lngamma(n+2*w) - 2*math::lngamma(w) - math::lnfactorial(n) - 4*w*log(2.0)) / (n+w);
            for(int m=l*mmin; m<=l*mmax; m+=mstep) {
                BasisSetExpRadialMult rmult(n, l, Alpha);
                DensitySphHarmIntegrand fnc(srcdensity, l, m, rmult, rscale);
                double xlower[2] = {fnc.scaledr(0), 0};
                double xupper[2] = {fnc.scaledr(INFINITY), 1};
                double result, error;
                int numEval;
                math::integrateNdim(fnc, 
                    xlower, xupper, EPSREL_POTENTIAL_INT, MAX_NUM_EVAL, &result, &error, &numEval);
                SHcoefs[n][l*(l+1)+m] = result * (m==0 ? 1 : M_SQRT2) / Inl;
            }
        }
}

void BasisSetExp::prepareCoefsDiscrete(const particles::ParticleArray<coord::PosSph> &points)
{
    SHcoefs.resize(Ncoefs_radial+1);
    for(unsigned int n=0; n<=Ncoefs_radial; n++)
        SHcoefs[n].assign(pow_2(1+Ncoefs_angular), 0);
    unsigned int npoints=points.size();
    if(Alpha<0.5)
        Alpha=1.;
    double legendre_array[MAX_NCOEFS_ANGULAR][MAX_NCOEFS_ANGULAR-1];
    double gegenpoly_array[MAX_NCOEFS_RADIAL];
    double Inl[MAX_NCOEFS_RADIAL][MAX_NCOEFS_ANGULAR];
    for(int l=0; l<=lmax; l+=lstep)
    {   // pre-compute coefficients
        double w=(2*l+1)*Alpha+0.5;
        for(unsigned int n=0; n<=Ncoefs_radial; n++)
            Inl[n][l] = 4*M_PI*Alpha * 
              exp( math::lngamma(n+2*w) - 2*math::lngamma(w) - math::lnfactorial(n) - 4*w*log(2.0)) /
              (n+w) * (4*(n+w)*(n+w)-1)/(8*Alpha*Alpha);
    }
    for(unsigned int i=0; i<npoints; i++) {
        const coord::PosSph& point = points.point(i);
        double massi = points.mass(i);
        double ralpha=pow(point.r, 1/Alpha);
        double xi=(ralpha-1)/(ralpha+1);
        double tau = cos(point.theta) / (1 + sin(point.theta));
        for(int m=0; m<=lmax; m+=mstep)
            math::sphHarmArray(lmax, m, tau, legendre_array[m]);
        for(int l=0; l<=lmax; l+=lstep) {
            double w=(2*l+1)*Alpha+0.5;
            double phil=pow(point.r, l) * pow(1+ralpha, -(2*l+1)*Alpha);
            math::gegenbauerArray(Ncoefs_radial, w, xi, gegenpoly_array);
            for(size_t n=0; n<=Ncoefs_radial; n++) {
                double mult= massi * gegenpoly_array[n] * phil * (2*M_SQRTPI) / Inl[n][l];
                for(int m=0; m<=l*mmax; m+=mstep)
                    SHcoefs[n][l*(l+1)+m] += mult * legendre_array[m][l-m] * cos(m*point.phi) * (m==0 ? 1 : M_SQRT2);
                if(mmin)
                    for(int m=mmin*l; m<0; m+=mstep)
                        SHcoefs[n][l*(l+1)+m] += mult * legendre_array[-m][l+m] * sin(-m*point.phi) * M_SQRT2;
            }
        }
    }
}

double BasisSetExp::enclosedMass(const double r) const
{
    if(r<=0) return 0;
    double ralpha=pow(r, 1/Alpha);
    double xi=(ralpha-1)/(ralpha+1);
    double gegenpoly_array[MAX_NCOEFS_RADIAL];
    math::gegenbauerArray(Ncoefs_radial, Alpha+0.5, xi, gegenpoly_array);
    double multr = pow(1+ralpha, -Alpha);
    double multdr= -ralpha/((ralpha+1)*r);
    double result=0;
    for(int n=0; n<=static_cast<int>(Ncoefs_radial); n++) {
        double dGdr=(n>0 ? (-n*xi*gegenpoly_array[n] + (n+2*Alpha)*gegenpoly_array[n-1])/(2*Alpha*r) : 0);
        result += SHcoefs[n][0] * multr * (multdr * gegenpoly_array[n] + dGdr);
    }
    return -result * r*r;   // d Phi(r)/d r = G M(r) / r^2
}

void BasisSetExp::computeSHCoefs(const double r, double coefsF[], double coefsdFdr[], double coefsd2Fdr2[]) const
{
    double ralpha=pow(r, 1/Alpha);
    double xi=(ralpha-1)/(ralpha+1);
    double gegenpoly_array[MAX_NCOEFS_RADIAL];
    if(coefsF)      for(size_t k=0; k<pow_2(Ncoefs_angular+1); k++) coefsF     [k] = 0;
    if(coefsdFdr)   for(size_t k=0; k<pow_2(Ncoefs_angular+1); k++) coefsdFdr  [k] = 0;
    if(coefsd2Fdr2) for(size_t k=0; k<pow_2(Ncoefs_angular+1); k++) coefsd2Fdr2[k] = 0;
    for(int l=0; l<=lmax; l+=lstep) {
        double w=(2*l+1)*Alpha+0.5;
        math::gegenbauerArray(Ncoefs_radial, w, xi, gegenpoly_array);
        double multr = -pow(r, l) * pow(1+ralpha, -(2*l+1)*Alpha);
        double multdr= (l-(l+1)*ralpha)/((ralpha+1)*r);
        for(unsigned int n=0; n<=Ncoefs_radial; n++) {
            double multdFdr = 0, multd2Fdr2 = 0;
            if(coefsdFdr!=NULL) {
                double dGdr = (n>0 ? (-xi*n*gegenpoly_array[n] + (n+2*w-1)*gegenpoly_array[n-1])/(2*Alpha*r) : 0);
                multdFdr = multdr * gegenpoly_array[n] + dGdr;
                if(coefsd2Fdr2!=NULL)
                    multd2Fdr2 = ( (l+1)*(l+2)*pow_2(ralpha) + 
                                   ( (1-2*l*(l+1)) - (2*n+1)*(2*l+1)/Alpha - n*(n+1)/pow_2(Alpha))*ralpha + 
                                   l*(l-1) 
                                 ) / pow_2( (1+ralpha)*r ) * gegenpoly_array[n] - dGdr*2/r;
            }
            for(int m=l*mmin; m<=l*mmax; m+=mstep) {
                int indx=l*(l+1)+m;
                double coef = SHcoefs[n][indx] * multr;
                if(coefsF)      coefsF     [indx] += coef * gegenpoly_array[n];
                if(coefsdFdr)   coefsdFdr  [indx] += coef * multdFdr;
                if(coefsd2Fdr2) coefsd2Fdr2[indx] += coef * multd2Fdr2;
            }
        }
    }
}

//----------------------------------------------------------------------------//
// Spherical-harmonic expansion of arbitrary potential, radial part is spline interpolated on a grid

// init coefs from point mass set
SplineExp::SplineExp(unsigned int _Ncoefs_radial, unsigned int _Ncoefs_angular, 
                     const particles::ParticleArray<coord::PosSph> &points, coord::SymmetryType _sym, 
    double smoothfactor, double Rmin, double Rmax):
    BasePotentialSphericalHarmonic(_Ncoefs_angular),
    Ncoefs_radial(std::max<size_t>(5,_Ncoefs_radial))
{
    setSymmetry(_sym);
    prepareCoefsDiscrete(points, smoothfactor, Rmin, Rmax);
}

// init from existing coefs
SplineExp::SplineExp(
    const std::vector<double> &_gridradii, const std::vector< std::vector<double> > &_coefs):
    BasePotentialSphericalHarmonic(_coefs.size()>0 ? static_cast<size_t>(sqrt(_coefs[0].size()*1.0)-1) : 0), 
    Ncoefs_radial(std::min<size_t>(MAX_NCOEFS_RADIAL-1, _coefs.size()-1))
{
    for(unsigned int n=0; n<_coefs.size(); n++)
        if(_coefs[n].size()!=pow_2(Ncoefs_angular+1))
            throw std::invalid_argument("SplineExp: incorrect size of coefficients array");
    initSpline(_gridradii, _coefs);
}

// init potential from analytic mass model
SplineExp::SplineExp(unsigned int _Ncoefs_radial, unsigned int _Ncoefs_angular, 
    const BaseDensity& srcdensity, double Rmin, double Rmax):
    BasePotentialSphericalHarmonic(_Ncoefs_angular), 
    Ncoefs_radial(std::max<unsigned int>(5,_Ncoefs_radial))
{
    setSymmetry(srcdensity.symmetry());
    prepareCoefsAnalytic(srcdensity, Rmin, Rmax);
}

/// radius-dependent multiplication factor for density integration in SplineExp potential
class SplineExpRadialMult: public math::IFunctionNoDeriv {
public:
    SplineExpRadialMult(int _n) : n(_n) {};
    virtual double value(double r) const {
        return math::powInt(r, n);
    }
private:
    const int n;
};

void SplineExp::prepareCoefsAnalytic(const BaseDensity& srcdensity, double Rmin, double Rmax)
{
    // find inner/outermost radius if they were not provided
    if(Rmin<0 || Rmax<0 || (Rmax>0 && Rmax<=Rmin*Ncoefs_radial))
        throw std::invalid_argument("SplineExp: invalid choice of min/max grid radii");
    double totalmass = srcdensity.totalMass();
    if(!isFinite(totalmass))
        throw std::invalid_argument("SplineExp: source density model has infinite mass");
    if(Rmax==0) {
        // how far should be the outer node (leave out this fraction of mass)
        double epsout = 0.1/sqrt(pow_2(Ncoefs_radial)+0.01*pow(Ncoefs_radial*1.0,4.0));
        Rmax = getRadiusByMass(srcdensity, totalmass*(1-epsout));
    }
    if(Rmin==0) {
        // how close can we get to zero, in terms of innermost grid node
        double epsin = 5.0/pow(Ncoefs_radial*1.0,3.0);
        Rmin  = getRadiusByMass(srcdensity, totalmass*epsin*0.1);
    }
    utils::msg(utils::VL_DEBUG, "SplineExp",
        "Grid in r=["+utils::toString(Rmin)+":"+utils::toString(Rmax)+"]");
    std::vector<double> radii = //math::createNonuniformGrid(Ncoefs_radial+1, Rmin, Rmax, true);
    math::createExpGrid(Ncoefs_radial, Rmin, Rmax);
    radii.insert(radii.begin(),0);
    std::vector< std::vector<double> > coefsArray(Ncoefs_radial+1);  // SHE coefficients to pass to initspline routine
    for(unsigned int i=0; i<=Ncoefs_radial; i++)
        coefsArray[i].assign(pow_2(1+Ncoefs_angular), 0);
    const double rscale = getRadiusByMass(srcdensity, 0.5*totalmass);  // scaling factor for integration in radius
    std::vector<double> coefsInner, coefsOuter;
    const double SPLINE_MIN_RADIUS = 1e-10;
    radii.front() = SPLINE_MIN_RADIUS*radii[1];  // to prevent log divergence for gamma=2 potentials
    for(int l=0; l<=lmax; l+=lstep) {
        for(int m=l*mmin; m<=l*mmax; m+=mstep) {
            // first precompute inner and outer density integrals at each radial grid point, summing contributions from each interval of radial grid
            coefsInner.assign(Ncoefs_radial+1, 0);
            coefsOuter.assign(Ncoefs_radial+1, 0);
            // loop over inner intervals
            double result, error;
            for(size_t c=0; c<Ncoefs_radial; c++) {
                SplineExpRadialMult rmult(l);
                DensitySphHarmIntegrand fnc(srcdensity, l, m, rmult, rscale);
                double xlower[2] = {fnc.scaledr(radii[c]), 0};
                double xupper[2] = {fnc.scaledr(radii[c+1]), 1};
                int numEval;
                math::integrateNdim(fnc, 
                    xlower, xupper, EPSREL_POTENTIAL_INT, MAX_NUM_EVAL, &result, &error, &numEval);
                coefsInner[c+1] = result + coefsInner[c];
            }
            // loop over outer intervals, starting from infinity backwards
            for(size_t c=Ncoefs_radial+1; c>(l==0 ? 0 : 1u); c--) {
                SplineExpRadialMult rmult(-l-1);
                DensitySphHarmIntegrand fnc(srcdensity, l, m, rmult, rscale);
                double xlower[2] = {fnc.scaledr(radii[c-1]), 0};
                double xupper[2] = {fnc.scaledr(c>Ncoefs_radial ? INFINITY : radii[c]), 1};
                int numEval;
                math::integrateNdim(fnc,
                    xlower, xupper, EPSREL_POTENTIAL_INT, MAX_NUM_EVAL, &result, &error, &numEval);
                coefsOuter[c-1] = result + (c>Ncoefs_radial?0:coefsOuter[c]);
            }
            // now compute the coefs of potential expansion themselves
            for(size_t c=0; c<=Ncoefs_radial; c++) {
                coefsArray[c][l*(l+1) + m] = ((c>0 ? coefsInner[c]*pow(radii[c], -l-1.0) : 0) + coefsOuter[c]*pow(radii[c], l*1.0)) *
                    -4*M_PI/(2*l+1) * (m==0 ? 1 : M_SQRT2);
            }
        }
    }
    radii.front()=0;
    initSpline(radii, coefsArray);
}

/// \cond INTERNAL_DOCS
inline bool compareParticleSph(
    const particles::ParticleArray<coord::PosSph>::ElemType& val1, 
    const particles::ParticleArray<coord::PosSph>::ElemType& val2) {
    return val1.first.r < val2.first.r;  }
/// \endcond

void SplineExp::computeCoefsFromPoints(const particles::ParticleArray<coord::PosSph> &srcPoints, 
    std::vector<double>& outputRadii, std::vector< std::vector<double> >& outputCoefs)
{
    double legendre_array[MAX_NCOEFS_ANGULAR][MAX_NCOEFS_ANGULAR-1];
    size_t npoints = srcPoints.size();
    for(size_t i=0; i<npoints; i++) {
        if(srcPoints.point(i).r<=0)
            throw std::invalid_argument("SplineExp: particles at r=0 are not allowed");
        if(srcPoints.mass(i)<0) 
            throw std::invalid_argument("SplineExp: input particles have negative mass");
    }

    // make a copy of input array to allow it to be sorted
    particles::ParticleArray<coord::PosSph> points(srcPoints);
    std::sort(points.data.begin(), points.data.end(), compareParticleSph);

    // having sorted particles in radius, may now initialize coefs
    outputRadii.resize(npoints);
    for(size_t i=0; i<npoints; i++)
        outputRadii[i] = points.point(i).r;

    // we need two intermediate arrays of inner and outer coefficients for each particle,
    // and in the end we output one array of 'final' coefficients for each particle.
    // We can use a trick to save memory, by allocating only one temporary array, 
    // and using the output array as the second intermediate one.
    std::vector< std::vector<double> > coefsInner(pow_2(Ncoefs_angular+1));  // this is the 1st temp array
    outputCoefs.resize(pow_2(Ncoefs_angular+1));  // this will be the final array
    // instead of allocating 2nd temp array, we use a reference to the already existing array
    std::vector< std::vector<double> >& coefsOuter = outputCoefs;
    // reserve memory only for those coefficients that are actually needed
    for(int l=0; l<=lmax; l+=lstep)
        for(int m=l*mmin; m<=l*mmax; m+=mstep)
            coefsOuter[l*(l+1)+m].assign(npoints, 0);  // reserve memory only for those coefs that will be used
    for(int l=0; l<=lmax; l+=lstep)
        for(int m=l*mmin; m<=l*mmax; m+=mstep)
            coefsInner[l*(l+1)+m].assign(npoints, 0);  // yes do it separately from the above, to allow contiguous block of memory to be freed after deleting CoefsInner

    // initialize SH expansion coefs at each point's location
    for(size_t i=0; i<npoints; i++) {
        double tau = cos(points.point(i).theta) / (1 + sin(points.point(i).theta));
        for(int m=0; m<=lmax; m+=mstep)
            math::sphHarmArray(lmax, m, tau, legendre_array[m]);
        for(int l=0; l<=lmax; l+=lstep)
            for(int m=l*mmin; m<=l*mmax; m+=mstep) {
                int coefind = l*(l+1)+m;
                int absm = math::abs(m);  // negative m correspond to sine, positive - to cosine
                double mult = -sqrt(4*M_PI)/(2*l+1) * (m==0 ? 1 : M_SQRT2) * points.mass(i) *
                    legendre_array[absm][l-absm] * 
                    (m>=0 ? cos(m*points.point(i).phi) : sin(-m*points.point(i).phi));
                coefsOuter[coefind][i] = mult * pow(points.point(i).r, -(1+l));
                coefsInner[coefind][i] = mult * pow(points.point(i).r, l);
            }
    }

    // sum inner coefs interior and outer coefs exterior to each point's location
    for(int l=0; l<=lmax; l+=lstep)
        for(int m=l*mmin; m<=l*mmax; m+=mstep)
        {
            int coefind=l*(l+1)+m;
            for(size_t i=1; i<npoints; i++)
                coefsInner[coefind][i] += coefsInner[coefind][i-1];
            for(size_t i=npoints-1; i>0; i--)
                coefsOuter[coefind][i-1] += coefsOuter[coefind][i];
        }

    // initialize potential expansion coefs by multiplying 
    // inner and outer coefs by r^(-1-l) and r^l, correspondingly
    for(size_t i=0; i<npoints; i++) {
        for(int l=0; l<=lmax; l+=lstep)
            for(int m=l*mmin; m<=l*mmax; m+=mstep) {
                int coefind = l*(l+1)+m;
                // note that here we are destroying the values of CoefsOuter, because this array
                // is aliased with outcoefs; but we do it from inside out, and for each i-th point
                // the coefficients from i+1 till the end of array are still valid.
                outputCoefs[coefind][i] = 
                    (i>0 ? coefsInner[coefind][i-1] * pow(points.point(i).r, -(1+l)) : 0) + 
                    coefsOuter[coefind][i] * pow(points.point(i).r, l);
            }
    }
    // local variable coefsInner will be automatically freed, but outputCoefs will remain
}

/** obtain the value of scaling radius for non-spherical harmonic coefficients `ascale`
    from the radial dependence of the l=0 coefficient, by finding the radius at which
    the value of this coefficient equals half of its value at r=0 */
double get_ascale(const std::vector<double>& radii, const std::vector<std::vector<double> >& coefsArray)
{
    assert(radii.size() == coefsArray.size());
    double targetVal = fabs(coefsArray[0][0])*0.5;
    double targetRad = NAN;
    for(size_t i=1; i<radii.size() && targetRad!=targetRad; i++) 
        if(fabs(coefsArray[i][0]) < targetVal && fabs(coefsArray[i-1][0]) >= targetVal) {
            // linearly interpolate
            targetRad = radii[i-1] + (radii[i]-radii[i-1]) *
                (targetVal-fabs(coefsArray[i-1][0])) / (fabs(coefsArray[i][0])-fabs(coefsArray[i-1][0]));
        }
    if(targetRad!=targetRad)  // shouldn't occur, but if it does, return some sensible value
        targetRad = radii[radii.size()/2];
    return targetRad;
}

void SplineExp::prepareCoefsDiscrete(const particles::ParticleArray<coord::PosSph> &points, 
    double smoothfactor, double innerBinRadius, double outerBinRadius)
{
    if(points.size() <= Ncoefs_radial*10)
        throw std::invalid_argument("SplineExp: number of particles is too small");
    if(innerBinRadius<0 || outerBinRadius<0 ||
        (outerBinRadius>0 && outerBinRadius<=innerBinRadius*Ncoefs_radial))
        throw std::invalid_argument("SplineExp: invalid choice of min/max grid radii");
    // radii of each point in ascending order
    std::vector<double> pointRadii;
    // note that array indexing is swapped w.r.t. coefsArray (i.e. pointCoefs[coefIndex][pointIndex])
    // to save memory by not initializing unnecessary coefs
    std::vector< std::vector<double> > pointCoefs;
    computeCoefsFromPoints(points, pointRadii, pointCoefs);

    // choose the radial grid parameters if they were not provided:
    // innermost cell contains minBinMass and outermost radial node should encompass cutoffMass.
    // spline definition region extends up to outerRadius which is 
    // ~several times larger than outermost radial node, 
    // however coefficients at that radius are not used in the potential computation later
    const size_t npoints = pointRadii.size();
    const size_t minBinPoints = 10;
    size_t npointsMargin = static_cast<size_t>(sqrt(npoints*0.1));
    // number of points within 1st grid radius
    size_t npointsInnerGrid = std::max<size_t>(minBinPoints, npointsMargin);
    // number of points within outermost grid radius
    size_t npointsOuterGrid = npoints - std::max<size_t>(minBinPoints, npointsMargin);
    if(innerBinRadius < pointRadii[0])
        innerBinRadius = pointRadii[npointsInnerGrid];
    if(outerBinRadius == 0 || outerBinRadius > pointRadii.back())
        outerBinRadius = pointRadii[npointsOuterGrid];
    std::vector<double> radii =     // radii of grid nodes to pass to initspline routine
        math::createNonuniformGrid(Ncoefs_radial+1, innerBinRadius, outerBinRadius, true);
    utils::msg(utils::VL_DEBUG, "SplineExp",
        "Grid in r=["+utils::toString(innerBinRadius)+":"+utils::toString(outerBinRadius)+"]");

    // find index of the inner- and outermost points which are used in b-spline fitting
    size_t npointsInnerSpline = 0;
    while(pointRadii[npointsInnerSpline]<radii[1])
        npointsInnerSpline++;
    npointsInnerSpline = std::min<size_t>(npointsInnerSpline-2,
        std::max<size_t>(minBinPoints, npointsInnerSpline/2));
    // index of last point used in b-spline fitting 
    // (it is beyond outer grid radius, since b-spline definition region
    // is larger than the range of radii for which the spline approximation
    // will eventually be constructed) -
    // roughly equally logarithmically spaced from the last two points
    double outerRadiusSpline = pow_2(radii[Ncoefs_radial])/radii[Ncoefs_radial-1];
    size_t npointsOuterSpline = npoints-1;
    while(pointRadii[npointsOuterSpline]>outerRadiusSpline)
        npointsOuterSpline--;
    //!!!FIXME: what happens if the outermost pointRadius is < outerBinRadius ?

    // outer and inner points are ignored
    size_t numPointsUsed = npointsOuterSpline-npointsInnerSpline;
    // including zero and outermost point; only interior nodes are actually used
    // for computing best-fit coefs (except l=0 coef, for which r=0 is also used)
    size_t numBSplineKnots = Ncoefs_radial+2;
    // transformed x- and y- values of original data points
    // which will be approximated by a spline regression
    std::vector<double> scaledPointRadii(numPointsUsed), scaledPointCoefs(numPointsUsed);
    // transformed x- and y- values of regression spline knots
    std::vector<double> scaledKnotRadii(numBSplineKnots);

    // SHE coefficients to pass to initspline routine
    std::vector< std::vector<double> > coefsArray(Ncoefs_radial+1);
    for(size_t i=0; i<=Ncoefs_radial; i++)
        coefsArray[i].assign(pow_2(Ncoefs_angular+1), 0);

    // open block so that temp variable "appr" will be destroyed upon closing this block
    {
        // first construct spline for zero-order term (radial dependence)
        potcenter=pointCoefs[0][0];  // value of potential at origin (times 1/2\sqrt{\pi} ?)
        for(size_t p=1; p<=Ncoefs_radial; p++)
            scaledKnotRadii[p] = log(radii[p]);
        scaledKnotRadii[Ncoefs_radial+1] = log(outerRadiusSpline);
        scaledKnotRadii[0] = log(pointRadii[npointsInnerSpline]);
        for(size_t i=0; i<numPointsUsed; i++)
        {
            scaledPointRadii[i] = log(pointRadii[i+npointsInnerSpline]);
            scaledPointCoefs[i] = log(1/(1/potcenter - 1/pointCoefs[0][i+npointsInnerSpline]));
        }
        math::CubicSpline spl(scaledKnotRadii,
            math::SplineApprox(scaledKnotRadii, scaledPointRadii).fit(scaledPointCoefs, 0));
        // now store fitted values in coefsArray to pass to initspline routine
        coefsArray[0][0] = potcenter;
        for(size_t c=1; c<=Ncoefs_radial; c++)
            coefsArray[c][0] = -1./(exp(-spl(scaledKnotRadii[c]))-1/potcenter);
    }
    if(lmax>0) {  // construct splines for all l>0 spherical-harmonic terms separately
        // first estimate the asymptotic power-law slope of coefficients at r=0 and r=infinity
        double gammaInner = 2-log((coefsArray[1][0]-potcenter)/(coefsArray[2][0]-potcenter))/log(radii[1]/radii[2]);
        if(gammaInner<0) gammaInner=0; 
        if(gammaInner>2) gammaInner=2;
        // this was the estimate of density slope. Now we need to convert it to the estimate of power-law slope of l>0 coefs
        gammaInner = 2.0-gammaInner;   // the same recipe as used later in initSpline
        double gammaOuter = -1.0;      // don't freak out, assume default value
        // init x-coordinates from scaling transformation
        ascale = get_ascale(radii, coefsArray);  // this uses only the l=0 term
        for(size_t p=0; p<=Ncoefs_radial; p++)
            scaledKnotRadii[p] = log(ascale+radii[p]);
        scaledKnotRadii[Ncoefs_radial+1] = log(ascale+outerRadiusSpline);
        for(size_t i=0; i<numPointsUsed; i++)
            scaledPointRadii[i] = log(ascale+pointRadii[i+npointsInnerSpline]);
        math::SplineApprox appr(scaledKnotRadii, scaledPointRadii);
        // loop over l,m
        for(int l=lstep; l<=lmax; l+=lstep)
        {
            for(int m=l*mmin; m<=l*mmax; m+=mstep)
            {
                int coefind=l*(l+1) + m;
                // init matrix of values to fit
                for(size_t i=0; i<numPointsUsed; i++)
                    scaledPointCoefs[i] = pointCoefs[coefind][i+npointsInnerSpline]/pointCoefs[0][i+npointsInnerSpline];
                double edf=0;  // equivalent number of free parameters in the fit; if it is ~2, fit is oversmoothed to death (i.e. to a linear regression, which means we should ignore it)

                math::CubicSpline spl(scaledKnotRadii,
                    appr.fitOversmooth(scaledPointCoefs, smoothfactor, NULL, &edf));
                // now store fitted values in coefsArray to pass to initspline routine
                coefsArray[0][coefind] = 0;  // unused
                for(size_t c=1; c<=Ncoefs_radial; c++)
                    coefsArray[c][coefind] = spl(scaledKnotRadii[c]) * coefsArray[c][0];  // scale back (multiply by l=0,m=0 coefficient)
                // correction to avoid fluctuation at first and last grid radius
                if( coefsArray[1][coefind] * coefsArray[2][coefind] < 0 || coefsArray[1][coefind]/coefsArray[2][coefind] > pow(radii[1]/radii[2], gammaInner))
                    coefsArray[1][coefind] = coefsArray[2][coefind] * pow(radii[1]/radii[2], gammaInner);   // make the smooth curve drop to zero at least as fast as gammaInner'th power of radius
                if( coefsArray[Ncoefs_radial][coefind] * coefsArray[Ncoefs_radial-1][coefind] < 0 || 
                    coefsArray[Ncoefs_radial][coefind] / coefsArray[Ncoefs_radial-1][coefind] > pow(radii[Ncoefs_radial]/radii[Ncoefs_radial-1], gammaOuter))
                    coefsArray[Ncoefs_radial][coefind] = coefsArray[Ncoefs_radial-1][coefind] * pow(radii[Ncoefs_radial]/radii[Ncoefs_radial-1], gammaOuter);
                if(edf<3.0)   // in case of error or an oversmoothed fit fallback to zero values
                    for(size_t c=0; c<=Ncoefs_radial; c++)
                        coefsArray[c][coefind] = 0;
            }
        }
    }
    initSpline(radii, coefsArray);
}

void SplineExp::checkSymmetry(const std::vector< std::vector<double> > &coefsArray)
{
    coord::SymmetryType sym=coord::ST_SPHERICAL;  // too optimistic:))
    // if ALL coefs of a certain subset of indices are below this value, assume some symmetry
    const double MINCOEF = 1e-8 * fabs(coefsArray[0][0]);
    for(size_t n=0; n<=Ncoefs_radial; n++)
    {
        for(int l=0; l<=(int)Ncoefs_angular; l++)
            for(int m=-l; m<=l; m++)
                if(fabs(coefsArray[n][l*(l+1)+m])>MINCOEF) 
                {   // nonzero coef.: check if that breaks any symmetry
                    if(l%2==1)  sym = (coord::SymmetryType)(sym & ~coord::ST_REFLECTION);
                    if(m<0 || m%2==1)  sym = (coord::SymmetryType)(sym & ~coord::ST_TRIAXIAL);
                    if(m!=0) sym = (coord::SymmetryType)(sym & ~coord::ST_ZROTATION);
                    if(l>0) sym = (coord::SymmetryType)(sym & ~coord::ST_ROTATION);
                }
    }
    setSymmetry(sym); 
}

/// \cond INTERNAL_DOCS
// auxiliary functions to find outer slope and 1st order correction to inner power-law slope for potential
class FindGammaOut: public math::IFunctionNoDeriv {
private:
    double r1,r2,r3,K;
public:
    FindGammaOut(double _r1, double _r2, double _r3, double _K) :
        r1(_r1), r2(_r2), r3(_r3), K(_K) {};
    virtual double value(double y) const {
        return( pow(r2, 3-y) - pow(r1, 3-y))/( pow(r3, 3-y) - pow(r2, 3-y)) - K;
    }
};
class FindBcorrIn: public math::IFunctionNoDeriv {
private:
    double r1,r2,r3,K2,K3;
public:
    FindBcorrIn(double _r1, double _r2, double _r3, double _K2, double _K3) :
        r1(_r1), r2(_r2), r3(_r3), K2(_K2), K3(_K3) {};
    virtual double value(double B) const {
        return (K2 - log( (1-B*r2)/(1-B*r1) ))*log(r3/r1) - (K3 - log( (1-B*r3)/(1-B*r1) ))*log(r2/r1);
    }
};
/// \endcond

void SplineExp::initSpline(const std::vector<double> &_radii, const std::vector< std::vector<double> > &_coefsArray)
{
    if(_radii[0]!=0)  // check if the innermost node is at origin
        throw std::invalid_argument("SplineExp: radii[0] != 0");
    if(_radii.size()!=Ncoefs_radial+1 || _coefsArray.size()!=Ncoefs_radial+1)
        throw std::invalid_argument("SplineExp: coefArray length != Ncoefs_radial+1");
    potcenter=_coefsArray[0][0];
    // safety measure: if zero-order coefs are the same as potcenter for r>0, skip these elements
    std::vector<double> newRadii;
    std::vector< std::vector<double> > newCoefsArray;
    size_t nskip=0;
    while(nskip+1<_coefsArray.size() && _coefsArray[nskip+1][0]==potcenter)
        nskip++;   // values of potential at r>0 should be strictly larger than at r=0
    if(nskip>0) {  // skip some elements
        newRadii=_radii;
        newRadii.erase(newRadii.begin()+1, newRadii.begin()+nskip+1);
        Ncoefs_radial=newRadii.size()-1;
        newCoefsArray=_coefsArray;
        newCoefsArray.erase(newCoefsArray.begin()+1, newCoefsArray.begin()+nskip+1);
        if(newRadii.size()<5) 
            throw std::invalid_argument("SplineExp: too few radial points");
    }
    const std::vector<double> &radii = nskip==0 ? _radii : newRadii;
    const std::vector< std::vector<double> > &coefsArray = nskip==0 ? _coefsArray : newCoefsArray;
    checkSymmetry(coefsArray);   // assign nontrivial symmetry class if some of coefs are equal or close to zero
    gridradii=radii;    // copy real radii
    minr=gridradii[1];
    maxr=gridradii.back();
    std::vector<double> spnodes(Ncoefs_radial);  // scaled radii
    std::vector<double> spvalues(Ncoefs_radial);
    splines.resize (pow_2(Ncoefs_angular+1));
    slopein. assign(pow_2(Ncoefs_angular+1), 1.);
    slopeout.assign(pow_2(Ncoefs_angular+1), -1.);

    // estimate outermost slope  (needed for accurate extrapolation beyond last grid point)
    const double Kout =
        ( coefsArray[Ncoefs_radial  ][0] * radii[Ncoefs_radial] - 
          coefsArray[Ncoefs_radial-1][0] * radii[Ncoefs_radial-1] ) / 
        ( coefsArray[Ncoefs_radial-1][0] * radii[Ncoefs_radial-1] - 
          coefsArray[Ncoefs_radial-2][0] * radii[Ncoefs_radial-2] );
    if(isFinite(Kout)) {
        FindGammaOut fout(radii[Ncoefs_radial], radii[Ncoefs_radial-1], radii[Ncoefs_radial-2], Kout);
        gammaout = math::findRoot(fout, 3.01, 10., ACCURACY_ROOT);
        if(gammaout != gammaout)
            gammaout = 4.0;
        coefout = fmax(0,
            (1 - coefsArray[Ncoefs_radial-1][0] * radii[Ncoefs_radial-1] /
                (coefsArray[Ncoefs_radial  ][0] * radii[Ncoefs_radial]) ) / 
            (pow(radii[Ncoefs_radial-1] / radii[Ncoefs_radial], 3-gammaout) - 1) );
    } else {
        gammaout=10.0;
        coefout=0;
    }

    // estimate innermost slope with 1st order correction for non-zero radii 
    const double K2 = log((coefsArray[2][0]-potcenter)/(coefsArray[1][0]-potcenter));
    const double K3 = log((coefsArray[3][0]-potcenter)/(coefsArray[1][0]-potcenter));
    FindBcorrIn fin(radii[1], radii[2], radii[3], K2, K3);
    double B = math::findRoot(fin, 0, 0.9/radii[3], ACCURACY_ROOT);
    if(B!=B)
        B = 0.;
    gammain = 2. - ( log((coefsArray[2][0]-potcenter)/(coefsArray[1][0]-potcenter)) - 
                     log((1-B*radii[2])/(1-B*radii[1])) ) / log(radii[2]/radii[1]);
    double gammainuncorr = 2. - log((coefsArray[2][0]-potcenter)/(coefsArray[1][0]-potcenter)) / 
        log(radii[2]/radii[1]);
    if(gammain>=1) gammain=gammainuncorr;
    if(gammain<0) gammain=0; 
    if(gammain>2) gammain=2;
    coefin = (1-coefsArray[1][0]/potcenter) / pow(radii[1], 2-gammain);
    utils::msg(utils::VL_DEBUG, "SplineExp",
        "Inner slope="+utils::toString(gammain)+", outer="+utils::toString(gammaout));

    potmax  = coefsArray.back()[0];
    potminr = coefsArray[1][0];
    // first init l=0 spline which has radial scaling "log(r)" and nontrivial transformation 1/(1/phi-1/phi0)
    for(size_t i=0; i<Ncoefs_radial; i++)
    {
        spnodes[i] = log(gridradii[i+1]);
        spvalues[i]= log(1/ (1/potcenter - 1/coefsArray[i+1][0]));
    }
    double derivLeft  = -(2-gammain)*potcenter/coefsArray[1][0];   // derivative at leftmost node
    double derivRight = - (1+coefout*(3-gammaout))/(1 - potmax/potcenter);  // derivative at rightmost node
    splines[0] = math::CubicSpline(spnodes, spvalues, derivLeft, derivRight);
    coef0(maxr, NULL, NULL, &der2out);

    // next init all higher-order splines which have radial scaling log(ascale+r) and value scaled to l=0,m=0 coefficient
    ascale = get_ascale(radii, coefsArray);
    for(size_t i=0; i<Ncoefs_radial; i++)
        spnodes[i] = log(ascale+gridradii[i+1]);
    double C00val, C00der;
    coef0(minr, &C00val, &C00der, NULL);
    for(int l=lstep; l<=lmax; l+=lstep)
    {
        for(int m=l*mmin; m<=l*mmax; m+=mstep)
        {
            int coefind=l*(l+1)+m;
            for(size_t i=0; i<Ncoefs_radial; i++)
                spvalues[i] = coefsArray[i+1][coefind]/coefsArray[i+1][0];
            slopein[coefind] = log(coefsArray[2][coefind]/coefsArray[1][coefind]) / log(gridradii[2]/gridradii[1]);   // estimate power-law slope of Clm(r) at r->0
            if(!isFinite(slopein[coefind]))
                slopein[coefind]=1.0;  // default
            slopein[coefind] = std::max<double>(slopein[coefind], std::min<double>(l, 2-gammain));  // the asymptotic power-law behaviour of the coefficient expected for power-law density profile
            derivLeft = spvalues[0] * (1+ascale/minr) * (slopein[coefind] - minr*C00der/C00val);   // derivative at innermost node
            slopeout[coefind] = log(coefsArray[Ncoefs_radial][coefind]/coefsArray[Ncoefs_radial-1][coefind]) / log(gridradii[Ncoefs_radial]/gridradii[Ncoefs_radial-1]) + 1;   // estimate slope of Clm(r)/C00(r) at r->infinity (+1 is added because C00(r) ~ 1/r at large r)
            if(!isFinite(slopeout[coefind]))
                slopeout[coefind]=-1.0;  // default
            slopeout[coefind] = std::min<double>(slopeout[coefind], std::max<double>(-l, 3-gammaout));
            derivRight = spvalues[Ncoefs_radial-1] * (1+ascale/maxr) * slopeout[coefind];   // derivative at outermost node
            splines[coefind] = math::CubicSpline(spnodes, spvalues, derivLeft, derivRight);
        }
    }
}

void SplineExp::getCoefs(
    std::vector<double> &radii, std::vector< std::vector<double> > &coefsArray) const
{
    radii.resize(Ncoefs_radial+1);
    for(size_t i=0; i<=Ncoefs_radial; i++)
        radii[i] = gridradii[i];
    coefsArray.resize(Ncoefs_radial+1);
    for(size_t i=0; i<=Ncoefs_radial; i++) {
        double rad = radii[i];
        double Coef00;
        coef0(rad, &Coef00, NULL, NULL);
        coefsArray[i].assign(pow_2(Ncoefs_angular+1), 0);
        coefsArray[i][0] = Coef00;
        double xi = log(ascale+rad);
        for(int l=lstep; l<=lmax; l+=lstep)
            for(int m=l*mmin; m<=l*mmax; m+=mstep) {
                int coefind=l*(l+1)+m;
                coeflm(coefind, rad, xi, &(coefsArray[i][l*(l+1)+m]), NULL, NULL, Coef00);
            }
    }
}

void SplineExp::coeflm(unsigned int lm, double r, double xi, double *val, double *der, double *der2, double c0val, double c0der, double c0der2) const  // works only for l2>0
{
    double cval=0, cder=0, cder2=0;   // value and derivatives of \tilde Clm = Clm(r)/C00(r)
    if(r < maxr)
    {
        if(r > minr)  // normal interpolation
        {
            if(der==NULL) {
                splines[lm].evalDeriv(xi, &cval, NULL, NULL);
            } else if(der2==NULL) {
                splines[lm].evalDeriv(xi, &cval, &cder, NULL);
                cder /= r+ascale;
            } else {
                splines[lm].evalDeriv(xi, &cval, &cder, &cder2);
                cder /= r+ascale;
                cder2 = (cder2/(r+ascale) - cder)/(r+ascale);
            }
        }
        else  // power-law asymptotics at r<minr
        {
            cval = splines[lm](splines[lm].xmin()) * potminr;
            if(val!=NULL)  *val = cval * pow(r/minr, slopein[lm]);
            if(der!=NULL){ *der = (*val) * slopein[lm]/r;
            if(der2!=NULL) *der2= (*der) * (slopein[lm]-1)/r; }
            return;   // for r<minr, Clm is not scaled by C00
        }
    }
    else  // power-law asymptotics for r>maxr
    {     // god knows what happens here...
        double ximax = splines[lm].xmax();
        double mval, mder, mder2;
        splines[lm].evalDeriv(ximax, &mval, &mder, &mder2);
        cval = mval * pow(r/maxr, slopeout[lm]);
        cder = cval * slopeout[lm]/r;
        cder2= cder * (slopeout[lm]-1)/r;
        double der2left = (mder2 - mder)/pow_2(r+ascale);
        double der2right = mval*slopeout[lm]*(slopeout[lm]-1)/pow_2(maxr);
        double acorr = (der2left-der2right)*0.5;
        double slopecorr = slopeout[lm]-4;
        double powcorr = pow(r/maxr, slopecorr);
        cval += acorr*powcorr*pow_2(r-maxr);
        cder += acorr*powcorr*(r-maxr)*(2 + slopecorr*(r-maxr)/r);
        cder2+= acorr*powcorr*(2 + 4*slopecorr*(r-maxr)/r + pow_2(1-maxr/r)*slopecorr*(slopecorr-1));
    }
    // scale by C00
    if(val!=NULL)  *val = cval*c0val;
    if(der!=NULL)  *der = cder*c0val + cval*c0der;
    if(der2!=NULL) *der2= cder2*c0val + 2*cder*c0der + cval*c0der2;
}

void SplineExp::coef0(double r, double *val, double *der, double *der2) const  // works only for l=0
{
    if(r<=maxr) {
        double logr=log(r);
        double sval, sder, sder2;
        if(r<minr) {
            double ratio = 1-coefin*pow(r, 2-gammain);  // C00(r)/C00(0)
            sval = log(-potcenter/coefin) - (2-gammain)*logr + log(ratio);
            sder = -(2-gammain)/ratio;
            sder2= -pow_2(sder)*(1-ratio);
        } else {
            splines[0].evalDeriv(logr, &sval, &sder, &sder2);
        }
        double sexp = (r>0? exp(-sval) : 0);
        double vval = 1./(sexp-1/potcenter);
        if(val!=NULL)  *val = -vval;
        if(der!=NULL)  *der = -vval*vval*sexp/r * sder;  // this would not work for r=0 anyway...
        if(der2!=NULL) *der2= -pow_2(vval/r)*sexp * (sder2 - sder + sder*sder*(2*vval*sexp-1) );
    } else {
        double r_over_maxr_g=pow(r/maxr, 3-gammaout);
        double der2right = -2*potmax*maxr/pow(r,3) * (1 - coefout*(r_over_maxr_g*(gammaout-1)*(gammaout-2)/2 - 1));
        double slopecorr = -gammaout-4;
        double acorr = 0*pow(r/maxr, slopecorr) * (der2out-der2right)*0.5;   // apparently unused, but why?
        if(val!=NULL)  *val = -(-potmax*maxr/r * (1 - coefout*(r_over_maxr_g-1))  + acorr*pow_2(r-maxr) );
        if(der!=NULL)  *der = -( potmax*maxr/r/r * (1 - coefout*(r_over_maxr_g*(gammaout-2) - 1))  + acorr*(r-maxr)*(2 + slopecorr*(r-maxr)/r) );
        if(der2!=NULL) *der2= -(der2right  + acorr*(2 + 4*slopecorr*(r-maxr)/r + pow_2(1-maxr/r)*slopecorr*(slopecorr-1)) );
    }
}

void SplineExp::computeSHCoefs(const double r, double coefsF[], double coefsdFdr[], double coefsd2Fdr2[]) const
{
    double xi = log(r+ascale);
    double val00, der00, der200;
    coef0(r, &val00, &der00, &der200);  // compute value and two derivatives of l=0,m=0 spline
    if(coefsF)    coefsF[0]    = val00;
    if(coefsdFdr) coefsdFdr[0] = der00;
    if(coefsd2Fdr2) coefsd2Fdr2[0] = der200;
    for(int l=lstep; l<=lmax; l+=lstep)
    {
        for(int m=l*mmin; m<=l*mmax; m+=mstep)
        {
            int coefind=l*(l+1)+m;
            coeflm(coefind, r, xi, 
                coefsF!=NULL ? coefsF+coefind : NULL, 
                coefsdFdr!=NULL ? coefsdFdr+coefind : NULL, 
                coefsd2Fdr2!=NULL ? coefsd2Fdr2+coefind : NULL, 
                val00, der00, der200);
        }
    }
}

double SplineExp::enclosedMass(const double r) const
{
    if(r<=0) return 0;
    double der;
    coef0(r, NULL, &der, NULL);
    return der * r*r;   // d Phi(r)/d r = G M(r) / r^2
}

}; // namespace
