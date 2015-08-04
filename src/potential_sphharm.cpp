#include "potential_sphharm.h"
#include "math_core.h"
#include "math_specfunc.h"
#include <cmath>
#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_gegenbauer.h>
#include <gsl/gsl_sf_gamma.h>
#ifdef HAVE_CUBATURE
#include <cubature.h>
#endif

namespace potential {

/// max number of basis-function expansion members (radial and angular).
const unsigned int MAX_NCOEFS_ANGULAR = 100;
const unsigned int MAX_NCOEFS_RADIAL  = 100;

/// auxiliary softening parameter for Spline potential to avoid singularities in some odd cases
const double SPLINE_MIN_RADIUS=1e-10;

//----------------------------------------------------------------------------//
// BasePotentialSphericalHarmonic -- parent class for all potentials 
// using angular expansion in spherical harmonics
void BasePotentialSphericalHarmonic::setSymmetry(SymmetryType sym)
{
    mysymmetry = sym;
    lmax = (mysymmetry & ST_SPHSYM)    ==ST_SPHSYM     ? 0 :     // if spherical model, use only l=0,m=0 term
        static_cast<int>(std::min<unsigned int>(Ncoefs_angular, MAX_NCOEFS_ANGULAR));
    lstep= (mysymmetry & ST_REFLECTION)==ST_REFLECTION ? 2 : 1;  // if reflection symmetry, use only even l
    mmax = (mysymmetry & ST_ZROTSYM)   ==ST_ZROTSYM    ? 0 : 1;  // if axisymmetric model, use only m=0 terms, otherwise all terms up to l (1 is the multiplying factor)
    mmin = (mysymmetry & ST_PLANESYM)  ==ST_PLANESYM   ? 0 :-1;  // if triaxial symmetry, do not use sine terms which correspond to m<0
    mstep= (mysymmetry & ST_PLANESYM)  ==ST_PLANESYM   ? 2 : 1;  // if triaxial symmetry, use only even m
}

/// \cond INTERNAL_DOCS
// helper functions for integration over angles (common for BSE and Spline potentials), used in computing coefficients from analytic density model
struct CPotentialParamSH{
    const BaseDensity* P;
    int sh_n, sh_l, sh_m;
    double costheta, sintheta, r, Alpha;
    double theta_max, phi_max;    // Pi/2, Pi or even 2*Pi, depending on potential symmetry
};
double intSH_phi(double phi, void* params)
{
    double costheta= ((CPotentialParamSH*)params)->costheta;
    double sintheta= ((CPotentialParamSH*)params)->sintheta;
    double r= ((CPotentialParamSH*)params)->r;
    const BaseDensity* P=((CPotentialParamSH*)params)->P;
    int sh_m=((CPotentialParamSH*)params)->sh_m;
    return P->density(coord::PosCar(r*sintheta*cos(phi), r*sintheta*sin(phi), r*costheta))
        * (sh_m>=0 ? cos(sh_m*phi) : sin(-sh_m*phi));
}
double intSH_theta(double theta, void* params)
{
    double result, error;
    size_t neval;
    ((CPotentialParamSH*)params)->costheta=cos(theta);
    ((CPotentialParamSH*)params)->sintheta=sin(theta);
    gsl_function F;
    F.function=&intSH_phi;
    F.params=params;
    if( (((CPotentialParamSH*)params)->P->symmetry() & ST_AXISYMMETRIC) == ST_AXISYMMETRIC)
    {   // don't integrate in phi
        result = ((CPotentialParamSH*)params)->sh_m != 0 ? 0 :
            ((CPotentialParamSH*)params)->phi_max * 
            math::legendrePoly(((CPotentialParamSH*)params)->sh_l, 0, theta)* 
            ((CPotentialParamSH*)params)->P->density(coord::PosCar(
                ((CPotentialParamSH*)params)->r*((CPotentialParamSH*)params)->sintheta, 0, 
                ((CPotentialParamSH*)params)->r*((CPotentialParamSH*)params)->costheta));
    } else {
        gsl_integration_qng(&F, 0, ((CPotentialParamSH*)params)->phi_max, 
            0, EPSREL_POTENTIAL_INT, &result, &error, &neval);
        result *= math::legendrePoly(((CPotentialParamSH*)params)->sh_l, abs(((CPotentialParamSH*)params)->sh_m), theta);
    }
    return result * sin(theta) / (2*M_SQRTPI);   // factor \sqrt{4\pi} coming from the definition of spherical function Y_l^m
}
/// \endcond

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
    for(int m=0; m<=mmax*lmax; m+=mstep) {
        math::legendrePolyArray(lmax, m, pos.theta, legendre_array, 
            grad!=NULL||hess!=NULL ? legendre_deriv_array : NULL, 
            hess!=NULL ? legendre_deriv2_array : NULL);
        double cosmphi = (m==0 ? 1 : cos(m*pos.phi)*M_SQRT2) * 2*M_SQRTPI;   // factor \sqrt{4\pi} from the definition of spherical function Y_l^m absorbed into this term
        double sinmphi = (sin(m*pos.phi)*M_SQRT2) * 2*M_SQRTPI;
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
    const particles::PointMassArray<coord::PosSph> &points, SymmetryType _sym):
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
    BasePotentialSphericalHarmonic(coefs.size()>0 ? static_cast<size_t>(sqrt(coefs[0].size()*1.0)-1) : 0), 
    Ncoefs_radial(std::min<size_t>(MAX_NCOEFS_RADIAL-1, static_cast<size_t>(coefs.size()-1))),
    Alpha(_Alpha)  // here Alpha!=0 - no autodetect
{
    if(_Alpha<0.5) 
        throw std::invalid_argument("BasisSetExp: invalid parameter Alpha");
    for(size_t n=0; n<coefs.size(); n++)
        if(coefs[n].size()!=pow_2(Ncoefs_angular+1))
            throw std::invalid_argument("BasisSetExp: incorrect size of coefficients array");
    SHcoefs = coefs;
    checkSymmetry();
}

BasisSetExp::BasisSetExp(double _Alpha, unsigned int _Ncoefs_radial, unsigned int _Ncoefs_angular, 
    const BaseDensity& srcdensity):    // init potential from analytic mass model
    BasePotentialSphericalHarmonic(_Ncoefs_angular), 
    Ncoefs_radial(std::min<size_t>(MAX_NCOEFS_RADIAL-1, _Ncoefs_radial)),
    Alpha(_Alpha)
{
    setSymmetry(srcdensity.symmetry());
    prepareCoefsAnalytic(srcdensity);
    checkSymmetry();
}

void BasisSetExp::checkSymmetry()
{ 
    SymmetryType sym=ST_SPHERICAL;  // too optimistic:))
    const double MINCOEF=1e-8;
    for(size_t n=0; n<=Ncoefs_radial; n++)
    {
        for(int l=0; l<=(int)Ncoefs_angular; l++)
            for(int m=-l; m<=l; m++)
                if(fabs(SHcoefs[n][l*(l+1)+m])>MINCOEF) 
                {   // nonzero coef.: check if that breaks any symmetry
                    if(l%2==1)  sym = (SymmetryType)(sym & ~ST_REFLECTION);
                    if(m<0 || m%2==1)  sym = (SymmetryType)(sym & ~ST_PLANESYM);
                    if(m!=0) sym = (SymmetryType)(sym & ~ST_ZROTSYM);
                    if(l>0) sym = (SymmetryType)(sym & ~ST_SPHSYM);
                }
    }
    // now set all coefs excluded by the inferred symmetry  to zero
    for(size_t n=0; n<=Ncoefs_radial; n++)
    {
        for(int l=0; l<=(int)Ncoefs_angular; l++)
            for(int m=-l; m<=l; m++)
                if( (l>0 && (sym & ST_SPHSYM)) ||
                    (m!=0 && (sym & ST_ZROTSYM)) ||
                    ((m<0 || m%2==1) && (sym & ST_PLANESYM)) ||
                    (l%2==1 && (sym & ST_REFLECTION)) ) 
                        SHcoefs[n][l*(l+1)+m] = 0;
    }
    setSymmetry(sym);
}

/// \cond INTERNAL_DOCS
double intBSE_xi(double xi, void* params)   // integrate over scaled radial variable
{
    if(xi>=1.0 || xi<=-1.0) return 0;
    double Alpha=((CPotentialParamSH*)params)->Alpha;
    double ralpha=(1+xi)/(1-xi);
    double r=pow(ralpha, Alpha);
    int n=((CPotentialParamSH*)params)->sh_n;
    int l=((CPotentialParamSH*)params)->sh_l;
    double phi_nl = gsl_sf_gegenpoly_n(n, (2*l+1)*Alpha+0.5, xi) * pow(r, l) * pow(1+ralpha, -(2*l+1)*Alpha);
    ((CPotentialParamSH*)params)->r=r;
    gsl_function F;
    F.function=&intSH_theta;
    F.params=params;
    double result, error;
    size_t neval;
    gsl_integration_qng(&F, 0, ((CPotentialParamSH*)params)->theta_max, 0, EPSREL_POTENTIAL_INT, &result, &error, &neval);
    return result * phi_nl * 8*M_PI*Alpha *r*r*r/(1-xi*xi);
}
/// \endcond

void BasisSetExp::prepareCoefsAnalytic(const BaseDensity& srcdensity)
{
    if(Alpha<0.5)
        Alpha = 1.;
    SHcoefs.resize(Ncoefs_radial+1);
    for(size_t n=0; n<=Ncoefs_radial; n++)
        SHcoefs[n].assign(pow_2(Ncoefs_angular+1), 0);
    CPotentialParamSH PP;
    PP.P = &srcdensity;
    PP.Alpha = Alpha;
    PP.theta_max = (symmetry() & ST_REFLECTION)==ST_REFLECTION ? M_PI_2 : M_PI;  // if symmetries exist, no need to integrate over whole space
    PP.phi_max   = (symmetry() & ST_PLANESYM)==ST_PLANESYM ? M_PI_2 : 2*M_PI;
    int multfactor = ((symmetry() & ST_PLANESYM)==ST_PLANESYM ? 4 : 1) * 
        ((symmetry() & ST_REFLECTION)==ST_REFLECTION ? 2 : 1);  // compensates integration of only half- or 1/8-space
    gsl_integration_workspace * ws = gsl_integration_workspace_alloc (1000);
    double interval[2]={-1.0, 1.0};
    gsl_function F;
    F.function=&intBSE_xi;
    F.params=&PP;
    for(size_t n=0; n<=Ncoefs_radial; n++)
        for(int l=0; l<=lmax; l+=lstep)
        {
            PP.sh_n=static_cast<int>(n);  PP.sh_l=l;
            double w=(2*l+1)*Alpha+0.5;
            double Knl = (4*pow_2(n+w)-1)/8/pow_2(Alpha);
            double Inl = Knl * 4*M_PI*Alpha * 
                exp( gsl_sf_lngamma(n+2*w) - 2*gsl_sf_lngamma(w) - gsl_sf_lnfact(PP.sh_n) - 4*w*log(2.0)) / (n+w);
            for(int m=l*mmin; m<=l*mmax; m+=mstep)
            {
                PP.sh_m=m;
                double result, error;
                gsl_integration_qagp(&F, interval, 2, 0, EPSREL_POTENTIAL_INT, 1000, ws, &result, &error);
                SHcoefs[n][l*(l+1)+m] = result * multfactor * (m==0 ? 1 : M_SQRT2) / Inl;
            }
        }
    gsl_integration_workspace_free (ws);
}

void BasisSetExp::prepareCoefsDiscrete(const particles::PointMassArray<coord::PosSph> &points)
{
    SHcoefs.resize(Ncoefs_radial+1);
    for(size_t n=0; n<=Ncoefs_radial; n++)
        SHcoefs[n].assign(pow_2(1+Ncoefs_angular), 0);
    size_t npoints=points.size();
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
              exp( gsl_sf_lngamma(n+2*w) - 2*gsl_sf_lngamma(w) - gsl_sf_lnfact(n) - 4*w*log(2.0)) /
              (n+w) * (4*(n+w)*(n+w)-1)/(8*Alpha*Alpha);
    }
    for(size_t i=0; i<npoints; i++)
    {
        const coord::PosSph& point = points.point(i);
        double massi = points.mass(i);
        double ralpha=pow(point.r, 1/Alpha);
        double xi=(ralpha-1)/(ralpha+1);
        for(int m=0; m<=lmax; m+=mstep)
            math::legendrePolyArray(lmax, m, point.theta, legendre_array[m]);

        for(int l=0; l<=lmax; l+=lstep)
        {
            double w=(2*l+1)*Alpha+0.5;
            double phil=pow(point.r, l) * pow(1+ralpha, -(2*l+1)*Alpha);
            gsl_sf_gegenpoly_array(static_cast<int>(Ncoefs_radial), w, xi, gegenpoly_array);
            for(size_t n=0; n<=Ncoefs_radial; n++)
            {
                double mult= massi * gegenpoly_array[n] * phil * 2*M_SQRTPI / Inl[n][l];
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
    gsl_sf_gegenpoly_array(static_cast<int>(Ncoefs_radial), Alpha+0.5, xi, gegenpoly_array);
    double multr = pow(1+ralpha, -Alpha);
    double multdr= -ralpha/((ralpha+1)*r);
    double result=0;
    for(int n=0; n<=static_cast<int>(Ncoefs_radial); n++)
    {
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
    for(int l=0; l<=lmax; l+=lstep)
    {
        double w=(2*l+1)*Alpha+0.5;
        gsl_sf_gegenpoly_array(static_cast<int>(Ncoefs_radial), w, xi, gegenpoly_array);
        double multr = -pow(r, l) * pow(1+ralpha, -(2*l+1)*Alpha);
        double multdr= (l-(l+1)*ralpha)/((ralpha+1)*r);
        for(int n=0; n<=(int)Ncoefs_radial; n++)
        {
            double multdFdr=0, multd2Fdr2=0, dGdr=0;
            if(coefsdFdr!=NULL) {
                dGdr=(n>0 ? (-n*xi*gegenpoly_array[n] + (n+2*w-1)*gegenpoly_array[n-1])/(2*Alpha*r) : 0);
                multdFdr= multdr * gegenpoly_array[n] + dGdr;
                if(coefsd2Fdr2!=NULL)
                    multd2Fdr2 = ( (l+1)*(l+2)*pow_2(ralpha) + 
                                   ( (1-2*l*(l+1)) - (2*n+1)*(2*l+1)/Alpha - n*(n+1)/pow_2(Alpha))*ralpha + 
                                   l*(l-1) 
                                 ) / pow_2( (1+ralpha)*r ) * gegenpoly_array[n] - dGdr*2/r;
            }
            for(int m=l*mmin; m<=l*mmax; m+=mstep)
            {
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
    const particles::PointMassArray<coord::PosSph> &points, SymmetryType _sym, 
    double smoothfactor, const std::vector<double> *radii):
    BasePotentialSphericalHarmonic(_Ncoefs_angular),
    Ncoefs_radial(std::max<size_t>(5,_Ncoefs_radial))
{
    setSymmetry(_sym);
    prepareCoefsDiscrete(points, smoothfactor, radii);
}

// init from existing coefs
SplineExp::SplineExp(
    const std::vector<double> &_gridradii, const std::vector< std::vector<double> > &_coefs):
    BasePotentialSphericalHarmonic(_coefs.size()>0 ? static_cast<size_t>(sqrt(_coefs[0].size()*1.0)-1) : 0), 
    Ncoefs_radial(std::min<size_t>(MAX_NCOEFS_RADIAL-1, _coefs.size()-1))
{
    for(size_t n=0; n<_coefs.size(); n++)
        if(_coefs[n].size()!=pow_2(Ncoefs_angular+1))
            throw std::invalid_argument("SplineExp: incorrect size of coefficients array");
    initSpline(_gridradii, _coefs);
}

// init potential from analytic mass model
SplineExp::SplineExp(unsigned int _Ncoefs_radial, unsigned int _Ncoefs_angular, 
    const BaseDensity& srcdensity, const std::vector<double> *radii):
    BasePotentialSphericalHarmonic(_Ncoefs_angular), 
    Ncoefs_radial(std::max<unsigned int>(5,_Ncoefs_radial))
{
    setSymmetry(srcdensity.symmetry());
    prepareCoefsAnalytic(srcdensity, radii);
}

/// \cond INTERNAL_DOCS
double intSpline_r(double r, void* params)
{
    if(r==0) return 0;
    int n=((CPotentialParamSH*)params)->sh_n;  // power index for s^n, either l+2 or 1-l
    double result, error;
    size_t neval;
    ((CPotentialParamSH*)params)->r=r;
    gsl_function F;
    F.function=&intSH_theta;
    F.params=params;
    gsl_integration_qng(&F, 0, ((CPotentialParamSH*)params)->theta_max, 0, EPSREL_POTENTIAL_INT, &result, &error, &neval);
    // workaround for the case when a coefficient is very small due to unnoticed symmetry (maybe a better solution exists?)
    if(((CPotentialParamSH*)params)->sh_l>2 && fabs(result)<EPSABS_POTENTIAL_INT)
        result=0;  // to avoid huge drop in performance for very small coef values
    return result * pow(r, n*1.0);
}
/// \endcond

#if 0 //#ifdef HAVE_CUBATURE
int intSplineCubature(unsigned int ndim, const double coords[],
    void* params, unsigned int /*fdim*/, double* output)
{
    const double r=coords[0]/(1-coords[0]);
    if(r==0) return 0;
    const double costheta=coords[1];
    const double rsintheta=r*sqrt(1-pow_2(costheta));
    const BaseDensity* P=((CPotentialParamSH*)params)->P;
    int n=((CPotentialParamSH*)params)->sh_n;  // power index for s^n, either l+2 or 1-l
    int sh_l=((CPotentialParamSH*)params)->sh_l;
    int sh_m=((CPotentialParamSH*)params)->sh_m;
    double result;
    if(ndim==N_DIM) {  // use phi only for non-axisymmetric potentials
        const double phi=coords[2];
        result = P->Rho(rsintheta*cos(phi), rsintheta*sin(phi), r*costheta) *
            gsl_sf_legendre_sphPlm(sh_l, abs(sh_m), costheta) * 
            (sh_m>=0 ? cos(sh_m*phi) : sin(-sh_m*phi));
    } else {
        result = P->Rho(rsintheta, 0, r*costheta) *
            gsl_sf_legendre_sphPlm(sh_l, 0, costheta) *
            (sh_m==0 ? ((CPotentialParamSH*)params)->phi_max : 0);
    }
    *output = result * pow(r, n*1.0) / pow_2(1-coords[0]) / (2*M_SQRTPI);   // factor \sqrt{4\pi} coming from the definition of spherical function Y_l^m
    return 0;
}
#endif

void SplineExp::prepareCoefsAnalytic(const BaseDensity& srcdensity, const std::vector<double> *srcradii)
{
    std::vector< std::vector<double> > coefsArray(Ncoefs_radial+1);  // SHE coefficients to pass to initspline routine
    std::vector<double> radii(Ncoefs_radial+1);  // true radii to pass to initspline routine
    for(unsigned int i=0; i<=Ncoefs_radial; i++)
        coefsArray[i].assign(pow_2(1+Ncoefs_angular), 0);
    bool initUserRadii = (srcradii!=NULL && srcradii->size()==Ncoefs_radial+1 && srcradii->front()==0);
    if(srcradii!=NULL && !initUserRadii)  // something went wrong with manually supplied radii
        throw std::invalid_argument("Invalid call to constructor of Spline potential");
    if(initUserRadii)
        radii= *srcradii;
    else {
        // find inner/outermost radius
        double totalmass = srcdensity.totalMass();
        if(!math::isFinite(totalmass))
            throw std::invalid_argument("SplineExp: source density model has infinite mass");
        // how far should be the outer node (leave out this fraction of mass)
        double epsout = 0.1/sqrt(pow_2(Ncoefs_radial)+0.01*pow(Ncoefs_radial*1.0,4.0));
        // how close can we get to zero, in terms of innermost grid node
        double epsin = 5.0/pow(Ncoefs_radial*1.0,3.0);
        // somewhat arbitrary choice for min/max radii, but probably reasonable
        double rout = getRadiusByMass(srcdensity, totalmass*(1-epsout));
        double rin = std::min<double>(epsin, getRadiusByMass(srcdensity, totalmass*epsin*0.1));
        math::createNonuniformGrid(Ncoefs_radial+1, rin, rout, true, radii); 
    }
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (1000);
    CPotentialParamSH PP;
    PP.P = &srcdensity;
    PP.theta_max = symmetry() & ST_REFLECTION ? M_PI_2 : M_PI;  // if symmetries exist, no need to integrate over whole space
    PP.phi_max = symmetry() & ST_PLANESYM ? M_PI_2 : 2*M_PI;
    int multfactor = (symmetry() & ST_PLANESYM ? 4 : 1) * (symmetry() & ST_REFLECTION ? 2 : 1);  // compensates integration of only half- or 1/8-space
    gsl_function F;
    F.function=&intSpline_r;
    F.params=&PP;
    std::vector<double> coefsInner, coefsOuter;
    radii.front()=SPLINE_MIN_RADIUS*radii[1];  // to prevent log divergence for gamma=2 potentials
    for(int l=0; l<=lmax; l+=lstep)
    {
        PP.sh_l=l;
        for(int m=l*mmin; m<=l*mmax; m+=mstep)
        {
            PP.sh_m=m;
            // first precompute inner and outer density integrals at each radial grid point, summing contributions from each interval of radial grid
            coefsInner.assign(Ncoefs_radial+1, 0);
            coefsOuter.assign(Ncoefs_radial+1, 0);
            // loop over inner intervals
            double result, error;
            PP.sh_n = l+2;
            for(size_t c=0; c<Ncoefs_radial; c++)
            {
#if 0 //#ifdef HAVE_CUBATURE
                double bmin[3]={radii[c]/(1+radii[c]), PP.theta_max>3.?-1.:0., 0.};  // integrate in theta up to pi/2 or pi
                double bmax[3]={radii[c+1]/(1+radii[c+1]), 1., PP.phi_max};  // integration box
                int ndim=(mysymmetry & ST_AXISYMMETRIC) == ST_AXISYMMETRIC ? 2 : 3;  // whether to integrate over phi or not
                hcubature(1, &intSplineCubature, &PP, ndim, bmin, bmax, 65536/*max_eval*/, 
                    EPSABS_POTENTIAL_INT, EPSREL_POTENTIAL_INT, ERROR_L1/*ignored*/, &result, &error);
#else                
                // the reason we give absolute error threshold is that for flat-density-core profiles 
                // the high-l,m coefs at small radii are extremely tiny and their exact calculation is impractically slow
                gsl_integration_qags(&F, radii[c], radii[c+1], EPSABS_POTENTIAL_INT, EPSREL_POTENTIAL_INT, 1000, w, &result, &error);
#endif
                coefsInner[c+1] = result + coefsInner[c];
            }
            // loop over outer intervals, starting from infinity backwards
            PP.sh_n = 1-l;
            for(size_t c=Ncoefs_radial+1; c>static_cast<size_t>(l==0 ? 0 : 1); c--)
            {
#if 0 //#ifdef HAVE_CUBATURE
                double bmin[3]={radii[c-1]/(1+radii[c-1]), PP.theta_max>3.?-1.:0., 0.};
                double bmax[3]={c>Ncoefs_radial ? 1.: radii[c]/(1+radii[c]), 1., PP.phi_max};
                int ndim=(mysymmetry & ST_AXISYMMETRIC) == ST_AXISYMMETRIC ? 2 : 3;
                hcubature(1, &intSplineCubature, &PP, ndim, bmin, bmax, 65536/*max_eval*/, 
                    EPSABS_POTENTIAL_INT*(c<=Ncoefs_radial?fabs(coefsOuter[c]):1), 
                    EPSREL_POTENTIAL_INT, ERROR_L1/*ignored*/, &result, &error);
#else
                if(c==Ncoefs_radial+1)
                    gsl_integration_qagiu(&F, radii.back(), EPSABS_POTENTIAL_INT, 
                        EPSREL_POTENTIAL_INT, 1000, w, &result, &error);
                else
                    gsl_integration_qags(&F, radii[c-1], radii[c], EPSREL_POTENTIAL_INT*fabs(coefsOuter[c]), 
                        EPSREL_POTENTIAL_INT, 1000, w, &result, &error);
#endif
                coefsOuter[c-1] = result + (c>Ncoefs_radial?0:coefsOuter[c]);
            }
            // now compute the coefs of potential expansion themselves
            for(size_t c=0; c<=Ncoefs_radial; c++)
            {
                coefsArray[c][l*(l+1) + m] = ((c>0 ? coefsInner[c]*pow(radii[c], -l-1.0) : 0) + coefsOuter[c]*pow(radii[c], l*1.0)) *
                    multfactor* -4*M_PI/(2*l+1) * (m==0 ? 1 : M_SQRT2);
            }
#ifdef DEBUGPRINT
            my_message(FUNCNAME, "l="+convertToString(l)+",m="+convertToString(m));
#endif
        }
    }
    radii.front()=0;
    gsl_integration_workspace_free (w);
    initSpline(radii, coefsArray);
}

/// \cond INTERNAL_DOCS
inline bool compareParticleSph(
    const particles::PointMassArray<coord::PosSph>::ElemType& val1, 
    const particles::PointMassArray<coord::PosSph>::ElemType& val2) {
    return val1.first.r < val2.first.r;  }
/// \endcond

void SplineExp::computeCoefsFromPoints(const particles::PointMassArray<coord::PosSph> &srcPoints, 
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
    particles::PointMassArray<coord::PosSph> points(srcPoints);
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
        for(int m=0; m<=lmax; m+=mstep)
            math::legendrePolyArray(lmax, m, points.point(i).theta, legendre_array[m]);
        for(int l=0; l<=lmax; l+=lstep)
            for(int m=l*mmin; m<=l*mmax; m+=mstep) {
                int coefind = l*(l+1)+m;
                int absm = abs(m);  // negative m correspond to sine, positive - to cosine
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
                    (i<npoints-1 ? coefsOuter[coefind][i+1] * pow(points.point(i).r, l) : 0);
            }
    }
    // local variable coefsInner will be automatically freed, but outputCoefs will remain
}


void SplineExp::prepareCoefsDiscrete(const particles::PointMassArray<coord::PosSph> &points, 
    double smoothfactor, const std::vector<double> *userradii)
{
    if(points.size() <= Ncoefs_radial*10)
        throw std::invalid_argument("SplineExp: number of particles is too small");
    // radii of each point in ascending order
    std::vector<double> pointRadii;
    // note that array indexing is swapped w.r.t. coefsArray (i.e. pointCoefs[coefIndex][pointIndex])
    // to save memory by not initializing unnecessary coefs
    std::vector< std::vector<double> > pointCoefs;
    computeCoefsFromPoints(points, pointRadii, pointCoefs);

    size_t npoints = pointRadii.size();
    std::vector<double> radii(Ncoefs_radial+1);                         // radii of grid nodes to pass to initspline routine
    std::vector< std::vector<double> > coefsArray(Ncoefs_radial+1);     // SHE coefficients to pass to initspline routine
    for(size_t i=0; i<=Ncoefs_radial; i++)
        coefsArray[i].assign(pow_2(Ncoefs_angular+1), 0);

    // choose the radial grid parameters: innermost cell contains minBinMass and outermost radial node should encompass cutoffMass
    // spline definition region extends up to outerRadius which is ~several times larger than outermost radial node, however coefficients at that radius are not used in the potential computation later
    const size_t minBinPoints=10;
    bool initUserRadii = (userradii!=NULL && userradii->size()==Ncoefs_radial+1 && userradii->front()==0 && userradii->at(1)>= pointRadii[minBinPoints]);
    if(userradii!=NULL && !initUserRadii)  // something went wrong with manually supplied radii
        throw std::invalid_argument("SplineExp: invalid radial grid");
    if(initUserRadii)
        radii= *userradii;
    else
    {
        size_t npointsMargin    = static_cast<size_t>(sqrt(npoints*0.1));
        size_t npointsInnerGrid = std::max<size_t>(minBinPoints, npointsMargin);    // number of points within 1st grid radius
        size_t npointsOuterGrid = npoints - std::max<size_t>(minBinPoints, npointsMargin);   // number of points within outermost grid radius
        double innerBinRadius = pointRadii[npointsInnerGrid];
        double outerBinRadius = pointRadii[npointsOuterGrid];
        math::createNonuniformGrid(Ncoefs_radial+1, innerBinRadius, outerBinRadius, true, radii);
    }
    // find index of the inner- and outermost points which are used in b-spline fitting
    size_t npointsInnerSpline = 0;
    while(pointRadii[npointsInnerSpline]<radii[1]) npointsInnerSpline++;
    npointsInnerSpline = std::min<size_t>(npointsInnerSpline-2, std::max<size_t>(minBinPoints, npointsInnerSpline/2));
    // index of last point used in b-spline fitting (it is beyond outer grid radius, since b-spline definition region is larger than the range of radii for which the spline approximation will eventually be constructed)
    double outerRadiusSpline = pow_2(radii[Ncoefs_radial])/radii[Ncoefs_radial-1];  // roughly equally logarithmically spaced from the last two points
    size_t npointsOuterSpline = npoints-1;
    while(pointRadii[npointsOuterSpline]>outerRadiusSpline) npointsOuterSpline--;
    //!!!FIXME: what happens if the outermost pointRadius is < outerBinRadius ?

    size_t numPointsUsed = npointsOuterSpline-npointsInnerSpline;  // outer and inner points are ignored
    size_t numBSplineKnots = Ncoefs_radial+2;  // including zero and outermost point; only interior nodes are actually used for computing best-fit coefs (except l=0 coef, for which r=0 is also used)
    std::vector<double> scaledPointRadii(numPointsUsed), scaledPointCoefs(numPointsUsed);  // transformed x- and y- values of original data points which will be approximated by a spline regression
    std::vector<double> scaledKnotRadii(numBSplineKnots), scaledSplineValues;              // transformed x- and y- values of regression spline knots

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
        math::SplineApprox appr(scaledPointRadii, scaledKnotRadii);
//        if(appr.isSingular())
//            my_message(FUNCNAME, 
//                "Warning, in Spline potential initialization: singular matrix for least-square fitting; fallback to a slow algorithm");
        double derivLeft, derivRight;
        appr.fitData(scaledPointCoefs, 0, scaledSplineValues, derivLeft, derivRight);
        // now store fitted values in coefsArray to pass to initspline routine
        coefsArray[0][0] = potcenter;
        for(size_t c=1; c<=Ncoefs_radial; c++)
            coefsArray[c][0] = -1./(exp(-scaledSplineValues[c])-1/potcenter);
    }
    if(lmax>0) {  // construct splines for all l>0 spherical-harmonic terms separately
        // first estimate the asymptotic power-law slope of coefficients at r=0 and r=infinity
        double gammaInner = 2-log((coefsArray[1][0]-potcenter)/(coefsArray[2][0]-potcenter))/log(radii[1]/radii[2]);
        if(gammaInner<0) gammaInner=0; 
        if(gammaInner>2) gammaInner=2;
        // this was the estimate of density slope. Now we need to convert it to the estimate of power-law slope of l>0 coefs
        gammaInner = 2.0-gammaInner;   // the same recipe as used later in initSpline
        double gammaOuter = -1.0;      // don't freak out, assume default value
#ifdef DEBUGPRINT
        my_message(FUNCNAME, 
            "Estimated slope: inner="+convertToString(gammaInner)+", outer="+convertToString(gammaOuter));
#endif
        // init x-coordinates from scaling transformation
        for(size_t p=0; p<=Ncoefs_radial; p++)
            scaledKnotRadii[p] = log(1+radii[p]);
        scaledKnotRadii[Ncoefs_radial+1] = log(1+outerRadiusSpline);
        for(size_t i=0; i<numPointsUsed; i++)
            scaledPointRadii[i] = log(1+pointRadii[i+npointsInnerSpline]);
        math::SplineApprox appr(scaledPointRadii, scaledKnotRadii);
//        if(appr.status()==CSplineApprox::AS_SINGULAR)
//            my_message(FUNCNAME, 
//                "Warning, in Spline potential initialization: singular matrix for least-square fitting; fallback to a slow algorithm without smoothing");
        // loop over l,m
        for(int l=lstep; l<=lmax; l+=lstep)
        {
            for(int m=l*mmin; m<=l*mmax; m+=mstep)
            {
                int coefind=l*(l+1) + m;
                // init matrix of values to fit
                for(size_t i=0; i<numPointsUsed; i++)
                    scaledPointCoefs[i] = pointCoefs[coefind][i+npointsInnerSpline]/pointCoefs[0][i+npointsInnerSpline];// *
                    //pow(std::min<double>(1.0, pointRadii[i+npointsInnerSpline]/radii[2]), gammaInner);    // a trick to downplay wild fluctuations at r->0
                double derivLeft, derivRight;
                double edf=0;  // equivalent number of free parameters in the fit; if it is ~2, fit is oversmoothed to death (i.e. to a linear regression, which means we should ignore it)
                appr.fitDataOversmooth(scaledPointCoefs, smoothfactor, scaledSplineValues, derivLeft, derivRight, NULL, &edf);
                if(edf<3.0)   // in case of error or an oversmoothed fit fallback to zero values
                    scaledSplineValues.assign(Ncoefs_radial+1, 0);
#ifdef DEBUGPRINT
                my_message(FUNCNAME, 
                    "l="+convertToString(l)+", m="+convertToString(m)+" - EDF="+convertToString(edf));
#endif
                // now store fitted values in coefsArray to pass to initspline routine
                coefsArray[0][coefind] = 0;  // unused
                for(size_t c=1; c<=Ncoefs_radial; c++)
                    coefsArray[c][coefind] = scaledSplineValues[c] * coefsArray[c][0];  // scale back (multiply by l=0,m=0 coefficient)
                // correction to avoid fluctuation at first and last grid radius
                if( coefsArray[1][coefind] * coefsArray[2][coefind] < 0 || coefsArray[1][coefind]/coefsArray[2][coefind] > pow(radii[1]/radii[2], gammaInner))
                    coefsArray[1][coefind] = coefsArray[2][coefind] * pow(radii[1]/radii[2], gammaInner);   // make the smooth curve drop to zero at least as fast as gammaInner'th power of radius
                if( coefsArray[Ncoefs_radial][coefind] * coefsArray[Ncoefs_radial-1][coefind] < 0 || 
                    coefsArray[Ncoefs_radial][coefind] / coefsArray[Ncoefs_radial-1][coefind] > pow(radii[Ncoefs_radial]/radii[Ncoefs_radial-1], gammaOuter))
                    coefsArray[Ncoefs_radial][coefind] = coefsArray[Ncoefs_radial-1][coefind] * pow(radii[Ncoefs_radial]/radii[Ncoefs_radial-1], gammaOuter);
            }
        }
    }
    initSpline(radii, coefsArray);
}

void SplineExp::checkSymmetry(const std::vector< std::vector<double> > &coefsArray)
{ 
    SymmetryType sym=ST_SPHERICAL;  // too optimistic:))
    const double MINCOEF=1e-8;   // if ALL coefs of a certain subset of indices are below this value, assume some symmetry
    for(size_t n=0; n<=Ncoefs_radial; n++)
    {
        for(int l=0; l<=(int)Ncoefs_angular; l++)
            for(int m=-l; m<=l; m++)
                if(fabs(coefsArray[n][l*(l+1)+m])>MINCOEF) 
                {   // nonzero coef.: check if that breaks any symmetry
                    if(l%2==1)  sym = (SymmetryType)(sym & ~ST_REFLECTION);
                    if(m<0 || m%2==1)  sym = (SymmetryType)(sym & ~ST_PLANESYM);
                    if(m!=0) sym = (SymmetryType)(sym & ~ST_ZROTSYM);
                    if(l>0) sym = (SymmetryType)(sym & ~ST_SPHSYM);
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
    while(nskip+1<_coefsArray.size() && _coefsArray[nskip+1][0]==potcenter) nskip++;
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
    if(math::isFinite(Kout)) {
        FindGammaOut fout(radii[Ncoefs_radial], radii[Ncoefs_radial-1], radii[Ncoefs_radial-2], Kout);
        gammaout = math::findRoot(fout, 3.01, 10., math::ACCURACY_ROOT);
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
    double B = math::findRoot(fin, 0, 0.9/radii[3], math::ACCURACY_ROOT);
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
#ifdef DEBUGPRINT
    my_message(FUNCNAME, "gammain="+convertToString(gammain)+
        " ("+convertToString(gammainuncorr)+");  gammaout="+convertToString(gammaout));
#endif
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
    // next init all higher-order splines which have radial scaling log(a+r) and value scaled to l=0,m=0 coefficient
    const double ascale=1.0;
    for(size_t i=0; i<Ncoefs_radial; i++)
        spnodes[i]=log(ascale+gridradii[i+1]);
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
            if(gsl_isnan(slopein[coefind])) slopein[coefind]=1.0;  // default
            slopein[coefind] = std::max<double>(slopein[coefind], std::min<double>(l, 2-gammain));  // the asymptotic power-law behaviour of the coefficient expected for power-law density profile
            derivLeft = spvalues[0] * (1+ascale/minr) * (slopein[coefind] - minr*C00der/C00val);   // derivative at innermost node
            slopeout[coefind] = log(coefsArray[Ncoefs_radial][coefind]/coefsArray[Ncoefs_radial-1][coefind]) / log(gridradii[Ncoefs_radial]/gridradii[Ncoefs_radial-1]) + 1;   // estimate slope of Clm(r)/C00(r) at r->infinity (+1 is added because C00(r) ~ 1/r at large r)
            if(gsl_isnan(slopeout[coefind])) slopeout[coefind]=-1.0;  // default
            slopeout[coefind] = std::min<double>(slopeout[coefind], std::max<double>(-l, 3-gammaout));
            derivRight = spvalues[Ncoefs_radial-1] * (1+ascale/maxr) * slopeout[coefind];   // derivative at outermost node
            splines[coefind] = math::CubicSpline(spnodes, spvalues, derivLeft, derivRight);
#ifdef DEBUGPRINT
            my_message(FUNCNAME, "l="+convertToString(l)+", m="+convertToString(m)+
                " - inner="+convertToString(slopein[coefind])+", outer="+convertToString(slopeout[coefind]));
#endif
        }
    }
#if 0
    bool densityNonzero = checkDensityNonzero();
    bool massMonotonic  = checkMassMonotonic();
    if(!massMonotonic || !densityNonzero) 
        my_message(FUNCNAME, "Warning, " + 
        std::string(!massMonotonic ? "mass does not monotonically increase with radius" : "") +
        std::string(!massMonotonic && !densityNonzero ? " and " : "") + 
        std::string(!densityNonzero ? "density drops to zero at a finite radius" : "") + "!");
#endif
}

void SplineExp::getCoefs(std::vector<double> *radii, std::vector< std::vector<double> > *coefsArray, bool useNodes) const
{
    if(radii==NULL || coefsArray==NULL) return;
    if(useNodes)
    {
        radii->resize(Ncoefs_radial+1);
        for(size_t i=0; i<=Ncoefs_radial; i++)
            (*radii)[i] = gridradii[i];
    }
    size_t numrad=radii->size();
    coefsArray->resize(numrad);
    for(size_t i=0; i<numrad; i++)
    {
        double rad=(*radii)[i];
        double xi=log(1+rad);
        double Coef00;
        coef0(rad, &Coef00, NULL, NULL);
        (*coefsArray)[i].assign(pow_2(Ncoefs_angular+1), 0);
        (*coefsArray)[i][0] = Coef00;
        for(int l=lstep; l<=lmax; l+=lstep)
            for(int m=l*mmin; m<=l*mmax; m+=mstep)
            {
                int coefind=l*(l+1)+m;
                coeflm(coefind, rad, xi, &((*coefsArray)[i][l*(l+1)+m]), NULL, NULL, Coef00);
            }
    }
}

void SplineExp::coeflm(unsigned int lm, double r, double xi, double *val, double *der, double *der2, double c0val, double c0der, double c0der2) const  // works only for l2>0
{
    const double ascale=1.0;
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
    const double ascale=1.0;
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
