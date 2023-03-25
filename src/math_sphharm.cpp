#include "math_sphharm.h"
#include "math_core.h"
#include "math_specfunc.h"
#include <cmath>
#include <cassert>
#include <stdexcept>
#ifndef _MSC_VER
#include <alloca.h>
#else
#include <malloc.h>
#endif

namespace math{

/** Calculate P_m^m(theta) from the analytic result:
    P_m^m(theta) = (-1)^m (2m-1)!! (sin(theta))^m , m > 0 ;
                 = 1 , m = 0 .
    store the pre-factor sqrt[ (2*m+1) / (4 pi (2m)!) ] in prefact,
    the value of Pmm in val, and optionally its first/second derivative w.r.t theta
    in der/der2 if they are not NULL.
*/
inline void legendrePmm(int m, double costheta, double sintheta, 
    double& prefact, double* value, double* der, double* der2)
{
    const int MMAX = 16;   // # of pre-computed coefs in the tables below for the most common values of m
    const double PREFACT[MMAX+1] = { 0.2820947917738782, 
        0.3454941494713355,    0.1287580673410632,    0.02781492157551894,   0.004214597070904597, 
        0.0004911451888263050, 4.647273819914057e-05, 3.700296470718545e-06, 2.542785532478802e-07,
        1.536743406172476e-08, 8.287860012085477e-10, 4.035298721198747e-11, 1.790656309174350e-12,
        7.299068453727266e-14, 2.751209457796109e-15, 9.643748535232993e-17, 3.159120301003413e-18 };
    const double COEF[MMAX+1] =  { 0.2820947917738782,
        -0.3454941494713355, 0.3862742020231896, -0.4172238236327841, 0.4425326924449826,
        -0.4641322034408582, 0.4830841135800662, -0.5000395635705506, 0.5154289843972843,
        -0.5295529414924496, 0.5426302919442215, -0.5548257538066191, 0.5662666637421912,
        -0.5770536647012670, 0.5872677968601020, -0.5969753602424046, 0.6062313441538353 };

    prefact = m<=MMAX ? PREFACT[m] : 0.5/M_SQRTPI * sqrt( (2*m+1) / factorial(2*m) );
    if(m == 0) {
        if(der)
            *der = 0;
        if(der2)
            *der2= 0;
        *value   = prefact;
        return;
    }
    if(m == 1) {
        if(der)
            *der = -costheta * prefact;
        if(der2)
            *der2=  sintheta * prefact;
        *value   = -sintheta * prefact;
        return;
    }
    double coef  = m<=MMAX ? COEF[m] : prefact * dfactorial(2*m-1) * (m%2 == 1 ? -1 : 1);
    double sinm2 = math::pow(sintheta, m-2);
    if(der)
        *der = m * coef * sinm2 * sintheta * costheta;
    if(der2)
        *der2= m * coef * sinm2 * (m * pow_2(costheta) - 1);
    *value   =     coef * sinm2 * pow_2(sintheta);
}

void sphHarmArray(const unsigned int lmax, const unsigned int m, const double tau,
    double* resultArray, double* derivArray, double* deriv2Array)
{
    if(m>lmax || resultArray==NULL || (deriv2Array!=NULL && derivArray==NULL))
        throw std::domain_error("Invalid parameters in sphHarmArray");
    if(lmax==0) {
        resultArray[0] = 0.5/M_SQRTPI;
        if(derivArray)
            derivArray[0] = 0;
        if(deriv2Array)
            deriv2Array[0] = 0;
        return;
    }
    const double ct =      2 * tau  / (1 + tau*tau);  // cos(theta)
    const double st = (1 - tau*tau) / (1 + tau*tau);  // sin(theta)
    double prefact; // will be initialized by legendrePmm
    legendrePmm(m, ct, st, prefact, resultArray, derivArray, deriv2Array);
    if(lmax == m)
        return;

    // values of two previous un-normalized polynomials needed in the recurrent relation
    double Plm1 = resultArray[0] / prefact, Plm = ct * (2*m+1) * Plm1, Plm2 = 0;
    // values of 2nd derivatives of un-normalized polynomials needed for the special case
    // m==1 and st<<1, since we need another recurrent relation for computing 2nd derivative -
    // the usual formula suffers from cancellation
    double d2Plm1 = st, d2Plm2 = 0, d2Plm = 12 * ct * st;
    // threshold in sin(theta) for applying asymptotic expressions for derivatives
    const double EPS = 1e-8;

    for(int l=m+1; l<=(int)lmax; l++) {
        unsigned int ind = l-m;  // index in the output array
        if(l>(int)m+1)  // skip first iteration which was assigned above
            Plm = (ct * (2*l-1) * Plm1 - (l+m-1) * Plm2) / (l-m);  // use recurrence for the rest
        prefact *= sqrt( (2*l+1.) / (2*l-1.) * (l-m) / (l+m) );
        resultArray[ind] = Plm * prefact;
        if(derivArray) {
            double dPlm = 0;
            if(st >= EPS || (m>2 && st>0))
                dPlm = (l * ct * Plm - (l+m) * Plm1) / st;
            else if(m==0)
                dPlm = -l*(l+1)/2 * st * (ct>0 || l%2==1 ? 1 : -1);
            else if(m==1)
                dPlm = -l*(l+1)/2 * (ct>0 || l%2==0 ? 1 : -1);
            else if(m==2)
                dPlm = l*(l+1)*(l+2)*(l-1)/4 * st * (ct>0 || l%2==1 ? 1 : -1);
            derivArray[ind] = prefact * dPlm;
        }
        if(deriv2Array!=NULL) {
            if(st >= EPS || (m>2 && st>0))
                deriv2Array[ind] = ct * derivArray[ind] / (-st) - (l*(l+1)-pow_2(m/st)) * resultArray[ind];
            else if(m==0)
                deriv2Array[ind] = -l*(l+1)/2 * prefact * (ct>0 || l%2==0 ? 1 : -1);
            else if(m==1) {
                if(l>(int)m+1) {
                    double twodPlm1 = -l*(l-1) * (ct>0 || l%2==1 ? 1 : -1);
                    d2Plm = ( (2*l-1) * (ct * (d2Plm1 - Plm1) - st * twodPlm1) - l * d2Plm2) / (l-1);
                }
                deriv2Array[ind] = prefact * d2Plm;
                d2Plm2 = d2Plm1;
                d2Plm1 = d2Plm;
            }
            else if(m==2)
                deriv2Array[ind] = l*(l+1)*(l+2)*(l-1)/4 * prefact * (ct>0 || l%2==0 ? 1 : -1);
            else
                deriv2Array[ind] = 0;
        }
        Plm2 = Plm1;
        Plm1 = Plm;
    }
}

void trigMultiAngle(const double phi, const unsigned int m, const bool needSine, double* outputArray)
{
    if(m<1)
        return;
    // accurate recurrence relation from section 5.4 of Num.Rec.3rd ed.
    double alpha, beta, sinphi, cosphi, sinphi1=0, cosphi1=1;
    sincos(phi, sinphi, cosphi);
    sincos(phi/2, alpha, beta);
    alpha *= alpha*2;
    beta = sinphi;
    for(unsigned int k=0; k<m; k++) {
        cosphi = cosphi1 - (alpha * cosphi1 + beta * sinphi1);
        sinphi = sinphi1 - (alpha * sinphi1 - beta * cosphi1);
        outputArray[k] = cosphi;
        if(needSine) outputArray[k+m] = sinphi;
        cosphi1 = cosphi;
        sinphi1 = sinphi;
    }
}

// ------ indexing scheme for spherical harmonics, encoding its symmetry properties ------ //

SphHarmIndices::SphHarmIndices(int _lmax, int _mmax, coord::SymmetryType _sym) :
    lmax(_lmax), mmax(_mmax),
    step(isZReflSymmetric(_sym) || isReflSymmetric(_sym) ? 2 : 1),
    sym(_sym)
{
    if(lmax<0 || mmax<0 || mmax>lmax)
        throw std::invalid_argument("SphHarmIndices: incorrect indexing scheme requested");
    if(isUnknown(sym))
        throw std::invalid_argument("SphHarmIndices: symmetry is not specified");
    // consistency check: if three plane symmetries are present, mirror symmetry is implied
    if(isXReflSymmetric(sym) && isYReflSymmetric(sym) && isZReflSymmetric(sym) && !isReflSymmetric(sym))
        throw std::invalid_argument("SphHarmIndices: invalid symmetry requested");
    if(mmax==0)
        sym = static_cast<coord::SymmetryType>
            (sym | coord::ST_ZROTATION | coord::ST_XREFLECTION | coord::ST_YREFLECTION);
    if(lmax==0)
        sym = static_cast<coord::SymmetryType>
            (sym | coord::ST_ROTATION | coord::ST_ZREFLECTION | coord::ST_REFLECTION);
    // fill the lmin array
    lmin_arr.resize(2*mmax+1);
    for(int m=-mmax; m<=mmax; m++) {
        int lminm = abs(m);   // by default start from the very first coefficient
        if(isReflSymmetric(sym) && m%2!=0)
            lminm = abs(m)+1; // in this case start from the next even l, because step in l is 2
        if( (isYReflSymmetric(sym) && m<0) ||
            (isXReflSymmetric(sym) && ((m<0) ^ (m%2!=0)) ) ||
            (isBisymmetric(sym)    && m%2!=0) )
            lminm = lmax+1;  // don't consider this m at all
        lmin_arr[m+mmax] = lminm;
    }
}

int SphHarmIndices::index_l(unsigned int c) 
{
    return (int)sqrt(c);
}

int SphHarmIndices::index_m(unsigned int c)
{
    int l=index_l(c);
    return (int)c-l*(l+1);
}

SphHarmIndices getIndicesFromCoefs(const std::vector<double> &C)
{
    int lmax = (int)sqrt((double)C.size())-1;
    if(lmax<0 || (int)C.size() != pow_2(lmax+1))
        throw std::invalid_argument("getIndicesFromCoefs: invalid size of coefs array");
    int sym  = coord::ST_SPHERICAL;
    int mmax = 0;
    for(unsigned int c=0; c<C.size(); c++) {
        if(!isFinite(C[c]))
            throw std::domain_error("getIndicesFromCoefs: coefficient not finite");
        if(C[c]!=0) {  // nonzero coefficient may break some of the symmetries, depending on l,m
            int l = SphHarmIndices::index_l(c);
            int m = SphHarmIndices::index_m(c);
            if(l%2 == 1)
                sym &= ~coord::ST_REFLECTION;
            if(m<0)
                sym &= ~coord::ST_YREFLECTION;
            if((l+m)%2 == 1)
                sym &= ~coord::ST_ZREFLECTION;
            if((m<0) ^ (m%2 != 0))
                sym &= ~coord::ST_XREFLECTION;
            if(m!=0) {
                sym &= ~coord::ST_ZROTATION;
                if(abs(m)>mmax)
                    mmax = abs(m);
            }
            if(l>0)
                sym &= ~coord::ST_ROTATION;
        }
    }
    return math::SphHarmIndices(lmax, mmax, static_cast<coord::SymmetryType>(sym));
}

std::vector<int> getIndicesAzimuthal(int mmax, coord::SymmetryType sym)
{
    if(isUnknown(sym))
        throw std::invalid_argument("getIndicesAzimuthal: symmetry is not specified");
    if(mmax<0)
        throw std::invalid_argument("getIndicesAzimuthal: mmax should be non-negative");
    std::vector<int> result(1, 0);  // m=0 is always present
    if((sym & coord::ST_ZROTATION) == coord::ST_ZROTATION)
        return result;  // in this case all m!=0 indices are zero
    for(int m=1; m<=mmax; m++) {
        // odd-m indices are excluded under the combination of z-reflection and mirror symmetry
        if(isBisymmetric(sym) && m%2 != 0)
            continue;
        bool addplusm = true, addminusm = true;
        // in case of y-reflection, only m>=0 indices are present
        if(isYReflSymmetric(sym))
            addminusm = false;
        // in case of x-reflection, negative-even and positive-odd indices are zero
        if(isXReflSymmetric(sym)) {
            if(m%2==1)
                addplusm = false;
            else
                addminusm = false;
        }
        if(addminusm)
            result.push_back(-m);
        if(addplusm)
            result.push_back(m);
    }
    return result;
}

// ------ classes for performing many transformations with identical setup ------ //

FourierTransformForward::FourierTransformForward(int _mmax, bool _useSine) :
    mmax(_mmax), useSine(_useSine)
{
    if(mmax<0)
        throw std::invalid_argument("FourierTransformForward: mmax must be non-negative");
    const int nphi = mmax+1;  // number of nodes in uniform grid in phi
    const int nfnc = useSine ? mmax*2+1 : mmax+1;  // number of trig functions for each phi-node
    trigFnc.resize(nphi * nfnc);
    // weight of a single value in uniform integration over phi
    double weight = M_PI / (mmax+0.5);
    // compute the values of trigonometric functions at nodes of phi-grid for all 0<=m<=mmax:
    // cos(m phi_k), and optionally sin(m phi_k) if terms with m<0 are non-trivial
    for(int k=0; k<nphi; k++) {
        trigFnc[k*nfnc] = 1.;  // cos(0*phi[k])
        if(mmax>0)
            trigMultiAngle(phi(k), mmax, useSine, &trigFnc[k*nfnc+1]);
        // if not using sines, then the grid in phi is 0 = phi_0 < ... < phi_{mmax} < pi,
        // so that all nodes except 0th should count twice.
        for(int m=0; m<nfnc; m++)
            trigFnc[k*nfnc+m] *= weight * (useSine || k==0 ? 1 : 2);
    }
}

void FourierTransformForward::transform(const double values[], double coefs[], int stride) const
{
    const int nfnc = useSine ? mmax*2+1 : mmax+1;  // number of trig functions for each phi-node
    for(int mm=0; mm<nfnc; mm++) {  // index in the output array
        coefs[mm] = 0;
        int m = useSine ? mm-mmax : mm;  // if use sines, m runs from -mmax to mmax
        for(int k=0; k<nfnc; k++) {
            int indphi = k<=mmax ? k : 2*mmax+1-k;  // index of angle phi_k is between 0 and mmax
            int indfnc = m>=0 ? m : mmax-m;  // index of trig function is between 0 and mmax or 2mmax
            double fnc = trigFnc[indphi*nfnc + indfnc];
            if(m<0 && k>mmax)  // sin(2pi-phi) = -sin(phi)
                fnc*=-1;
            coefs[mm] += fnc * values[k*stride];
        }
    }
}

// index of Legendre function P_{lm}(theta_j) in the `legFnc` array
inline unsigned int indLeg(const SphHarmIndices& ind, int j, int l, int m)
{
    int ntheta = ind.lmax/2+1;
    int nlegfn = (ind.lmax+1) * (ind.mmax+1);
    int absm = abs(m);
    int absj = j<ntheta ? j : ind.lmax-j;
    return absj * nlegfn + absm * (ind.lmax+1) + l;
}

SphHarmTransformForward::SphHarmTransformForward(const SphHarmIndices& _ind):
    ind(_ind),
    fourier(ind.mmax, ind.mmin()<0)
{
    int ngrid  = ind.lmax+1;    // # of nodes of GL grid on [-1:1]
    int ntheta = ind.lmax/2+1;  // # of theta values to compute Plm
    int nlegfn = (ind.lmax+1) * (ind.mmax+1);  // # of Legendre functions for each theta-node

    legFnc.resize(ntheta * nlegfn);
    costhnodes.resize(ngrid);
    // obtain nodes and weights of Gauss-Legendre quadrature of degree lmax+1 on [-1:1] for cos(theta)
    std::vector<double> nodes(ngrid), weights(ngrid);
    prepareIntegrationTableGL(-1, 1, ngrid, &nodes.front(), &weights.front());
    // compute the values of associated Legendre functions at nodes of theta-grid
    for(int j=0; j<ngrid; j++) {  // loop over nodes of theta-grid
        costhnodes[j] = nodes[ngrid-1-j];
        if(j>=ntheta)  // don't consider nodes with theta>pi/2, 
            continue;  // as the values of Plm for them are known from symmetry properties
        // loop over m and compute all functions of order up to lmax for each m
        for(int m=0; m<=ind.mmax; m++) {
            double tau = costhnodes[j] / (sqrt(1 - pow_2(costhnodes[j])) + 1);
            sphHarmArray(ind.lmax, m, tau, &legFnc[indLeg(ind, j, m, m)]);
            // multiply the values of all Legendre functions at theta[i]
            // by the weight of this node in GL quadrature and by additional prefactor
            for(int l=m; l<=ind.lmax; l++)
                legFnc[indLeg(ind, j, l, m)] *= weights[j] * (0.5/M_SQRTPI) * (m>0 ? M_SQRT2 : 1);
        }
    }
}

void SphHarmTransformForward::transform(const double values[], double coefs[], int stride) const
{
    if(ind.size() == 1) {
        // shortcut in the spherically-symmetric case
        coefs[0] = values[0];
        return;
    }
    if(stride<=0)
        throw std::invalid_argument("stride must be positive");
    std::fill(coefs, coefs+ind.size(), 0.);
    int mmin  = ind.mmin();
    assert(mmin == 0 || mmin == -ind.mmax);
    int ngrid = ind.lmax+1;      // # of nodes of GL grid for integration in theta on (0:pi)
    int nfour = ind.mmax-mmin+1; // # of Fourier harmonic terms - either mmax+1 or 2*mmax+1
    int nsamp = thetasize();     // # of samples taken in theta (is less than ngrid in case of z-symmetry)
    // azimuthal Fourier coefficients F_jm for each value of theta_j and m.
    // indexing scheme: val_m[ j * nfour + m+ind.mmax ] = F_jm, 0<=j<nsamp, -mmax<=m<=mmax.
    // allocate the temp.array on the stack, will be automatically freed on exit
    double* val_m = static_cast<double*>(alloca(thetasize() * nfour * sizeof(double)));

    // first step: perform integration in phi for each value of theta, using Fourier transform
    for(unsigned int j=0; j<thetasize(); j++)
        fourier.transform( &values[j * fourier.size() * stride], &val_m[j * nfour], stride );

    // second step: perform integration in theta for each m
    for(int m=ind.mmin(); m<=ind.mmax; m++) {
        for(int l=ind.lmin(m); l<=ind.lmax; l+=ind.step) {
            unsigned int c = ind.index(l, m);
            for(int j=0; j<ngrid; j++) {
                // take the sample at |z| if z<0 and have z-reflection symmetry
                unsigned int jsamp = j<nsamp ? j : ngrid-1-j;
                double Plm = legFnc[indLeg(ind, j, l, m)];
                if( (l+m)%2 == 1 && j>ind.lmax/2 )
                    Plm *= -1;   // Plm(-x) = (-1)^{l+m) Plm(x), here x=cos(theta) and theta > pi/2
                coefs[c] += Plm * val_m[ jsamp * nfour + m-mmin ];
            }
        }
    }
}

double sphHarmTransformInverse(const SphHarmIndices& ind, const double coefs[],
    const double tau, const double phi)
{
    // here we create a temporary array on the stack, without dynamic memory allocation.
    // it will be automatically freed upon return from this routine, just as any local stack variable.
    // An alternative is to use a statically sized array with some maximum size set at compile time...
    int size = 2*ind.mmax + ind.lmax + 1;
    double* tmptrig = static_cast<double*>(alloca(size * sizeof(double)));
    double* tmpleg  = tmptrig + 2*ind.mmax;  // part of the temporary array for the Legendre transform
    const bool useSine = ind.mmin()<0;
    if(ind.mmax>0)
        trigMultiAngle(phi, ind.mmax, useSine, tmptrig);
    double result = 0;
    for(int m=ind.mmin(); m<=ind.mmax; m++) {
        int lmin = ind.lmin(m);
        if(lmin>ind.lmax)
            continue;  // empty m-harmonic
        int absm = abs(m);
        double trig = m==0 ?       (2*M_SQRTPI) : // extra numerical factors from the definition of sph.harm.
            m>0 ? tmptrig[m-1]   * (2*M_SQRTPI * M_SQRT2) :
            tmptrig[ind.mmax-m-1]* (2*M_SQRTPI * M_SQRT2);
        sphHarmArray(ind.lmax, absm, tau, tmpleg);
        for(int l=lmin; l<=ind.lmax; l+=ind.step) {
            double leg = tmpleg[l-absm];
            result += coefs[ind.index(l, m)] * leg * trig;
        }
    }
    return result;
}

}  // namespace math