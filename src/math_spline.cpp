#include "math_spline.h"
#include "math_core.h"
#include "math_fit.h"
#include "utils.h"
#include <cmath>
#include <algorithm>
#include <cassert>
#include <stdexcept>

namespace math {

// ------- Some machinery for B-splines ------- //
namespace {

/// linear interpolation on a grid with a special treatment for indices outside the grid
static inline double linInt(const double x, const double grid[], int size, int i1, int i2)
{
    double x1 = grid[i1<0 ? 0 : i1>=size ? size-1 : i1];
    double x2 = grid[i2<0 ? 0 : i2>=size ? size-1 : i2];
    if(x1==x2) {
        return x==x1 ? 1 : 0;
    } else {
        return (x-x1) / (x2-x1);
    }
}

/** Compute the values of B-spline functions used for 1d interpolation.
    For any point inside the grid, at most N+1 basis functions are non-zero out of the entire set
    of (N_grid+N-1) basis functions; this routine reports only the nontrivial ones.
    \tparam N   is the degree of spline basis functions;
    \param[in]  x  is the input position on the grid;
    \param[in]  grid  is the array of grid nodes;
    \param[in]  size  is the length of this array;
    \param[out] B  are the values of N+1 possibly nonzero basis functions at this point,
    if the point is outside the grid then all values are zeros;
    \return  the index of the leftmost out of N+1 nontrivial basis functions.
*/
template<int N>
static inline int bsplineValues(const double x, const double grid[], int size, double B[])
{
    const int ind = binSearch(x, grid, size);
    if(ind<0 || ind>=size) {
        for(int i=0; i<=N; i++)
            B[i] = 0;
        return 0;
    }

    // de Boor's algorithm:
    // 0th degree basis functions are all zero except the one on the grid segment `ind`
    for(int i=0; i<=N; i++)
        B[i] = i==N ? 1 : 0;
    for(int l=1; l<=N; l++) {
        double Bip1=0;
        for(int i=N, j=ind; j>=ind-l; i--, j--) {
            double Bi = B[i] * linInt(x, grid, size, j, j+l)
                      + Bip1 * linInt(x, grid, size, j+l+1, j+1);
            Bip1 = B[i];
            B[i] = Bi;
        }
    }
    return ind;
}

/// subexpression in b-spline derivatives (inverse distance between grid nodes or 0 if they coincide)
static inline double denom(const double grid[], int size, int i1, int i2)
{
    double x1 = grid[i1<0 ? 0 : i1>=size ? size-1 : i1];
    double x2 = grid[i2<0 ? 0 : i2>=size ? size-1 : i2];
    return x1==x2 ? 0 : 1 / (x2-x1);
}

/// recursive template definition for B-spline derivatives through B-spline derivatives
/// of lower degree and order;
/// the arguments are the same as for `bsplineValues`, and `order` is the order of derivative.
template<int N, int Order>
static inline int bsplineDerivs(const double x, const double grid[], int size, double B[])
{
    int ind = bsplineDerivs<N-1, Order-1>(x, grid, size, B+1);
    B[0] = 0;
    for(int i=0, j=ind-N; i<=N; i++, j++) {
        B[i] = N * (B[i]   * denom(grid, size, j, j+N)
            + (i<N? B[i+1] * denom(grid, size, j+N+1, j+1) : 0) );
    }
    return ind;
}

/// the above recursion terminates when the order of derivative is zero, returning B-spline values;
/// however, C++ rules do not permit to declare a partial function template specialization
/// (for arbitrary N and order 0), therefore we use full specializations for several values of N
template<>
inline int bsplineDerivs<0,0>(const double x, const double grid[], int size, double B[]) {
    return bsplineValues<0>(x, grid, size, B);
}
template<>
inline int bsplineDerivs<1,0>(const double x, const double grid[], int size, double B[]) {
    return bsplineValues<1>(x, grid, size, B);
}
template<>
inline int bsplineDerivs<2,0>(const double x, const double grid[], int size, double B[]) {
    return bsplineValues<2>(x, grid, size, B);
}
template<>
inline int bsplineDerivs<3,0>(const double x, const double grid[], int size, double B[]) {
    return bsplineValues<3>(x, grid, size, B);
}
template<>
inline int bsplineDerivs<0,1>(const double, const double[], int, double[]) {
    assert(!"Should not be called");
    return 0;
}
template<>
inline int bsplineDerivs<0,2>(const double, const double[], int, double[]) {
    assert(!"Should not be called");
    return 0;
}

/** Similar to bsplineValues, but uses linear extrapolation outside the grid domain */
template<int N>
static inline int bsplineValuesExtrapolated(const double x, const double grid[], int size, double B[])
{
    double x0 = fmax(grid[0], fmin(grid[size-1], x));
    int ind = bsplineValues<N>(x0, grid, size, B);
    if(x != x0) {   // extrapolate using the derivatives
        double D[N+1];
        bsplineDerivs<N,1>(x0, grid, size, D);
        for(int i=0; i<=N; i++)
            B[i] += D[i] * (x-x0);
    }
    return ind;
}

/** Compute the matrix of overlap integrals for the array of 1d B-spline functions or their derivs.
    Let N>=1 be the degree of B-splines, and D - the order of derivative in question.
    There are numBasisFnc = numKnots+N-1 basis functions B_p(x) on the entire interval spanned by knots,
    and each of them is nonzero on at most N+1 consecutive sub-intervals between knots.
    Define the matrix M_{pq}, 0<=p<=q<numBasisFnc, to be the symmetric matrix of overlap integrals:
    \f$  M_{pq} = \int dx B^(D)_p(x) B^(D)_q(x)  \f$, where the integrand is nonzero on at most q-p+N+1 
    consecutive sub-intervals, and B^(D) is the D'th derivative of the corresponding function.
*/
template<int N, int D>
static Matrix<double> computeOverlapMatrix(const std::vector<double> &knots)
{
    int numKnots = knots.size(), numBasisFnc = numKnots+N-1;
    // B-spline of degree N is a polynomial of degree N, so its D'th derivative is a polynomial
    // of degree N-D. To compute the integral of a product of two such functions over a sub-interval,
    // it is sufficient to employ a Gauss-Legendre quadrature rule with the number of nodes = N-D+1.
    const int Nnodes = std::max<int>(N-D+1, 0);
    double glnodes[Nnodes], glweights[Nnodes];
    prepareIntegrationTableGL(0, 1, Nnodes, glnodes, glweights);

    // Collect the values of all possibly non-zero basis functions (or their D'th derivatives)
    // at Nnodes points of each sub-interval between knots. There are at most N+1 such non-zero functions,
    // so these values are stored in a 2d array [N+1] x [number of subintervals * number of GL nodes].
    Matrix<double> values(N+1, (numKnots-1)*Nnodes);
    for(int k=0; k<numKnots-1; k++) {
        double der[N+1];
        for(int n=0; n<Nnodes; n++) {
            // evaluate the possibly non-zero functions and keep track of the index of the leftmost one
            int ind = bsplineDerivs<N, D> ( knots[k] + (knots[k+1] - knots[k]) * glnodes[n],
                &knots.front(), numKnots, der);
            for(int b=0; b<=N; b++)
                values(b, k*Nnodes+n) = der[b+k-ind];
        }
    }

    // evaluate overlap integrals and store them in the symmetric matrix M_pq, which is a banded matrix
    // with nonzero values only within N+1 cells from the diagonal
    Matrix<double> mat(numBasisFnc, numBasisFnc, 0);
    for(int p=0; p<numBasisFnc; p++) {
        int kmin = std::max<int>(p-N, 0);   // index of leftmost knot of the integration sub-intervals
        int kmax = std::min<int>(p+1, numKnots-1);     // same for the rightmost
        int qmax = std::min<int>(p+N+1, numBasisFnc);  // max index of the column of the banded matrix
        for(int q=p; q<qmax; q++) {
            double result = 0;
            // loop over sub-intervals where the integrand might be nonzero
            for(int k=kmin; k<kmax; k++) {
                double dx = knots[k+1]-knots[k];
                // loop over nodes of GL quadrature rule over the sub-interval
                for(int n=0; n<Nnodes; n++) {
                    double P = p>=k && p<=k+N ? values(p-k, k*Nnodes+n) : 0;
                    double Q = q>=k && q<=k+N ? values(q-k, k*Nnodes+n) : 0;
                    result  += P * Q * glweights[n] * dx;
                }
            }
            mat(p, q) = result;
            mat(q, p) = result;  // it is symmetric
        }
    }
    return mat;
}

#if 0
/** Convert the weighted combination of 1d B-spline functions of degree N=3
    \f$  f(x) = \sum_{i=0}^{K+1}  A_i  B_i(x)  \f$,  defined by grid nodes X[k] (0<=k<K),
    into the input data for an ordinary clamped cubic spline:
    the values of f(x) at grid knots plus two endpoint derivatives.
    \param[in]  ampl  is the array of K+2 amplitudes of each B-spline basis function;
    \param[in]  grid  is the array of K grid nodes;
    \param[out] fvalues  will contain the values of f at the grid nodes;
    \param[out] derivLeft, derifRight  will contain the two derivatives at X[0] and X[K-1].
    \throws  std::invalid_argument exception if the input grid is not in ascending order
    or does not match the size of amplitudes array.
*/
static void convertToCubicSpline(const std::vector<double> &ampl, const std::vector<double> &grid,
    std::vector<double> &fvalues, double &derivLeft, double &derivRight)
{
    unsigned int numKnots = grid.size();
    if(ampl.size() != numKnots+2)
        throw std::invalid_argument("convertToCubicSpline: ampl.size() != grid.size()+2");
    fvalues.assign(numKnots, 0);
    for(unsigned int k=0; k<numKnots; k++) {
        if(k>0 && grid[k]<=grid[k-1])
            throw std::invalid_argument("convertToCubicSpline: grid nodes must be in ascending order");
        // for any x, at most 4 basis functions are non-zero, starting from ind
        double Bspl[4];
        int ind = bsplineValues<3>(grid[k], &grid[0], numKnots, Bspl);
        double val=0;
        for(int p=0; p<=3; p++)
            val += Bspl[p] * ampl[p+ind];
        fvalues[k] = val;
        if(k==0 || k==numKnots-1) {  // at endpoints also compute derivatives
            bsplineDerivs<3,1>(grid[k], &grid[0], numKnots, Bspl);
            double der=0;
            for(int p=0; p<=3; p++)
                der += Bspl[p] * ampl[p+ind];
            if(k==0)
                derivLeft = der;
            else
                derivRight = der;
        }
    }
}
#endif

/// definite integral of x^(m+n)
class MonomialIntegral: public IFunctionIntegral {
    const int n;
public:
    MonomialIntegral(int _n) : n(_n) {};
    virtual double integrate(double x1, double x2, int m=0) const {
        return m+n+1==0 ? log(x2/x1) : (powInt(x2, m+n+1) - powInt(x1, m+n+1)) / (m+n+1);
    }
};

//---- spline evaluation routines ----//
/// compute the value, derivative and 2nd derivative of (possibly several, K>=1) cubic spline(s);
/// input arguments contain the value(s) and 2nd derivative(s) of these splines
/// at the boundaries of interval [xl..xh] that contain the point x.
template<unsigned int K>
static inline void evalCubicSplines(
    const double xl,   // input:   lower boundary of the interval
    const double xh,   // input:   upper boundary of the interval
    const double x,    // input:   value of x at which the spline is computed (xl <= x <= xu)
    const double* fl,  // input:   f_k(xl)
    const double* fh,  // input:   f_k(xh)
    const double* cl,  // input:   d2f_k(xl)
    const double* ch,  // input:   d2f_k(xh)
    double* f,         // output:  f_k(x)      if f   != NULL
    double* df,        // output:  df_k/dx     if df  != NULL
    double* d2f)       // output:  d^2f_k/dx^2 if d2f != NULL
{
    const double
        h  = xh - xl,
        hi = 1 / h,
        t  = (x-xl) / h,
        T  = 1-t,
        b  = -1./6 * h*h * t * T,
        bh = h * (0.5 * t*t - 1./6),
        bl = h * (t - 0.5) - bh;
    for(unsigned int k=0; k<K; k++) {
        const double c = cl[k] * T + ch[k] * t;
        if(f)
            f[k]   = fl[k] * T  +  fh[k] * t  +  (cl[k] + ch[k] + c) * b;
        if(df)
            df[k]  = (fh[k] - fl[k]) * hi  +  cl[k] * bl  +  ch[k] * bh;
        if(d2f)
            d2f[k] = c;
    }
}

/// compute the value, derivative and 2nd derivative of (possibly several, K>=1) Hermite spline(s);
/// input arguments contain the value(s) and 1st derivative(s) of these splines
/// at the boundaries of interval [xl..xh] that contain the point x.
template<unsigned int K>
static inline void evalHermiteSplines(
    const double xl,   // input:   lower boundary of the interval
    const double xh,   // input:   upper boundary of the interval
    const double x,    // input:   value of x at which the spline is computed (xl <= x <= xu)
    const double* fl,  // input:   f_k (xl)
    const double* fh,  // input:   f_k (xh)
    const double* dl,  // input:   df_k/dx (xl)
    const double* dh,  // input:   df_k/dx (xh)
    double* f,         // output:  f_k(x)      if f   != NULL
    double* df,        // output:  df_k/dx     if df  != NULL
    double* d2f)       // output:  d^2f_k/dx^2 if d2f != NULL
{
    const double
        h      =  xh - xl,
        hi     =  1 / h,
        t      =  (x-xl) / h,  // NOT (x-xl)*hi, because this doesn't always give an exact result if x==xh
        T      =  1-t,
        tq     =  t*t,
        Tq     =  T*T,
        f_fl   =  Tq * (1+2*t),
        f_fh   =  tq * (1+2*T),
        f_dl   =  Tq * (x-xl),
        f_dh   = -tq * (xh-x),
        df_dl  =  T  * (1-3*t),
        df_dh  =  t  * (1-3*T),
        df_dif =  6  * t*T * hi,
        d2f_dl =  (6*t-4) * hi,
        d2f_dh = -(6*T-4) * hi,
        d2f_dif= -(d2f_dl + d2f_dh) * hi;
    for(unsigned int k=0; k<K; k++) {
        const double dif = fh[k] - fl[k];
        if(f)
            f[k]   = dl[k] *   f_dl  +  dh[k] *   f_dh  +  fl[k] * f_fl  +  fh[k] * f_fh;
        if(df)
            df[k]  = dl[k] *  df_dl  +  dh[k] *  df_dh  +  dif *  df_dif;
        if(d2f)
            d2f[k] = dl[k] * d2f_dl  +  dh[k] * d2f_dh  +  dif * d2f_dif;
    }
}

/// compute the value, derivative and 2nd derivative of (possibly several, K>=1) quintic spline(s);
/// input arguments contain the value(s), 1st and 3rd derivative(s) of these splines
/// at the boundaries of interval [xl..xh] that contain the point x.
template<unsigned int K>
static inline void evalQuinticSplines(
    const double xl,   // input:   lower boundary of the interval
    const double xh,   // input:   upper boundary of the interval
    const double x,    // input:   value of x at which the spline is computed (xl <= x <= xu)
    const double* fl,  // input:   f_k(xl)
    const double* fh,  // input:   f_k(xh)
    const double* f1l, // input:   df_k(xl)
    const double* f1h, // input:   df_k(xh)
    const double* f3l, // input:   d3f_k(xl)
    const double* f3h, // input:   d3f_k(xh)
    double* f,         // output:  f_k(x)      if f   != NULL
    double* df,        // output:  df_k/dx     if df  != NULL
    double* d2f,       // output:  d^2f_k/dx^2 if d2f != NULL
    double* d3f=NULL)  // output:  d^3f_k/dx^3 if d3f != NULL
{
    const double
        h   = xh - xl,
        hi  = 1 / h,
        hq  = h * h,
        hiq6= hi * hi * 6,
        B   = (x - xl) / h,
        Bq  = B * B,
        A   = 1 - B,
        Aq  = A * A,
        AB  = A * B,
        C   = h * Aq * B,
        D   =-h * Bq * A,
        E   = hq * C * (2*Aq-A-1) * (1./48),
        F   = hq * D * (2*Bq-B-1) * (1./48),
        Cp  = A  * (A - 2*B),
        Dp  = B  * (B - 2*A),
        Ep  = hq * AB * (1./24) * (1+A-5*Aq),
        Fp  = hq * AB * (1./24) * (1+B-5*Bq),
        Cpp = hi * 2 * (B - 2*A),
        Dpp =-hi * 2 * (A - 2*B),
        Epp = h  * (1./12*Aq * (9*B-A) - 1./24),
        Fpp = h  * (1./12*Bq * (B-9*A) + 1./24);
    for(unsigned int k=0; k<K; k++) {
        double
            d1m = (fh[k] - fl[k]) * hi,
            d1l = f1l[k] - d1m,
            d1h = f1h[k] - d1m,
            d3m = (d1l + d1h) * hiq6,
            d3l = f3l[k] - d3m,
            d3h = f3h[k] - d3m;
        if(f)
            f[k]   = A * fl[k]  +  B * fh[k]  +  C * d1l  +  D * d1h  +  E * d3l  +  F * d3h;
        if(df)
            df[k]  = Cp * f1l[k]  +  Dp * f1h[k]  +  6*AB * d1m  +  Ep * d3l  +  Fp * d3h;
        if(d2f)
            d2f[k] = Cpp * d1l  +  Dpp * d1h  +  Epp * d3l  +  Fpp * d3h;
        if(d3f)
            d3f[k] = (2.5*Aq-1.5*A) * f3l[k]  +  (2.5*Bq-1.5*B) * f3h[k]  +  5*AB * d3m;

    }
}
}  // internal namespace


BaseInterpolator1d::BaseInterpolator1d(const std::vector<double>& xv, const std::vector<double>& fv) :
    xval(xv), fval(fv)
{
    if(xv.size() < 2)
        throw std::invalid_argument("Error in 1d interpolator: number of nodes should be >=2");
    for(unsigned int i=1; i<xv.size(); i++)
        if(xv[i] <= xv[i-1])
            throw std::invalid_argument("Error in 1d interpolator: "
                "x values must be monotonically increasing");
}

LinearInterpolator::LinearInterpolator(const std::vector<double>& xv, const std::vector<double>& yv) :
    BaseInterpolator1d(xv, yv)
{
    if(fval.size() != xval.size())
        throw std::invalid_argument("LinearInterpolator: input arrays are not equal in length");
}

void LinearInterpolator::evalDeriv(const double x, double* value, double* deriv, double* deriv2) const
{
    int i = std::max<int>(0, std::min<int>(xval.size()-2, binSearch(x, &xval[0], xval.size())));
    //TODO: extrapolate linearly, not as a constant, and provide the derivative as well
    if(value)
        *value = linearInterp(x, xval[i], xval[i+1], fval[i], fval[i+1]);
    if(deriv)
        *deriv = 0;
    if(deriv2)
        *deriv2 = 0;
}


//-------------- CUBIC SPLINE --------------//

CubicSpline::CubicSpline(const std::vector<double>& _xval,
    const std::vector<double>& _fval, double der1, double der2) :
    BaseInterpolator1d(_xval, _fval)
{
    unsigned int numPoints = xval.size();
    if(fval.size() == numPoints+2 && der1!=der1 && der2!=der2) {
        // initialize from the amplitudes of B-splines defined at these nodes
        std::vector<double> ampl(_fval);  // temporarily store the amplitudes of B-splines
        fval.assign(numPoints, 0);
        fder2.resize(numPoints);
        for(unsigned int i=0; i<numPoints; i++) {
            // compute values and second derivatives of B-splines at grid nodes
            double val[4], der2[4];
            int ind = bsplineValues<3>(xval[i], &xval[0], numPoints, val);
            bsplineDerivs<3,2>(xval[i], &xval[0], numPoints, der2);
            for(int p=0; p<=3; p++) {
                fval [i] += val [p] * ampl[p+ind];
                fder2[i] += der2[p] * ampl[p+ind];
            }
        }
        return;
    }
    if(fval.size() != numPoints)
        throw std::invalid_argument("CubicSpline: input arrays are not equal in length");
    std::vector<double> rhs(numPoints-2), diag(numPoints-2), offdiag(numPoints-3);
    for(unsigned int i = 1; i < numPoints-1; i++) {
        const double
        dxm = xval[i] - xval[i-1],
        dxp = xval[i+1] - xval[i],
        dym = (fval[i] - fval[i-1]) / dxm,
        dyp = (fval[i+1] - fval[i]) / dxp;
        if(i < numPoints-2)
            offdiag[i-1] = dxp;
        diag[i-1] = 2.0 * (dxp + dxm);
        rhs [i-1] = 6.0 * (dyp - dym);
        if(i == 1 && isFinite(der1)) {
            diag[i-1] = 1.5 * dxm + 2.0 * dxp;
            rhs [i-1] = 6.0 * dyp - 9.0 * dym + 3.0 * der1;
        }
        if(i == numPoints-2 && isFinite(der2)) {
            diag[i-1] = 1.5 * dxp + 2.0 * dxm;
            rhs [i-1] = 9.0 * dyp - 6.0 * dym - 3.0 * der2;
        }
    }

    if(numPoints == 2) {
        fder2.assign(2, 0.);
    } else if(numPoints == 3) {
        fder2.assign(3, 0.);
        fder2[1] = rhs[0] / diag[0];
    } else {
        fder2 = solveTridiag(diag, offdiag, offdiag, rhs);
        fder2.insert(fder2.begin(), 0.);  // for natural cubic spline,
        fder2.push_back(0.);              // 2nd derivatives are zero at endpoints;
    }
    if(isFinite(der1))                    // but for a clamped spline they are not.
        fder2[0] = ( 3 * (fval[1]-fval[0]) / (xval[1]-xval[0]) 
            -3 * der1 - 0.5 * fder2[1] * (xval[1]-xval[0]) ) / (xval[1]-xval[0]);
    if(isFinite(der2))
        fder2[numPoints-1] = (
            -3 * (fval[numPoints-1]-fval[numPoints-2]) / (xval[numPoints-1]-xval[numPoints-2]) 
            +3 * der2 - 0.5 * fder2[numPoints-2] * (xval[numPoints-1]-xval[numPoints-2]) ) /
            (xval[numPoints-1]-xval[numPoints-2]);
}

void CubicSpline::evalDeriv(const double x, double* val, double* deriv, double* deriv2) const
{
    if(xval.size() == 0)
        throw std::range_error("Empty spline");
    if(x < xval.front()) {
        double dx  =  xval[1]-xval[0];
        double der = (fval[1]-fval[0]) / dx - dx * (1./6 * fder2[1] + 1./3 * fder2[0]);
        if(val)
            // if der==0, correct result even for infinite x
            *val   = fval[0] + (der==0 ? 0 : der * (x-xval[0]));
        if(deriv)
            *deriv = der;
        if(deriv2)
            *deriv2= 0;
        return;
    }
    if(x > xval.back()) {
        const unsigned int size = xval.size();
        double dx  =  xval[size-1]-xval[size-2];
        double der = (fval[size-1]-fval[size-2]) / dx + dx * (1./6 * fder2[size-2] + 1./3 * fder2[size-1]);
        if(val)
            *val   = fval[size-1] + (der==0 ? 0 : der * (x-xval[size-1]));
        if(deriv)
            *deriv = der;
        if(deriv2)
            *deriv2= 0;
        return;
    }

    unsigned int index = binSearch(x, &xval.front(), xval.size());
    evalCubicSplines<1> (xval[index], xval[index+1], x,
        &fval[index], &fval[index+1], &fder2[index], &fder2[index+1],
        /*output*/ val, deriv, deriv2);
}

bool CubicSpline::isMonotonic() const
{
    if(xval.size()==0)
        throw std::range_error("Empty spline");
    bool ismonotonic=true;
    for(unsigned int index=0; ismonotonic && index < xval.size()-1; index++) {
        const double
        dx = xval[index + 1] - xval[index],
        dy = fval[index + 1] - fval[index],
        cl = fder2[index],
        ch = fder2[index+1],
        a  = dx * (ch - cl) * 0.5,
        b  = dx * cl,
        c  = (dy / dx) - dx * (1./6 * ch + 1./3 * cl),
        // derivative is  a * t^2 + b * t + c,  with 0<=t<=1 on the given interval.
        D  = b*b-4*a*c;  // discriminant of the above quadratic equation
        if(D>=0) {       // need to check roots
            double chi1 = (-b-sqrt(D)) / (2*a);
            double chi2 = (-b+sqrt(D)) / (2*a);
            if( (chi1>0 && chi1<1) || (chi2>0 && chi2<1) )
                ismonotonic=false;    // there is a root ( y'=0 ) somewhere on the given interval
        }  // otherwise there are no roots
    }
    return ismonotonic;
}

double CubicSpline::integrate(double x1, double x2, int n) const {
    return integrate(x1, x2, MonomialIntegral(n));
}

double CubicSpline::integrate(double x1, double x2, const IFunctionIntegral& f) const
{
    if(x1==x2)
        return 0;
    if(x1>x2)
        return integrate(x2, x1, f);
    double result = 0;
    if(x1 <= xval.front()) {    // spline is linearly extrapolated at x<xval[0]
        double dx  =  xval[1]-xval[0];
        double der = (fval[1]-fval[0]) / dx - dx * (1./6 * fder2[1] + 1./3 * fder2[0]);
        double X2  = fmin(x2, xval.front());
        result +=
            f.integrate(x1, X2, 0) * (fval.front() - der * xval.front()) +
            f.integrate(x1, X2, 1) * der;
        if(x2<=xval.front())
            return result;
        x1 = xval.front();
    }
    if(x2 >= xval.back()) {    // same for x>xval[end]
        unsigned int size = xval.size();
        double dx  =  xval[size-1]-xval[size-2];
        double der = (fval[size-1]-fval[size-2]) / dx + dx * (1./6 * fder2[size-2] + 1./3 * fder2[size-1]);
        double X1  = fmax(x1, xval.back());
        result +=
            f.integrate(X1, x2, 0) * (fval.back() - der * xval.back()) +
            f.integrate(X1, x2, 1) * der;
        if(x1>=xval.back())
            return result;
        x2 = xval.back();
    }
    unsigned int i1 = binSearch(x1, &xval.front(), xval.size());
    unsigned int i2 = binSearch(x2, &xval.front(), xval.size());
    for(unsigned int i=i1; i<=i2; i++) {
        double x  = xval[i];
        double h  = xval[i+1] - x;
        double a  = fval[i];
        double c  = fder2[i];
        double b  = (fval[i+1] - a) / h - h * (1./6 * fder2[i+1] + 1./3 * c);
        double d  = (fder2[i+1] - c) / h;
        // spline(x) = fval[i] + dx * (b + dx * (c/2 + dx*d/6)), where dx = x-xval[i]
        double X1 = i==i1 ? x1 : x;
        double X2 = i==i2 ? x2 : xval[i+1];
        result   +=
            f.integrate(X1, X2, 0) * (a - x * (b - 1./2 * x * (c - 1./3 * x * d))) +
            f.integrate(X1, X2, 1) * (b - x * (c - 1./2 * x * d)) +
            f.integrate(X1, X2, 2) * (c - x * d) / 2 +
            f.integrate(X1, X2, 3) * d / 6;
    }
    return result;
}

// ------ Hermite cubic spline ------ //

HermiteSpline::HermiteSpline(const std::vector<double>& _xval,
    const std::vector<double>& _fval, const std::vector<double>& _fder) :
    BaseInterpolator1d(_xval, _fval), fder(_fder)
{
    unsigned int numPoints = xval.size();
    if(fval.size() != numPoints || fder.size() != numPoints)
        throw std::invalid_argument("Error in spline initialization: input arrays are not equal in length");
}


void HermiteSpline::evalDeriv(const double x, double* val, double* deriv, double* deriv2) const
{
    if(xval.size() == 0)
        throw std::range_error("Empty spline");
    if(x < xval.front()) {
        if(val)
            *val   = fval.front() +
            (fder.front()==0 ? 0 : fder.front() * (x-xval.front()));
            // if der==0, will give correct result even for infinite x
        if(deriv)
            *deriv = fder.front();
        if(deriv2)
            *deriv2= 0;
        return;
    }
    if(x > xval.back()) {
        if(val)
            *val   = fval.back() + (fder.back()==0 ? 0 : fder.back() * (x-xval.back()));
        if(deriv)
            *deriv = fder.back();
        if(deriv2)
            *deriv2= 0;
        return;
    }
    unsigned int index = binSearch(x, &xval.front(), xval.size());
    evalHermiteSplines<1> (xval[index], xval[index+1], x,
        &fval[index], &fval[index+1], &fder[index], &fder[index+1],
        /*output*/ val, deriv, deriv2);
}

// ------ Quintic spline ------- //
QuinticSpline::QuinticSpline(const std::vector<double>& _xval, 
    const std::vector<double>& _fval, const std::vector<double>& _fder): 
    BaseInterpolator1d(_xval, _fval), fder(_fder), fder3(xval.size())
{
    unsigned int numPoints = xval.size();
    if(fval.size() != numPoints || fder.size() != numPoints)
        throw std::invalid_argument("Error in spline initialization: input arrays are not equal in length");
    std::vector<double> v(numPoints-1);
    double dx = xval[1]-xval[0];
    double dy = fval[1]-fval[0];
    fder3[0]  = v[0] = 0.;
    for(unsigned int i=1; i<numPoints-1; i++) {
        double dx1 = xval[i+1] - xval[i];
        double dx2 = xval[i+1] - xval[i-1];
        double dy1 = fval[i+1] - fval[i];
        double sig = dx/dx2;
        double p   = sig*v[i-1] - 3;
        double der3= 12 * ( 7*fder[i]*dx2 / (dx*dx1) 
            + 3 * (fder[i-1]/dx + fder[i+1]/dx1)
            - 10* (dy / (dx*dx) + dy1 / (dx1*dx1)) ) / dx2;
        fder3[i]   = (der3 - sig*fder3[i-1] ) / p;
        v[i]       = (sig-1)/p;
        dx = dx1;
        dy = dy1;
    }
    fder3[numPoints-1] = 0.;
    for(unsigned int i=numPoints-1; i>0; i--)
        fder3[i-1] += v[i-1]*fder3[i];
}

void QuinticSpline::evalDeriv(const double x, double* val, double* deriv, double* deriv2) const
{
    if(xval.size() == 0)
        throw std::range_error("Empty spline");
    if(x < xval.front()) {
        if(val)
            *val   = fval[0] + (fder[0]==0 ? 0 : fder[0] * (x-xval[0]));
        if(deriv)
            *deriv = fder[0];
        if(deriv2)
            *deriv2= 0;
        return;
    }
    if(x > xval.back()) {
        const unsigned int size = xval.size();
        if(val)
            *val   = fval[size-1] + (fder[size-1]==0 ? 0 : fder[size-1] * (x-xval[size-1]));
        if(deriv)
            *deriv = fder[size-1];
        if(deriv2)
            *deriv2= 0;
        return;
    }
    unsigned int index = binSearch(x, &xval.front(), xval.size());
    evalQuinticSplines<1> (xval[index], xval[index+1], x,
        &fval[index], &fval[index+1], &fder[index], &fder[index+1], &fder3[index], &fder3[index+1],
        /*output*/ val, deriv, deriv2);
}

double QuinticSpline::deriv3(const double x) const
{
    if(xval.size() == 0)
        throw std::range_error("Empty spline");
    if(x < xval.front() || x > xval.back())
        return 0;
    unsigned int index = binSearch(x, &xval.front(), xval.size());
    double der3;
    evalQuinticSplines<1> (xval[index], xval[index+1], x,
        &fval[index], &fval[index+1], &fder[index], &fder[index+1], &fder3[index], &fder3[index+1],
        /*output*/ NULL, NULL, NULL, &der3);
    return der3;
}


// ------ B-spline interpolator ------ //

template<int N>
BsplineInterpolator1d<N>::BsplineInterpolator1d(const std::vector<double>& xgrid) :
    xnodes(xgrid), numComp(xnodes.size()+N-1)
{
    if(xnodes.size()<2)
        throw std::invalid_argument("BsplineInterpolator1d: number of nodes is too small");
    bool monotonic = true;
    for(unsigned int i=1; i<xnodes.size(); i++)
        monotonic &= xnodes[i-1] < xnodes[i];
    if(!monotonic)
        throw std::invalid_argument("BsplineInterpolator1d: grid nodes must be sorted in ascending order");
}

template<int N>
unsigned int BsplineInterpolator1d<N>::nonzeroComponents(const double x, double values[]) const
{
    return bsplineValues<N>(x, &xnodes[0], xnodes.size(), values);
}

template<int N>
double BsplineInterpolator1d<N>::interpolate(
    const double x, const std::vector<double> &amplitudes) const
{
    if(amplitudes.size() != numComp)
        throw std::range_error("BsplineInterpolator1d: invalid size of amplitudes array");
    double bspl[N+1];
    unsigned int leftInd = bsplineValues<N>(x, &xnodes[0], xnodes.size(), bspl);
    double val=0;
    for(int i=0; i<=N; i++)
        val += bspl[i] * amplitudes[i+leftInd];
    return val;
}

template<int N>
void BsplineInterpolator1d<N>::eval(const double* x, double values[]) const
{
    std::fill(values, values+numComp, 0);
    double bspl[N+1];
    unsigned int leftInd = bsplineValues<N>(*x, &xnodes[0], xnodes.size(), bspl);
    for(int i=0; i<=N; i++)
        values[i+leftInd] = bspl[i];
}

template<int N>
double BsplineInterpolator1d<N>::integrate(double x1, double x2,
    const std::vector<double> &amplitudes, int n) const
{
    double sign = 1.;
    if(x1>x2) {  // swap limits of integration
        double tmp=x2;
        x2 = x1;
        x1 = tmp;
        sign = -1.;
    }

    // find out the min/max indices of grid segments that contain the integration interval
    const double* xgrid = &xnodes.front();
    const int Ngrid = xnodes.size();
    int i1 = std::max<int>(binSearch(x1, xgrid, Ngrid), 0);
    int i2 = std::min<int>(binSearch(x2, xgrid, Ngrid), Ngrid-2);

    // B-spline of degree N is a piecewise polynomial of degree N, thus to compute the integral
    // of B-spline times x^n on each grid segment, it is sufficient to employ a Gauss-Legendre
    // quadrature rule with the number of nodes = floor((N+n)/2)+1.
    const int NnodesGL = (N+n)/2+1;
    std::vector<double> glnodes(NnodesGL), glweights(NnodesGL);
    prepareIntegrationTableGL(0, 1, NnodesGL, &glnodes[0], &glweights[0]);

    // loop over segments
    double result = 0;
    for(int i=i1; i<=i2; i++) {
        double X1 = i==i1 ? x1 : xgrid[i];
        double X2 = i==i2 ? x2 : xgrid[i+1];
        double bspl[N+1];
        for(int k=0; k<NnodesGL; k++) {
            const double x = X1 + (X2 - X1) * glnodes[k];
            // evaluate the possibly non-zero functions and keep track of the index of the leftmost one
            int leftInd = bsplineValues<N>(x, xgrid, Ngrid, bspl);
            // add the contribution of this GL point to the integral of x^n * \sum A_j B_j(x),
            // where the index j runs from leftInd to leftInd+N
            double fval = 0;
            for(int b=0; b<=N; b++)
                fval += bspl[b] * amplitudes[b+leftInd];
            result += (X2-X1) * fval * glweights[k] * powInt(x, n);
        }
    }
    return result * sign;
}

template<int N>
Matrix<double> BsplineInterpolator1d<N>::computeOverlapMatrix(const unsigned int D) const
{
    switch(D) {
        case 0: return math::computeOverlapMatrix<N, 0>(xnodes);
        case 1: return math::computeOverlapMatrix<N, 1>(xnodes);
        case 2: return math::computeOverlapMatrix<N, 2>(xnodes);
        default:
            throw std::invalid_argument("computeOverlapMatrix: invalid order of derivative");
    }
}

template<int N>
std::vector<double> createBsplineInterpolator1dArray(const IFunction& F,
    const std::vector<double>& xnodes, int NnodesGL)
{
    const double* xgrid = &xnodes.front();
    const int Ngrid = xnodes.size();
    BsplineInterpolator1d<N> bspline(xnodes);
    NnodesGL = std::max<int>(NnodesGL, N/2+3);
    std::vector<double> glnodes(NnodesGL), glweights(NnodesGL);
    prepareIntegrationTableGL(0, 1, NnodesGL, &glnodes[0], &glweights[0]);

    // loop over segments
    std::vector<double> integrals(bspline.numValues());
    for(int i=0; i<Ngrid-1; i++) {
        double bsplval[N+1];
        for(int k=0; k<NnodesGL; k++) {
            const double x = xgrid[i] + (xgrid[i+1] - xgrid[i]) * glnodes[k];
            const double v = (xgrid[i+1] - xgrid[i]) * glweights[k] * F(x);
            // evaluate the possibly non-zero basis functions
            int leftInd = bsplineValues<N>(x, xgrid, Ngrid, bsplval);
            // add the contribution of this GL point to the integrals of f(x) B_j(x)
            // where the index j runs from leftInd to leftInd+N
            for(int b=0; b<=N; b++)
                integrals[b+leftInd] += v * bsplval[b];
        }
    }
    return CholeskyDecomp(bspline.computeOverlapMatrix(0)).solve(integrals);
}

// force template instantiations for several values of N
template std::vector<double> createBsplineInterpolator1dArray<0>(
    const IFunction&, const std::vector<double>&, int);
template std::vector<double> createBsplineInterpolator1dArray<1>(
    const IFunction&, const std::vector<double>&, int);
template std::vector<double> createBsplineInterpolator1dArray<2>(
    const IFunction&, const std::vector<double>&, int);
template std::vector<double> createBsplineInterpolator1dArray<3>(
    const IFunction&, const std::vector<double>&, int);
template class BsplineInterpolator1d<0>;
template class BsplineInterpolator1d<1>;
template class BsplineInterpolator1d<2>;
template class BsplineInterpolator1d<3>;


// ------ INTERPOLATION IN 2D ------ //

BaseInterpolator2d::BaseInterpolator2d(
    const std::vector<double>& xgrid, const std::vector<double>& ygrid,
    const Matrix<double>& fvalues) :
    xval(xgrid), yval(ygrid), fval(fvalues.data(), fvalues.data() + fvalues.size())
{
    const unsigned int xsize = xgrid.size();
    const unsigned int ysize = ygrid.size();
    if(xsize<2 || ysize<2)
        throw std::invalid_argument(
            "Error in 2d interpolator initialization: number of nodes should be >=2 in each direction");
    if(fvalues.rows() != xsize)
        throw std::invalid_argument(
            "Error in 2d interpolator initialization: x and f array lengths differ");
    if(fvalues.cols() != ysize)
        throw std::invalid_argument(
            "Error in 2d interpolator initialization: y and f array lengths differ");
}

// ------- Bilinear interpolation in 2d ------- //

void LinearInterpolator2d::evalDeriv(const double x, const double y, 
     double *z, double *deriv_x, double *deriv_y,
     double* deriv_xx, double* deriv_xy, double* deriv_yy) const
{
    if(isEmpty())
        throw std::range_error("Empty 2d interpolator");
    // 2nd derivatives are always zero
    if(deriv_xx)
        *deriv_xx = 0;
    if(deriv_xy)
        *deriv_xy = 0;
    if(deriv_yy)
        *deriv_yy = 0;
    // no interpolation outside the 2d grid
    if(x<xval.front() || x>xval.back() || y<yval.front() || y>yval.back()) {
        if(z)
            *z = NAN;
        if(deriv_x)
            *deriv_x = NAN;
        if(deriv_y)
            *deriv_y = NAN;
        return;
    }
    const unsigned int
        nx  = xval.size(),
        ny  = yval.size(),
        xi  = binSearch(x, &xval.front(), nx),
        yi  = binSearch(y, &yval.front(), ny),
        // indices of corner nodes in the flattened 2d array
        ill = xi * ny + yi, // xlow,ylow
        ilu = ill + 1,      // xlow,yupp
        iul = ill + ny,     // xupp,ylow
        iuu = iul + 1;      // xupp,yupp
    const double
        zlowlow = fval[ill],
        zlowupp = fval[ilu],
        zupplow = fval[iul],
        zuppupp = fval[iuu],
        // width and height of the grid cell
        dx = xval[xi+1] - xval[xi],
        dy = yval[yi+1] - yval[yi],
        // relative positions within the grid cell [0:1], in units of grid cell size
        t = (x - xval[xi]) / dx,
        u = (y - yval[yi]) / dy;
    if(z)
        *z = (1-t)*(1-u) * zlowlow + t*(1-u) * zupplow + (1-t)*u * zlowupp + t*u * zuppupp;
    if(deriv_x)
        *deriv_x = (-(1-u) * zlowlow + (1-u) * zupplow - u * zlowupp + u * zuppupp) / dx;
    if(deriv_y)
        *deriv_y = (-(1-t) * zlowlow - t * zupplow + (1-t) * zlowupp + t * zuppupp) / dy;
}


//------------ 2D CUBIC SPLINE -------------//

CubicSpline2d::CubicSpline2d(const std::vector<double>& xgrid, const std::vector<double>& ygrid,
    const Matrix<double>& fvalues,
    double deriv_xmin, double deriv_xmax, double deriv_ymin, double deriv_ymax) :
    BaseInterpolator2d(xgrid, ygrid, fvalues),
    fx (fvalues.size()),
    fy (fvalues.size()),
    fxy(fvalues.size())
{
    const unsigned int xsize = xgrid.size();
    const unsigned int ysize = ygrid.size();
    std::vector<double> tmpvalues(ysize);
    // step 1. for each x_i, construct cubic splines for f(x_i, y) in y and assign df/dy at grid nodes
    for(unsigned int i=0; i<xsize; i++) {
        for(unsigned int j=0; j<ysize; j++)
            tmpvalues[j] = fval[i * ysize + j];
        CubicSpline spl(yval, tmpvalues, deriv_ymin, deriv_ymax);
        for(unsigned int j=0; j<ysize; j++)
            spl.evalDeriv(yval[j], NULL, &fy[i * ysize + j]);
    }
    tmpvalues.resize(xsize);
    // step 1. for each y_j, construct cubic splines for f(x, y_j) in x and assign df/dx at grid nodes
    for(unsigned int j=0; j<ysize; j++) {
        for(unsigned int i=0; i<xsize; i++)
            tmpvalues[i] = fval[i * ysize + j];
        CubicSpline spl(xval, tmpvalues, deriv_xmin, deriv_xmax);
        for(unsigned int i=0; i<xsize; i++)
            spl.evalDeriv(xval[i], NULL, &fx[i * ysize + j]);
        // step 3. assign the mixed derivative d2f/dxdy:
        // if derivs at the boundary are specified and constant, 2nd deriv must be zero
        if( (j==0 && isFinite(deriv_ymin)) || (j==ysize-1 && isFinite(deriv_ymax)) ) {
            for(unsigned int i=0; i<xsize; i++)
                fxy[i * ysize + j] = 0.;
        } else {
            // otherwise construct cubic splines for df/dy(x,y_j) in x and assign d2f/dydx
            for(unsigned int i=0; i<xsize; i++)
                tmpvalues[i] = fy[i * ysize + j];
            CubicSpline spl(xval, tmpvalues,
                isFinite(deriv_xmin) ? 0. : NAN, isFinite(deriv_xmax) ? 0. : NAN);
            for(unsigned int i=0; i<xsize; i++)
                spl.evalDeriv(xval[i], NULL, &fxy[i * ysize + j]);
        }
    }
}

void CubicSpline2d::evalDeriv(const double x, const double y, 
    double *z, double *z_x, double *z_y, double *z_xx, double *z_xy, double *z_yy) const
{
    if(isEmpty())
        throw std::range_error("Empty 2d spline");
    if(x<xval.front() || x>xval.back() || y<yval.front() || y>yval.back()) {
        if(z)
            *z = NAN;
        if(z_x)
            *z_x = NAN;
        if(z_y)
            *z_y = NAN;
        if(z_xx)
            *z_xx = NAN;
        if(z_xy)
            *z_xy = NAN;
        if(z_yy)
            *z_yy = NAN;
        return;
    }
    const unsigned int
        nx = xval.size(),
        ny = yval.size(),
        // indices of grid cell in x and y
        xi = binSearch(x, &xval.front(), nx),
        yi = binSearch(y, &yval.front(), ny),
        // indices in flattened 2d arrays: 
        ill = xi * ny + yi, // xlow,ylow
        ilu = ill + 1,      // xlow,yupp
        iul = ill + ny,     // xupp,ylow
        iuu = iul + 1;      // xupp,yupp
    const double
        // coordinates of corner points
        xlow = xval[xi],
        xupp = xval[xi+1],
        ylow = yval[yi],
        yupp = yval[yi+1],
        // values and derivatives for the intermediate Hermite splines
        flow  [4] = { fval[ill], fval[iul], fx [ill], fx [iul] },
        fupp  [4] = { fval[ilu], fval[iuu], fx [ilu], fx [iuu] },
        dflow [4] = { fy  [ill], fy  [iul], fxy[ill], fxy[iul] },
        dfupp [4] = { fy  [ilu], fy  [iuu], fxy[ilu], fxy[iuu] };
    double F  [4];  // {   f    (xlow, y),   f    (xupp, y),  df/dx   (xlow, y),  df/dx   (xupp, y) }
    double dF [4];  // {  df/dy (xlow, y),  df/dy (xupp, y), d2f/dxdy (xlow, y), d2f/dxdy (xupp, y) }
    double d2F[4];  // { d2f/dy2(xlow, y), d2f/dy2(xupp, y), d3f/dxdy2(xlow, y), d3f/dxdy2(xupp, y) }
    bool der  = z_y!=NULL || z_xy!=NULL;
    bool der2 = z_yy!=NULL;
    // intermediate interpolation along y direction
    evalHermiteSplines<4> (ylow, yupp, y,  flow, fupp, dflow, dfupp,
        /*output*/ F, der? dF : NULL, der2? d2F : NULL);
    // final interpolation along x direction
    evalHermiteSplines<1> (xlow, xupp, x,  &F[0], &F[1], &F[2], &F[3],
        /*output*/ z, z_x, z_xx);
    if(der)
        evalHermiteSplines<1> (xlow, xupp, x,  &dF[0], &dF[1], &dF[2], &dF[3],
            /*output*/ z_y, z_xy, NULL);
    if(der2)
        evalHermiteSplines<1> (xlow, xupp, x,  &d2F[0], &d2F[1], &d2F[2], &d2F[3],
            /*output*/ z_yy, NULL, NULL);
}


//------------ 2D QUINTIC SPLINE -------------//

QuinticSpline2d::QuinticSpline2d(const std::vector<double>& xgrid, const std::vector<double>& ygrid,
    const Matrix<double>& fvalues, const Matrix<double>& dfdx, const Matrix<double>& dfdy) :
    BaseInterpolator2d(xgrid, ygrid, fvalues),
    fx    (dfdx.data(), dfdx.data() + dfdx.size()),
    fy    (dfdy.data(), dfdy.data() + dfdy.size()),
    fxxx  (fvalues.size()),
    fyyy  (fvalues.size()),
    fxyy  (fvalues.size()),
    fxxxyy(fvalues.size())
{
    const unsigned int xsize = xgrid.size();
    const unsigned int ysize = ygrid.size();
    // step 1. for each y_j construct 1d quintic spline for f in x, and record d^3f/dx^3
    std::vector<double> t(xsize), t1(xsize);
    for(unsigned int j=0; j<ysize; j++) {
        for(unsigned int i=0; i<xsize; i++) {
            t[i]  = fval[i * ysize + j];
            t1[i] = fx  [i * ysize + j];
        }
        QuinticSpline s(xval, t, t1);
        for(unsigned int i=0; i<xsize; i++)
            fxxx[i * ysize + j] = s.deriv3(xval[i]);
    }
    // 2. for each x_i construct 1d quintic spline for f and cubic splines for df/dx, d^3f/dx^3 in y
    t.resize(ysize);
    t1.resize(ysize);
    for(unsigned int i=0; i<xsize; i++) {
        for(unsigned int j=0; j<ysize; j++) {
            t[j]  = fval[i * ysize + j];
            t1[j] = fy  [i * ysize + j];
        }
        QuinticSpline s(yval, t, t1);
        for(unsigned int j=0; j<ysize; j++) {
            t [j] = fx  [i * ysize + j];
            t1[j] = fxxx[i * ysize + j];
        }
        CubicSpline u(yval, t);
        CubicSpline v(yval, t1);
        for(unsigned int j=0; j<ysize; j++) {
            fyyy[i * ysize + j] = s.deriv3(yval[j]);
            u.evalDeriv(yval[j], NULL, NULL, &fxyy  [i * ysize + j]);
            v.evalDeriv(yval[j], NULL, NULL, &fxxxyy[i * ysize + j]);
        }
    }
}

void QuinticSpline2d::evalDeriv(const double x, const double y,
    double* z, double* z_x, double* z_y,
    double* z_xx, double* z_xy, double* z_yy) const
{
    if(isEmpty())
        throw std::range_error("Empty 2d spline");
    if(x<xval.front() || x>xval.back() || y<yval.front() || y>yval.back()) {
        if(z)
            *z    = NAN;
        if(z_x)
            *z_x  = NAN;
        if(z_y)
            *z_y  = NAN;
        if(z_xx)
            *z_xx = NAN;
        if(z_xy)
            *z_xy = NAN;
        if(z_yy)
            *z_yy = NAN;
        return;
    }
    const unsigned int
        nx = xval.size(),
        ny = yval.size(),
        // indices of grid cell in x and y
        xi = binSearch(x, &xval.front(), nx),
        yi = binSearch(y, &yval.front(), ny),
        // indices in flattened 2d arrays: 
        ill = xi * ny + yi, // xlow,ylow
        ilu = ill + 1,      // xlow,yupp
        iul = ill + ny,     // xupp,ylow
        iuu = iul + 1;      // xupp,yupp
    const double
        // coordinates of corner points
        xlow = xval[xi],
        xupp = xval[xi+1],
        ylow = yval[yi],
        yupp = yval[yi+1],
        // values and derivatives for the intermediate splines
        fl [2] = { fval[ill], fval[iul] },
        fh [2] = { fval[ilu], fval[iuu] },
        f1l[2] = { fy  [ill], fy  [iul] },
        f1h[2] = { fy  [ilu], fy  [iuu] },
        f3l[2] = { fyyy[ill], fyyy[iul] },
        f3h[2] = { fyyy[ilu], fyyy[iuu] },
        gl [4] = { fx  [ill], fx  [iul], fxxx  [ill], fxxx  [iul] },
        gh [4] = { fx  [ilu], fx  [iuu], fxxx  [ilu], fxxx  [iuu] },
        g2l[4] = { fxyy[ill], fxyy[iul], fxxxyy[ill], fxxxyy[iul] },
        g2h[4] = { fxyy[ilu], fxyy[iuu], fxxxyy[ilu], fxxxyy[iuu] };
    bool der  = z_y!=NULL || z_xy!=NULL;
    bool der2 = z_yy!=NULL;
    // compute intermediate splines
    double
        F  [2],  // {   f      (xlow, y),   f      (xupp, y) }
        dF [2],  // {  df/dy   (xlow, y),  df/dy   (xupp, y) }
        d2F[2],  // { d2f/dy2  (xlow, y), d2f/dy2  (xupp, y) }
        G  [4],  // {  df/dx   (xlow, y),  df/dx   (xupp, y), d3f/dx3   (xlow, y), d3f/dx3   (xupp, y) }
        dG [4],  // { d2f/dxdy (xlow, y), d2f/dxdy (xupp, y), d4f/dx3dy (xlow, y), d4f/dx3dy (xupp, y) }
        d2G[4];  // { d3f/dxdy2(xlow, y), d3f/dxdy2(xupp, y), d5f/dx3dy2(xlow, y), d5f/dx3dy2(xupp, y) }
    evalQuinticSplines<2> (ylow, yupp, y,  fl, fh, f1l, f1h, f3l, f3h,
        /*output*/ F, der? dF : NULL, der2? d2F : NULL);
    evalCubicSplines<4>   (ylow, yupp, y,  gl, gh, g2l, g2h,
        /*output*/ G, der? dG : NULL, der2? d2G : NULL);
    // compute and output requested values and derivatives
    evalQuinticSplines<1> (xlow, xupp, x,  &F[0], &F[1], &G[0], &G[1], &G[2], &G[3],
        /*output*/ z, z_x, z_xx);
    if(z_y || z_xy)
        evalQuinticSplines<1> (xlow, xupp, x,
            &dF[0], &dF[1], &dG[0], &dG[1], &dG[2], &dG[3],
            /*output*/ z_y, z_xy, NULL);
    if(z_yy)
        evalQuinticSplines<1> (xlow, xupp, x,
            &d2F[0], &d2F[1], &d2G[0], &d2G[1], &d2G[2], &d2G[3],
            /*output*/ z_yy, NULL, NULL);
}


// ------- Interpolation in 3d ------- //

template<int N>
BsplineInterpolator3d<N>::BsplineInterpolator3d(
    const std::vector<double>& xgrid, const std::vector<double>& ygrid, const std::vector<double>& zgrid) :
    xnodes(xgrid), ynodes(ygrid), znodes(zgrid),
    numComp(indComp(xnodes.size()+N-2, ynodes.size()+N-2, znodes.size()+N-2)+1)
{
    if(xnodes.size()<2 || ynodes.size()<2 || znodes.size()<2)
        throw std::invalid_argument("BsplineInterpolator3d: number of nodes is too small");
    bool monotonic = true;
    for(unsigned int i=1; i<xnodes.size(); i++)
        monotonic &= xnodes[i-1] < xnodes[i];
    for(unsigned int i=1; i<ynodes.size(); i++)
        monotonic &= ynodes[i-1] < ynodes[i];
    for(unsigned int i=1; i<znodes.size(); i++)
        monotonic &= znodes[i-1] < znodes[i];
    if(!monotonic)
        throw std::invalid_argument("BsplineInterpolator3d: grid nodes must be sorted in ascending order");
}

template<int N>
void BsplineInterpolator3d<N>::nonzeroComponents(const double point[3],
    unsigned int leftIndices[3], double values[]) const
{
    double weights[3][N+1];
    for(int d=0; d<3; d++) {
        const std::vector<double>& nodes = d==0? xnodes : d==1? ynodes : znodes;
        leftIndices[d] = bsplineValues<N>(point[d], &nodes[0], nodes.size(), weights[d]);
    }
    for(int i=0; i<=N; i++)
        for(int j=0; j<=N; j++)
            for(int k=0; k<=N; k++)
                values[(i * (N+1) + j) * (N+1) + k] = weights[0][i] * weights[1][j] * weights[2][k];
}

template<int N>
double BsplineInterpolator3d<N>::interpolate(
    const double point[3], const std::vector<double> &amplitudes) const
{
    if(amplitudes.size() != numComp)
        throw std::range_error("BsplineInterpolator3d: invalid size of amplitudes array");
    double weights[(N+1)*(N+1)*(N+1)];
    unsigned int leftInd[3];
    nonzeroComponents(point, leftInd, weights);
    double val=0;
    for(int i=0; i<=N; i++)
        for(int j=0; j<=N; j++)
            for(int k=0; k<=N; k++)
                val += weights[ (i * (N+1) + j) * (N+1) + k ] *
                    amplitudes[ indComp(i+leftInd[0], j+leftInd[1], k+leftInd[2]) ];
    return val;
}

template<int N>
void BsplineInterpolator3d<N>::eval(const double point[3], double values[]) const
{
    unsigned int leftInd[3];
    double weights[(N+1)*(N+1)*(N+1)];
    nonzeroComponents(point, leftInd, weights);
    std::fill(values, values+numComp, 0);
    for(int i=0; i<=N; i++)
        for(int j=0; j<=N; j++)
            for(int k=0; k<=N; k++)
                values[ indComp(i+leftInd[0], j+leftInd[1], k+leftInd[2]) ] =
                    weights[(i * (N+1) + j) * (N+1) + k];
}

template<int N>
double BsplineInterpolator3d<N>::valueOfComponent(const double point[3], unsigned int indComp) const
{
    if(indComp>=numComp)
        throw std::range_error("BsplineInterpolator3d: component index out of range");
    unsigned int leftInd[3], indices[3];
    double weights[(N+1)*(N+1)*(N+1)];
    nonzeroComponents(point, leftInd, weights);
    decomposeIndComp(indComp, indices);
    if( indices[0]>=leftInd[0] && indices[0]<=leftInd[0]+N &&
        indices[1]>=leftInd[1] && indices[1]<=leftInd[1]+N &&
        indices[2]>=leftInd[2] && indices[2]<=leftInd[2]+N )
        return weights[ ((indices[0]-leftInd[0]) * (N+1)
                         +indices[1]-leftInd[1]) * (N+1) + indices[2]-leftInd[2] ];
    else
        return 0;
}

template<int N>
void BsplineInterpolator3d<N>::nonzeroDomain(unsigned int indComp,
    double xlower[3], double xupper[3]) const
{
    if(indComp>=numComp)
        throw std::range_error("BsplineInterpolator3d: component index out of range");
    unsigned int indices[3];
    decomposeIndComp(indComp, indices);
    for(int d=0; d<3; d++) {
        const std::vector<double>& nodes = d==0? xnodes : d==1? ynodes : znodes;
        xlower[d] = nodes[ indices[d]<N ? 0 : indices[d]-N ];
        xupper[d] = nodes[ std::min<unsigned int>(indices[d]+1, nodes.size()-1) ];
    }
}

template<int N>
SpMatrix<double> BsplineInterpolator3d<N>::computeRoughnessPenaltyMatrix() const
{
    std::vector<Triplet> values;      // elements of sparse matrix will be accumulated here
    Matrix<double>
    X0(computeOverlapMatrix<N,0>(xnodes)),  // matrices of products of 1d basis functions or derivs
    X1(computeOverlapMatrix<N,1>(xnodes)),
    X2(computeOverlapMatrix<N,2>(xnodes)),
    Y0(computeOverlapMatrix<N,0>(ynodes)),
    Y1(computeOverlapMatrix<N,1>(ynodes)),
    Y2(computeOverlapMatrix<N,2>(ynodes)),
    Z0(computeOverlapMatrix<N,0>(znodes)),
    Z1(computeOverlapMatrix<N,1>(znodes)),
    Z2(computeOverlapMatrix<N,2>(znodes));
    for(unsigned int index1=0; index1<numComp; index1++) {
        unsigned int ind[3];
        decomposeIndComp(index1, ind);
        // use the fact that in each dimension, the overlap matrix elements are zero if
        // |rowIndex-colIndex| > N (i.e. it is a band matrix with width 2N+1).
        unsigned int
        imin = ind[0]<N ? 0 : ind[0]-N,
        jmin = ind[1]<N ? 0 : ind[1]-N,
        kmin = ind[2]<N ? 0 : ind[2]-N,
        imax = std::min<unsigned int>(ind[0]+N+1, xnodes.size()+N-1),
        jmax = std::min<unsigned int>(ind[1]+N+1, ynodes.size()+N-1),
        kmax = std::min<unsigned int>(ind[2]+N+1, znodes.size()+N-1);
        for(unsigned int i=imin; i<imax; i++) {
            for(unsigned int j=jmin; j<jmax; j++) {
                for(unsigned int k=kmin; k<kmax; k++) {
                    unsigned int index2 = indComp(i, j, k);
                    if(index2>index1)
                        continue;  // will initialize from a symmetric element
                    double val =
                        X2(ind[0], i) * Y0(ind[1], j) * Z0(ind[2], k) +
                        X0(ind[0], i) * Y2(ind[1], j) * Z0(ind[2], k) + 
                        X0(ind[0], i) * Y0(ind[1], j) * Z2(ind[2], k) + 
                        X1(ind[0], i) * Y1(ind[1], j) * Z0(ind[2], k) * 2 +
                        X0(ind[0], i) * Y1(ind[1], j) * Z1(ind[2], k) * 2 +
                        X1(ind[0], i) * Y0(ind[1], j) * Z1(ind[2], k) * 2;
                    values.push_back(Triplet(index1, index2, val));
                    if(index1!=index2)
                        values.push_back(Triplet(index2, index1, val));
                }
            }
        }
    }
    return SpMatrix<double>(numComp, numComp, values);
}

template<int N>
std::vector<double> createBsplineInterpolator3dArray(const IFunctionNdim& F,
    const std::vector<double>& xnodes,
    const std::vector<double>& ynodes,
    const std::vector<double>& znodes)
{
    if(F.numVars() != 3 || F.numValues() != 1)
        throw std::invalid_argument(
            "createBsplineInterpolator3dArray: input function must have numVars=3, numValues=1");
    BsplineInterpolator3d<N> interp(xnodes, ynodes, znodes);
    
    // collect the function values at all nodes of 3d grid
    std::vector<double> fncvalues(interp.numValues());
    double point[3];
    for(unsigned int i=0; i<xnodes.size(); i++) {
        point[0] = xnodes[i];
        for(unsigned int j=0; j<ynodes.size(); j++) {
            point[1] = ynodes[j];
            for(unsigned int k=0; k<znodes.size(); k++) {
                point[2] = znodes[k];
                unsigned int index = interp.indComp(i, j, k);
                F.eval(point, &fncvalues[index]);
            }
        }
    }
    if(N==1)
        // in this case no further action is necessary: the values of function at grid nodes
        // are identical to the amplitudes used in the interpolation
        return fncvalues;

    // the matrix of values of basis functions at grid nodes (could be *BIG*, although it is sparse)
    std::vector<Triplet> values;  // elements of sparse matrix will be accumulated here
    const std::vector<double>* nodes[3] = {&xnodes, &ynodes, &znodes};
    // values of 1d B-splines at each grid node in each of the three dimensions, or -
    // for the last two rows in each matrix - 2nd derivatives of B-splines at the first/last grid nodes
    Matrix<double> weights[3];
    // indices of first non-trivial B-spline functions at each grid node in each dimension
    std::vector<int> leftInd[3];

    // collect the values of all basis functions at each grid node in each dimension
    for(int d=0; d<3; d++) {
        unsigned int Ngrid = nodes[d]->size();
        weights[d]=math::Matrix<double>(Ngrid+N-1, N+1);
        leftInd[d].resize(Ngrid+N-1);
        const double* arr = &(nodes[d]->front());
        for(unsigned int n=0; n<Ngrid; n++)
            leftInd[d][n] = bsplineValues<N>(arr[n], arr, Ngrid, &weights[d](n, 0));
        // collect 2nd derivatives at the endpoints
        leftInd[d][Ngrid]   = bsplineDerivs<N,2>(arr[0],       arr, Ngrid, &weights[d](Ngrid,   0));
        leftInd[d][Ngrid+1] = bsplineDerivs<N,2>(arr[Ngrid-1], arr, Ngrid, &weights[d](Ngrid+1, 0));
    }
    // each row of the matrix corresponds to the value of source function at a given grid point,
    // or to its the second derivative at the endpoints of grid which is assumed to be zero
    // (i.e. natural cubic spline boundary condition);
    // each column corresponds to the weights of each element of amplitudes array,
    // which is formed as a product of non-zero 1d basis functions in three dimensions,
    // or their 2nd derivs at extra endpoint nodes
    for(unsigned int i=0; i<xnodes.size()+N-1; i++) {
        for(unsigned int j=0; j<ynodes.size()+N-1; j++) {
            for(unsigned int k=0; k<znodes.size()+N-1; k++) {
                unsigned int indRow = interp.indComp(i, j, k);
                for(int ti=0; ti<=N; ti++) {
                    for(int tj=0; tj<=N; tj++) {
                        for(int tk=0; tk<=N; tk++) {
                            unsigned int indCol = interp.indComp(
                                ti + leftInd[0][i], tj + leftInd[1][j], tk + leftInd[2][k]);
                            values.push_back(Triplet(indRow, indCol, 
                                weights[0](i, ti) * weights[1](j, tj) * weights[2](k, tk)));
                        }
                    }
                }
            }
        }
    }

    // solve the linear system (could take *LONG* )
    return LUDecomp(SpMatrix<double>(interp.numValues(), interp.numValues(), values)).solve(fncvalues);
}

template<int N>
std::vector<double> createBsplineInterpolator3dArrayFromSamples(
    const Matrix<double>& points, const std::vector<double>& pointWeights,
    const std::vector<double>& /*xnodes*/,
    const std::vector<double>& /*ynodes*/,
    const std::vector<double>& /*znodes*/)
{
    if(points.rows() != pointWeights.size() || points.cols() != 3)
        throw std::invalid_argument(
            "createBsplineInterpolator3dArrayFromSamples: invalid size of input arrays");
    throw std::runtime_error("createBsplineInterpolator3dArrayFromSamples NOT IMPLEMENTED");
}

// force the template instantiations to compile
template class BsplineInterpolator3d<1>;
template class BsplineInterpolator3d<3>;

template std::vector<double> createBsplineInterpolator3dArray<1>(const IFunctionNdim& F,
    const std::vector<double>& xnodes,
    const std::vector<double>& ynodes,
    const std::vector<double>& znodes);
template std::vector<double> createBsplineInterpolator3dArray<3>(const IFunctionNdim& F,
    const std::vector<double>& xnodes,
    const std::vector<double>& ynodes,
    const std::vector<double>& znodes);
template std::vector<double> createBsplineInterpolator3dArrayFromSamples<1>(
    const Matrix<double>& points, const std::vector<double>& pointWeights,
    const std::vector<double>& xnodes,
    const std::vector<double>& ynodes,
    const std::vector<double>& znodes);
template std::vector<double> createBsplineInterpolator3dArrayFromSamples<3>(
    const Matrix<double>& points, const std::vector<double>& pointWeights,
    const std::vector<double>& xnodes,
    const std::vector<double>& ynodes,
    const std::vector<double>& znodes);


//-------------- PENALIZED SPLINE APPROXIMATION ---------------//

/// Implementation of penalized spline approximation
class SplineApproxImpl {
    const std::vector<double> knots;   ///< b-spline knots  X[k], k=0..numKnots-1
    const std::vector<double> xvalues; ///< x[i], i=0..numDataPoints-1
    const unsigned int numKnots;       ///< number of X[k] knots in the fitting spline;
    ///< the number of basis functions is  numBasisFnc = numKnots+2
    const unsigned int numDataPoints;  ///< number of x[i],y[i] pairs (original data)

    /// sparse matrix  C  containing the values of each basis function at each data point:
    /// (size: numDataPoints rows, numBasisFnc columns, with only 4 nonzero values in each row)
    SpMatrix<double> CMatrix;

    /// in the non-singular case, the matrix A = C^T C  of the system of normal equations is formed,
    /// and the lower triangular matrix L contains its Cholesky decomposition (size: numBasisFnc^2)
    Matrix<double> LMatrix;

    /// matrix "M" is the transformed version of roughness matrix R, which contains
    /// integrals of product of second derivatives of basis functions (size: numBasisFnc^2)
    Matrix<double> MMatrix;

    /// part of the decomposition of the matrix M (size: numBasisFnc)
    std::vector<double> singValues;

public:
    /// Auxiliary data used in the fitting process, pre-initialized for each set of data points `y`
    /// (these data cannot be members of the class, since they are not constant)
    struct FitData {
        std::vector<double> zRHS;  ///< C^T y, right hand side of the system of normal equations
        std::vector<double> MTz;   ///< the product  M^T z
        double ynorm2;             ///< the squared norm of vector y
    };

    /** Prepare internal tables for fitting the data points at the given set of x-coordinates
        and the given array of knots which determine the basis functions */
    SplineApproxImpl(const std::vector<double> &knots, const std::vector<double> &xvalues);

    /** find the amplitudes of basis functions that provide the best fit to the data points `y`
        for the given value of smoothing parameter `lambda`, determined indirectly by EDF.
        \param[in]  yvalues  are the data values corresponding to x-coordinates
        that were provided to the constructor;
        \param[in]  EDF>=0  is the equivalent number of degrees of freedom (2<=EDF<=numBasisFnc);
        \param[out] ampl  will contain the computed amplitudes of basis functions;
        \param[out] RSS  will contain the residual sum of squared differences between data and appxox;
    */
    void solveForAmplitudesWithEDF(const std::vector<double> &yvalues, double EDF,
        std::vector<double> &ampl, double &RSS) const;

    /** find the amplitudes of basis functions that provide the best fit to the data points `y`
        with the Akaike information criterion (AIC) being offset by deltaAIC from its minimum value
        (the latter corresponding to the case of optimal smoothing).
        \param[in]  yvalues  are the data values;
        \param[in]  deltaAIC is the offset of AIC (0 means the optimally smoothed spline);
        \param[out] ampl  will contain the computed amplitudes of basis functions;
        \param[out] RSS,EDF  same as in the previous function;
    */
    void solveForAmplitudesWithAIC(const std::vector<double> &yvalues, double deltaAIC,
        std::vector<double> &ampl, double &RSS, double &EDF) const;

    /** Obtain the best-fit solution for the given value of smoothing parameter lambda
        (this method is called repeatedly in the process of finding the optimal value of lambda).
        \param[in]  fitData contains the pre-initialized auxiliary arrays constructed by `initFit()`;
        \param[in]  lambda is the smoothing parameter;
        \param[out] ampl  will contain the computed amplitudes of basis functions;
        \param[out] RSS,EDF  same as in the previous function;
        \return  the value of AIC (Akaike information criterion) corresponding to these RSS and EDF
    */
    double computeAmplitudes(const FitData &fitData, double lambda,
        std::vector<double> &ampl, double &RSS, double &EDF) const;

private:
    /** Initialize temporary arrays used in the fitting process for the provided data vector y,
        in the case that the normal equations are not singular.
        \param[in]  yvalues is the vector of data values `y` at each data point;
        \returns    the data structure used by other methods later in the fitting process
    */
    FitData initFit(const std::vector<double> &yvalues) const;
};

namespace{
// compute the number of equivalent degrees of freedom
static double computeEDF(const std::vector<double>& singValues, double lambda)
{
    if(!isFinite(lambda))  // infinite smoothing leads to a straight line (2 d.o.f)
        return 2;
    else if(lambda==0)     // no smoothing means the number of d.o.f. equal to the number of basis fncs
        return singValues.size()*1.;
    else {
        double EDF = 0;
        for(unsigned int c=0; c<singValues.size(); c++)
            EDF += 1 / (1 + lambda * singValues[c]);
        return EDF;
    }
}
//-------- helper classes for root-finders -------//
class SplineEDFRootFinder: public IFunctionNoDeriv {
    const std::vector<double>& singValues;
    double targetEDF;
public:
    SplineEDFRootFinder(const std::vector<double>& _singValues, double _targetEDF) :
        singValues(_singValues), targetEDF(_targetEDF) {}
    virtual double value(double lambda) const {
        return computeEDF(singValues, lambda) - targetEDF;
    }
};

class SplineAICRootFinder: public IFunctionNoDeriv {
    const SplineApproxImpl& impl; ///< the fitting interface
    const SplineApproxImpl::FitData& fitData; ///< data for the fitting procedure
    const double targetAIC;       ///< target value of AIC for root-finder
public:
    SplineAICRootFinder(const SplineApproxImpl& _impl,
        const SplineApproxImpl::FitData& _fitData, double _targetAIC) :
        impl(_impl), fitData(_fitData), targetAIC(_targetAIC) {};
    virtual double value(double lambda) const {
        std::vector<double> ampl;
        double RSS, EDF;
        double AIC = impl.computeAmplitudes(fitData, lambda, ampl, RSS, EDF);
        return AIC - targetAIC;
    }
};
}  // internal namespace

SplineApproxImpl::SplineApproxImpl(const std::vector<double> &_knots, const std::vector<double> &_xvalues) :
    knots(_knots), xvalues(_xvalues),
    numKnots(_knots.size()), numDataPoints(_xvalues.size())
{
    for(unsigned int k=1; k<numKnots; k++)
        if(knots[k]<=knots[k-1])
            throw std::invalid_argument("SplineApprox: knots must be in ascending order");

    // compute the roughness matrix R (integrals over products of second derivatives of basis functions)
    Matrix<double> RMatrix(computeOverlapMatrix<3,2>(knots));

    // initialize b-spline matrix C
    std::vector<Triplet> Cvalues;
    Cvalues.reserve(numDataPoints * 4);
    for(unsigned int i=0; i<numDataPoints; i++) {
        // for each input point, at most 4 basis functions are non-zero, starting from index 'ind'
        double B[4];
        unsigned int ind = bsplineValuesExtrapolated<3>(xvalues[i], &knots.front(), numKnots, B);
        assert(ind<=numKnots-2);
        // store non-zero elements of the matrix
        for(int k=0; k<4; k++)
            Cvalues.push_back(Triplet(i, k+ind, B[k]));
    }
    CMatrix = SpMatrix<double>(numDataPoints, numKnots+2, Cvalues);
    Cvalues.clear();

    SpMatrix<double> SpA(numKnots+2, numKnots+2);  // temporary sparse matrix containing A = C^T C
    blas_dgemm(CblasTrans, CblasNoTrans, 1, CMatrix, CMatrix, 0, SpA);
    Matrix<double> AMatrix(SpA);  // convert to a dense matrix

    // to prevent a failure of Cholesky decomposition in the case if A is not positive definite,
    // we add a small multiple of R to A (following the recommendation in Ruppert,Wand&Carroll)
    blas_daxpy(1e-10, RMatrix, AMatrix);  // TODO: make it scale-invariant (proportional to norm(A)?)

    // pre-compute matrix L which is the Cholesky decomposition of matrix of normal equations A
    CholeskyDecomp CholA(AMatrix);
    LMatrix = CholA.L();

    // transform the roughness matrix R into a more suitable form M+singValues:
    // obtain Q = L^{-1} R L^{-T}, where R is the roughness penalty matrix (replace R by Q)
    blas_dtrsm(CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, 1, LMatrix, RMatrix);
    blas_dtrsm(CblasRight, CblasLower,  CblasTrans, CblasNonUnit, 1, LMatrix, RMatrix);

    // decompose this Q via singular value decomposition: Q = U * diag(S) * V^T
    SVDecomp SVD(RMatrix);
    singValues = SVD.S();       // vector of singular values of matrix Q:
    singValues[numKnots] = 0;   // the smallest two singular values must be zero;
    singValues[numKnots+1] = 0; // set it explicitly to avoid roundoff error

    // precompute M = L^{-T} U  which is used in computing amplitudes of basis functions.
    MMatrix = SVD.U();
    blas_dtrsm(CblasLeft, CblasLower, CblasTrans, CblasNonUnit, 1, LMatrix, MMatrix);
    // now M is finally in place, and the amplitudes for any lambda are given by
    // w = M (I + lambda * diag(singValues))^{-1} M^T  z
}

// initialize the temporary arrays used in the fitting process
SplineApproxImpl::FitData SplineApproxImpl::initFit(const std::vector<double> &yvalues) const
{
    if(yvalues.size() != numDataPoints) 
        throw std::invalid_argument("SplineApprox: input array sizes do not match");
    FitData fitData;
    fitData.ynorm2  = blas_ddot(yvalues, yvalues);
    fitData.zRHS.resize(numKnots+2);
    fitData.MTz. resize(numKnots+2);
    blas_dgemv(CblasTrans, 1, CMatrix, yvalues, 0, fitData.zRHS);     // precompute z = C^T y
    blas_dgemv(CblasTrans, 1, MMatrix, fitData.zRHS, 0, fitData.MTz); // precompute M^T z
    return fitData;
}

// obtain solution of linear system for the given smoothing parameter,
// using the pre-computed matrix M^T z, where z = C^T y is the rhs of the system of normal equations;
// output the amplitudes of basis functions and other relevant quantities (RSS, EDF); return AIC
double SplineApproxImpl::computeAmplitudes(const FitData &fitData, double lambda,
    std::vector<double> &ampl, double &RSS, double &EDF) const
{
    std::vector<double> tempv(numKnots+2);
    for(unsigned int p=0; p<numKnots+2; p++) {
        double sv = singValues[p];
        tempv[p]  = fitData.MTz[p] / (1 + (sv>0 ? sv*lambda : 0));
    }
    ampl.resize(numKnots+2);
    blas_dgemv(CblasNoTrans, 1, MMatrix, tempv, 0, ampl);
    // compute the residual sum of squares (note: may be prone to cancellation errors?)
    tempv = ampl;
    blas_dtrmv(CblasLower, CblasTrans, CblasNonUnit, LMatrix, tempv); // tempv = L^T w
    double wTz = blas_ddot(ampl, fitData.zRHS);
    RSS = (fitData.ynorm2 - 2*wTz + blas_ddot(tempv, tempv));
    EDF = computeEDF(singValues, lambda);  // equivalent degrees of freedom
    return log(RSS) + 2*EDF / (numDataPoints-EDF-1);  // AIC
}

void SplineApproxImpl::solveForAmplitudesWithEDF(const std::vector<double> &yvalues, double EDF,
    std::vector<double> &ampl, double &RSS) const
{
    if(EDF==0)
        EDF = numKnots+2;
    else if(EDF<2 || EDF>numKnots+2)
        throw std::invalid_argument("SplineApprox: incorrect number of equivalent degrees of freedom");
    double lambda = findRoot(SplineEDFRootFinder(singValues, EDF), 0, INFINITY, 1e-6);
    computeAmplitudes(initFit(yvalues), lambda, ampl, RSS, EDF);
}

void SplineApproxImpl::solveForAmplitudesWithAIC(const std::vector<double> &yvalues, double deltaAIC,
    std::vector<double> &ampl, double &RSS, double &EDF) const
{
    double lambda=0;
    FitData fitData = initFit(yvalues);
    if(deltaAIC < 0)
        throw std::invalid_argument("SplineApprox: deltaAIC must be non-negative");
    if(deltaAIC == 0) {  // find the value of lambda corresponding to the optimal fit
        lambda = findMin(SplineAICRootFinder(*this, fitData, 0),
            0, INFINITY, NAN /*no initial guess*/, 1e-6);
        if(lambda!=lambda)
            lambda = 0;  // no smoothing in case of weird problems
    } else {  // find an oversmoothed solution
        // the reference value of AIC at lambda=0 (NOT the value that minimizes AIC, but very close to it)
        double AIC0 = computeAmplitudes(fitData, 0, ampl, RSS, EDF);
        // find the value of lambda so that AIC is larger than the reference value by the required amount
        lambda = findRoot(SplineAICRootFinder(*this, fitData, AIC0 + deltaAIC),
            0, INFINITY, 1e-6);
        if(!isFinite(lambda))   // root does not exist, i.e. AIC is everywhere lower than target value
            lambda = INFINITY;  // basically means fitting with a linear regression
    }
    // compute the amplitudes for the final value of lambda
    computeAmplitudes(fitData, lambda, ampl, RSS, EDF);
}

//----------- DRIVER CLASS FOR PENALIZED SPLINE APPROXIMATION ------------//

SplineApprox::SplineApprox(const std::vector<double> &grid, const std::vector<double> &xvalues)
{
    impl = new SplineApproxImpl(grid, xvalues);
}

SplineApprox::~SplineApprox()
{
    delete impl;
}

std::vector<double> SplineApprox::fit(
    const std::vector<double> &yvalues, const double edf,
    double *rms) const
{
    std::vector<double> ampl;
    double RSS;
    impl->solveForAmplitudesWithEDF(yvalues, edf, ampl, RSS);
    if(rms)
        *rms = sqrt(RSS / yvalues.size());
    return ampl;
}

std::vector<double> SplineApprox::fitOversmooth(
    const std::vector<double> &yvalues, const double deltaAIC,
    double *rms, double* edf) const
{
    std::vector<double> ampl;
    double RSS, EDF;
    impl->solveForAmplitudesWithAIC(yvalues, deltaAIC, ampl, RSS, EDF);
    if(rms)
        *rms = sqrt(RSS / yvalues.size());
    if(edf)
        *edf = EDF;
    return ampl;
}


//------------ LOG-SPLINE DENSITY ESTIMATOR ------------//
namespace {

/** Data for SplineLogDensity fitting procedure that is changing during the fit */
struct SplineLogFitParams {
    std::vector<double> ampl; ///< array of amplitudes used to start the multidimensional minimizer
    double lambda;            ///< smoothing parameter
    double targetLogL;        ///< target value of likelihood for the case with smoothing
    double best;              ///< highest cross-validation score or smallest offset from root
    double gradNorm;          ///< normalization factor for determining the root-finder tolerance
    SplineLogFitParams() : lambda(0), targetLogL(0), best(-INFINITY), gradNorm(0) {}
};

/** The engine of log-spline density estimator relies on the maximization of log-likelihood
    of input samples by varying the parameters of the estimator.

    Let  x_i, w_i; i=0..N_{data}-1  be the coordinates and weights of samples drawn from
    an unknown density distribution that we wish to estimate by constructing a function P(x).
    The total weight of all samples is M = \sum_{i=0}^{N_{data}-1} w_i  (does not need to be unity),
    and we stipulate that  \int P(x) dx = M.

    The logarithm of estimated density P(x) is represented as
    \ln P(x) = \sum_{k=0}^{N_{basis}-1}  A_k B_k(x) - \ln G_0 + \ln M = Q(x) - \ln G_0 + \ln M,
    where  A_k  are the amplitudes -- free parameters that are adjusted during the fit,
    B_k(x)  are basis functions (B-splines of degree N defined by grid nodes),
    Q(x) = \sum_k  A_k B_k(x)  is the weighted sum of basis function, and
    G_0  = \int \exp[Q(x)] dx  is the normalization constant determined from the condition
    that the integral of P(x) over the entire domain equals to M.
    If we shift the entire weighted sum of basis functions Q(x) up or down by a constant,
    this will have no effect on P(x), because this shift will be counterbalanced by G_0.
    Therefore, there is an extra gauge freedom of choosing {A_k};
    we elimitate it by fixing the amplitude of the last B-spline to zero: A_{N_{basis}-1} = 0.
    In the end, there are N_{ampl} = N_{basis}-1 free parameters that are adjusted during the fit.

    The total likelihood of the model given the amplitudes {A_k} is
    \ln L = \sum_{i=0}^{N_{data}-1}  w_i  \ln P(x_i)
          = \sum_{i=0}^{N_{data}-1}  w_i (\sum_{k=0}^{N_{ampl}-1} A_k B_k(x_i) - \ln G_0 + \ln M)
          = \sum_{k=0}^{N_{ampl}-1}  A_k  L_k  - M \ln G_0({A_k})  + M \ln M,
    where  L_k = \sum_{i=0}^{N_{data}-1} w_i B_k(x_i)  is an array of 'basis likelihoods'
    that is computed from input samples only once at the beginning of the fitting procedure.

    Additionally, we may impose a penalty for unsmoothness of the estimated P(x), by adding a term
    -\lambda \int (\ln P(x)'')^2 dx  into the above expression.
    Here lambda>=0 is the smoothing parameter, and the integral of squared second derivative
    of the sum of B-splines is expressed as a quadratic form in amplitudes:
    \sum_k \sum_l  A_k A_l R_{kl} ,  where R_{kl} = \int B_k''(x) B_l''(x) dx
    is the 'roughhness matrix', again pre-computed at the beginning of the fitting procedure.
    This addition decreases the overall likelihood, but makes the estimated ln P(x) more smooth.
    To find the suitable value of lambda, we use the following consideration:
    if P(x) were the true density distribution, then the likelihood of the finite number of samples
    is a random quantity with mean E and rms scatter D.
    We can tolerate the decrease in the likelihood by an amount comparable to D,
    if this makes the estimate more smooth.
    Therefore, we first find the values of amplitudes that maximize the likelihood without smoothing,
    and then determine the value of lambda that decreases log L by the prescribed amount.
    The mean and rms scatter of ln L are given (for equal-weight samples) by
    E   = \int P(x) \ln P(x) dx
        = M (G_1 + \ln M - \ln G_0),
    D   = \sqrt{ M \int P(x) [\ln P(x)]^2 dx  -  E^2 } / \sqrt{N_{data}}
        = M \sqrt{ G_2/G_0 - (G_1/G_0)^2 } / \sqrt{N_{data}} ,  where we defined
    G_d = \int \exp[Q(x)] [Q(x)]^d dx  for d=0,1,2.

    We minimize  -log(L)  by varying the amplitudes A_k (for a fixed value of lambda),
    using a nonlinear multidimensional root-finder with derivatives to locate the point
    where d log(L) / d A_k = 0 for all A_k.
    This class implements the interface needed for `findRootNdimDeriv()`: computation of
    gradient and hessian of log(L) w.r.t. each of the free parameters A_k, k=0..N_{ampl}-1.
*/
template<int N>
class SplineLogDensityFitter: public IFunctionNdimDeriv {
public:
    SplineLogDensityFitter(
        const std::vector<double>& xvalues, const std::vector<double>& weights,
        const std::vector<double>& grid, FitOptions options,
        SplineLogFitParams& params);

    /** Return the array of properly normalized amplitudes, such that the integral of
        P(x) over the entire domain is equal to the sum of sample weights M.
    */
    std::vector<double> getNormalizedAmplitudes(const std::vector<double>& ampl) const;

    /** Compute the expected rms scatter in log-likelihood for a density function defined
        by the given amplitudes, for the given number of samples */
    double logLrms(const std::vector<double>& ampl) const;

    /** Compute the log-likelihood of the data given the amplitudes.
        \param[in]  ampl  is the array of amplitudes A_k, k=0..numAmpl-1.
        \return     \ln L = \sum_k A_k V_k - M \ln G_0({A_k}) + M \ln M ,  where
        G_0 is the normalization constant that depends on all A_k,
        V_k is the pre-computed array Vbasis.
    */
    double logL(const std::vector<double>& ampl) const;

    /** Compute the cross-validation likelihood of the data given the amplitudes.
        \param[in]  ampl  is the array of amplitudes.
        \return     \ln L_CV = \ln L - tr(H^{-1} B^T B) + (d \ln G_0 / d A) H^{-1} W.
    */
    double logLcv(const std::vector<double>& ampl) const;

private:
    /** Compute the gradient and hessian of the full log-likelihood function
        (including the roughness penalty) multiplied by a constant:
        \ln L_full = \ln L - \lambda \sum_k \sum_l A_k A_l R_{kl},
        where \ln L is defined in logL(),  R_{kl} is the pre-computed roughnessMatrix,
        and lambda is taken from an external SplineLogFitParams variable.
        This routine is used in the nonlinear root-finder to determine the values of A_k
        that correspond to grad=0.
        \param[in]  ampl  is the array of amplitudes A_k that are varied during the fit;
        \param[out] grad  = (-1/M) d \ln L / d A_k;
        \param[out] hess  = (-1/M) d^2 \ln L / d A_k d A_l.
    */
    virtual void evalDeriv(const double ampl[], double grad[], double hess[]) const;
    virtual unsigned int numVars() const { return numAmpl; }
    virtual unsigned int numValues() const { return numAmpl; }

    /** Compute the values and derivatives of  G_d = \int \exp(Q(x)) [Q(x)]^d  dx,  where
        Q(x) = \sum_{k=0}^{N_{ampl}-1}  A_k B_k(x)  is the weighted sum of basis functions,
        B_k(x) are basis functions (B-splines of degree N defined by the grid nodes),
        and the integral is taken over the finite or (semi-) infinite interval,
        depending on the boolean constants leftInfinite, rightInfinite
        (if any of them is false, the corresponding boundary is the left/right-most grid point,
        otherwise it is +-infinity).
        \param[in]  ampl  is the array of A_k.
        \param[out] deriv if not NULL, will contain the derivatives of  \ln(G_0) w.r.t. A_k;
        \param[out] deriv2 if not NULL, will contain the second derivatives:
        d^2 \ln G_0 / d A_k d A_l.
        \param[out] GdG0  if not NULL, will contain  G_1/G_0  and  G_2/G_0.
        \return     \ln G_0.
    */
    double logG(const double ampl[], double deriv[]=NULL, double deriv2[]=NULL, double GdG0[]=NULL) const;

    const std::vector<double> grid;   ///< grid nodes that define the B-splines
    const unsigned int numNodes;      ///< shortcut for grid.size()
    const unsigned int numBasisFnc;   ///< shortcut for the number of B-splines (numNodes+N-1)
    const unsigned int numAmpl;       ///< the number of amplitudes that may be varied (numBasisFnc-1)
    const unsigned int numData;       ///< number of sample points
    const FitOptions options;         ///< whether the definition interval extends to +-inf
    static const int GL_ORDER = 8;    ///< order of GL quadrature for computing the normalization
    double GLnodes[GL_ORDER], GLweights[GL_ORDER];  ///< nodes and weights of GL quadrature
    std::vector<double> Vbasis;       ///< basis likelihoods: V_k = \sum_i w_i B_k(x_i)
    std::vector<double> Wbasis;       ///< W_k = \sum_i w_i^2 B_k(x_i) 
    Matrix<double> BTBmatrix;         ///< matrix B^T B, where B_{ik} = w_i B_k(x_i)
    double sumWeights;                ///< sum of weights of input points (M)
    Matrix<double> roughnessMatrix;   ///< roughness penalty matrix - integrals of B_k''(x) B_l''(x)
    SplineLogFitParams& params;       ///< external parameters that may be changed during the fit
};

template<int N>
SplineLogDensityFitter<N>::SplineLogDensityFitter(
    const std::vector<double>& _grid,
    const std::vector<double>& xvalues,
    const std::vector<double>& weights,
    FitOptions _options,
    SplineLogFitParams& _params) :
    grid(_grid),
    numNodes(grid.size()),
    numBasisFnc(numNodes + N - 1),
    numAmpl(numBasisFnc - 1),
    numData(xvalues.size()),
    options(_options),
    params(_params)
{
    if(numData <= 0)
        throw std::invalid_argument("splineLogDensity: no data");
    if(numData != weights.size())
        throw std::invalid_argument("splineLogDensity: sizes of input arrays are not equal");
    if(numNodes<2)
        throw std::invalid_argument("splineLogDensity: grid size should be at least 2");
    for(unsigned int k=1; k<numNodes; k++)
        if(grid[k-1] >= grid[k])
            throw std::invalid_argument("splineLogDensity: grid nodes are not monotonic");
    prepareIntegrationTableGL(0, 1, GL_ORDER, GLnodes, GLweights);

    // prepare the roughness penalty matrix
    // (integrals over products of certain derivatives of basis functions)
    if((options & FO_PENALTY_3RD_DERIV) == FO_PENALTY_3RD_DERIV)
        roughnessMatrix = computeOverlapMatrix<N,3>(grid);
    else
        roughnessMatrix = computeOverlapMatrix<N,2>(grid);

    // prepare the log-likelihoods of each basis fnc and other useful arrays
    Vbasis.assign(numBasisFnc, 0);
    Wbasis.assign(numAmpl, 0);
    std::vector<Triplet> Bvalues;
    Bvalues.reserve(numData * (N+1));
    sumWeights = 0;
    double minWeight = INFINITY;
    double avgx = 0, avgx2 = 0;
    for(unsigned int p=0; p<numData; p++) {
        if(weights[p] < 0)
            throw std::invalid_argument("splineLogDensity: sample weights may not be negative");
        if(weights[p] == 0)
            continue;
        // if the interval is (semi-)finite, samples beyond its boundaries are ignored
        if( (xvalues[p] < grid[0]          && (options & FO_INFINITE_LEFT)  != FO_INFINITE_LEFT) ||
            (xvalues[p] > grid[numNodes-1] && (options & FO_INFINITE_RIGHT) != FO_INFINITE_RIGHT) )
            continue;
        double Bspl[N+1];
        int ind = bsplineValuesExtrapolated<N>(xvalues[p], &grid[0], numNodes, Bspl);
        for(unsigned int b=0; b<=N; b++) {
            Vbasis[b+ind] += weights[p] * Bspl[b];
            if(b+ind<numAmpl) {
                Wbasis[b+ind] += pow_2(weights[p]) * Bspl[b];
                Bvalues.push_back(Triplet(p, b+ind, weights[p] * Bspl[b]));
            }
        }
        sumWeights += weights[p];
        minWeight = fmin(minWeight, weights[p]);
        avgx += weights[p] * xvalues[p];
        avgx2+= weights[p] * pow_2(xvalues[p]);
    }
    // sanity check    
    if(sumWeights==0)
        throw std::invalid_argument("splineLogDensity: sum of sample weights should be positive");

    // sanity check: all of basis functions must have a contribution from sample points,
    // otherwise the problem is singular and the max-likelihood solution is unattainable
    bool isSingular = false;
    for(unsigned int k=0; k<numBasisFnc; k++) {
        isSingular |= Vbasis[k]==0;
        params.gradNorm = fmax(params.gradNorm, fabs(Vbasis[k]));
    }
    if(isSingular) {
        // add fake contributions to all basis functions that would have arisen from
        // a uniformly distributed minWeight over each grid segment
        minWeight *= 1. / (numNodes-1);
        for(unsigned int j=0; j<numNodes-1; j++) {
            for(int s=0; s<GL_ORDER; s++) {
                double x = grid[j] + GLnodes[s] * (grid[j+1]-grid[j]);
                double Bspl[N+1];
                int ind = bsplineValues<N>(x, &grid[0], numNodes, Bspl);
                for(unsigned int b=0; b<=N; b++)
                    Vbasis[b+ind] += minWeight * Bspl[b] * GLweights[s];
            }
            sumWeights += minWeight;
        }
    }
    // chop off the last basis function whose amplitude is not varied in the fitting procedure
    Vbasis.resize(numAmpl);

    // compute B^T B that is used in cross-validation
    SpMatrix<double> Bmatrix(numData, numAmpl, Bvalues);
    SpMatrix<double> SpBTB(numAmpl, numAmpl);  // temporary sparse matrix containing B^T B
    blas_dgemm(CblasTrans, CblasNoTrans, 1, Bmatrix, Bmatrix, 0, SpBTB);
    BTBmatrix = Matrix<double>(SpBTB);   // convert to a dense matrix

    // compute the mean and dispersion of input samples
    avgx /= sumWeights;
    avgx2/= sumWeights;
    double dispx = fmax(avgx2 - pow_2(avgx), 0.01 * pow_2(grid.back()-grid.front()));
    avgx  = fmin(fmax(avgx, grid.front()), grid.back());
    // assign the initial guess for amplitudes using a Gaussian density distribution
    params.ampl.assign(numBasisFnc, 0);
    for(int k=0; k<(int)numBasisFnc; k++) {
        double xnode = grid[ std::min<int>(numNodes-1, std::max(0, k-N/2)) ];
        params.ampl[k] = -pow_2(xnode-avgx) / 2 / dispx;
    }
    // make sure that we start with a density that is declining when extrapolated
    if((options & FO_INFINITE_LEFT) == FO_INFINITE_LEFT)
        params.ampl[0] = fmin(params.ampl[0], params.ampl[1] - (grid[1]-grid[0]));
    if((options & FO_INFINITE_RIGHT) == FO_INFINITE_RIGHT)
        params.ampl[numBasisFnc-1] = fmin(params.ampl[numBasisFnc-1],
            params.ampl[numBasisFnc-2] - (grid[numNodes-1]-grid[numNodes-2]));

    // now shift all amplitudes by the value of the rightmost one, which is always kept equal to zero
    for(unsigned int k=0; k<numAmpl; k++)
        params.ampl[k] -= params.ampl.back();
    // and eliminate the last one, since it does not take part in the fitting process
    params.ampl.pop_back();
}

template<int N>
std::vector<double> SplineLogDensityFitter<N>::getNormalizedAmplitudes(
    const std::vector<double>& ampl) const
{
    assert(ampl.size() == numAmpl);
    std::vector<double> result(numBasisFnc);
    double C = log(sumWeights) - logG(&ampl[0]);
    for(unsigned int n=0; n<numBasisFnc; n++)
        result[n] = (n<numAmpl ? ampl[n] : 0) + C;
    return result;
}

template<int N>
double SplineLogDensityFitter<N>::logLrms(const std::vector<double>& ampl) const
{
    assert(ampl.size() == numAmpl);
    double GdG0[2];
    logG(&ampl[0], NULL, NULL, GdG0);
    double rms = sumWeights * sqrt((GdG0[1] - pow_2(GdG0[0])) / numData);
    if(utils::verbosityLevel >= utils::VL_VERBOSE) {
        double avg = sumWeights * (GdG0[0] + log(sumWeights) - logG(&ampl[0]));
        utils::msg(utils::VL_VERBOSE, "splineLogDensity",
            "Expected log L="+utils::toString(avg)+" +- "+utils::toString(rms));
    }
    return rms;
}

template<int N>
double SplineLogDensityFitter<N>::logL(const std::vector<double>& ampl) const
{
    assert(ampl.size() == numAmpl);
    double val = sumWeights * (log(sumWeights) - logG(&ampl[0]));
    for(unsigned int k=0; k<numAmpl; k++)
        val += Vbasis[k] * ampl[k];
    return val;
}

template<int N>
double SplineLogDensityFitter<N>::logLcv(const std::vector<double>& ampl) const
{
    assert(ampl.size() == numAmpl);
    std::vector<double> grad(numAmpl);
    Matrix<double> hess(numAmpl, numAmpl);
    double val = sumWeights * (log(sumWeights) - logG(&ampl[0], &grad[0], hess.data()));
    for(unsigned int k=0; k<numAmpl; k++) {
        val += Vbasis[k] * ampl[k];
        for(unsigned int l=0; l<numAmpl; l++) {
            hess(k, l) = sumWeights * hess(k, l) + 2 * params.lambda * roughnessMatrix(k, l);
        }
    }
    try{
        CholeskyDecomp hessdec(hess);
        Matrix<double> hessL(hessdec.L()), tmpmat(BTBmatrix);
        // tmpmat = H^{-1} (B^T B)
        blas_dtrsm(CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, 1, hessL, tmpmat);
        blas_dtrsm(CblasLeft, CblasLower, CblasTrans,   CblasNonUnit, 1, hessL, tmpmat);
        double add = 0;
        for(unsigned int k=0; k<numAmpl; k++) {
            add -= tmpmat(k, k);  // trace of H^{-1} B^T B
        }
        std::vector<double> Hm1W = hessdec.solve(Wbasis);  // H^{-1} W
        add += blas_ddot(grad, Hm1W);  // dG/dA H^{-1} W
        // don't allow the cross-validation likelihood to be higher than log L itself
        val += fmin(add, 0);  // (this shouldn't occur under normal circumstances)
    }
    catch(std::exception&) {  // may happen if the fit did not converge, i.e. gradient != 0
        utils::msg(utils::VL_WARNING, "splineLogDensity",
            "Hessian is not positive-definite");
        val -= 1e10;   // this will never be a good fit
    }
    return val;
}

template<int N>
void SplineLogDensityFitter<N>::evalDeriv(const double ampl[], double deriv[], double deriv2[]) const
{
    logG(ampl, deriv, deriv2);
    if(deriv!=NULL) {  // (-1/M)  d (log L) / d A_k
        for(unsigned int k=0; k<numAmpl; k++) {
            deriv[k] -= Vbasis[k] / sumWeights;
        }
    }
    // roughness penalty (taking into account symmetry and sparseness of the matrix)
    if(params.lambda!=0) {
        for(unsigned int k=0; k<numAmpl; k++) {
            for(unsigned int l=k; l<std::min(numAmpl, k+N+1); l++) {
                double v = 2 * roughnessMatrix(k, l) * params.lambda / sumWeights;
                if(deriv2!=NULL) {
                    deriv2[k * numAmpl + l] += v;
                    if(k!=l)
                        deriv2[l * numAmpl + k] += v;
                }
                if(deriv!=NULL) {
                    deriv[k] += v * ampl[l];
                    if(k!=l)
                        deriv[l] += v * ampl[k];
                }
            }
        }
    }
}

template<int N>
double SplineLogDensityFitter<N>::logG(
    const double ampl[], double deriv_arg[], double deriv2[], double GdG0[]) const
{
    std::vector<double> deriv_tmp;
    double* deriv = deriv_arg;
    if(deriv_arg==NULL && deriv2!=NULL) {  // need a temporary workspace for the gradient vector
        deriv_tmp.resize(numAmpl);
        deriv = &deriv_tmp.front();
    }
    // accumulator for the integral  G_d = \int \exp( Q(x) ) [Q(x)]^d  dx,
    // where  Q = \sum_k  A_k B_k(x),  and d ranges from 0 to 2
    double integral[3] = {0};
    // accumulator for d G_0 / d A_k
    if(deriv)
        std::fill(deriv, deriv+numAmpl, 0);
    // accumulator for d^2 G_0 / d A_k d A_l
    if(deriv2)
        std::fill(deriv2, deriv2+pow_2(numAmpl), 0);
    // determine the constant offset needed to keep the magnitude in a reasonable range
    double offset = 0;
    for(unsigned int k=0; k<numAmpl; k++)
        offset = fmax(offset, ampl[k]);

    // loop over grid segments...
    for(unsigned int k=0; k<numNodes-1; k++) {
        double segwidth = grid[k+1] - grid[k];
        // ...and over sub-nodes of Gauss-Legendre quadrature rule within each grid segment
        for(int s=0; s<GL_ORDER; s++) {
            double x = grid[k] + GLnodes[s] * segwidth;
            double Bspl[N+1];
            // obtain the values of all nontrivial basis function at this point,
            // and the index of the first of these functions.
            int ind = bsplineValues<N>(x, &grid[0], numNodes, Bspl);
            // sum the contributions to Q(x) from each basis function,
            // weighted with the provided amplitudes; 
            // here we substitute zero in place of the last (numBasisFnc-1)'th amplitude.
            double Q = 0;
            for(unsigned int b=0; b<=N && b+ind<numAmpl; b++) {
                Q += Bspl[b] * ampl[b+ind];
            }
            // the contribution of this point to the integral is weighted according to the GL quadrature;
            // the value of integrand is exp(Q) * Q^d,
            // but to avoid possible overflows, we instead compute  exp(Q-offset) Q^d.
            double val = GLweights[s] * segwidth * exp(Q-offset);
            for(int d=0; d<=2; d++)
                integral[d] += val * powInt(Q, d);
            // contribution of this point to the integral of derivatives is further multiplied
            // by the value of each basis function at this point.
            if(deriv) {
                for(unsigned int b=0; b<=N && b+ind<numAmpl; b++)
                    deriv[b+ind] += val * Bspl[b];
            }
            // and contribution to the integral of second derivatives is multiplied
            // by the product of two basis functions
            if(deriv2) {
                for(unsigned int b=0; b<=N && b+ind<numAmpl; b++)
                    for(unsigned int c=0; c<=N && c+ind<numAmpl; c++)
                        deriv2[(b+ind) * numAmpl + c+ind] += val * Bspl[b] * Bspl[c];
            }
        }
    }

    // if the interval is (semi-)infinite, need to add contributions from the tails beyond the grid
    bool   infinite[2] = {(options & FO_INFINITE_LEFT)  == FO_INFINITE_LEFT,
                          (options & FO_INFINITE_RIGHT) == FO_INFINITE_RIGHT};
    double endpoint[2] = {grid[0], grid[numNodes-1]};
    double signder [2] = {+1, -1};
    for(int p=0; p<2; p++) {
        if(!infinite[p])
            continue;
        double Bspl[N+1], Bder[N+1];
        int ind = bsplineValues<N>(endpoint[p], &grid[0], numNodes, Bspl);
        bsplineDerivs<N,1>(endpoint[p], &grid[0], numNodes, Bder);
        double Q = 0, Qder = 0;
        for(unsigned int b=0; b<=N && b+ind<numAmpl; b++) {
            Q    += Bspl[b] * ampl[b+ind];
            Qder += Bder[b] * ampl[b+ind];
        }
        if(signder[p] * Qder <= 0) {  
            // the extrapolated function rises as x-> -inf, so the integral does not exist
            if(deriv)
                deriv[0] = INFINITY;
            if(deriv2)
                deriv2[0] = INFINITY;
            return INFINITY;
        }
        double val = signder[p] * exp(Q-offset) / Qder;
        integral[0] += val;
        integral[1] += val * (Q-1);
        integral[2] += val * (pow_2(Q-1)+1);
        if(deriv) {
            for(unsigned int b=0; b<=N && b+ind<numAmpl; b++)
                deriv[b+ind] += val * (Bspl[b] - Bder[b] / Qder);
        }
        if(deriv2) {
            for(unsigned int b=0; b<=N && b+ind<numAmpl; b++)
                for(unsigned int c=0; c<=N && c+ind<numAmpl; c++)
                    deriv2[(b+ind) * numAmpl + c+ind] +=
                        val * ( Bspl[b] * Bspl[c] -
                        (Bspl[b] * Bder[c] + Bspl[c] * Bder[b]) / Qder +
                        2 * Bder[b] * Bder[c] / pow_2(Qder) );
        }
    }

    // output the log-derivative: d (ln G_0) / d A_k = (d G_0 / d A_k) / G_0
    if(deriv) {
        for(unsigned int k=0; k<numAmpl; k++)
            deriv[k] /= integral[0];
    }
    // d^2 (ln G_0) / d A_k d A_l = d^2 G_0 / d A_k d A_l - (d ln G_0 / d A_k) (d ln G_0 / d A_l)
    if(deriv2) {
        for(unsigned int kl=0; kl<pow_2(numAmpl); kl++)
            deriv2[kl] = deriv2[kl] / integral[0] - deriv[kl / numAmpl] * deriv[kl % numAmpl];
    }
    // if necessary, return G_d/G_0, d=1,2
    if(GdG0) {
        GdG0[0] = integral[1] / integral[0];
        GdG0[1] = integral[2] / integral[0];
    }
    // put back the offset in the logarithm of the computed value of G_0
    return log(integral[0]) + offset;
}


/** Class for performing the search of the smoothing parameter lambda that meets some goal.
    There are two regimes:
    1) find the maximum of cross-validation score (used with one-dimensional findMin routine);
    2) search for lambda that yields the required value of log-likelihood (if params.targetLogL != 0).
    In either case, we find the best-fit amplitudes of basis functions for the current choice of lambda,
    and then if the fit converged and the goal is closer (i.e. the cross-validation score is higher
    or the difference between logL and targetLogL is smaller than any previous value),
    we also update the best-fit amplitudes in params.ampl, so that on the next iteration the search
    would start from a better initial point. This also improves the robustness of the entire procedure.
*/
template<int N>
class SplineLogDensityLambdaFinder: public IFunctionNoDeriv {
public:
    SplineLogDensityLambdaFinder(const SplineLogDensityFitter<N>& _fitter, SplineLogFitParams& _params) :
        fitter(_fitter), params(_params) {}
private:
    virtual double value(const double scaledLambda) const
    {
        bool useCV = params.targetLogL==0;   // whether we are in the minimizer or root-finder mode
        params.lambda = exp( 1 / scaledLambda - 1 / (1-scaledLambda) );
        std::vector<double> result(params.ampl);
        int numIter   = findRootNdimDeriv(fitter, &params.ampl[0],
            1e-8*params.gradNorm, 100, &result[0]);
        double logL   = fitter.logL(result);
        double logLcv = fitter.logLcv(result);
        bool converged= numIter>0;  // check for convergence (numIter positive)
        if(utils::verbosityLevel >= utils::VL_VERBOSE) {
            utils::msg(utils::VL_VERBOSE, "splineLogDensity",
                "lambda="+utils::toString(params.lambda)+", #iter="+utils::toString(numIter)+
                ", logL= "+utils::toString(logL)+", CV="+utils::toString(logLcv)+
                (!converged ? " did not converge" : params.best < logLcv ? " improved" : ""));
        }
        if(useCV) {  // we are searching for the highest cross-validation score
            if( params.best < logLcv && converged)
            {   // update the best-fit params and the starting point for fitting
                params.best = logLcv;
                params.ampl = result;
            }
            return -logLcv;
        } else {  // we are searching for the target value of logL
            double difference = params.targetLogL - logL;
            if(fabs(difference) < params.best && converged) {
                params.best = fabs(difference);
                params.ampl = result;
            }
            return difference;
        }
    }
    const SplineLogDensityFitter<N>& fitter;
    SplineLogFitParams& params;
};
}  // internal namespace

template<int N>
std::vector<double> splineLogDensity(const std::vector<double> &grid,
    const std::vector<double> &xvalues, const std::vector<double> &weights,
    FitOptions options, double smoothing)
{
    SplineLogFitParams params;
    const SplineLogDensityFitter<N> fitter(grid, xvalues,
        weights.empty()? std::vector<double>(xvalues.size(), 1./xvalues.size()) : weights,
        options, params);
    if(N==1) { // find the best-fit amplitudes without any smoothing
        std::vector<double> result(params.ampl);
        int numIter = findRootNdimDeriv(fitter, &params.ampl[0], 1e-8*params.gradNorm, 100, &result[0]);
        if(numIter>0)  // check for convergence
            params.ampl = result;
        utils::msg(utils::VL_VERBOSE, "splineLogDensity", 
            "#iter="+utils::toString(numIter)+", logL="+utils::toString(fitter.logL(result))+
            ", CV="+utils::toString(fitter.logLcv(result))+(numIter<=0 ? " did not converge" : ""));
    } else {
        // Find the value of lambda and corresponding amplitudes that maximize the cross-validation score.
        // Normally lambda is a small number ( << 1), but it ranges from 0 to infinity,
        // so the root-finder uses a scaling transformation, such that scaledLambda=1
        // corresponds to lambda=0 and scaledLambda=0 -- to lambda=infinity.
        // However, we don't use the entire interval from 0 to infinity, to avoid singularities:
        const double MINSCALEDLAMBDA = 0.12422966;  // corresponds to lambda = 1000, rather arbitrary
        const double MAXSCALEDLAMBDA = 0.971884607; // corresponds to lambda = 1e-15
        // Since the minimizer first computes the function at the left endpoint of the interval
        // and then at the right endpoint, this leads to first performing an oversmoothed fit
        // (large lambda), which should yield a reasonable 'gaussian' first approximation,
        // then a fit with almost no smoothing, which starts with an already more reasonable
        // initial guess and thus has a better chance to converge.
        const SplineLogDensityLambdaFinder<N> finder(fitter, params);
        findMin(finder, MINSCALEDLAMBDA, MAXSCALEDLAMBDA, NAN, 1e-4);
        if(smoothing>0) {
            // target value of log-likelihood is allowed to be worse than
            // the best value for the case of no smoothing by an amount
            // that is proportional to the expected rms variation of logL
            params.best = smoothing * fitter.logLrms(params.ampl);
            params.targetLogL = fitter.logL(params.ampl) - params.best;
            findRoot(finder, MINSCALEDLAMBDA, MAXSCALEDLAMBDA, 1e-4);
        }
    }
    return fitter.getNormalizedAmplitudes(params.ampl);
}

// force the template instantiations to compile
template std::vector<double> splineLogDensity<1>(
    const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, FitOptions, double);
template std::vector<double> splineLogDensity<3>(
    const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, FitOptions, double);

//------------ GENERATION OF UNEQUALLY SPACED GRIDS ------------//

std::vector<double> createUniformGrid(unsigned int nnodes, double xmin, double xmax)
{
    if(nnodes<2 || xmax<=xmin)
        throw std::invalid_argument("Invalid parameters for grid creation");
    std::vector<double> grid(nnodes);
    for(unsigned int k=1; k<nnodes-1; k++)
        grid[k] = (xmin * (nnodes-1-k) + xmax * k) / (nnodes-1);
    grid.front() = xmin;
    grid.back()  = xmax;
    return grid;
}

std::vector<double> createExpGrid(unsigned int nnodes, double xmin, double xmax)
{
    if(nnodes<2 || xmin<=0 || xmax<=xmin)
        throw std::invalid_argument("Invalid parameters for grid creation");
    double logmin = log(xmin), logmax = log(xmax);
    std::vector<double> grid(nnodes);
    grid.front() = xmin;
    grid.back()  = xmax;
    for(unsigned int k=1; k<nnodes-1; k++)
        grid[k] = exp(logmin + k*(logmax-logmin)/(nnodes-1));
    return grid;
}

// Creation of grid with cells increasing first near-linearly, then near-exponentially
class GridSpacingFinder: public IFunctionNoDeriv {
public:
    GridSpacingFinder(double _dynrange, int _nnodes) : dynrange(_dynrange), nnodes(_nnodes) {};
    virtual double value(const double A) const {
        return (A==0) ? nnodes-dynrange :
            (exp(A*nnodes)-1)/(exp(A)-1) - dynrange;
    }
private:
    double dynrange;
    int nnodes;
};

std::vector<double> createNonuniformGrid(unsigned int nnodes, double xmin, double xmax, bool zeroelem)
{   // create grid so that x_k = B*(exp(A*k)-1)
    if(nnodes<2 || xmin<=0 || xmax<=xmin)
        throw std::invalid_argument("Invalid parameters for grid creation");
    double A, B, dynrange=xmax/xmin;
    std::vector<double> grid(nnodes);
    int indexstart=zeroelem?1:0;
    if(zeroelem) {
        grid[0] = 0;
        nnodes--;
    }
    if(fcmp(static_cast<double>(nnodes), dynrange, 1e-6)==0) { // no need for non-uniform grid
        for(unsigned int i=0; i<nnodes; i++)
            grid[i+indexstart] = xmin+(xmax-xmin)*i/(nnodes-1);
        return grid;
    }
    // solve for A:  dynrange = (exp(A*nnodes)-1)/(exp(A)-1)
    GridSpacingFinder F(dynrange, nnodes);
    // first localize the root coarsely, to avoid overflows in root solver
    double Amin=0, Amax=0;
    double step=1;
    while(step>10./nnodes)
        step/=2;
    if(dynrange>nnodes) {
        while(Amax<10 && F(Amax)<=0)
            Amax+=step;
        Amin = Amax-step;
    } else {
        while(Amin>-10 && F(Amin)>=0)
            Amin-=step;
        Amax = Amin+step;
    }
    A = findRoot(F, Amin, Amax, 1e-4);
    B = xmin / (exp(A)-1);
    for(unsigned int i=0; i<nnodes; i++)
        grid[i+indexstart] = B*(exp(A*(i+1))-1);
    grid[nnodes-1+indexstart] = xmax;
    return grid;
}

/// creation of a grid with minimum guaranteed number of input points per bin
static void makegrid(std::vector<double>::iterator begin, std::vector<double>::iterator end,
    double startval, double endval)
{
    double step=(endval-startval)/(end-begin-1);
    while(begin!=end){
        *begin=startval;
        startval+=step;
        ++begin;
    }
    *(end-1)=endval;  // exact value
}

std::vector<double> createAlmostUniformGrid(unsigned int gridsize, 
    const std::vector<double> &srcpoints_unsorted, unsigned int minbin)
{
    if(srcpoints_unsorted.size()==0)
        throw std::invalid_argument("Error in creating a grid: input points array is empty");
    if(gridsize < 2 || (gridsize-1)*minbin > srcpoints_unsorted.size())
        throw std::invalid_argument("Invalid grid size");
    std::vector<double> srcpoints(srcpoints_unsorted);
    std::sort(srcpoints.begin(), srcpoints.end());
    std::vector<double> grid(gridsize);
    std::vector<double>::iterator gridbegin=grid.begin(), gridend=grid.end();
    std::vector<double>::const_iterator srcbegin=srcpoints.begin(), srcend=srcpoints.end();
    std::vector<double>::const_iterator srciter;
    std::vector<double>::iterator griditer;
    bool ok=true, directionBackward=false;
    int numChangesDirection=0;
    do{
        makegrid(gridbegin, gridend, *srcbegin, *(srcend-1));
        ok=true; 
        // find the index of bin with the largest number of points
        int largestbin=-1;
        unsigned int maxptperbin=0;
        for(srciter=srcbegin, griditer=gridbegin; griditer!=gridend-1; ++griditer) {
            unsigned int ptperbin=0;
            while(srciter+ptperbin!=srcend && *(srciter+ptperbin) < *(griditer+1)) 
                ++ptperbin;
            if(ptperbin>maxptperbin) {
                maxptperbin=ptperbin;
                largestbin=griditer-grid.begin();
            }
            srciter+=ptperbin;
        }
        // check that all bins contain at least minbin srcpoints
        if(!directionBackward) {  // forward scan
            srciter = srcbegin;
            griditer = gridbegin;
            while(ok && griditer!=gridend-1) {
                unsigned int ptperbin=0;
                while(srciter+ptperbin!=srcend && *(srciter+ptperbin) < *(griditer+1)) 
                    ptperbin++;
                if(ptperbin>=minbin)  // ok, move to the next one
                {
                    ++griditer;
                    srciter+=ptperbin;
                } else {  // assign minbin points and decrease the available grid interval from the front
                    if(griditer-grid.begin() < largestbin) { 
                        // bad bin is closer to the grid front; move gridbegin forward
                        while(ptperbin<minbin && srciter+ptperbin!=srcend) 
                            ptperbin++;
                        if(srciter+ptperbin==srcend)
                            directionBackward=true; // oops, hit the end of array..
                        else {
                            srcbegin=srciter+ptperbin;
                            gridbegin=griditer+1;
                        }
                    } else {
                        directionBackward=true;
                    }   // will restart scanning from the end of the grid
                    ok=false;
                }
            }
        } else {  // backward scan
            srciter = srcend-1;
            griditer = gridend-1;
            while(ok && griditer!=gridbegin) {
                unsigned int ptperbin=0;
                while(srciter+1-ptperbin!=srcbegin && *(srciter-ptperbin) >= *(griditer-1))
                    ptperbin++;
                if(ptperbin>=minbin)  // ok, move to the previous one
                {
                    --griditer;
                    if(srciter+1-ptperbin==srcbegin)
                        srciter=srcbegin;
                    else
                        srciter-=ptperbin;
                } else {  // assign minbin points and decrease the available grid interval from the back
                    if(griditer-grid.begin() <= largestbin) { 
                        // bad bin is closer to the grid front; reset direction to forward
                        directionBackward=false;
                        numChangesDirection++;
                        if(numChangesDirection>10) {
//                            my_message(FUNCNAME, "grid creation seems not to converge?");
                            return grid;  // don't run forever but would not fulfill the minbin condition
                        }
                    } else {
                        // move gridend backward
                        while(ptperbin<minbin && srciter-ptperbin!=srcbegin) 
                            ++ptperbin;
                        if(srciter-ptperbin==srcbegin) {
                            directionBackward=false;
                            numChangesDirection++;
                            if(numChangesDirection>10) {
//                                my_message(FUNCNAME, "grid creation seems not to converge?");
                                return grid;  // don't run forever but would not fulfill the minbin condition
                            }
                        } else {
                            srcend=srciter-ptperbin+1;
                            gridend=griditer;
                        }
                    }
                    ok=false;
                }
            }
        }
    } while(!ok);
    return grid;
}

std::vector<double> mirrorGrid(const std::vector<double> &input)
{
    unsigned int size = input.size();
    if(size==0 || input[0]!=0)
        throw std::invalid_argument("incorrect input in mirrorGrid");
    std::vector<double> output(size*2-1);
    output[size-1] = 0;
    for(unsigned int i=1; i<size; i++) {
        if(input[i] <= input[i-1])
            throw std::invalid_argument("incorrect input in mirrorGrid");
        output[size-1-i] = -input[i];
        output[size-1+i] =  input[i];
    }
    return output;
}

}  // namespace
