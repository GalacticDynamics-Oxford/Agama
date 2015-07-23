#include "math_spline.h"
#include "math_core.h"
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_bspline.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multifit.h>

namespace math {

//-------------- PENALIZED SPLINE APPROXIMATION ---------------//

/// Implementation of penalized spline approximation
class SplineApproxImpl {
private:
    const size_t numDataPoints;     ///< number of x[i],y[i] pairs (original data)
    const size_t numKnots;          ///< number of X[k] knots in the fitting spline; the number of basis functions is numKnots+2
    gsl_vector* knots;              ///< b-spline knots  X[k], k=0..numKnots-1
    gsl_vector* xvalues;            ///< x[i], i=0..numDataPoints-1
    gsl_vector* yvalues;            ///< y[i], overwritten each time loadYvalues is called
    gsl_vector* weightCoefs;        ///< w_p, weight coefficients for basis functions to be found in the process of fitting 
    gsl_vector* zRHS;               ///< z_p = C^T y, right hand side of linear system
    gsl_matrix* bsplineMatrix;      ///< matrix "C_ip" used in fitting process; i=0..numDataPoints-1, p=0..numBasisFnc-1
    gsl_matrix* LMatrix;            ///< lower triangular matrix L is Cholesky decomposition of matrix A = C^T C, of size numBasisFnc*numBasisFnc
    gsl_matrix* MMatrix;            ///< matrix "M" which is the transformed version of roughness matrix "R_pq" of integrals of product of second derivatives of basis functions; p,q=0..numBasisFnc-1
    gsl_vector* singValues;         ///< part of the decomposition of the roughness matrix
    gsl_vector* MTz;                ///< pre-computed M^T z
    gsl_bspline_workspace*
        bsplineWorkspace;           ///< workspace for b-spline evaluation
    gsl_vector* bsplineValues;      ///< to compute values of all b-spline basis functions at a given point x
    gsl_bspline_deriv_workspace*
        bsplineDerivWorkspace;      ///< workspace for derivative computation
    gsl_matrix* bsplineDerivValues; ///< to compute values and derivatives of basis functions
    gsl_vector* tempv;              ///< some routines require temporary storage
    double ynorm2;                  ///< |y|^2 - used to compute residual sum of squares (RSS)

public:
    SplineApproxImpl(const std::vector<double> &_xvalues, const std::vector<double> &_knots);
    ~SplineApproxImpl();

    /** load the y-values of data points and precompute zRHS */
    void loadyvalues(const std::vector<double> &_yvalues);

    /** compute integrals over products of second derivatives of basis functions, 
        and transform R to M+singValues  */
    void initRoughnessMatrix();

    /** compute the weights of basis function for the given value of smoothing parameter */
    void computeWeights(double lambda=0);

    /** compute the RMS scatter of data points about the approximating spline,
        and the number of effective degrees of freedom (EDF) */
    void computeRMSandEDF(double lambda, double* rmserror=NULL, double* edf=NULL) const;

    /** compute the value of Akaike information criterion (AIC)
        for the given value of smoothing parameter 'lambda'  */
    double computeAIC(double lambda);

    /** compute Y-values at spline knots X[k], and also two endpoint derivatives, 
        after the weights w have been determined  */
    void computeYvalues(std::vector<double>& splineValues, double& der_left, double& der_right) const;

    /** compute values of spline at an arbitrary set of points  */
    void computeRegressionAtPoints(const std::vector<double> &xpoints, std::vector<double> &ypoints) const;

    /** check if the basis matrix L is singular */
    bool isSingular() const { return LMatrix==NULL; }

private:
    /** In the unfortunate case that the fit matrix appears to be singular, another algorithm
        is used which is based on the GSL multifit routine, which performs SVD of bsplineMatrix.
        It is much slower and cannot accomodate nonzero smoothing. */
    void computeWeightsSingular();
    
    SplineApproxImpl& operator= (const SplineApproxImpl&);  ///< assignment operator forbidden
    SplineApproxImpl(const SplineApproxImpl&);              ///< copy constructor forbidden
};

SplineApproxImpl::SplineApproxImpl(const std::vector<double> &_xvalues, const std::vector<double> &_knots) :
    numDataPoints(_xvalues.size()), numKnots(_knots.size())
{
    knots=xvalues=yvalues=singValues=weightCoefs=zRHS=bsplineValues=NULL;
    bsplineMatrix=LMatrix=MMatrix=NULL;
    bsplineWorkspace=NULL;
    bsplineDerivWorkspace=NULL;

    // first check for validity of input range
    bool range_ok = (numDataPoints>2 && numKnots>2);
    for(size_t k=1; k<numKnots; k++)
        range_ok &= (_knots[k-1]<_knots[k]);  // knots must be in ascending order
    if(!range_ok)
        throw std::invalid_argument("Error in SplineApprox initialization: knots must be in ascending order");
    xvalues = gsl_vector_alloc(numDataPoints);
    knots = gsl_vector_alloc(numKnots);
    if(xvalues==NULL || knots==NULL) {
        gsl_vector_free(xvalues);
        gsl_vector_free(knots);
        throw std::bad_alloc();
    }
    for(size_t i=0; i<numDataPoints; i++)
        gsl_vector_set(xvalues, i, _xvalues[i]);
    for(size_t k=0; k<numKnots; k++)
        gsl_vector_set(knots, k, _knots[k]);
    if(gsl_vector_min(xvalues) < _knots.front() || gsl_vector_max(xvalues) > _knots.back()) 
        throw std::invalid_argument("Error in SplineApprox initialization: "
            "source data points must lie within spline definition region");

    // next allocate b-splines and other matrices
    bsplineWorkspace      = gsl_bspline_alloc(4, numKnots);
    bsplineValues         = gsl_vector_alloc(numKnots+2);
    bsplineDerivWorkspace = gsl_bspline_deriv_alloc(4);
    bsplineDerivValues    = gsl_matrix_alloc(numKnots+2, 3);
    bsplineMatrix= gsl_matrix_alloc(numDataPoints, numKnots+2); // matrix C_ip -- this is the largest chunk of memory to be used
    yvalues      = gsl_vector_alloc(numDataPoints);
    LMatrix      = gsl_matrix_alloc(numKnots+2, numKnots+2);    // lower triangular matrix L obtained by Cholesky decomposition of matrix A = C^T C
    weightCoefs  = gsl_vector_calloc(numKnots+2);               // weight coefficients at basis functions, which are the unknowns in the linear system
    zRHS         = gsl_vector_alloc(numKnots+2);                // z = C^T y, RHS of the linear system
    MTz          = gsl_vector_alloc(numKnots+2);
    tempv        = gsl_vector_alloc(numKnots+2);
    if(bsplineWorkspace==NULL || bsplineValues==NULL || bsplineDerivWorkspace==NULL || 
        bsplineDerivValues==NULL || yvalues==NULL || bsplineMatrix==NULL || LMatrix==NULL || 
        weightCoefs==NULL || zRHS==NULL || MTz==NULL || tempv==NULL) {
        gsl_bspline_free(bsplineWorkspace);
        gsl_bspline_deriv_free(bsplineDerivWorkspace);
        gsl_vector_free(bsplineValues);
        gsl_matrix_free(bsplineDerivValues);
        gsl_vector_free(xvalues);
        gsl_vector_free(yvalues);
        gsl_vector_free(knots);
        gsl_vector_free(weightCoefs);
        gsl_vector_free(zRHS);
        gsl_vector_free(MTz);
        gsl_vector_free(tempv);
        gsl_matrix_free(bsplineMatrix);
        gsl_matrix_free(LMatrix);
        throw std::bad_alloc();
    }
    ynorm2=gsl_nan();  // to indicate that no y-values have been loaded yet

    // initialize b-spline matrix C 
    gsl_bspline_knots(knots, bsplineWorkspace);
    for(size_t i=0; i<numDataPoints; i++) {
        gsl_bspline_eval(_xvalues[i], bsplineValues, bsplineWorkspace);
        for(size_t p=0; p<numKnots+2; p++)
            gsl_matrix_set(bsplineMatrix, i, p, gsl_vector_get(bsplineValues, p));
    }

    // pre-compute matrix L
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1, bsplineMatrix, bsplineMatrix, 0, LMatrix);
    try {
        gsl_linalg_cholesky_decomp(LMatrix);
    }
    catch(std::domain_error&) {   // means that the matrix is not positive definite, i.e. fit is singular
        gsl_matrix_free(LMatrix);
        LMatrix = NULL;
    }
}

SplineApproxImpl::~SplineApproxImpl()
{
    gsl_bspline_free(bsplineWorkspace);
    gsl_bspline_deriv_free(bsplineDerivWorkspace);
    gsl_vector_free(bsplineValues);
    gsl_matrix_free(bsplineDerivValues);
    gsl_vector_free(xvalues);
    gsl_vector_free(yvalues);
    gsl_vector_free(knots);
    gsl_vector_free(weightCoefs);
    gsl_vector_free(zRHS);
    gsl_vector_free(MTz);
    gsl_vector_free(tempv);
    gsl_matrix_free(bsplineMatrix);
    gsl_matrix_free(LMatrix);
    gsl_matrix_free(MMatrix);
    gsl_vector_free(singValues);
}

void SplineApproxImpl::loadyvalues(const std::vector<double> &_yvalues)
{
    if(_yvalues.size() != numDataPoints) 
        throw std::invalid_argument("SplineApprox: input array sizes do not match");
    ynorm2=0;
    for(size_t i=0; i<numDataPoints; i++) {
        gsl_vector_set(yvalues, i, _yvalues[i]);
        ynorm2 += pow_2(_yvalues[i]);
    }
    if(!isSingular())    // precompute z = C^T y
        gsl_blas_dgemv(CblasTrans, 1, bsplineMatrix, yvalues, 0, zRHS);
}

/// convenience function returning values from band matrix or zero if indexes are outside the band
double getVal(const gsl_matrix* deriv, size_t row, size_t col)
{
    if(row<col || row>=col+3) return 0; 
    else return gsl_matrix_get(deriv, row-col, col);
}

void SplineApproxImpl::initRoughnessMatrix()
{
    if(MMatrix != NULL) {  // already computed
        gsl_blas_dgemv(CblasTrans, 1, MMatrix, zRHS, 0, MTz);  // precompute M^T z
        return;
    }
    // init matrix with roughness penalty (integrals of product of second derivatives of basis functions)
    MMatrix = gsl_matrix_calloc(numKnots+2, numKnots+2);   // matrix R_pq
    singValues = gsl_vector_alloc(numKnots+2);   // vector S
    gsl_matrix* tempm  = gsl_matrix_alloc(numKnots+2, numKnots+2);
    gsl_matrix* derivs = gsl_matrix_calloc(3, numKnots);
    if(MMatrix==NULL || singValues==NULL || tempm==NULL || derivs==NULL) { 
        gsl_matrix_free(derivs);
        gsl_matrix_free(tempm);
        gsl_matrix_free(MMatrix);
        gsl_vector_free(singValues);
        throw std::bad_alloc();
    }
    for(size_t k=0; k<numKnots; k++)
    {
        size_t istart, iend;
        gsl_bspline_deriv_eval_nonzero(gsl_vector_get(knots, k), 2, bsplineDerivWorkspace->dB, &istart, &iend, bsplineWorkspace, bsplineDerivWorkspace);
        for(size_t b=0; b<3; b++)
            gsl_matrix_set(derivs, b, k, gsl_matrix_get(bsplineDerivWorkspace->dB, b+k-istart, 2));
    }
    for(size_t p=0; p<numKnots+2; p++)
    {
        size_t kmin = p>3 ? p-3 : 0;
        size_t kmax = std::min<size_t>(p+3,knots->size-1);
        for(size_t q=p; q<std::min<size_t>(p+4,numKnots+2); q++)
        {
            double result=0;
            for(size_t k=kmin; k<kmax; k++)
            {
                double x0 = gsl_vector_get(knots, k);
                double x1 = gsl_vector_get(knots, k+1);
                double Gp = getVal(derivs,p,k)*x1 - getVal(derivs,p,k+1)*x0;
                double Hp = getVal(derivs,p,k+1)  - getVal(derivs,p,k);
                double Gq = getVal(derivs,q,k)*x1 - getVal(derivs,q,k+1)*x0;
                double Hq = getVal(derivs,q,k+1)  - getVal(derivs,q,k);
                result += (Hp*Hq*(pow(x1,3.0)-pow(x0,3.0))/3.0 + (Gp*Hq+Gq*Hp)*(pow_2(x1)-pow_2(x0))/2.0 + Gp*Gq*(x1-x0)) / pow_2(x1-x0);
            }
            gsl_matrix_set(MMatrix, p, q, result);
            gsl_matrix_set(MMatrix, q, p, result);  // it is symmetric
        }
    }

    // now transform the roughness matrix R into more suitable form (so far MMatrix contains R)
    // obtain Q = L^{-1} R L^{-T}, where R is the roughness penalty matrix (store Q instead of R)
    gsl_blas_dtrsm(CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, 1, LMatrix, MMatrix);
    gsl_blas_dtrsm(CblasRight, CblasLower, CblasTrans, CblasNonUnit, 1, LMatrix, MMatrix);   // now MMatrix contains Q = L^{-1} R L^(-T}

    // next decompose this Q via singular value decomposition: Q = U * diag(SV) * V^T
    gsl_linalg_SV_decomp(MMatrix, tempm, singValues, tempv);   // now MMatrix contains U, and workm contains V^T
    // Because Q was symmetric and positive definite, we expect that U=V, but don't actually check it.
    gsl_vector_set(singValues, numKnots, 0);   // the smallest two singular values must be zero; set explicitly to avoid roundoff error
    gsl_vector_set(singValues, numKnots+1, 0);

    // precompute M = L^{-T} U  which is used in computing basis weight coefs.
    gsl_blas_dtrsm(CblasLeft, CblasLower, CblasTrans, CblasNonUnit, 1, LMatrix, MMatrix);   // now M is finally in place
    // now the weight coefs for any lambda are given by  w = M (I + lambda*diag(singValues))^{-1} M^T  z

    gsl_matrix_free(tempm);
    gsl_matrix_free(derivs);

    gsl_blas_dgemv(CblasTrans, 1, MMatrix, zRHS, 0, MTz);  // precompute M^T z
}

// obtain solution of linear system for the given smoothing parameter, store the weights of basis functions in weightCoefs
void SplineApproxImpl::computeWeights(double lambda)
{
    if(isSingular()) {
        computeWeightsSingular();
        return;
    }
    if(lambda==0)  // simple case, no need to use roughness penalty matrix
        gsl_linalg_cholesky_solve(LMatrix, zRHS, weightCoefs);
    else {
        for(size_t p=0; p<numKnots+2; p++) {
            double sv = gsl_vector_get(singValues, p);
            gsl_vector_set(tempv, p, gsl_vector_get(MTz, p) / (1 + (sv>0 ? sv*lambda : 0)));
        }
        gsl_blas_dgemv(CblasNoTrans, 1, MMatrix, tempv, 0, weightCoefs);
    }
}

// compute weights of basis functions in the case that the matrix is singular
void SplineApproxImpl::computeWeightsSingular()
{
    gsl_matrix* covarMatrix  = gsl_matrix_alloc(numKnots+2, numKnots+2);
    gsl_multifit_linear_workspace* fitWorkspace = gsl_multifit_linear_alloc(numDataPoints, numKnots+2);
    if(covarMatrix==NULL || fitWorkspace==NULL) {
        gsl_multifit_linear_free(fitWorkspace);
        gsl_matrix_free(covarMatrix);
        throw std::bad_alloc();
    }
    double chisq;
    size_t rank;
    gsl_multifit_linear_svd(bsplineMatrix, yvalues, 1e-8, &rank, weightCoefs, covarMatrix, &chisq, fitWorkspace);
    gsl_multifit_linear_free(fitWorkspace);
    gsl_matrix_free(covarMatrix);
    ynorm2 = chisq/numDataPoints;
}

void SplineApproxImpl::computeRMSandEDF(double lambda, double* rmserror, double* edf) const
{
    if(rmserror == NULL && edf == NULL)
        return;
    if(isSingular()) {
        if(rmserror)
            *rmserror = ynorm2;
        if(*edf)
            *edf = static_cast<double>(numKnots+2);
        return;
    }
    gsl_vector_memcpy(tempv, weightCoefs);
    gsl_blas_dtrmv(CblasLower, CblasTrans, CblasNonUnit, LMatrix, tempv);
    double wTz;
    gsl_blas_ddot(weightCoefs, zRHS, &wTz);
    if(rmserror)
        *rmserror = (ynorm2 - 2*wTz + pow_2(gsl_blas_dnrm2(tempv))) / numDataPoints;
    if(edf == NULL)
        return;
    // equivalent degrees of freedom
    *edf = 0;
    if(!gsl_finite(lambda))
        *edf = 2;
    else if(lambda>0) 
        for(size_t c=0; c<numKnots+2; c++)
            *edf += 1/(1+lambda*gsl_vector_get(singValues, c));
    else
        *edf = static_cast<double>(numKnots+2);
}

double SplineApproxImpl::computeAIC(double lambda) {
    double rmserror, edf;
    computeWeights(lambda);
    computeRMSandEDF(lambda, &rmserror, &edf);
    return log(rmserror) + 2*edf/(numDataPoints-edf-1);
}

/// after the weights of basis functions have been determined, evaluate the values of approximating spline 
/// at its nodes, and additionally its derivatives at endpoints
void SplineApproxImpl::computeYvalues(std::vector<double>& splineValues, double& der_left, double& der_right) const
{
    splineValues.assign(numKnots, 0);
    for(size_t k=1; k<numKnots-1; k++) {  // loop over interior nodes
        gsl_bspline_eval(gsl_vector_get(knots, k), bsplineValues, bsplineWorkspace);
        double val=0;
        for(size_t p=0; p<numKnots+2; p++)
            val += gsl_vector_get(bsplineValues, p) * gsl_vector_get(weightCoefs, p);
        splineValues[k] = val;
    }
    for(size_t k=0; k<numKnots; k+=numKnots-1) {  // two endpoints: values and derivatives
        gsl_bspline_deriv_eval(gsl_vector_get(knots, k), 1, bsplineDerivValues, bsplineWorkspace, bsplineDerivWorkspace);
        double val=0, der=0;
        for(size_t p=0; p<numKnots+2; p++) {
            val += gsl_matrix_get(bsplineDerivValues, p, 0) * gsl_vector_get(weightCoefs, p);
            der += gsl_matrix_get(bsplineDerivValues, p, 1) * gsl_vector_get(weightCoefs, p);
        }
        splineValues[k] = val;
        if(k==0)
            der_left = der;
        else
            der_right = der;
    }
}

void SplineApproxImpl::computeRegressionAtPoints(const std::vector<double> &xpoints, std::vector<double> &ypoints) const
{
    ypoints.assign(xpoints.size(), NAN);  // default value for nodes outside the definition range
    for(size_t i=0; i<xpoints.size(); i++)  // loop over interior nodes
        if(xpoints[i]>=gsl_vector_get(knots, 0) && xpoints[i]<=gsl_vector_get(knots, numKnots-1)) {
            gsl_bspline_eval(xpoints[i], bsplineValues, bsplineWorkspace);
            double val=0;
            for(size_t p=0; p<numKnots+2; p++)
                val += gsl_vector_get(bsplineValues, p) * gsl_vector_get(weightCoefs, p);
            ypoints[i] = val;
        }
}

//-------- helper class for root-finder -------//
class SplineAICRootFinder: public IFunctionNoDeriv {
public:
    SplineAICRootFinder(SplineApproxImpl& _impl, double _targetAIC) :
        impl(_impl), targetAIC(_targetAIC) {};
    virtual double value(double lambda) const {
        return impl.computeAIC(lambda) - targetAIC;
    }
private:
    SplineApproxImpl& impl;
    double targetAIC;       ///< target value of AIC for root-finder
};

//----------- DRIVER CLASS FOR PENALIZED SPLINE APPROXIMATION ------------//

SplineApprox::SplineApprox(const std::vector<double> &xvalues, const std::vector<double> &knots)
{
    impl = new SplineApproxImpl(xvalues, knots);
}

SplineApprox::~SplineApprox()
{
    delete static_cast<SplineApproxImpl*>(impl);
}

bool SplineApprox::isSingular() const {
    return static_cast<SplineApproxImpl*>(impl)->isSingular();
}

void SplineApprox::fitData(const std::vector<double> &yvalues, const double lambda,
    std::vector<double>& splineValues, double& deriv_left, double& deriv_right, double *rmserror, double* edf)
{
    SplineApproxImpl& imp = *static_cast<SplineApproxImpl*>(impl);
    imp.loadyvalues(yvalues);
    if(imp.isSingular() || lambda==0)
        imp.computeWeights();
    else {
        imp.initRoughnessMatrix();
        imp.computeWeights(lambda);
    }
    imp.computeYvalues(splineValues, deriv_left, deriv_right);
    imp.computeRMSandEDF(lambda, rmserror, edf);
}

void SplineApprox::fitDataOversmooth(const std::vector<double> &yvalues, const double deltaAIC,
    std::vector<double>& splineValues, double& deriv_left, double& deriv_right, 
    double *rmserror, double* edf, double *lambda)
{
    SplineApproxImpl& imp = *static_cast<SplineApproxImpl*>(impl);
    imp.loadyvalues(yvalues);
    double lambdaFit = 0;
    if(imp.isSingular()) {
        imp.computeWeights();
    } else {
        imp.initRoughnessMatrix();
        if(deltaAIC <= 0) {  // find optimal fit
            SplineAICRootFinder fnc(imp, 0);
            lambdaFit = findMin(fnc, 0, INFINITY, NAN, 1e-6);  // no initial guess
        } else {  // allow for somewhat higher AIC value, to smooth more than minimum necessary amount
            SplineAICRootFinder fnc(imp, imp.computeAIC(0) + deltaAIC);
            lambdaFit = findRoot(fnc, 0, INFINITY, 1e-6);
            if(!isFinite(lambdaFit))   // root does not exist, i.e. function is everywhere lower than target value
                lambdaFit = INFINITY;  // basically means fitting with a linear regression
        }
    }
    imp.computeYvalues(splineValues, deriv_left, deriv_right);
    imp.computeRMSandEDF(lambdaFit, rmserror, edf);
    if(lambda!=NULL)
        *lambda=lambdaFit;
}

void SplineApprox::fitDataOptimal(const std::vector<double> &yvalues,
    std::vector<double>& splineValues, double& deriv_left, double& deriv_right, 
    double *rmserror, double* edf, double *lambda) 
{
    fitDataOversmooth(yvalues, 0.0, splineValues, deriv_left, deriv_right, rmserror, edf, lambda);
}


//-------------- CUBIC SPLINE --------------//

/*  Clamped or natural cubic splines;
    the implementation is based on the code for natural cubic splines from GSL, original author:  G. Jungman
*/

// if one wants to have a 'natural' spline boundary condition then pass NaN as the value of derivative.
CubicSpline::CubicSpline(const std::vector<double>& xa, const std::vector<double>& ya, double der1, double der2) :
    xval(xa), yval(ya)
{
    size_t num_points = xa.size();
    if(ya.size() != num_points)
        throw std::invalid_argument("Error in spline initialization: input arrays are not equal in length");
    if(num_points < 3)
        throw std::invalid_argument("Error in spline initialization: number of nodes should be >=3");
    size_t max_index = num_points - 1;  /* Engeln-Mullges + Uhlig "n" */
    size_t sys_size = max_index - 1;    /* linear system is sys_size x sys_size */
    cval.assign(num_points, 0);
    std::vector<double> g(sys_size), diag(sys_size), offdiag(sys_size);  // temporary arrays

    for (size_t i = 0; i < sys_size; i++) {
        const double h_i   = xa[i + 1] - xa[i];
        const double h_ip1 = xa[i + 2] - xa[i + 1];
        if(h_i<=0 || h_ip1<=0)
            throw std::invalid_argument("Error in spline initialization: x values are not monotonic");
        const double ydiff_i   = ya[i + 1] - ya[i];
        const double ydiff_ip1 = ya[i + 2] - ya[i + 1];
        const double g_i = (h_i != 0.0) ? 1.0 / h_i : 0.0;
        const double g_ip1 = (h_ip1 != 0.0) ? 1.0 / h_ip1 : 0.0;
        offdiag[i] = h_ip1;
        diag[i] = 2.0 * (h_ip1 + h_i);
        g[i] = 3.0 * (ydiff_ip1 * g_ip1 -  ydiff_i * g_i);
        if(i == 0 && der1==der1) {
            diag[i] = 1.5 * h_i + 2.0 * h_ip1;
            g[i] = 3.0 * (ydiff_ip1 * g_ip1 - 1.5 * ydiff_i * g_i + 0.5 * der1);
        }
        if(i == sys_size-1 && der2==der2) {
            diag[i] = 1.5 * h_ip1 + 2.0 * h_i;
            g[i] = 3.0 * (1.5 * ydiff_ip1 * g_ip1 - 0.5 * der2 - ydiff_i * g_i);
        }
    }

    if (sys_size == 1) {
        cval[1] = g[0] / diag[0];
    } else {
        gsl_vector_view g_vec = gsl_vector_view_array(&(g.front()), sys_size);
        gsl_vector_view diag_vec = gsl_vector_view_array(&(diag.front()), sys_size);
        gsl_vector_view offdiag_vec = gsl_vector_view_array(&(offdiag.front()), sys_size - 1);
        gsl_vector_view solution_vec = gsl_vector_view_array(&(cval[1]), sys_size); 
        int status = gsl_linalg_solve_symm_tridiag(&diag_vec.vector, &offdiag_vec.vector, &g_vec.vector, &solution_vec.vector);
        if(status != GSL_SUCCESS)
            throw std::runtime_error("Error in spline initialization");
        if(der1==der1) 
            cval[0] = ( 3.0*(ya[1]-ya[0])/(xa[1]>xa[0] ? xa[1]-xa[0] : 1) 
                - 3.0*der1 - cval[1]*(xa[1]-xa[0]) )*0.5/(xa[1]>xa[0] ? xa[1]-xa[0] : 1);
        else cval[0]=0.0;
        if(der2==der2)
            cval[max_index] = -( 3*(ya[max_index]-ya[max_index-1])/(xa[max_index]-xa[max_index-1]) 
                - 3*der2 + cval[max_index-1]*(xa[max_index]-xa[max_index-1]) )*0.5/(xa[max_index]-xa[max_index-1]);
        else cval[max_index]=0.0;
    }
}

// evaluate spline value, derivative and 2nd derivative at once (faster than doing it separately)
void CubicSpline::evalDeriv(const double x, double* val, double* deriv, double* deriv2) const
{
    const size_t size = xval.size();
    if(size==0)
        throw std::range_error("Empty spline");
    if(x <= xval[0]) {
        double dx  =  xval[1]-xval[0];
        double der = (yval[1]-yval[0])/dx - dx*(cval[1]+2*cval[0])/3.0;
        if(val)
            *val   = yval[0] + der*(x-xval[0]);
        if(deriv)
            *deriv = der;
        if(deriv2)
            *deriv2= 0;
        return;
    }
    if(x >= xval[size-1]) {
        double dx  =  xval[size-1]-xval[size-2];
        double der = (yval[size-1]-yval[size-2])/dx + dx*(cval[size-2]+2*cval[size-1])/3.0;
        if(val)
            *val   = yval[size-1] + der*(x-xval[size-1]);
        if(deriv)
            *deriv = der;
        if(deriv2)
            *deriv2= 0;
        return;
    }

    size_t index = 0;
    size_t indhi = size-1;
    while(indhi > index + 1) {    // binary search to determine the spline segment that contains x
        size_t i = (indhi + index)/2;
        if(xval[i] > x)
            indhi = i;
        else
            index = i;
    }
    double x_hi = xval[index + 1];
    double x_lo = xval[index];
    double dx   = x_hi - x_lo;
    double y_lo = yval[index];
    double y_hi = yval[index + 1];
    double dy   = y_hi - y_lo;
    double delx = x - x_lo;
    double c_i  = cval[index];
    double c_ip1= cval[index+1];
    double b_i  = (dy / dx) - dx * (c_ip1 + 2.0 * c_i) / 3.0;
    double d_i  = (c_ip1 - c_i) / (3.0 * dx);
    if(val)
        *val    = y_lo + delx * (b_i + delx * (c_i + delx * d_i));
    if(deriv)
        *deriv  = b_i + delx * (2.0 * c_i + 3.0 * d_i * delx);
    if(deriv2)
        *deriv2 = 2.0 * c_i + 6.0 * d_i * delx;
}

bool CubicSpline::isMonotonic() const
{
    if(xval.size()==0)
        throw std::range_error("Empty spline");
    bool ismonotonic=true;
    for(size_t index=0; ismonotonic && index < xval.size()-1; index++) {
        double dx = xval[index + 1] - xval[index];
        double dy = yval[index + 1] - yval[index];
        double c_i   = cval[index];
        double c_ip1 = cval[index+1];
        double a = dx * (c_ip1 - c_i);
        double b = 2 * dx * c_i;
        double c = (dy / dx) - dx * (c_ip1 + 2.0 * c_i) / 3.0;
        // derivative is  a * chi^2 + b * chi + c,  with 0<=chi<=1 on the given interval.
        double D = b*b-4*a*c;
        if(D>=0) { // need to check roots
            double chi1 = (-b-sqrt(D))/(2*a);
            double chi2 = (-b+sqrt(D))/(2*a);
            if( (chi1>=0 && chi1<=1) || (chi2>=0 && chi2<=1) )
                ismonotonic=false;    // there is a root ( y'=0 ) somewhere on the given interval
        }  // otherwise there are no roots
    }
    return ismonotonic;
}


//------------ GENERATION OF UNEQUALLY SPACED GRIDS ------------//

// Creation of grid with exponentially increasing cells
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

void createNonuniformGrid(size_t nnodes, double xmin, double xmax, bool zeroelem, std::vector<double>& grid)
{   // create grid so that x_k = B*(exp(A*k)-1)
    double A, B, dynrange=xmax/xmin;
    grid.resize(nnodes);
    int indexstart=zeroelem?1:0;
    if(zeroelem) {
        grid[0] = 0;
        nnodes--;
    }
    if(fcmp(static_cast<double>(nnodes), dynrange, 1e-6)==0) { // no need for non-uniform grid
        for(size_t i=0; i<nnodes; i++)
            grid[i+indexstart] = xmin+(xmax-xmin)*i/(nnodes-1);
        return;
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
    for(size_t i=0; i<nnodes; i++)
        grid[i+indexstart] = B*(exp(A*(i+1))-1);
    grid[nnodes-1+indexstart] = xmax;
}

/// creation of a grid with minimum guaranteed number of input points per bin
static void makegrid(std::vector<double>::iterator begin, std::vector<double>::iterator end, double startval, double endval)
{
    double step=(endval-startval)/(end-begin-1);
    while(begin!=end){
        *begin=startval;
        startval+=step;
        begin++;
    }
    *(end-1)=endval;  // exact value
}

void createAlmostUniformGrid(const std::vector<double> &srcpoints, size_t minbin, size_t& gridsize, std::vector<double>& grid)
{
    if(srcpoints.size()==0)
        throw std::invalid_argument("Error in creating a grid: input points array is empty");
    gridsize = std::max<size_t>(2, std::min<size_t>(gridsize, static_cast<size_t>(srcpoints.size()/minbin)));
    grid.resize(gridsize);
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
        size_t maxptperbin=0;
        for(srciter=srcbegin, griditer=gridbegin; griditer!=gridend-1; griditer++) {
            size_t ptperbin=0;
            while(srciter+ptperbin!=srcend && *(srciter+ptperbin) < *(griditer+1)) 
                ptperbin++;
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
                size_t ptperbin=0;
                while(srciter+ptperbin!=srcend && *(srciter+ptperbin) < *(griditer+1)) 
                    ptperbin++;
                if(ptperbin>=minbin)  // ok, move to the next one
                {
                    griditer++;
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
                size_t ptperbin=0;
                while(srciter+1-ptperbin!=srcbegin && *(srciter-ptperbin) >= *(griditer-1))
                    ptperbin++;
                if(ptperbin>=minbin)  // ok, move to the previous one
                {
                    griditer--;
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
                            return;  // don't run forever but would not fulfill the minbin condition
                        }
                    } else {
                        // move gridend backward
                        while(ptperbin<minbin && srciter-ptperbin!=srcbegin) 
                            ptperbin++;
                        if(srciter-ptperbin==srcbegin) {
                            directionBackward=false;
                            numChangesDirection++;
                            if(numChangesDirection>10) {
//                                my_message(FUNCNAME, "grid creation seems not to converge?");
                                return;  // don't run forever but would not fulfill the minbin condition
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
}

}  // namespace
