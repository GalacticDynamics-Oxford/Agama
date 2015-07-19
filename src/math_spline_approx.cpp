#include "math_spline_approx.h"
#include "mathutils.h"
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_bspline.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multifit.h>

namespace mathutils {
/* ------------- class for performing penalized spline approximation --------------- */

class SplineApproxImpl: public IFunction {
private:
    const size_t numDataPoints;     ///< number of x[i],y[i] pairs (original data)
    const size_t numKnots;          ///< number of X[k] knots in the fitting spline; the number of basis functions is numKnots+2
    gsl_vector* knots;              ///< b-spline knots  X[k], k=0..numKnots-1
    gsl_vector* xvalues;            ///< x[i], i=0..numDataPoints-1
    gsl_vector* yvalues;            ///< y[i], overwritten each time loadYvalues is called
    mutable gsl_vector* weightCoefs;///< w_p, weight coefficients for basis functions to be found in the process of fitting 
    gsl_vector* zRHS;               ///< z_p = C^T y, right hand side of linear system
    gsl_matrix* bsplineMatrix;      ///< matrix "C_ip" used in fitting process; i=0..numDataPoints-1, p=0..numBasisFnc-1
    gsl_matrix* LMatrix;            ///< lower triangular matrix L is Cholesky decomposition of matrix A = C^T C, of size numBasisFnc*numBasisFnc
    gsl_matrix* MMatrix;            ///< matrix "M" which is the transformed version of roughness matrix "R_pq" of integrals of product of second derivatives of basis functions; p,q=0..numBasisFnc-1
    gsl_vector* singValues;         ///< part of the decomposition of the roughness matrix
    gsl_vector* MTz;                ///< pre-computed M^T z
    mutable gsl_vector* tempv;      ///< some routines require temporary storage
    gsl_bspline_workspace*
        bsplineWorkspace;           ///< workspace for b-spline evaluation
    gsl_vector* bsplineValues;      ///< to compute values of all b-spline basis functions at a given point x
    gsl_bspline_deriv_workspace*
        bsplineDerivWorkspace;      ///< workspace for derivative computation
    gsl_matrix* bsplineDerivValues; ///< to compute values and derivatives of basis functions
    mutable double ynorm2;          ///< |y|^2 - used to compute residual sum of squares (RSS)
    double targetAIC;               ///< target value of AIC for root-finder

public:
    SplineApproxImpl(const std::vector<double> &_xvalues, const std::vector<double> &_knots);
    ~SplineApproxImpl();

    void loadyvalues(const std::vector<double> &_yvalues);

    /// compute integrals over products of second derivatives of basis functions, and transform R to M+singValues
    void initRoughnessMatrix();

    void computeWeights(double lambda=0) const;

    void computeRMSandEDF(double lambda, double* rmserror=NULL, double* edf=NULL) const;

    void setTargetAIC(double AIC) { targetAIC=AIC; }

    /** IFunction interface implements the computation of Akaike information criterion (AIC)
        for the given value of smoothing parameter 'lambda'.  */
    virtual void eval_deriv(double lambda, double* AIC=NULL, double* =NULL, double* =NULL) const;

    /// compute Y-values at spline knots X[k], and also two endpoint derivatives, after the weights w have been determined
    void computeYvalues(std::vector<double>& splineValues, double& der_left, double& der_right) const;

    /// compute values of spline at an arbitrary set of points
    void computeRegressionAtPoints(const std::vector<double> &xpoints, std::vector<double> &ypoints) const;

    virtual int numDerivs() const { return 0; }

    bool isSingular() const { return LMatrix==NULL; }

private:
    /** In the unfortunate case that the fit matrix appears to be singular, another algorithm
        is used which is based on the GSL multifit routine, which performs SVD of bsplineMatrix.
        It is much slower and cannot accomodate nonzero smoothing. */
    void computeWeightsSingular() const;

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
    if(gsl_linalg_cholesky_decomp(LMatrix) != GSL_SUCCESS) {
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
void SplineApproxImpl::computeWeights(double lambda) const
{
    if(isSingular()) {
        computeWeightsSingular();
        return;
    }
    if(lambda==0)  // simple case, no need to use roughness penalty matrix
        gsl_linalg_cholesky_solve(LMatrix, zRHS, weightCoefs);
    else {
        for(size_t p=0; p<numKnots+2; p++)
            gsl_vector_set(tempv, p, gsl_vector_get(MTz, p)/(1+lambda*gsl_vector_get(singValues, p)));
        gsl_blas_dgemv(CblasNoTrans, 1, MMatrix, tempv, 0, weightCoefs);
    }
}

// compute weights of basis functions in the case that the matrix is singular
void SplineApproxImpl::computeWeightsSingular() const
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

/// compute the value of AIC (Akaike Information criterion) for the given smoothing parameter lambda
void SplineApproxImpl::eval_deriv(double lambda, double* result, double*, double*) const {
    assert(result!=NULL);
    double rmserror, edf;
    computeWeights(lambda);
    computeRMSandEDF(lambda, &rmserror, &edf);
    double AIC = log(rmserror) + 2*edf/(numDataPoints-edf-1);
    *result = AIC - targetAIC;
}

/// after the weights of basis functions have been determined, evaluate the values of approximating spline 
/// at its nodes, and additionally its derivatives at endpoints
void SplineApproxImpl::computeYvalues(std::vector<double>& splineValues, double& der_left, double& der_right) const
{
    splineValues.assign(numKnots, 0);
    for(size_t k=1; k<numKnots-1; k++) {  // loop over interior nodes
        gsl_bspline_eval(gsl_vector_get(knots, k), bsplineValues, bsplineWorkspace);
        double value=0;
        for(size_t p=0; p<numKnots+2; p++)
            value += gsl_vector_get(bsplineValues, p) * gsl_vector_get(weightCoefs, p);
        splineValues[k] = value;
    }
    for(size_t k=0; k<numKnots; k+=numKnots-1) {  // two endpoints: values and derivatives
        gsl_bspline_deriv_eval(gsl_vector_get(knots, k), 1, bsplineDerivValues, bsplineWorkspace, bsplineDerivWorkspace);
        double value=0, deriv=0;
        for(size_t p=0; p<numKnots+2; p++) {
            value += gsl_matrix_get(bsplineDerivValues, p, 0) * gsl_vector_get(weightCoefs, p);
            deriv += gsl_matrix_get(bsplineDerivValues, p, 1) * gsl_vector_get(weightCoefs, p);
        }
        splineValues[k] = value;
        if(k==0)
            der_left = deriv;
        else
            der_right = deriv;
    }
}

void SplineApproxImpl::computeRegressionAtPoints(const std::vector<double> &xpoints, std::vector<double> &ypoints) const
{
    ypoints.assign(xpoints.size(), NAN);  // default value for nodes outside the definition range
    for(size_t i=0; i<xpoints.size(); i++)  // loop over interior nodes
        if(xpoints[i]>=gsl_vector_get(knots, 0) && xpoints[i]<=gsl_vector_get(knots, numKnots-1)) {
            gsl_bspline_eval(xpoints[i], bsplineValues, bsplineWorkspace);
            double value=0;
            for(size_t p=0; p<numKnots+2; p++)
                value += gsl_vector_get(bsplineValues, p) * gsl_vector_get(weightCoefs, p);
            ypoints[i] = value;
        }
}

//----------- DRIVER CLASS ------------//

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
    SplineApproxImpl& fnc = *static_cast<SplineApproxImpl*>(impl);
    fnc.loadyvalues(yvalues);
    if(fnc.isSingular() || lambda==0)
        fnc.computeWeights();
    else {
        fnc.initRoughnessMatrix();
        fnc.computeWeights(lambda);
    }
    fnc.computeYvalues(splineValues, deriv_left, deriv_right);
    fnc.computeRMSandEDF(lambda, rmserror, edf);
}

void SplineApprox::fitDataOversmooth(const std::vector<double> &yvalues, const double deltaAIC,
    std::vector<double>& splineValues, double& deriv_left, double& deriv_right, 
    double *rmserror, double* edf, double *lambda)
{
    SplineApproxImpl& fnc = *static_cast<SplineApproxImpl*>(impl);
    fnc.loadyvalues(yvalues);
    double lambdaFit = 0;
    if(fnc.isSingular()) {
        fnc.computeWeights();
    } else {
        fnc.initRoughnessMatrix();
        fnc.setTargetAIC(0);
        if(deltaAIC <= 0) {  // find optimal fit
            lambdaFit = findMin(fnc, 0, INFINITY, NAN, 0.1);  // no initial guess, and mediocre accuracy of 10% is enough
        } else {
            double AIC0 = fnc.value(0);
            fnc.setTargetAIC(AIC0 + deltaAIC);  // allow for somewhat higher AIC value, to smooth more than minimum necessary amount
            lambdaFit = findRoot(fnc, 0, INFINITY, 1e-2);
            if(!isFinite(lambdaFit))   // root does not exist, i.e. function is everywhere lower than target value
                lambdaFit = INFINITY;  // basically means fitting with a linear regression
        }
    }
    fnc.computeYvalues(splineValues, deriv_left, deriv_right);
    fnc.computeRMSandEDF(lambdaFit, rmserror, edf);
    if(lambda!=NULL)
        *lambda=lambdaFit;
}

void SplineApprox::fitDataOptimal(const std::vector<double> &yvalues,
    std::vector<double>& splineValues, double& deriv_left, double& deriv_right, 
    double *rmserror, double* edf, double *lambda) 
{
    fitDataOversmooth(yvalues, 0.0, splineValues, deriv_left, deriv_right, rmserror, edf, lambda);
}

}  // namespace
