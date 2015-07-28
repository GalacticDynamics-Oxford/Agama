#include "math_core.h"
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_trig.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_min.h>
#include <gsl/gsl_poly.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_fit.h>
#include <gsl/gsl_odeiv2.h>
#include <stdexcept>
#include <cassert>

namespace math{

const int MAXITER = 42;  ///< upper limit on the number of iterations in root-finders, minimizers, etc.

// ------ error handling ------ //

static void exceptionally_awesome_gsl_error_handler (const char *reason, const char * /*file*/, int /*line*/, int gsl_errno)
{
    if( // list error codes that are non-critical and don't need to be reported
        gsl_errno == GSL_ETOL ||
        gsl_errno == GSL_EROUND ||
        gsl_errno == GSL_ESING ||
        gsl_errno == GSL_EDIVERGE )
        return;  // do nothing
    if( gsl_errno == GSL_ERANGE ||
        gsl_errno == GSL_EOVRFLW )
        throw std::range_error(std::string("GSL range error: ")+reason);
    if( gsl_errno == GSL_EDOM )
        throw std::domain_error(std::string("GSL domain error: ")+reason);
    if( gsl_errno == GSL_EINVAL )
        throw std::invalid_argument(std::string("GSL invalid argument error: ")+reason);
    throw std::runtime_error(std::string("GSL error: ")+reason);
}

// a static variable that initializes our error handler
bool error_handler_set = gsl_set_error_handler(&exceptionally_awesome_gsl_error_handler);

// ------ math primitives -------- //

static double functionWrapper(double x, void* param){
    return static_cast<IFunction*>(param)->operator()(x);
}
#if 0
static double functionDerivWrapper(double x, void* param){
    double der;
    static_cast<IFunction*>(param)->evalDeriv(x, NULL, &der);
    return der;
}

static void functionAndDerivWrapper(double x, void* param, double* f, double *df) {
    static_cast<IFunction*>(param)->evalDeriv(x, f, df);
}
#endif
bool isFinite(double x) {
    return gsl_finite(x);
}

int fcmp(double x, double y, double eps) {
    if(x==0)
        return y<-eps ? -1 : y>eps ? +1 : 0;
    if(y==0)
        return x<-eps ? -1 : x>eps ? +1 : 0;
    return gsl_fcmp(x, y, eps);
}

double wrapAngle(double x) {
    return gsl_sf_angle_restrict_pos(x);
}

double unwrapAngle(double x, double xprev) {
    double diff=(x-xprev)/(2*M_PI);
    double nwraps=0;
    if(diff>0.5) 
        modf(diff+0.5, &nwraps);
    else if(diff<-0.5) 
        modf(diff-0.5, &nwraps);
    return x - 2*M_PI * nwraps;
}
    
// ------ root finder routines ------//

// used in hybrid root-finder to predict the root location by Hermite interpolation
inline double interpHermiteMonotonic(double x, double x1, double f1, double dfdx1, 
    double x2, double f2, double dfdx2)
{
    if(!gsl_finite(dfdx1+dfdx2) || dfdx1*dfdx2<0)  // derivatives must exist and have the same sign
        return NAN;
    const double dx = x2-x1, sixdf = 6*(f2-f1);
    const double t = (x-x1)/dx;
    // check if the interpolant is monotonic on t=[0:1]
    double t1, t2;
    int nroots = gsl_poly_solve_quadratic(-sixdf+3*dx*(dfdx1+dfdx2), 
        sixdf-2*dx*(2*dfdx1+dfdx2), dx*dfdx1, &t1, &t2);
    if(nroots>0 && ((t1>=0 && t1<=1) || (t2>=0 && t2<=1)) )
        return NAN;   // will not produce a non-monotonic result
    return pow_2(1-t) * ( (1+2*t)*f1 + t * dfdx1*dx )
         + pow_2(t) * ( (3-2*t)*f2 + (t-1)*dfdx2*dx );
}

// a hybrid between Brent's method and interpolation of root using function derivatives
// it is based on the implementation from GSL, original authors: Reid Priedhorsky, Brian Gough
static double findRootHybrid(const IFunction& fnc, 
    const double x_lower, const double x_upper, const double reltoler)
{
    double a = x_lower;
    double b = x_upper;
    double fa, fb;
    double fdera = NAN, fderb = NAN;
    bool have_derivs = fnc.numDerivs()>=1;
    fnc.evalDeriv(a, &fa, have_derivs? &fdera : NULL);
    fnc.evalDeriv(b, &fb, have_derivs? &fderb : NULL);

    if ((fa < 0.0 && fb < 0.0) || (fa > 0.0 && fb > 0.0) || !isFinite(fa+fb))
        return NAN;   // endpoints do not bracket root
    /*  b  is the current estimate of the root,
        c  is the counter-point (i.e. f(b)*f(c)<0, and |f(b)|<|f(c)| ),
        a  is the previous estimate of the root:  either
           (1) a==c, or 
           (2) f(a) has the same sign as f(b), |f(a)|>|f(b)|,
               and 'a, b, c' form a monotonic sequence.
    */
    double c = a;
    double fc = fa;
    double fderc = fdera;
    double d = b - c;   // always holds the (signed) length of current interval
    double e = b - c;   // this is used to estimate roundoff (?)
    if (fabs (fc) < fabs (fb)) {  // swap b and c so that |f(b)| < |f(c)|
        a = b;
        b = c;
        c = a;
        fa = fb;
        fb = fc;
        fc = fa;
        fdera = fderb;
        fderb = fderc;
        fderc = fdera;
    }
    int numIter = 0;
    bool converged = false;
    double abstoler = fabs(x_lower-x_upper) * reltoler;
    do {
        double tol = 0.5 * GSL_DBL_EPSILON * fabs (b);
        double m = 0.5 * (c - b);
        if (fb == 0 || fabs (m) <= tol) 
            return b;  // the ROOT
        if (fabs (e) < tol || fabs (fa) <= fabs (fb)) { 
            d = m;            /* use bisection */
            e = m;
        } else {
            double dd = NAN;
            if(have_derivs && fderb*fderc>0)  // derivs exist and have the same sign
            {   // attempt to obtain the approximation by Hermite interpolation
                dd = interpHermiteMonotonic(0, fb, 0, 1/fderb, fc, 2*m, 1/fderc);
            }
            if(dd == dd) {  // Hermite interpolation is successful
                d = dd;
            } else {        // proceed as usual in the Brent method
                double p, q, r;
                double s = fb / fa;
                if (a == c) {     // secant method (linear interpolation)
                    p = 2 * m * s;
                    q = 1 - s;
                } else {          // inverse quadratic interpolation
                    q = fa / fc;
                    r = fb / fc;
                    p = s * (2 * m * q * (q - r) - (b - a) * (r - 1));
                    q = (q - 1) * (r - 1) * (s - 1);
                }
                if (p > 0)
                    q = -q;
                else
                    p = -p;
                if (2 * p < GSL_MIN (3 * m * q - fabs (tol * q), fabs (e * q))) { 
                    e = d;
                    d = p / q;
                } else {
                    /* interpolation failed, fall back to bisection */
                    d = m;
                    e = m;
                }
            }
        }

        a = b;
        fa = fb;
        fdera = fderb;
        if (fabs (d) > tol)
            b += d;
        else
            b += (m > 0 ? +tol : -tol);

        fnc.evalDeriv(b, &fb, have_derivs? &fderb : NULL);
        if(!isFinite(fb))
            return NAN;

        /* Update the best estimate of the root and bounds on each iteration */
        if ((fb < 0 && fc < 0) || (fb > 0 && fc > 0)) {   // the root is between 'a' and the new 'b'
            c = a;       // so the new counterpoint is moved to the old 'a'
            fc = fa;
            fderc = fdera;
            d = b - c;
            e = b - c;
        }
        if (fabs (fc) < fabs (fb)) {   // ensure that 'b' is close to zero than 'c'
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
            fdera = fderb;
            fderb = fderc;
            fderc = fdera;
        }

        numIter++;
        if(fabs(b-c) <= abstoler) { // convergence criterion from bracketing algorithm
            converged = true;
            double offset = fb*(b-c)/(fc-fb);  // offset from b to the root via secant
            if((offset>0 && offset<c-b) || (offset<0 && offset>c-b))      
                b += offset;        // final secant step
        } else if(have_derivs) {    // convergence from derivatives
            double offset = -fb / fderb;       // offset from b to the root via Newton's method
            bool bracketed = (offset>0 && offset<c-b) || (offset<0 && offset>c-b);
            if(bracketed && fabs(offset) < abstoler) {
                converged = true;
                b += offset;        // final Newton step
            }
        }
        if(numIter >= MAXITER)
            converged = true;  // not quite ready, but can't loop forever
    } while(!converged);
    return b;  // best approximation
}

/** scaling transformation of input function for the case that the interval is (semi-)infinite:
    it replaces the original argument  x  with  y in the range [0:1],  and implements the transformation of derivative.
*/
class ScaledFunction: public IFunction {
public:
    const IFunction& F;
    double x_edge, x_scaling;
    bool inf_lower, inf_upper;
    ScaledFunction(const IFunction& _F, double xlower, double xupper) : F(_F) {
        assert(xlower < xupper);
        inf_lower = xlower==-INFINITY;
        inf_upper = xupper== INFINITY;
        if(inf_upper) {
            if(inf_lower) {
                x_scaling = 1;
            } else {
                x_edge = xlower;
                x_scaling = fmax(xlower, 1.);  // quite an arbitrary choice
            }
        } else {
            if(inf_lower) {
                x_edge = xupper;
                x_scaling = fmax(-xupper, 1.);
            } else {
                x_edge = xlower;
                x_scaling = xupper;
            }
        }
    };

    virtual int numDerivs() const { return F.numDerivs()>1 ? 1 : F.numDerivs(); }

    // return the scaled variable y for the given original variable x
    double y_from_x(const double x) const {
        if(inf_upper) {
            if(inf_lower) {                   // x in (-inf,inf)
                return  fabs(x/x_scaling)<1 ? // two cases depending on whether |x| is small or large
                    1/(1 + sqrt(1+pow_2(x*0.5/x_scaling)) - x*0.5/x_scaling) :  // x is close to zero
                    0.5 + sqrt(0.25+pow_2(x_scaling/x))*sign(x) - x_scaling/x;  // x is large
            } else {                          // x in [x_edge, inf)
                assert(x>=x_edge);
                return 1 - 1/(1 + (x-x_edge)/x_scaling);
            }
        } else {
            if(inf_lower) {                   // x in (-inf, x_edge]
                assert(x<=x_edge);
                return 1/(1 + (x_edge-x)/x_scaling);
            } else {                          // x in [x_edge, x_scaling]
                assert(x>=x_edge && x<=x_scaling);
                return (x-x_edge) / (x_scaling-x_edge);
            }
        }
    }

    // return the original variable x for the given scaled variable y in [0,1]
    double x_from_y(const double y) const {
        if(y!=y)
            return NAN;
        assert(y>=0 && y<=1);
        return inf_upper ?
            (  inf_lower ?
                x_scaling*(1/(1-y)-1/y) :     // x in (-inf,inf)
                x_edge + y/(1-y)*x_scaling    // x in [x_edge, inf)
            ) : ( inf_lower ?
                x_edge - x_scaling*(1-y)/y :  // x in (-inf, x_edge]
                x_edge*(1-y) + x_scaling*y    // x in [x_edge, x_scaling]
            );
    }

    // return the derivative of the original variable over the scaled one
    double dxdy_from_y(const double y) const {
        return inf_upper ?
            (  inf_lower ?
                x_scaling*(1/pow_2(1-y)+1/(y*y)) : // (-inf,inf)
                x_scaling/pow_2(1-y)               // [x_edge, inf)
            ) : ( inf_lower ?
                x_scaling/pow_2(y) :               // (-inf, x_edge]
                x_scaling-x_edge                   // [x_edge, x_scaling]
            );
    }

    // compute the original function for the given value of scaled argument
    virtual void evalDeriv(const double y, double* val=0, double* der=0, double* der2=0) const {
        double x = x_from_y(y), f, dfdx;
        F.evalDeriv(x, val ? &f : NULL, der ? &dfdx : NULL);
        if(val)
            *val = f;
        if(der)
            *der = dfdx * dxdy_from_y(y);
        if(der2)
            *der2= NAN;  // not implemented
    }
};

// root-finder with optional scaling
double findRoot(const IFunction& fnc, double xlower, double xupper, double reltoler)
{
    if(reltoler<=0)
        throw std::invalid_argument("findRoot: relative tolerance must be positive");
    if(xlower>=xupper) {
        double z=xlower;
        xlower=xupper;
        xupper=z;
    }
    if(xlower==-INFINITY || xupper==INFINITY) {   // apply internal scaling procedure
        ScaledFunction F(fnc, xlower, xupper);
        double scroot = findRootHybrid(F, 0., 1., reltoler);
        return F.x_from_y(scroot);
    } else {  // no scaling - use the original function
        return findRootHybrid(fnc, xlower, xupper, reltoler);
    }
}

// 1d minimizer
static double findMinKnown(const IFunction& fnc, double xlower, double xupper, double xinit, double reltoler)
{
    gsl_function F;
    F.function = &functionWrapper;
    F.params = const_cast<IFunction*>(&fnc);
    gsl_min_fminimizer *minser = gsl_min_fminimizer_alloc(gsl_min_fminimizer_brent);
    double xroot = NAN;
    double abstoler = reltoler*fabs(xupper-xlower);
    if(gsl_min_fminimizer_set(minser, &F, xinit, xlower, xupper) == GSL_SUCCESS) {
        int status=0, iter=0;
        do {
            iter++;
            try {
                gsl_min_fminimizer_iterate (minser);
            }
            catch(std::runtime_error&) {
                xroot = NAN;
                break;
            }
            xroot  = gsl_min_fminimizer_x_minimum (minser);
            xlower = gsl_min_fminimizer_x_lower (minser);
            xupper = gsl_min_fminimizer_x_upper (minser);
            status = gsl_root_test_interval (xlower, xupper, 0, reltoler);
        }
        while (fabs(xlower-xupper) < abstoler && iter < MAXITER);
    }
    gsl_min_fminimizer_free(minser);
    return xroot;
}

// 1d minimizer without prior knowledge of minimum location
double findMin(const IFunction& fnc, double xlower, double xupper, double xinit, double reltoler)
{
    if(reltoler<=0)
        throw std::invalid_argument("findMin: relative tolerance must be positive");
    if(xlower>=xupper) {
        double z=xlower;
        xlower=xupper;
        xupper=z;
    }
    if(xinit==xinit && (xinit<xlower || xinit>xupper))
        throw std::invalid_argument("findMin: initial guess is outside the search interval");
    ScaledFunction F(fnc, xlower, xupper);  // transform the original range into [0:1], even if it was (semi-)infinite
    xlower = F.y_from_x(xlower);
    xupper = F.y_from_x(xupper);
    if(xinit == xinit) 
        xinit  = F.y_from_x(xinit);
    else {    // initial guess not provided
        xinit = (xlower+xupper)/2;
        double ylower = F(xlower);
        double yupper = F(xupper);
        double yinit  = F(xinit);
        if(!isFinite(ylower+yupper+yinit))
            return NAN;
        double abstoler = reltoler*fabs(xupper-xlower);
        int iter = 0;
        while( (yinit>ylower || yinit>yupper) && iter<MAXITER && fabs(xlower-xupper)>abstoler) {
            if(yinit<ylower) {
                xlower=xinit;
                ylower=yinit;
            } else {
                if(yinit<yupper) {
                    xupper=xinit;
                    yupper=yinit;
                } else {  // pathological case - initial guess was higher than both ends
                    double xmin1 = findMin(F, xlower, xinit,  NAN, reltoler);
                    double xmin2 = findMin(F, xinit,  xupper, NAN, reltoler);
                    double ymin1 = F(xmin1);
                    double ymin2 = F(xmin2);
                    if(!isFinite(ymin1+ymin2))
                        return NAN;
                    return F.x_from_y(ymin1<ymin2 ? xmin1 : xmin2);
                }
            }
            xinit=(xlower+xupper)/2;
            yinit=F(xinit);
            if(!isFinite(yinit))
                return NAN;
            iter++;
        }
        if(yinit>=ylower && yinit<=yupper)  // couldn't locate a minimum inside the interval,
            return F.x_from_y(xlower);
        if(yinit>=yupper && yinit<=ylower)  // so return one of endpoints
            return F.x_from_y(xupper);
    }
    return F.x_from_y(findMinKnown(F, xlower, xupper, xinit, reltoler));  // normal min-search
}

// ------- integration routines ------- //

double integrate(const IFunction& fnc, double x1, double x2, double reltoler)
{
    if(x1==x2)
        return 0;
    gsl_function F;
    F.function = functionWrapper;
    F.params = const_cast<IFunction*>(&fnc);
    double result, error;
    if(reltoler==0) {  // don't care about accuracy -- use the fastest integration rule
#if 1
        const int N=10;  // tables up to N=20 are hard-coded in the library, no overhead
        gsl_integration_glfixed_table* t = gsl_integration_glfixed_table_alloc(N);
        result = gsl_integration_glfixed(&F, x1, x2, t);
        gsl_integration_glfixed_table_free(t);
#else
        double dummy;  // 15-point Gauss-Kronrod
        gsl_integration_qk15(&F, x1, x2, &result, &error, &dummy, &dummy);
#endif
    } else {  // use adaptive method with limited max # of points (87)
        size_t neval;
        gsl_integration_qng(&F, x1, x2, 0, reltoler, &result, &error, &neval);
    }
    return result;
}

/** The integral \int_{x1}^{x2} f(x) dx is transformed into 
    \int_{y1}^{y2} f(x(y)) (dx/dy) dy,  where x(y) = x_low + (x_upp-x_low) y^2 (3-2y),
    and x1=x(y1), x2=x(y2).   */

class ScaledIntegrand: public IFunction {
public:
    ScaledIntegrand(const IFunction& _F, double low, double upp) : 
        F(_F), x_low(low), x_upp(upp) {};
    virtual int numDerivs() const { return 0; }
private:
    const IFunction& F;
    double x_low, x_upp;
    virtual void evalDeriv(const double y, double* val=0, double* =0, double* =0) const {
        const double x = x_low + (x_upp-x_low) * y*y*(3-2*y);
        const double dx = (x_upp-x_low) * 6*y*(1-y);
        *val = F(x) * dx;
    }
};

// invert the above relation between x and y by solving a cubic equation
static double solveForScaled_y(double x, double x_low, double x_upp) {
    assert(x>=x_low && x<=x_upp);
    if(x==x_low) return 0;
    if(x==x_upp) return 1;
    double phi=acos(1-2*(x-x_low)/(x_upp-x_low))/3.0;
    return (1 - cos(phi) + M_SQRT3*sin(phi))/2.0;
}

double integrateScaled(const IFunction& fnc, double x1, double x2, 
    double x_low, double x_upp, double rel_toler)
{
    if(x1==x2) return 0;
    if(x1>x2 || x1<x_low || x2>x_upp || x_low>=x_upp)
        throw std::invalid_argument("Error in integrate_scaled: arguments out of range");
    ScaledIntegrand transf(fnc, x_low, x_upp);
    double y1=solveForScaled_y(x1, x_low, x_upp);
    double y2=solveForScaled_y(x2, x_low, x_upp);
    return integrate(transf, y1, y2, rel_toler);
}

// ----- derivatives and related fncs ------- //

PointNeighborhood::PointNeighborhood(const IFunction& fnc, double x0)
{
    delta = fmax(fabs(x0) * GSL_SQRT_DBL_EPSILON, 16*GSL_DBL_EPSILON);
    // we assume that the function can be computed at all points, but the derivatives not necessarily can
    double fplusd = NAN, fderplusd = NAN, fminusd = NAN;
    f0 = fder = fder2=NAN;
    if(fnc.numDerivs()>=2) {
        fnc.evalDeriv(x0, &f0, &fder, &fder2);
        if(isFinite(fder+fder2))
            return;  // no further action necessary
    }
    if(!isFinite(f0))  // haven't called it yet
        fnc.evalDeriv(x0, &f0, fnc.numDerivs()>=1 ? &fder : NULL);
    fnc.evalDeriv(x0+delta, &fplusd, fnc.numDerivs()>=1 ? &fderplusd : NULL);
    if(isFinite(fder)) {
        if(isFinite(fderplusd)) {  // have 1st derivative at both points
            fder2 = (6*(fplusd-f0)/delta - (4*fder+2*fderplusd))/delta;
            return;
        }
    } else if(isFinite(fderplusd)) {  // have 1st derivative at one point only
        fder = 2*(fplusd-f0)/delta - fderplusd;
        fder2= 2*( fderplusd - (fplusd-f0)/delta )/delta;
        return;
    }
    // otherwise we don't have any derivatives computed
    fminusd= fnc(x0-delta);
    fder = (fplusd-fminusd)/(2*delta);
    fder2= (fplusd+fminusd-2*f0)/(delta*delta);
}

double PointNeighborhood::dx_to_posneg(double sgn) const
{
    double s0 = sgn*f0, sder = sgn*fder, sder2 = sgn*fder2;
    if(s0>0)
        return 0;  // already there
    if(sder==0) {
        if(sder2<=0)
            return NAN;  // we are at a maximum already
        else
            return fmax(sqrt(-s0/sder2), delta);
    }
    // now we know that s0<=0 and sder!=0
    if(sder2>=0)  // may only curve towards zero, so a tangent is a safe estimate
        return -s0/sder + delta*sign(sder);
    double discr = sder*sder - 2*s0*sder2;
    if(discr<=0)
        return NAN;  // never cross zero
    return sign(sder) * (delta + 2*s0/(sqrt(discr)+abs(sder)) );
}

double PointNeighborhood::dx_to_nearest_root() const
{
    double dx_nearest_root = -f0/fder;  // nearest root by linear extrapolation
    //dx_farthest_root= dx_nearest_root>0 ? gsl_neginf() : gsl_posinf();
    if(f0*fder2<0) {  // use quadratic equation to find two nearest roots
        double discr = sqrt(fder*fder - 2*f0*fder2);
        if(fder<0) {
            dx_nearest_root = 2*f0/(discr-fder);
            //dx_farthest_root = (discr-fder)/fder2;
        } else {
            dx_nearest_root = 2*f0/(-discr-fder);
            //dx_farthest_root = (-discr-fder)/fder2;
        }
    }
    return dx_nearest_root;
}

// ----- other stuff ------- //
double linearFitZero(unsigned int N, const double x[], const double y[], double* rms)
{
    double c, cov, sumsq;
    gsl_fit_mul(x, 1, y, 1, N, &c, &cov, &sumsq);
    if(rms!=NULL)
        *rms = sqrt(sumsq/N);
    return c;
}

void linearFit(unsigned int N, const double x[], const double y[], 
    double& slope, double& intercept, double* rms)
{
    double cov00, cov11, cov01, sumsq;
    gsl_fit_linear(x, 1, y, 1, N, &intercept, &slope, &cov00, &cov01, &cov11, &sumsq);
    if(rms!=NULL)
        *rms = sqrt(sumsq/N);
}

// ------ ODE solver ------- //
// Simple ODE integrator using Runge-Kutta Dormand-Prince 8 adaptive stepping
// dy_i/dt = f_i(t) where int (*f)(double t, const double y, double f, void *params)

static int functionWrapperODE(double t, const double y[], double dydt[], void* param){
    static_cast<IOdeSystem*>(param)->eval(t, y, dydt);
    return GSL_SUCCESS;
}

struct OdeImpl{
    gsl_odeiv2_step * s;
    gsl_odeiv2_control * c;
    gsl_odeiv2_evolve * e;
    gsl_odeiv2_system sys;
};

OdeSolver::OdeSolver(const IOdeSystem& F, double abstoler, double reltoler)
{
    OdeImpl* data = new OdeImpl;
    data->sys.function  = functionWrapperODE;
    data->sys.jacobian  = NULL;
    data->sys.dimension = F.size();
    data->sys.params    = const_cast<IOdeSystem*>(&F);
    data->s = gsl_odeiv2_step_alloc(gsl_odeiv2_step_rk8pd, data->sys.dimension);
    data->c = gsl_odeiv2_control_y_new(abstoler, reltoler);
    data->e = gsl_odeiv2_evolve_alloc(data->sys.dimension);
    impl=data;
}

OdeSolver::~OdeSolver() {
    OdeImpl* data=static_cast<OdeImpl*>(impl);
    gsl_odeiv2_evolve_free(data->e);
    gsl_odeiv2_control_free(data->c);
    gsl_odeiv2_step_free(data->s);
    delete data;
}

int OdeSolver::advance(double tstart, double tfinish, double *y){
    OdeImpl* data = static_cast<OdeImpl*>(impl);
    double h = tfinish-tstart;
    double direction=(h>0?1.:-1.);
    int numstep=0;
    while ((tfinish-tstart)*direction>0 && numstep<ODE_MAX_NUM_STEP) {
        int status = gsl_odeiv2_evolve_apply (data->e, data->c, data->s, &(data->sys), &tstart, tfinish, &h, y);
        // check if computation is broken
        double test=0;
        for(unsigned int i=0; i<data->sys.dimension; i++) 
            test += y[i];
        if (status != GSL_SUCCESS || !isFinite(test)) {
            numstep = -1;
            break;
        }
        numstep++;
    }
    if(numstep>=ODE_MAX_NUM_STEP)
        throw std::runtime_error("ODE solver: number of sub-steps exceeds maximum");
    return numstep;
}

#if 0
//=================================================================================================
// SPECIAL FUNCTIONS //
inline double erf(double x){return gsl_sf_erf (x);}
inline double erfc(double x){return gsl_sf_erfc (x);}
inline double besselI(double x, int n){return gsl_sf_bessel_In (n,x);}
inline double besselJ(double x, int n){return gsl_sf_bessel_Jn (n,x);}
inline double gamma(double x){return gsl_sf_gamma (x);}
inline double ellint_first(double phi, double k){ return gsl_sf_ellint_F(phi,k,(gsl_mode_t)1e-15);}
// F(\phi,k) = \int_0^\phi \d t \, \frac{1}{\sqrt{1-k^2\sin^2 t}}
inline double ellint_second(double phi, double k){ return gsl_sf_ellint_E(phi,k,(gsl_mode_t)1e-15);}
// E(\phi,k) = \int_0^\phi \d t \, \sqrt{1-k^2\sin^2 t}
inline double ellint_third(double phi, double k, double n){ return gsl_sf_ellint_P(phi,k,n,(gsl_mode_t)1e-15);}
// \Pi(\phi,k,n) = \int_0^\phi \d t \, \frac{1}{(1+n\sin^2 t)\sqrt{1-k^2\sin^2 t}}
#endif

}  // namespace
