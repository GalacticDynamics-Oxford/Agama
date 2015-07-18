// An interface to GSL - hides all the gsl calls behind easier to use functions
// Jason Sanders
#if 0
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_sort_double.h>
#include <gsl/gsl_permute.h>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_ieee_utils.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_min.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_erf.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_ellint.h>
#include <gsl/gsl_monte_vegas.h>
#include <gsl/gsl_siman.h>
#include <vector>
#endif

#include "mathutils.h"
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_trig.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_poly.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_deriv.h>
#include <gsl/gsl_fit.h>
#include <gsl/gsl_odeiv2.h>
#include <stdexcept>
#include <cassert>

namespace mathutils{

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
    return static_cast<IFunction*>(param)->value(x);
}
#if 0
static double functionDerivWrapper(double x, void* param){
    double der;
    static_cast<IFunction*>(param)->eval_deriv(x, NULL, &der);
    return der;
}

static void functionAndDerivWrapper(double x, void* param, double* f, double *df) {
    static_cast<IFunction*>(param)->eval_deriv(x, f, df);
}
#endif
bool isFinite(double x) {
    return gsl_finite(x);
}

int fcmp(double x, double y, double eps) {
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
double findRootHybrid(const IFunction& fnc, 
    const double x_lower, const double x_upper, const double reltoler)
{
    double a = x_lower;
    double b = x_upper;
    double fa, fb;
    double fdera = NAN, fderb = NAN;
    bool have_derivs = fnc.numDerivs()>=1;
    fnc.eval_deriv(a, &fa, have_derivs? &fdera : NULL);
    fnc.eval_deriv(b, &fb, have_derivs? &fderb : NULL);
    
    if ((fa < 0.0 && fb < 0.0) || (fa > 0.0 && fb > 0.0))
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
            {  // attempt to obtain the approximation by Hermite interpolation
                dd = interpHermiteMonotonic(0, fb, 0, 1/fderb, fc, 2*m, 1/fderc);
            }
            if(gsl_finite(dd)) {  // Hermite interpolation is successful
                d = dd;
            } else {  // proceed as usual in the Brent method
                double p, q, r;
                double s = fb / fa;
                if (a == c) {   // secant method (linear interpolation)
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

        fnc.eval_deriv(b, &fb, have_derivs? &fderb : NULL);

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
        if(numIter >= 42)
            converged = true;  // not quite ready, but can't loop forever
    } while(!converged);
    return b;  // best approximation
}

class ScaledRootFinder: public IFunction {
public:
    const IFunction& F;
    double x_edge, x_scaling;
    bool inf_lower, inf_upper;
    explicit ScaledRootFinder(const IFunction& _F) : F(_F) {};
    virtual int numDerivs() const { return F.numDerivs(); }
    double scaledArg(const double y) const {      // x(y) where y is the scaled argument in [0,1]
        return inf_upper ? (inf_lower ? 
            x_scaling*(1/(1-y)-1/y) :             // x in (-inf,inf)
            x_edge + y/(1-y)*x_scaling) :         // x in [x_edge, inf)
            x_edge - x_scaling*(1-y)/y;           // x in (-inf, x_edge]
    }
    double scaledDer(const double y) const {      // dx/dy
        return inf_upper ? (inf_lower ? 
            x_scaling*(1/pow_2(1-y)+1/pow_2(y)) : // (-inf,inf)
            x_scaling/pow_2(1-y) ) :              // [x_edge, inf)
            x_scaling/pow_2(y);                   // (-inf, x_edge]
    }
    virtual void eval_deriv(const double y, double* val=0, double* der=0, double* der2=0) const {
        double x = scaledArg(y), f, dfdx;
        F.eval_deriv(x, val ? &f : NULL, der ? &dfdx : NULL);
        if(val)
            *val = f;
        if(der)
            *der = dfdx * scaledDer(y);
        if(der2)
            *der2= NAN;
    }
};

double findRoot(const IFunction& fnc, double xlower, double xupper, double reltoler)
{
    if(reltoler<=0)
        throw std::invalid_argument("findRoot: relative tolerance must be positive");
    if(xlower>=xupper) {
        double z=xlower;
        xlower=xupper;
        xupper=z;
    }
    if(xlower==gsl_neginf() || xupper==gsl_posinf()) {   // apply internal scaling procedure
        ScaledRootFinder srf(fnc);
        srf.inf_lower = xlower==gsl_neginf();
        srf.inf_upper = xupper==gsl_posinf();
        if(srf.inf_upper && !srf.inf_lower) {
            srf.x_edge = xlower;
            srf.x_scaling = fmax(xlower, 1.);  // quite an arbitrary choice
        } else if(srf.inf_lower && !srf.inf_upper) {
            srf.x_edge = xupper;
            srf.x_scaling = fmax(-xupper, 1.);
        } else
            srf.x_scaling=1;
        double scroot = findRootHybrid(srf, 0., 1., reltoler);
        return srf.scaledArg(scroot);
    } else {  // no scaling - use the original function
        return findRootHybrid(fnc, xlower, xupper, reltoler);
    }
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
    virtual void eval_deriv(const double y, double* val=0, double* =0, double* =0) const {
        const double x = x_low + (x_upp-x_low) * y*y*(3-2*y);
        const double dx = (x_upp-x_low) * 6*y*(1-y);
        *val = F.value(x) * dx;
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
        fnc.eval_deriv(x0, &f0, &fder, &fder2);
        if(isFinite(fder+fder2))
            return;  // no further action necessary
    }
    if(!isFinite(f0))  // haven't called it yet
        fnc.eval_deriv(x0, &f0, fnc.numDerivs()>=1 ? &fder : NULL);
    fnc.eval_deriv(x0+delta, &fplusd, fnc.numDerivs()>=1 ? &fderplusd : NULL);
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
    fminusd= fnc.value(x0-delta);
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
// RANDOM NUMBERS //
// random number generators - rand_uniform returns random numbers uniformly distributed in the
// interval [0,1], rand_gaussian returns gaussian distributed random numbers with sd = sigma

class rand_base{
	private:
		const gsl_rng_type * TYPE;
		unsigned long int seed;
    public:
       	gsl_rng * r;
     	rand_base(unsigned long int s){
     	// construct random number generator with seed s
     		seed = s;
     		gsl_rng_env_setup();
     	    TYPE = gsl_rng_default;
       		r    = gsl_rng_alloc (TYPE);
       		gsl_rng_set(r, seed);
       		}
       	~rand_base(){gsl_rng_free (r);}
       	void reseed(unsigned long int newseed){
       	// give new seed
       		seed = newseed;
       		gsl_rng_set(r,newseed);
       		}
};

class rand_uniform:public rand_base{
	public:
		rand_uniform(unsigned long int SEED=0):rand_base(SEED){}
		~rand_uniform(){}
		// return uniformly distributed random numbers
		double nextnumber(){return gsl_rng_uniform (r);}
};

class rand_gaussian:public rand_base{
	public:
		double sigma;
		rand_gaussian(double s, unsigned long int SEED=0):rand_base(SEED){sigma = s;}
		~rand_gaussian(){}
		// return gaussian distributed random numbers
		double nextnumber(){return gsl_ran_gaussian (r,sigma);}
		void newsigma(double newsigma){sigma=newsigma;}
};

class rand_exponential:public rand_base{
    public:
        double scale;
        rand_exponential(double scale, unsigned long int SEED=0):rand_base(SEED),scale(scale){}
        ~rand_exponential(){}
        // return exponentially distributed random numbers
        double nextnumber(){return gsl_ran_exponential (r,scale);}
        void new_scale(double newscale){scale=newscale;}
};

//=================================================================================================
// ROOT FINDING	  //
// finds root by Brent's method. Constructor initialises function and tolerances, findroot finds
// root in given interval. Function must be of form double(*func)(double,void*)

class root_find{
	private:
};

//=================================================================================================
// INTEGRATION //
// Simple 1d numerical integration using adaptive Gauss-Kronrod which can deal with singularities
// constructor takes function and tolerances and integrate integrates over specified region.
// integrand function must be of the form double(*func)(double,void*)
class integrator{
	private:
		gsl_integration_workspace *w;
		void *p;
		double result,err,eps;
		gsl_function F;
		size_t neval;
	public:
		integrator(double eps): eps(eps){
			neval    = 1000;
			w        = gsl_integration_workspace_alloc (neval);
			F.params = &p;
       		}
		~integrator(){gsl_integration_workspace_free (w);}
		double integrate(double(*func)(double,void*),double xa, double xb){
		    F.function = func;
			gsl_integration_qags (&F, xa, xb, 0, eps, neval,w, &result, &err);
			//gsl_integration_qng(&F, xa, xb, 0, eps, &result, &err, &neval);
			return result;
			}
		double error(){return err;}
};

inline double integrate(double(*func)(double,void*),double xa, double xb, double eps){
	double result,err; size_t neval;void *p;gsl_function F;F.function = func;F.params = &p;
	gsl_integration_qng(&F, xa, xb, 0, eps, &result, &err, &neval);
	return result;
}

class MCintegrator{
	private:
		gsl_monte_vegas_state *s;
		const gsl_rng_type *T;
 		gsl_rng *r;
 		size_t dim;
 	public:
 		MCintegrator(size_t Dim){
 			dim=Dim;
 			gsl_rng_env_setup ();
  			T = gsl_rng_default;
  			r = gsl_rng_alloc (T);
 			s = gsl_monte_vegas_alloc(Dim);
 		}
 		~MCintegrator(){
 			gsl_monte_vegas_free(s);
 			gsl_rng_free(r);
 		}
 		double integrate(double(*func)(double*,size_t,void*),double *xlow, double *xhigh,
 		size_t calls, double *err, int burnin=10000){
 			gsl_monte_function G = { func, dim, 0 }; double res;
 			if(burnin)gsl_monte_vegas_integrate(&G,xlow,xhigh,dim,burnin,r,s,&res,err);
 			gsl_monte_vegas_integrate(&G,xlow,xhigh,dim,calls,r,s,&res,err);
 			return res;
 		}
};

//=================================================================================================
// 1D INTERPOLATION //
// Interpolation using cubic splines

class interpolator{
	private:
		gsl_interp_accel *acc;
		gsl_spline *spline;
	public:
		interpolator(double *x, double *y, int n){
			acc = gsl_interp_accel_alloc();
			spline = gsl_spline_alloc(gsl_interp_cspline,n);
			gsl_spline_init (spline, x, y, n);
		}
		~interpolator(){
			gsl_spline_free (spline);
         	gsl_interp_accel_free (acc);
        }
        double interpolate(double xi){
        	return gsl_spline_eval (spline, xi, acc);
        }
        double derivative(double xi){
        	return gsl_spline_eval_deriv(spline, xi, acc);
        }
        void new_arrays(double *x, double *y,int n){
        	spline = gsl_spline_alloc(gsl_interp_cspline,n);
        	gsl_spline_init (spline, x, y, n);
        }
};

//=================================================================================================
// SORTING //
// sorting algorithm
// sort2 sorts first argument and then applies the sorted permutation to second list
class sorter{
	private:
		const gsl_rng_type * T;
       	gsl_rng * r;
    public:
    	sorter(){
    		gsl_rng_env_setup();
            T = gsl_rng_default;
     		r = gsl_rng_alloc (T);
     		}
     	~sorter(){gsl_rng_free (r);}
     	void sort(double *data, int n){
     		gsl_sort(data,1,n);
     	}
     	void sort2(double *data, int n, double *data2){
     		size_t p[n];
     		gsl_sort_index(p,data,1,n);
     		gsl_permute(p,data2,1,n);
     	}

};

//=================================================================================================
// ODE SOLVER //


//=================================================================================================
// MINIMISER //
// finds a minimum of a function of the form double(*func)(const gsl_vector *v, void *params)
// using a downhill simplex algorithm. Setup minimiser with initial guesses and required tolerance
// with constructor and then minimise with minimise().
class minimiser{
	private:
		const gsl_multimin_fminimizer_type *T; ;
		gsl_multimin_fminimizer *s;
		gsl_vector *ss, *x;
		gsl_multimin_function minex_func;
		size_t iter; int status,N_params; double size;
		double eps;
	public:
		minimiser(double(*func)(const gsl_vector *v, void *params),double *parameters,
		 int N, double *sizes, double eps, void *params):N_params(N), eps(eps){

			T = gsl_multimin_fminimizer_nmsimplex2rand;
			ss = gsl_vector_alloc (N_params);x = gsl_vector_alloc (N_params);
			for(int i=0;i<N_params;i++){
				gsl_vector_set (x, i, parameters[i]);gsl_vector_set(ss,i,sizes[i]);}

			minex_func.n = N_params; minex_func.f = func; minex_func.params = params;
			s = gsl_multimin_fminimizer_alloc (T, N_params);
			gsl_multimin_fminimizer_set (s, &minex_func, x, ss);
			status = 0; iter = 0;

		}

		minimiser(double(*func)(const gsl_vector *v, void *params),std::vector<double> parameters,
		 std::vector<double> sizes, double eps,void *params): eps(eps){

			N_params = parameters.size();
			T = gsl_multimin_fminimizer_nmsimplex2rand;
			ss = gsl_vector_alloc (N_params);x = gsl_vector_alloc (N_params);
			for(int i=0;i<N_params;i++){
				gsl_vector_set (x, i, parameters[i]);gsl_vector_set(ss,i,sizes[i]);}

			minex_func.n = N_params; minex_func.f = func; minex_func.params = params;
			s = gsl_multimin_fminimizer_alloc (T, N_params);
			gsl_multimin_fminimizer_set (s, &minex_func, x, ss);
			status = 0; iter = 0;
		}

		~minimiser(){
			gsl_vector_free(x);
			gsl_vector_free(ss);
			gsl_multimin_fminimizer_free (s);
		}

		double minimise(double *results,unsigned int maxiter,bool vocal){
			do
			  {
				iter++; status = gsl_multimin_fminimizer_iterate(s);
				if(status)break;
				size = gsl_multimin_fminimizer_size (s);
				status = gsl_multimin_test_size (size, eps);
//				if(vocal){	std::cout<<iter<<" ";
//							for(int i=0; i<N_params;i++)std::cout<<gsl_vector_get(s->x,i)<<" ";
//							std::cout<<s->fval<<" "<<size<<std::endl;
//							}
			}
			while (status == GSL_CONTINUE && iter < maxiter);
			for(int i=0;i<N_params;i++){results[i] = gsl_vector_get(s->x,i);}
			return s->fval;
		}

		double minimise(std::vector<double> *results,unsigned int maxiter,bool vocal){
			do
			  {
				iter++; status = gsl_multimin_fminimizer_iterate(s);
				if(status)break;
				size = gsl_multimin_fminimizer_size (s);
				status = gsl_multimin_test_size (size, eps);
//				if(vocal){	std::cout<<iter<<" ";
//							for(int i=0; i<N_params;i++)std::cout<<gsl_vector_get(s->x,i)<<" ";
//							std::cout<<s->fval<<" "<<size<<std::endl;
//							}
			}
			while (status == GSL_CONTINUE && iter < maxiter);
			for(int i=0;i<N_params;i++) results->push_back(gsl_vector_get(s->x,i));
			return s->fval;
		}
};

class minimiser1D{
	private:
		const gsl_min_fminimizer_type *T; ;
		gsl_min_fminimizer *s;
		size_t iter; int status;
		double m, a, b, eps;
	public:
		minimiser1D(double(*func)(double, void *params), double m, double a, double b, double eps, void* params)
			:m(m), a(a), b(b), eps(eps){

			gsl_function F;F.function = func;F.params = params;
			T = gsl_min_fminimizer_brent;
			s = gsl_min_fminimizer_alloc (T);
			gsl_min_fminimizer_set (s, &F, m, a, b);
			status = 0; iter = 0;
		}
		~minimiser1D(){
			gsl_min_fminimizer_free (s);
		}
		double minimise(unsigned int maxiter){
			do
			  {
				iter++;
				status = gsl_min_fminimizer_iterate(s);
				m = gsl_min_fminimizer_x_minimum (s);
           		a = gsl_min_fminimizer_x_lower (s);
           		b = gsl_min_fminimizer_x_upper (s);
				status = gsl_min_test_interval (a, b, eps, 0.0);
			}
			while (status == GSL_CONTINUE && iter < maxiter);
			return m;
		}
};
/*
double Distance(void *xp, void *yp){
       double x = *((double *) xp);
       double y = *((double *) yp);
       return fabs(x - y);
}

void Step(const gsl_rng * r, void *xp, double step_size){
    double old_x = *((double *) xp);
    double new_x;

    double u = gsl_rng_uniform(r);
    new_x = u * 2 * step_size - step_size + old_x;

    memcpy(xp, &new_x, sizeof(new_x));
}

void Print(void *xp){
    printf ("%12g", *((double *) xp));
}

class sim_anneal{
	private:
		const gsl_rng_type * T;
    	gsl_rng * r;
    	int N_TRIES, ITER_FIXED_T;
    	double STEP_SIZE, K, T_INITIAL, MU_T, T_MIN;
    	gsl_siman_params_t params;
    public:
    	sim_anneal(int N_TRIES, int ITER_FIXED_T, double STEP_SIZE):
    	N_TRIES(N_TRIES), ITER_FIXED_T(ITER_FIXED_T),STEP_SIZE(STEP_SIZE){
    		gsl_rng_env_setup();
            T = gsl_rng_default;
     	  	r = gsl_rng_alloc(T);
     	  	//params[0]=N_TRIES;params[1]=ITER_FIZED_T;params[2]=STEP_SIZE;
     	  	//K=1.; params[3]=K; T_INITIAL=0.008; params[4]=T_INITIAL;
     	  	//MU_T=1.003; params[5]=MU_T; T_MIN=2.0e-6; params[6]=T_MIN;
     	  	gsl_siman_params_t params
       = {N_TRIES, ITERS_FIXED_T, STEP_SIZE,
          K, T_INITIAL, MU_T, T_MIN};
    	}
    	~sim_anneal(){
    		gsl_rng_free(r);
    	}
    	double minimise(double(*func)(void *xp), double x){
    		double x_initial=x;
    		gsl_siman_solve(r, &x_initial, &func, Step, Distance, Print,
                       		NULL, NULL, NULL,
                       		sizeof(double), params);
    		return x_initial;
    	}
};*/

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
