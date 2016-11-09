#include "math_core.h"
#include "utils.h"
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_trig.h>
#include <gsl/gsl_min.h>
#include <gsl/gsl_poly.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_rng.h>
#include <stdexcept>
#include <cassert>
#include <vector>

#ifdef HAVE_CUBA
#include <cuba.h>
#else
#include "cubature.h"
#endif

#ifdef _OPENMP
#if defined(__APPLE__) && __GNUC__==4 && __GNUC_MINOR__==2
// this is apparently a bug in Apple compiler that forces us to disable correct OpenMP support in this context
// (linker reports undefined symbols _gomp_thread_attr and _gomp_tls_key)
#warning Use of random() is not thread-safe due to broken OpenMP implementation
#else
#define HAVE_VALID_OPENMP
#endif
#endif
#ifdef HAVE_VALID_OPENMP
#include <omp.h>
#endif

namespace math{

const int MAXITER = 64;  ///< upper limit on the number of iterations in root-finders, minimizers, etc.

const int MAXINTEGRPOINTS = 1000;  ///< size of workspace for adaptive integration

// ------ error handling ------ //

static void exceptionally_awesome_gsl_error_handler(const char *reason, 
    const char * /*file*/, int /*line*/, int gsl_errno)
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
    throw std::runtime_error("GSL error "+utils::toString(gsl_errno)+": "+reason);
}

// a static variable that initializes our error handler
bool error_handler_set = gsl_set_error_handler(&exceptionally_awesome_gsl_error_handler);

// ------ math primitives -------- //

static double functionWrapper(double x, void* param)
{
    return static_cast<IFunction*>(param)->value(x);
}

int fcmp(double x, double y, double eps)
{
    if(x==0)
        return y<-eps ? -1 : y>eps ? +1 : 0;
    if(y==0)
        return x<-eps ? -1 : x>eps ? +1 : 0;
    if(x!=x)
        return -2;
    if(y!=y)
        return +2;
    return gsl_fcmp(x, y, eps);
}

double powInt(double x, int n)
{
    return gsl_pow_int(x, n);
}

double wrapAngle(double x)
{
    return isFinite(x) ? gsl_sf_angle_restrict_pos(x) : x;
}

double unwrapAngle(double x, double xprev)
{
    double diff=(x-xprev)/(2*M_PI);
    double nwraps=0;
    if(diff>0.5) 
        modf(diff+0.5, &nwraps);
    else if(diff<-0.5) 
        modf(diff-0.5, &nwraps);
    return x - 2*M_PI * nwraps;
}

template<typename NumT>
unsigned int binSearch(const NumT x, const NumT arr[], unsigned int size)
{
    if(size<1 || x<arr[0])
        return -1;
    if(x>arr[size-1] || size<2)
        return size;
    // first guess the likely location in the case that the input grid is equally-spaced
    unsigned int index = static_cast<unsigned int>( (x-arr[0]) / (arr[size-1]-arr[0]) * (size-1) );
    unsigned int indhi = size-1;
    if(index==size-1)
        return size-2;     // special case -- we are exactly at the end of array, return the previous node
    if(x>=arr[index]) {
        if(x<arr[index+1])
            return index;  // guess correct, exiting
        // otherwise the search is restricted to [ index .. indhi ]
    } else {
        indhi = index;  // search restricted to [ 0 .. index ]
        index = 0;
    }
    while(indhi > index + 1) {
        unsigned int i = (indhi + index)/2;
        if(arr[i] > x)
            indhi = i;
        else
            index = i;
    }
    return index;
}

// template instantiations
template unsigned int binSearch(const double x, const double arr[], unsigned int size);
template unsigned int binSearch(const float x, const float arr[], unsigned int size);
template unsigned int binSearch(const int x, const int arr[], unsigned int size);
template unsigned int binSearch(const long x, const long arr[], unsigned int size);
template unsigned int binSearch(const unsigned int x, const unsigned int arr[], unsigned int size);
template unsigned int binSearch(const unsigned long x, const unsigned long arr[], unsigned int size);

/* --------- random numbers -------- */
class RandGenStorage{
#ifdef HAVE_VALID_OPENMP
    // in the case of OpenMP, we have as many independent pseudo-random number generators
    // as there are threads, and each thread uses its own instance, to avoid race condition
    // and maintain deterministic output
    int maxThreads;
    std::vector<gsl_rng*> randgen;
public:
    RandGenStorage() {
        maxThreads = std::max(1, omp_get_max_threads());
        randgen.resize(maxThreads);
        for(int i=0; i<maxThreads; i++) {
            randgen[i] = gsl_rng_alloc(gsl_rng_default);
            gsl_rng_set(randgen[i], i);  // assign a different but deterministic seed to each thread
        }
    }
    ~RandGenStorage() {
        for(int i=0; i<maxThreads; i++)
            gsl_rng_free(randgen[i]);
    }
    void randomize(unsigned int seed) {
        if(!seed)
            seed = (unsigned int)time(NULL);
        for(int i=0; i<maxThreads; i++)
            gsl_rng_set(randgen[i], seed+i);
    }
    inline double random() {
        int i = std::min(omp_get_thread_num(), maxThreads-1);
        return gsl_rng_uniform(randgen[i]);
    }
#else
    gsl_rng* randgen;
public:
    RandGenStorage() {
        randgen = gsl_rng_alloc(gsl_rng_default);
    }
    ~RandGenStorage() {
        gsl_rng_free(randgen);
    }
    void randomize(unsigned int seed) {
        gsl_rng_set(randgen, seed ? seed : (unsigned int)time(NULL));
    }
    inline double random() {
        return gsl_rng_uniform(randgen);
    }
#endif
};

// global instance of random number generator -- created at program startup and destroyed
// at program exit. Note that the order of initialization of different modules is undefined,
// thus no other static variable initializer may use the random() function.
// Moving the initializer into the first call of random() is not a remedy either,
// since it may already be called from a parallel section and will not determine
// the number of threads correctly.
static RandGenStorage randgen;

void randomize(unsigned int seed)
{
    randgen.randomize(seed);
}

// generate a random number using the global generator
double random()
{
    return randgen.random();
}

// generate 2 random numbers with normal distribution, using Box-Muller approach
void getNormalRandomNumbers(double& num1, double& num2)
{
    double p1 = random();
    double p2 = random();
    if(p1>0&&p1<=1)
        p1 = sqrt(-2*log(p1));
    num1 = p1*sin(2*M_PI*p2);
    num2 = p1*cos(2*M_PI*p2);
}

double quasiRandomHalton(unsigned int ind, unsigned int base)
{
    double val = 0, fac = 1.;
    while(ind > 0) {
        fac /= base;
        val += fac * (ind % base);
        ind /= base;
    }
    return val;
}

// ------- tools for analyzing the behaviour of a function around a particular point ------- //
// this comes handy in root-finding and related applications, when one needs to ensure that 
// the endpoints of an interval strictly bracked the root: 
// if f(x) is exactly zero at one of the endpoints, and we want to locate the root inside the interval,
// then we need to shift slightly the endpoint to ensure that f(x) is strictly positive (or negative).

PointNeighborhood::PointNeighborhood(const IFunction& fnc, double x0) : absx0(fabs(x0))
{
    // small offset used in computing numerical derivatives, if the analytic ones are not available
    double delta = fmax(fabs(x0) * GSL_ROOT3_DBL_EPSILON, 16*GSL_DBL_EPSILON);
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

double PointNeighborhood::dxToPosneg(double sgn) const
{
    // safety factor to make sure we overshoot in finding the value of opposite sign
    double s0 = sgn*f0 * 1.1;
    double sder = sgn*fder, sder2 = sgn*fder2;
    // offset should be no larger than the scale of variation of the function,
    // but no smaller than the minimum resolvable distance between floating point numbers
    const double delta = fmin(fabs(fder/fder2)*0.5,
        fmax(1000*GSL_DBL_EPSILON * absx0, fabs(f0)) / fabs(sder));  //TODO!! this is not satisfactory
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
    return sign(sder) * (delta - 2*s0/(sqrt(discr)+fabs(sder)) );
}

double PointNeighborhood::dxToNearestRoot() const
{
    if(f0==0) return 0;  // already there
    double dx_nearest_root = -f0/fder;  // nearest root by linear extrapolation, if fder!=0
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

double PointNeighborhood::dxBetweenRoots() const
{
    if(f0==0 && fder==0) return 0;  // degenerate case
    return sqrt(fder*fder - 2*f0*fder2) / fabs(fder2);  // NaN if discriminant<0 - no roots
}

double deriv2(double x0, double x1, double x2, double f0, double f1, double f2,
    double df0, double df1, double df2)
{
    // construct a divided difference table to evaluate 2nd derivative via Hermite interpolation
    double dx10 = x1-x0, dx21 = x2-x1, dx20 = x2-x0;
    double df10 = (f1   - f0  ) / dx10;
    double df21 = (f2   - f1  ) / dx21;
    double dd10 = (df10 - df0 ) / dx10;
    double dd11 = (df1  - df10) / dx10;
    double dd21 = (df21 - df1 ) / dx21;
    double dd22 = (df2  - df21) / dx21;
    return ( -2 * (pow_2(dx21)*(dd10-2*dd11) + pow_2(dx10)*(dd22-2*dd21)) +
            4*dx10*dx21 * (dx10*dd21 + dx21*dd11) / dx20 ) / pow_2(dx20);
}

// ------ root finder routines ------//

/// used in hybrid root-finder to predict the root location by Hermite interpolation:
/// compute the value of f(x) given its values and derivatives at two points x1,x2
/// (x1<=x<=x2 or x1>=x>=x2 is implied but not checked), if the function is expected to be
/// monotonic on this interval (i.e. its derivative does not have roots on x1..x2),
/// otherwise return NAN
inline double interpHermiteMonotonic(double x, double x1, double f1, double dfdx1,
    double x2, double f2, double dfdx2)
{
    // derivatives must exist and have the same sign
    // (but shouldn't bee too large, otherwise we have an overflow -- apparently a bug in gsl_poly_solve)
    if(!isFinite(dfdx1+dfdx2) || dfdx1*dfdx2<0 || fabs(dfdx1)>1e100 || fabs(dfdx2)>1e100)
        return NAN;
    const double dx = x2-x1, sixdf = 6*(f2-f1);
    const double t = (x-x1)/dx;
    // check if the interpolant is monotonic on t=[0:1]
    double t1, t2;
    int nroots = gsl_poly_solve_quadratic(-sixdf+3*dx*(dfdx1+dfdx2), 
        sixdf-2*dx*(2*dfdx1+dfdx2), dx*dfdx1, &t1, &t2);
    if(nroots>0 && ((t1>=0 && t1<=1) || (t2>=0 && t2<=1)) )
        return NAN;   // will produce a non-monotonic result
    return pow_2(1-t) * ( (1+2*t)*f1 + t * dfdx1*dx )
         + pow_2(t) * ( (3-2*t)*f2 + (t-1)*dfdx2*dx );
}

/// a hybrid between Brent's method and interpolation of root using function derivatives;
/// it is based on the implementation from GSL, original authors: Reid Priedhorsky, Brian Gough
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
        if(numIter >= MAXITER) {
            converged = true;  // not quite ready, but can't loop forever
            utils::msg(utils::VL_WARNING, "findRoot", "max # of iterations exceeded: "
                "x="+utils::toString(b,15)+" +- "+utils::toString(b-c)+
                " on interval ["+utils::toString(x_lower,15)+":"+utils::toString(x_upper,15)+
                "], req.toler.="+utils::toString(abstoler));
        }
    } while(!converged);
    return b;  // best approximation
}

/** scaling transformation of input function for the case that the interval is (semi-)infinite:
    it replaces the original argument  x  with  y in the range [0:1],  
    and implements the transformation of 1st derivative.
    TODO: change the transformation to logarithmic in case of infinite intervals,
    to promote a (nearly) scale-free behaviour.
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
                x_edge = 0;
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

    virtual unsigned int numDerivs() const { return F.numDerivs()>1 ? 1 : F.numDerivs(); }

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

// 1d minimizer with known initial point
static double findMinKnown(const IFunction& fnc, 
    double xlower, double xupper, double xinit,
    double flower, double fupper, double finit, double reltoler)
{
    gsl_function F;
    F.function = &functionWrapper;
    F.params = const_cast<IFunction*>(&fnc);
    gsl_min_fminimizer *minser = gsl_min_fminimizer_alloc(gsl_min_fminimizer_brent);
    double xroot = NAN;
    double abstoler = reltoler*fabs(xupper-xlower);
    gsl_min_fminimizer_set_with_values(minser, &F, xinit, finit, xlower, flower, xupper, fupper);
    int iter=0;
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
    }
    while (fabs(xlower-xupper) > abstoler && iter < MAXITER);
    gsl_min_fminimizer_free(minser);
    return xroot;
}

static inline double minGuess(double x1, double x2, double y1, double y2)
{
    const double golden = 0.618034;
    if(y1<y2)
        return x1 * golden + x2 * (1-golden);
    else
        return x2 * golden + x1 * (1-golden);
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
    // transform the original range into [0:1], even if it was (semi-)infinite
    ScaledFunction F(fnc, xlower, xupper);
    xlower = F.y_from_x(xlower);
    xupper = F.y_from_x(xupper);
    double ylower = F(xlower);
    double yupper = F(xupper);
    if(xinit == xinit) {
        xinit  = F.y_from_x(xinit);
    } else {    // initial guess not provided - try to find it somewhere inside the interval
        xinit = minGuess(xlower, xupper, ylower, yupper);
    }
    double yinit  = F(xinit);
    if(!isFinite(ylower+yupper+yinit))
        return NAN;
    int iter = 0;
    while( (yinit>=ylower || yinit>=yupper) && iter<MAXITER && fabs(xlower-xupper)>reltoler) {
        // if the initial guess does not enclose minimum, provide a new guess inside a smaller range
        if(ylower<yupper) {
            xupper = xinit;
            yupper = yinit;
        } else {
            xlower = xinit;
            ylower = yinit;
        }
        xinit = minGuess(xlower, xupper, ylower, yupper);
        yinit = F(xinit);
        if(!isFinite(yinit))
            return NAN;
        iter++;
    }
    if(yinit>=ylower || yinit>=yupper)  // couldn't locate a minimum inside the interval,
        return F.x_from_y(ylower<yupper ? xlower : xupper);  // so return one of endpoints
    return F.x_from_y(findMinKnown(F, xlower, xupper, xinit, ylower, yupper, yinit, reltoler));
}

// ------- integration routines ------- //

double integrate(const IFunction& fnc, double x1, double x2, double reltoler, 
    double* error, int* numEval)
{
    if(x1==x2)
        return 0;
    gsl_function F;
    F.function = functionWrapper;
    F.params = const_cast<IFunction*>(&fnc);
    double result, dummy;
    size_t neval;
    gsl_integration_qng(&F, x1, x2, 0, reltoler, &result, error!=NULL ? error : &dummy, &neval);
    if(numEval!=NULL)
        *numEval = neval;
    return result;
}

double integrateAdaptive(const IFunction& fnc, double x1, double x2, double reltoler, 
    double* error, int* numEval)
{
    if(x1==x2)
        return 0;
    gsl_function F;
    F.function = functionWrapper;
    F.params = const_cast<IFunction*>(&fnc);
    double result, dummy;
    size_t neval;
    gsl_integration_cquad_workspace* ws=gsl_integration_cquad_workspace_alloc(MAXINTEGRPOINTS);
    gsl_integration_cquad(&F, x1, x2, 0, reltoler, ws, &result, error!=NULL ? error : &dummy, &neval);
    gsl_integration_cquad_workspace_free(ws);
    if(numEval!=NULL)
        *numEval = neval;
    return result;
}

double integrateGL(const IFunction& fnc, double x1, double x2, unsigned int N)
{
    if(x1==x2)
        return 0;
    gsl_function F;
    F.function = functionWrapper;
    F.params = const_cast<IFunction*>(&fnc);
    // tables up to N=20 are hard-coded in the library, no overhead
    gsl_integration_glfixed_table* t = gsl_integration_glfixed_table_alloc(N);
    double result = gsl_integration_glfixed(&F, x1, x2, t);
    gsl_integration_glfixed_table_free(t);
    return result;
}

void prepareIntegrationTableGL(double x1, double x2, int N, double* coords, double* weights)
{
    gsl_integration_glfixed_table* gltable = gsl_integration_glfixed_table_alloc(N);
    for(int i=0; i<N; i++)
        gsl_integration_glfixed_point(x1, x2, i, &(coords[i]), &(weights[i]), gltable);
    gsl_integration_glfixed_table_free(gltable);
}

// integration transformation classes

double ScaledIntegrandEndpointSing::x_from_y(const double y) const 
{
    if(y<0 || y>1)
        throw std::invalid_argument("value out of range [0,1]");
    return x_low + (x_upp-x_low) * y*y*(3-2*y);
}

// invert the transformation relation between x and y by solving a cubic equation
double ScaledIntegrandEndpointSing::y_from_x(const double x) const 
{
    if(x<x_low || x>x_upp)
        throw std::invalid_argument("value out of range [x_low,x_upp]");
    if(x==x_low) return 0;
    if(x==x_upp) return 1;
    double phi=acos(1-2*(x-x_low)/(x_upp-x_low))/3.0;
    return (1 - cos(phi) + M_SQRT3*sin(phi))/2.0;
}

double ScaledIntegrandEndpointSing::value(const double y) const 
{
    const double x = x_from_y(y);
    const double dx = (x_upp-x_low) * 6*y*(1-y);
    return dx==0 ? 0 : F(x)*dx;
}

// ------- multidimensional integration ------- //
#ifdef HAVE_CUBA
// wrapper for Cuba library
struct CubaParams {
    const IFunctionNdim& F;      ///< the original function
    const double xlower[];        ///< lower limits of integration
    const double xupper[];        ///< upper limits of integration
    std::vector<double> xvalue;  ///< temporary storage for un-scaling input point 
                                 ///< from [0:1]^N to the original range
    std::string error;           ///< store error message in case of exception
    CubaParams(const IFunctionNdim& _F, const double* _xlower, const double* _xupper) :
        F(_F), xlower(_xlower), xupper(_xupper) 
    { xvalue.resize(F.numVars()); };
};
static int integrandNdimWrapperCuba(const int *ndim, const double xscaled[],
    const int *ncomp, double fval[], void *v_param)
{
    CubaParams* param = static_cast<CubaParams*>(v_param);
    assert(*ndim == (int)param->F.numVars() && *ncomp == (int)param->F.numValues());
    try {
        for(int n=0; n< *ndim; n++)
            param->xvalue[n] = param->xlower[n] + (param->xupper[n]-param->xlower[n])*xscaled[n];
        param->F.eval(&param->xvalue.front(), fval);
        double result=0;
        for(int i=0; i< *ncomp; i++)
            result+=fval[i];
        if(!isFinite(result)) {
            param->error = "integrateNdim: invalid function value encountered at";
            for(int n=0; n< *ndim; n++)
                param->error += " "+utils::toString(param->xvalue[n], 15);
            return -1;
        }
        return 0;   // success
    }
    catch(std::exception& e) {
        param->error = std::string("integrateNdim: ") + e.what();
        return -999;  // signal of error
    }
}
#else
// wrapper for Cubature library
struct CubatureParams {
    const IFunctionNdim& F; ///< the original function
    int numEval;            ///< count the number of function evaluations
    std::string error;      ///< store error message in case of exception
    explicit CubatureParams(const IFunctionNdim& _F) :
        F(_F), numEval(0){};
};
static int integrandNdimWrapperCubature(unsigned int ndim, const double *x, void *v_param,
    unsigned int fdim, double *fval)
{
    CubatureParams* param = static_cast<CubatureParams*>(v_param);
    assert(ndim == param->F.numVars() && fdim == param->F.numValues());
    try {
        param->numEval++;
        param->F.eval(x, fval);
        double result=0;
        for(unsigned int i=0; i<fdim; i++)
            result+=fval[i];
        if(!isFinite(result)) {
            param->error = "integrateNdim: invalid function value encountered at";
            for(unsigned int n=0; n<ndim; n++)
                param->error += " "+utils::toString(x[n], 15);
            return -1;
        }
        return 0;   // success
    }
    catch(std::exception& e) {
        param->error = std::string("integrateNdim: ") + e.what();
        return -1;  // signal of error
    }
}
#endif

void integrateNdim(const IFunctionNdim& F, const double xlower[], const double xupper[], 
    const double relToler, const unsigned int maxNumEval, 
    double result[], double outError[], int* numEval)
{
    const unsigned int numVars = F.numVars();
    const unsigned int numValues = F.numValues();
    const double absToler = 0;  // maybe should be more flexible?
    // storage for errors in the case that user doesn't need them
    std::vector<double> tempError(numValues);
    double* error = outError!=NULL ? outError : &tempError.front();
#ifdef HAVE_CUBA
    CubaParams param(F, xlower, xupper);
    std::vector<double> tempProb(numValues);  // unused
    int nregions, neval, fail;
    const int NVEC = 1, FLAGS = 0, KEY = 0, minNumEval = 0;
    cubacores(0, 0);
    Cuhre(numVars, numValues, &integrandNdimWrapperCuba, &param, NVEC,
        relToler, absToler, FLAGS, minNumEval, maxNumEval, 
        KEY, NULL/*STATEFILE*/, NULL/*spin*/,
        &nregions, numEval!=NULL ? numEval : &neval, &fail, 
        result, error, &tempProb.front());
    if(fail==-1)
        throw std::runtime_error("integrateNdim: number of dimensions is too large");
    if(fail==-2)
        throw std::runtime_error("integrateNdim: number of components is too large");
    // need to scale the result to account for coordinate transformation [xlower:xupper] => [0:1]
    double scaleFactor = 1.;
    for(unsigned int n=0; n<numVars; n++)
        scaleFactor *= (xupper[n]-xlower[n]);
    for(unsigned int m=0; m<numValues; m++) {
        result[m] *= scaleFactor;
        error[m] *= scaleFactor;
    }
    if(!param.error.empty())
        throw std::runtime_error(param.error);
#else
    CubatureParams param(F);
    hcubature(numValues, &integrandNdimWrapperCubature, &param,
        numVars, xlower, xupper, maxNumEval, absToler, relToler,
        ERROR_INDIVIDUAL, result, error);
    if(numEval!=NULL)
        *numEval = param.numEval;
    if(!param.error.empty())
        throw std::runtime_error(param.error);
#endif
}

}  // namespace
