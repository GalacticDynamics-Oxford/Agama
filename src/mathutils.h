#pragma once

namespace mathutils{

/** prototype of a function taking one argument and a pointer to some other parameters */
typedef double(*function)(double,void*);

/** prototype of a function that returns derivatives for an ODE system */
typedef int (*odefunction)(double,const double *,double*,void*);

/** default relative accuracy of root-finder */
const double ACCURACY_ROOT=1e-6;

/** default relative accuracy of integration */
const double ACCURACY_INTEGR=1e-6;

/** limit on the maximum number of steps in ODE solver */
const int ODE_MAX_NUM_STEP=1e6;

/** test if a number is not infinity or NaN */
bool isFinite(double x);

/** compare two numbers with a relative accuracy eps: 
    \return -1 if x<y, +1 if x>y, or 0 if x and y are approximately equal */
int fcmp(double x, double y, double eps=1e-15);

/** return sign of a number */
inline double sign(double x) { return x>0?1.:x<0?-1.:0; }

/** ensure that the angle lies in [0,2pi) */
double wrapAngle(double x);

/** create a nearly monotonic sequence of angles by adding or subtracting 2Pi
    as many times as needed to bring the current angle to within Pi from the previous one.
    This may be used in a loop over elements of an array, by providing the previous 
    element, already processed by this function, as xprev. 
    Note that this usage scenario is not stable against error accumulation. */
double unwrapAngle(double x, double xprev);

/** find a root of function on the interval [x1,x2].
    function must be finite at the ends of interval and have opposite signs (or be zero),
    otherwise NaN is returned.
    Throws an exception if the function evaluates to a non-finite number inside the interval,
    or if there are other problems in locating the root.
*/
double findRoot(function F, void* params, double x1, double x2, double rel_toler=ACCURACY_ROOT);

/** find a root of function on the open interval [x1,x2].
    function is never evaluated at the ends of interval (thus it may be undefined there), 
    and the interval may be (semi-)infinite. 
    However, one must provide some information about the function, namely: 
    whether it is expected to decrease or increase on the interval, and 
    an initial guess (used to determine whether to look left or right of this point).
    \return the root, or NaN if cannot bracket the root on the interval. 
*/
double findRootGuess(function F, void* params, double x1, double x2, 
    double xinit, bool increasing, double rel_toler=ACCURACY_ROOT);

/** integrate a (well-behaved) function on a finite interval */
double integrate(function F, void* params, double x1, double x2, double rel_toler=ACCURACY_INTEGR);

/** integrate a function with a transformation that removes possible singularities
    at the endpoints [x_low,x_upp], and the integral is computed over the interval [x1,x2] 
    such than x_low<=x1<=x2<=x_upp.
*/
double integrateScaled(function F, void* params, double x1, double x2, 
    double x_low, double x_upp, double rel_toler=ACCURACY_INTEGR);
    
/** numerically find a derivative of a function at a given point.
    \param[in] h is the initial step-size of differentiation, 
    \param[in] dir is the direction (1 - forward/right, -1 - backward/left, 0 - symmetric)
*/
double deriv(function F, void* params, double x, double h, int dir);

/** find a point x_1 where a function attains a strictly positive value.
    \param[in]  x_0 is the initial guess point,
    \param[out] f_1 if not NULL, will contain the value F(x_1);
    \param[out] der if not NULL, and if the initial value was non-positive, 
    an estimate of function derivative at x_1 will be returned in this variable;
    \return x_1 such that F(x_1)>0, or NaN if cannot find such point. 
*/
double findPositiveValue(function F, void* params, double x_0, double* f_1=0, double* der=0);

/** perform a linear least-square fit without constant term (i.e., y=c*x) */
double linearFitZero(unsigned int N, const double x[], const double y[]);

/** solve a system of differential equations */
class OdeSolver {
public:
    OdeSolver(odefunction F, void* params, int numvars, double abstoler, double reltoler);
    ~OdeSolver();
    int advance(double tstart, double tfinish, double *y);
private:
    void* impl;
};

}  // namespace
