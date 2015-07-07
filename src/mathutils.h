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

/** default per-step relative accuracy of ODE solver */
const double ACCURACY_ODE=1e-8;

/** limit on the maximum number of steps in ODE solver */
const int ODE_MAX_NUM_STEP=1e6;

/** test if a number is not infinity or NaN */
bool is_finite(double x);

/** compare two numbers with a relative accuracy eps: 
    \return -1 if x<y, +1 if x>y, or 0 if x and y are approximately equal */
int fcmp(double x, double y, double eps=1e-15);

/** find a root of function on the interval [x1,x2].
    function must be finite at the ends of interval and have opposite signs (or be zero).
*/
double findroot(function F, void* params, double x1, double x2, double rel_toler=ACCURACY_ROOT);

/** find a root of function on the open interval [x1,x2].
    function is never evaluated at the ends of interval (thus it may be undefined there), 
    and the interval may be (semi-)infinite. 
    However, one must provide some information about the function, namely: 
    whether it is expected to decrease or increase on the interval, and 
    an initial guess (used to determine whether to look left or right of this point).
    Throws an exception if cannot bracket the root on the interval.
*/
double findroot_guess(function F, void* params, double x1, double x2, 
    double xinit, bool increasing, double rel_toler=ACCURACY_ROOT);

/** integrate a (well-behaved) function on a finite interval */
double integrate(function F, void* params, double x1, double x2, double rel_toler=ACCURACY_INTEGR);

/** solve a system of differential equations */
class odesolver {
public:
    odesolver(odefunction F, void* params, int numvars, double rel_toler=ACCURACY_ODE);
    ~odesolver();
    int advance(double tstart, double tfinish, double *y);
private:
    void* impl;
};

}  // namespace
