#include "math_spline.h"
#include <cmath>
#include <stdexcept>
#include <gsl/gsl_linalg.h>
/*  Clamped or natural cubic splines;
    the implementation is based on the code for natural cubic splines from GSL, original author:  G. Jungman
*/

namespace mathutils {
/* ----------- spline functions --------------- */

typedef struct
{
  std::vector<double> c;
  std::vector<double> g;
  std::vector<double> diag;
  std::vector<double> offdiag;
} cspline_state_t;

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
void CubicSpline::eval_deriv(const double x, double* value, double* deriv, double* deriv2) const
{
    const size_t size = xval.size();
    if(x <= xval[0]) {
        double dx  =  xval[1]-xval[0];
        double der = (yval[1]-yval[0])/dx - dx*(cval[1]+2*cval[0])/3.0;
        if(value)
            *value = yval[0] + der*(x-xval[0]);
        if(deriv)
            *deriv = der;
        if(deriv2)
            *deriv2= 0;
        return;
    }
    if(x >= xval[size-1]) {
        double dx  =  xval[size-1]-xval[size-2];
        double der = (yval[size-1]-yval[size-2])/dx + dx*(cval[size-2]+2*cval[size-1])/3.0;
        if(value)
            *value = yval[size-1] + der*(x-xval[size-1]);
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
    if(value)
        *value  = y_lo + delx * (b_i + delx * (c_i + delx * d_i));
    if(deriv)
        *deriv  = b_i + delx * (2.0 * c_i + 3.0 * d_i * delx);
    if(deriv2)
        *deriv2 = 2.0 * c_i + 6.0 * d_i * delx;
}

bool CubicSpline::isMonotonic() const
{
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

}  // namespace
