/** \file    math_gausshermite.h
    \brief   Gauss-Hermite expansion
    \date    2017-2019
    \author  Eugene Vasiliev

*/
#pragma once
#include "math_base.h"
#include "math_spline.h"

namespace math{

/** Representation of a velocity distribution function in terms of Gauss-Hermite expansion */
class GaussHermiteExpansion: public IFunctionNoDeriv {
    double Ampl;   ///< overall amplitude
    double Center; ///< position of the center of expansion
    double Width;  ///< width of the gaussian
    std::vector<double> moments;  ///< values of Gauss-Hermite moments
public:
    /// initialize the function from previously computed coefficients
    GaussHermiteExpansion(const std::vector<double>& coefs,
        double ampl, double center, double width) :
        Ampl(ampl), Center(center), Width(width), moments(coefs) {}

    /// find the best-fit coefficients for a given function.
    /// If the parameters gamma, center and sigma are not provided, they are estimated
    /// by finding the best-fit Gaussian without higher-order terms; in this case
    /// the first three GH moments should be (1,0,0) to within integration accuracy.
    GaussHermiteExpansion(const IFunction& fnc,
        unsigned int order, double gamma=NAN, double center=NAN, double sigma=NAN);

    /// same as the above constructor, but specialized to the case of a piecewise-polynomial function
    GaussHermiteExpansion(const BaseInterpolator1d& fnc,
        unsigned int order, double gamma=NAN, double center=NAN, double sigma=NAN);

    /// evaluate the expansion at the given point
    virtual double value(double x) const;

    /// return the array of Gauss-Hermite coefficients
    inline const std::vector<double>& coefs() const { return moments; }

    inline double ampl()   const { return Ampl;   }  ///< return the overall amplitude
    inline double center() const { return Center; }  ///< return the center of expansion
    inline double width()  const { return Width;  }  ///< return the width of the base gaussian

    /// return the normalization constant \f$ N_n = \int_{-\infty}^\infty exp(-x^2/2) H_n(x) dx \f$
    static double normn(unsigned int n);

    /// return the integral of the function over the entire real axis
    double norm() const;
};

/** Construct the matrix that converts the velocity distribution represented by its B-spline amplitudes
    into the Gauss-Hermite moments for a single aperture with known parameters of GH expansion.
    \param[in]  N      is the degree of B-spline (0 to 3);
    \param[in]  grid   is the grid in velocity space defining the B-spline;
    \param[in]  order  is the order M of GH expansion (i.e. it has order+1 coefficients h_0..h_M);
    \param[in]  ampl   is the overall amplitude of the gaussian;
    \param[in]  center is the central point of the gaussian;
    \param[in]  width  is the width of the gaussian;
    \return  a matrix G with (order+1) rows and bspl.numValues() columns.
    To obtain the GH moments, multiply this matrix by the vector of amplitudes for a single aperture.
*/
Matrix<double> computeGaussHermiteMatrix(int N, const std::vector<double>& grid,
    unsigned int order, double ampl, double center, double width);


}  // namespace
