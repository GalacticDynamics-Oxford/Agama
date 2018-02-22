/** \file   math_sphharm.h
    \brief  Legendre polynomials and spherical-harmonic transformations
    \date   2015-2016
    \author Eugene Vasiliev
*/
#pragma once
#include "coord.h"
#include <vector>
#include <utility>

namespace math {

/** Array of normalized associate Legendre polynomials W and their derivatives for l=m..lmax
    (theta-dependent factors in spherical-harmonic expansion):
    \f$  Y_l^m(\theta, \phi) = W_l^m(\theta) \{\sin,\cos\}(m\phi) ,
         W_l^m = \sqrt{\frac{ (2l+1) (l-m)! }{ 4\pi (l+m)! }} P_l^m(\cos(\theta))  \f$,
    where P are un-normalized associated Legendre functions.
    \param[in]  lmax - the maximum degree l of computed polynomials.
    \param[in]  m    - the order m: 0 <= m <= lmax.
    \param[in]  tau  - the argument of the function: tau = cos(theta) / (sin(theta) + 1),
    where theta is the usual polar angle. The rationale for choosing this combination,
    which is simply tan( (pi/2 - theta) / 2), is twofold: on the one hand, using cos(theta)
    as the argument leads to loss of precision when |cos|->1, which happens already for
    theta as large as 1e-8; on the other hand, using theta as argument would incur additional
    expense to compute both sin and cos theta. By contrast, they are simply computed from tau
    as sin(theta) = (1 - tau^2) / (1 + tau^2), cos(theta) = 2*tau / (1 + tau^2),  and
    at the same time |d tau / d theta| is always between 1/2 and 1, without loss of precision.
    \param[out] resultArray - the computed array of W_l^m,  which must be pre-allocated
    to contain lmax-m+1 elements, stored as  l=m, m+1, ..., lmax;
    \param[out] derivArray - the computed array of derivatives dW_l^m / d theta
    (not cos theta, nor tau); if NULL, they are not computed, otherwise must be pre-allocated
    with the same size as resultArray. 
    \param[out] deriv2Array - the computed array of second derivatives d2W_l^m / d theta^2;
    if NULL, they are not computed, and if not NULL, derivArray must be not NULL too.
    The derivatives for |tau| -> 1 (i.e. theta->0 or theta->pi) are computed accurately
    using asymptotic expressions.
*/
void sphHarmArray(const unsigned int lmax, const unsigned int m, const double tau,
    double* resultArray, double* derivArray=NULL, double* deriv2Array=NULL);

/** Compute the values of cosines and optionally sines of an arithmetic progression of angles:
    cos(phi), cos(2 phi), ..., cos(m phi), [ sin(phi), sin(2 phi), ..., sin(m phi) ].
    \param[in]  phi - the angle;
    \param[in]  m   - the number of multiples of this angle to process, must be >=1;
    \param[in]  needSine - whether to compute sines as well (if false then only cosines are computed);
    \param[out] outputArray - pointer to an existing array of length m (if needSine==false)
    or 2m (if needSine==true) that will store the output values.
*/
void trigMultiAngle(const double phi, const unsigned int m, const bool needSine, double* outputArray);

/** Indexing scheme for spherical-harmonic transformation.
    It defines the maximum order of expansion in theta (lmax) and phi (mmax),
    that should satisfy 0 <= mmax <= lmax (0 means using only one term),
    and also defines which coefficients to skip (they are assumed to be identically zero
    due to particular symmetries of the function), specified by step, lmin and mmin.
    Namely, the loop over non-zero coefficients should look like
    \code
    for(int m=ind.mmin(); m<=ind.mmax; m++)
       for(int l=ind.lmin(m); l<=ind.lmax; l+=ind.step)
           doSomethingWithCoefficient(ind.index(l, m));
    \endcode
    This scheme is used in the forward and inverse SH transformation routines.

    The coefficients are arranged in the triangular shape, as follows:
    \code
    m=   -4  -3  -2  -1   0   1   2   3   4
    l=0                   0
    l=1               1   2   3
    l=2           4   5   6   7   8
    l=3       9  10  11  12  13  14  15
    l=4  16  17  18  19  20  21  22  23  24
    \endcode
    m>=0 correspond to cos(m phi) and m<0 - to sin(|m| phi).
    Regardless of mmax, the length of the coefficients array is always (lmax+1)^2.

    Various symmetries imply that some of the terms must be zero, as detailed in the table below,
    where 0 stands for a coefficient that must be zero, and dot - for a possibly non-zero one
    \code
    +-----------------------------------------------------------+
    | invariance       |       coefficients that are zero       |
    | under transform  |     l is odd    |      l is even  |    |
    |                  |m>=0 m>0 m<0  m<0|m>=0 m>0 m<0  m<0|m!=0|
    |                  |even odd even odd|even odd even odd|    |
    |------------------+----------------------------------------|
    |x => -x           | .    0   0    .    .   0   0    .    . | ST_XREFLECTION
    |y => -y           | .    .   0    0    .   .   0    0    . | ST_YREFLECTION
    |z => -z           | 0    .   0    .    .   0   .    0    . | ST_ZREFLECTION
    |x,y,z => -x,-y,-z | 0    0   0    0    .   .   .    .    . | ST_REFLECTION
    |x,y=>-x,-y | z=>-z| 0    0   0    0    .   0   .    0    . | ST_BISYMMETRIC
    |triaxial sym.     | 0    0   0    0    .   0   0    0    . | ST_TRIAXIAL
    |z-axis rotation   | .    0   0    0    .   0   0    0    0 | ST_ZROTATION
    |axisymmetric      | 0    0   0    0    .   0   0    0    0 | ST_AXISYMMETRIC
    |spherical         |     only  l=0, m=0 remains nonzero     | ST_SPHERICAL
    +-----------------------------------------------------------+
    \endcode
    Another way of representing these symmetries is shown on the diagrams below:
    \code
                x-reflection                    z-reflection
    m=   -4 -3 -2 -1  0  1  2  3  4      -4 -3 -2 -1  0  1  2  3  4
    l=0               .                               .
    l=1            .  .  0                         .  0  .
    l=2         0  .  .  0  .                   .  0  .  0  .
    l=3      .  0  .  .  0  .  0             .  0  .  0  .  0  .
    l=4   0  .  0  .  .  0  .  0  .       .  0  .  0  .  0  .  0  .

                y-reflection                mirror (xyz-reflection)
    m=   -4 -3 -2 -1  0  1  2  3  4       -4 -3 -2 -1  0  1  2  3  4
    l=0               .                                .
    l=1            0  .  .                          0  0  0
    l=2         0  0  .  .  .                    .  .  .  .  .
    l=3      0  0  0  .  .  .  .              0  0  0  0  0  0  0
    l=4   0  0  0  0  .  .  .  .  .        .  .  .  .  .  .  .  .  .

    \endcode
*/
class SphHarmIndices {
public:
    const int
    lmax,  ///< order of expansion in theta (>=0)
    mmax,  ///< order of expansion in phi (0<=mmax<=lmax)
    step;  ///< 1 if all l terms are used, 2 if only every other l term for each m is used

    /** Create the spherical-harmonic indexing scheme for the given symmetry type
        and expansion order.
        \param[in] sym  - the type of symmetry that determines which coefficients to omit;
        \param[in] lmax - order of expansion in polar angle (theta);
        \param[in] mmax - order of expansion in azimuthal angle (phi);
        \returns   the instance of indexing scheme to be passed to spherical harmonic transform.
    */
    SphHarmIndices(int lmax, int mmax, coord::SymmetryType sym);

    /// return symmetry properties of this index set
    coord::SymmetryType symmetry() const { return sym; }

    /// number of elements in the array of spherical-harmonic coefficients
    inline unsigned int size() const { return (lmax+1)*(lmax+1); }

    /// index of coefficient with the given l and m
    /// (0<=l<=lmax, -l<=m<=l, no range check performed!)
    static unsigned int index(int l, int m) { return l*(l+1)+m; }

    /// decode the l-index from the combined index of a coefficient
    static int index_l(unsigned int c);

    /// decode the m-index from the combined index of a coefficient
    static int index_m(unsigned int c);

    /// minimum l-index for the given m (if larger than lmax, it means that this value of m is not used)
    inline int lmin(int m) const { return m>=-mmax && m<=mmax ? lmin_arr[m+mmax] : lmax+1; }

    /// minimum m-index
    inline int mmin() const { return isYReflSymmetric(sym) ? 0 : -mmax; }

private:
    coord::SymmetryType sym;   ///< symmetry properties of this index set
    std::vector<int> lmin_arr; ///< array of minimum l-indices for each m
};

/** Determine the order of expansion and its symmetry properties from the list of coefficients,
    by analyzing which of them are non-zero.
    \param[in]  C - an array of coefficients with length (lmax+1)^2;
    \returns    an instance of indexing scheme.
*/
SphHarmIndices getIndicesFromCoefs(const std::vector<double> &C);

/** Collect the list of azimuthal harmonic indices (m) that are non-zero
    for the given symmetry type and expansion order */
std::vector<int> getIndicesAzimuthal(int mmax, coord::SymmetryType sym);

/** Class for performing forward Fourier transformation
    (computing Fourier coefficients from the values of a function at equally spaced angles).
    The transformation may involve only cosine terms, or both sine and cosine terms:
    \f$  C_m    = \int_0^{2\pi} f(\phi) \cos( m \phi) d\phi  \f$,
    \f$  C_{-m} = \int_0^{2\pi} f(\phi) \sin(|m|\phi) d\phi  \f$, 0 <= m <= mmax.
    For the given expansion order and symmetry requirement, it computes transformation
    coefficients and stores in an internal table; the user then may perform many transformations
    with the same setup, by collecting the values of function at the grid of angles and
    calling `transform()` method, as follows:
    \code
    FourierTransformForward trans(mmax, useSine);
    std::vector<double> input_values(trans.size());
    for(unsigned int i=0; i<trans.size(); i++)
        input_values[i] = my_function(trans.phi(i));
    std::vector<double> output_coefs(trans.size());
    trans.transform(&values.front(), &output_coefs.front());
    \endcode
    If useSine flag is set, the transformation includes both sine and cosine terms,
    which are stored in the following order:
    output_coefs[mmax-m] = C_{-m}, 1 <= m <= mmax  (sine terms),
    output_coefs[mmax+m] = C_m,    0 <= m <= mmax  (cosine terms).
    If useSine is false, then the output contains only cosine terms:
    output_coefs[m] = C_m,  0 <= m <= mmax.
*/
class FourierTransformForward {
public:
    /** create the transformation of given order and symmetry */
    FourierTransformForward(int mmax, bool useSine);

    /** return the size of both input array of function values
        and the output array of coefficients */
    inline unsigned int size() const { return useSine ? mmax*2+1 : mmax+1; }

    /** return the i-th node of angular grid:
        if useSine=true, 0 <= phi_i < 2 pi, otherwise 0 <= phi_i < pi */
    inline double phi(unsigned int i) const { return i*M_PI/(mmax+0.5); }

    /** perform the transform on the collected function values.
        \param[in]  values is the array of function values at a regular grid in phi,
        arranged so that  values[i*stride] = f(phi(i)), 0 <= i < size()
        \param[out] coefs must point to an existing array of length `size()`,
        which will be filled with Fourier coefficients as described above
        \param[in]  stride (optional) is the spacing between consecutive elements
        in the input array (no stride in the output)
    */
    void transform(const double values[] /*in*/, double coefs[] /*out*/, int stride=1) const;
private:
    const int mmax;               ///< order of expansion
    const bool useSine;           ///< whether to use sine terms (if no then only cosines)
    std::vector<double> trigFnc;  ///< values of sine/cosine at the nodes of angular grid
};

/** Class for performing forward spherical-harmonic transformation.
    For the given coefficient indexing scheme, specified by an instance of `SphHarmIndices`,
    it computes the S-H coefficients C_lm from the values of function f(theta,phi)
    at nodes of a 2d grid in theta and phi.
    The workflow is the following:
      - create the instance of forward transform class for the given indexing scheme;
      - for each function f(theta,phi) the user should collect its values at the nodes of grid
        specified by member functions `theta(i)` and `phi(i)` into an array with length `size()`:
        \code
        SphHarmTransformForward trans(ind);
        std::vector<double> input_values(trans.size());
        for(unsigned int i=0; i<trans.size(); i++)
            input_values[i] = my_function(trans.theta(i), trans.phi(i));
        std::vector<double> output_coefs(ind.size());
        trans.transform(&values.front(), &output_coefs.front());
        \endcode
    Depending on the symmetry properties specified by the indexing scheme, not all elements
    of input_values need to be filled by the user; the transform routine takes this into account.
    The implementation uses 'naive' summation approach without any FFT or fast Legendre transform
    algorithms, has complexity O(lmax^2*mmax) and is only suitable for lmax <~ few dozen.
    The transformation is 'lossless' (to machine precision) if the original function is
    band-limited, i.e. given by a sum of spherical harmonics with order up to lmax and mmax.
*/
class SphHarmTransformForward {
public:
    /// initialize the grid in theta and the table of transformation coefficients
    SphHarmTransformForward(const SphHarmIndices& ind);

    /// return the required size of input array for the forward transformation
    inline unsigned int size() const { return thetasize() * fourier.size(); }

    /// return the cos(theta) coordinate (-1:1) of i-th element of input array, 0 <= i < size()
    inline double costheta(unsigned int i) const { return costhnodes[i / fourier.size()]; }

    /// return the phi coordinate [0:2pi) of i-th element of input array, 0 <= i < size()
    inline double phi(unsigned int i) const { return fourier.phi(i % fourier.size()); }

    /** perform the transformation of input array (values) into the array of coefficients.
        \param[in]  values is the array of function values at a rectangular grid in (theta,phi),
        arranged so that  values[i*stride] = f(theta(i), phi(i)), 0 <= i < size()
        \param[out] coefs must point to an existing array of length `ind.size()`,
        which will be filled with spherical-harmonic expansion coefficients as follows:
        coefs[ind.index(l,m)] = C_lm, 0 <= l <= lmax, -l <= m <= l
        \param[in]  stride (optional) is the spacing between consecutive elements in the input array
    */
    void transform(const double values[] /*in*/, double coefs[] /*out*/, int stride=1) const;

private:
    /// coefficient indexing scheme (including lmax and mmax)
    const SphHarmIndices ind;

    /// Fourier transform in azimuthal angle
    const FourierTransformForward fourier;

    /// coordinates of the grid nodes in theta on (0:pi/2]
    std::vector<double> costhnodes;

    /// values of all associated Legendre functions of order <= lmax,mmax at nodes of theta-grid
    std::vector<double> legFnc;

    /// number of sample points in theta, spanning either (0:pi/2] or (0:pi)
    unsigned int thetasize() const { return isZReflSymmetric(ind.symmetry()) ? ind.lmax/2+1 : ind.lmax+1; }
};

/** Routine for performing inverse spherical-harmonic transformation.
    Given the array of coefficients obtained by the forward transformation,
    it computes the value of function at the given position on unit sphere (theta,phi).
    \param[in]  ind   - coefficient indexing scheme, defining lmax, mmax and skipped coefs;
    \param[in]  coefs - the array of coefficients;
    \param[in]  tau   - cos(theta) / (1 + sin(theta)), where theta is the polar angle;
    \param[in]  phi   - the azimuthal angle;
    \returns    the value of function at (theta,phi)
*/
double sphHarmTransformInverse(const SphHarmIndices& ind, const double coefs[],
    const double tau, const double phi);

}  // namespace math
