#include "galaxymodel_fokkerplanck.h"
#include "galaxymodel_spherical.h"
#include "df_spherical.h"
#include "math_core.h"
#include "potential_multipole.h"
#include "utils.h"
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <algorithm>

namespace galaxymodel{

namespace {

/// default grid size in log phase volume
static const size_t DEFAULT_GRID_SIZE = 200;

/// width of the log-normal source function describing the star formation
static const double SOURCE_WIDTH = 0.2;


/// scaling transformation for the grid in phase volume (a uniform grid in scaled variable is used):
/// scaled variable is x = SCALEH(h)
#define SCALEH    log
/// accordingly, h = UNSCALEH(x)
#define UNSCALEH  exp
/// and dh(x)/dx = DHDSCALEH(x)
#define DHDSCALEH exp

// a few auxiliary routines:

/// construct a new vector with values converted by the given mathematical function
inline std::vector<double> convert(const std::vector<double>& vec, double(*fnc)(double))
{
    std::vector<double> newvec(vec.size());
    for(size_t i=0, size=vec.size(); i<size; i++)
        newvec[i] = fnc(vec[i]);
    return newvec;
}

/// find the minimum element of a vector
inline double minElement(const std::vector<double>& vec)
{
    return *std::min_element(vec.begin(), vec.end());
}

/// find the maximum element of a vector
inline double maxElement(const std::vector<double>& vec)
{
    return *std::max_element(vec.begin(), vec.end());
}

/// compute the sum the elements of a vector
inline double sumElements(const std::vector<double>& vec)
{
    double result = 0;
    for(size_t i=0, size=vec.size(); i<size; i++)
        result += vec[i];
    return result;
}

/// compute element-wise sum of several vectors
inline std::vector<double> sumVectors(const std::vector< std::vector<double> > vec)
{
    std::vector<double> result(vec[0]);
    for(unsigned int c=1; c<vec.size(); c++)
        math::blas_daxpy(1., vec[c], result);
    return result;
}


/// Scaling transformation for a function expressed in scaled coordinates and its derivative
class ScaledFunction: public math::IFunction {
    math::PtrFunction fnc;  ///< function in scaled coordinates
public:
    ScaledFunction(const math::PtrFunction& f) : fnc(f) {}
    // the input is un-scaled coordinate, and the output derivative is also w.r.t. unscaled coord
    virtual void evalDeriv(const double h, double* val=NULL, double* der=NULL, double* =NULL) const
    {
        double x = SCALEH(h);
        fnc->evalDeriv(x, val, der);
        // convert the derivative w.r.t. x to deriv w.r.t. h
        if(der)
            *der /= DHDSCALEH(x);
        // do not let the interpolated value to drop below zero
        if(val && *val < 0.) {
            *val = 0.;
            if(der) *der = 0.;
        }
    }
    virtual unsigned int numDerivs() const { return 1; }
};

/// A log-normal function describing the source (rate of mass increase per unit time)
class LogNormal: public math::IFunctionNoDeriv {
    const double mean, width;
public:
    LogNormal(double _mean, double _width) :
        mean(_mean), width(_width) {}
    virtual double value(const double h) const {
        return exp(-0.5 * pow_2((log(h / mean)) / width)) / (M_SQRT2 * M_SQRTPI * width * h);
    }
};

/// The function whose projection operator yields the mass associated with each basis element
class FncMass: public math::IFunctionNoDeriv {
public:
    virtual double value(const double /*x*/) const { return 1; }
};

/// Same for the energy
class FncEnergy: public math::IFunctionNoDeriv {
    const potential::PhaseVolume& phasevol;
public:
    FncEnergy(const potential::PhaseVolume& pv) : phasevol(pv) {}
    virtual double value(const double h) const { return phasevol.E(h); }
};

/// The function determining the loss-cone draining rate
/// (fraction of mass lost per unit time at the given h)
class FncDrainRate: public math::IFunctionNoDeriv {
    const potential::BasePotential& pot;      ///< gravitational potential
    const potential::PhaseVolume&  phasevol;  ///< conversion between h and E
    const double Lcapt2;                      ///< angular momentum of the loss cone
    const math::IFunction& difCoefLC;         ///< diffusion coefficient in angular momentum D(h)
public:
    FncDrainRate(
        const potential::BasePotential& _pot,
        const potential::PhaseVolume& _phasevol,
        const double _Lcapt2,
        const math::IFunction& _difCoefLC) :
        pot(_pot), phasevol(_phasevol), Lcapt2(_Lcapt2), difCoefLC(_difCoefLC) {}

    virtual double value(const double h) const {
        double
        g, E  = phasevol.E(h, &g),
        Lcirc2= pow_2(L_circ(pot, E)),
        Trad  = g / (4*M_PI*M_PI * Lcirc2),
        difLC = difCoefLC(h),
        q     = difLC * Trad * Lcirc2 / Lcapt2,
        alpha = sqrt(q * sqrt(1 + q*q)),
        Rcapt = Lcapt2 / Lcirc2;
        if(Rcapt < 1)
            return -difLC / ((alpha - 1) * (1 - Rcapt) - log(Rcapt));
        else  // at this energy, all orbits lie inside the loss cone, i.e. should be depopulated entirely
            return -INFINITY;
    }
};

/// an extremely simple function that is a sum of several components
class FncSum: public math::IFunctionNoDeriv {
public:
    std::vector<math::PtrFunction> comps;
    explicit FncSum(unsigned int size): comps(size) {}
    virtual double value(const double x) const {
        double sum = 0;
        for(unsigned int c=0; c<comps.size(); c++)
            sum += comps[c]->value(x);
        return sum;
    }
};

/// solve the Poisson equation and construct the spherical potential
potential::PtrPotential computePotential(
    double Mbh,
    const math::IFunction* modelDensity,
    double rmin, double rmax,
    /*output*/ double& Phi0)
{
    std::vector<double> rad;
    std::vector<std::vector<double> > Phi, dPhi;
    if(modelDensity) {
        // compute the potential from the density, i.e. solve the Poisson equation
        shared_ptr<const potential::Multipole> modelPotential =
        potential::Multipole::create(potential::FunctionToDensityWrapper(*modelDensity),
            coord::ST_SPHERICAL, /*lmax*/ 0, /*mmax*/ 0, /*gridsize*/ 100, rmin, rmax);

        // diagnostic output: stellar potential at origin
        Phi0 = modelPotential->value(coord::PosCyl(0,0,0));

        // if no other components, use the stellar potential only
        if(Mbh == 0)
            return modelPotential;

        // otherwise obtain the potential coefficients and later add the black hole contribution
        modelPotential->getCoefs(rad, Phi, dPhi);
    } else {
        if(Mbh == 0)
            throw std::runtime_error(
                "FokkerPlanckSolver: need an external potential if self-gravity is disabled");

        // no stellar potential - just create an empty array of coefficients
        rad.resize(3);  // pure Newtonian potential is scale-free, use an arbitrary radial grid
        rad[0] = 0.5; rad[1] = 1.; rad[2] = 2.;
        Phi.resize(1, std::vector<double>(3));
        dPhi=Phi;
        Phi0=0.;
    }

    // add the contribution from the central black hole
    for(unsigned int i=0; i<rad.size(); i++) {
        Phi [0][i] -= Mbh / rad[i];
        dPhi[0][i] += Mbh / pow_2(rad[i]);
    }

    // construct the spherical interpolator from the total potential
    return potential::PtrPotential(new potential::Multipole(rad, Phi, dPhi));
}

/// make sure that the interpolated function is well-behaved
/// (i.e. has converging asymptotics and no negative values)
template<int N>
std::vector<double> regularizeAmplitudes(
    const math::BsplineInterpolator1d<N>& interp, const std::vector<double>& amplitudes)
{
    std::vector<double> result(amplitudes);
    int numAmpl = result.size();
    // first eliminate negative values
    for(int i=0; i<numAmpl; i++)
        result[i] = fmax(0., result[i]);
    // next check that the log-slopes at the inner/outermost points are physically valid
    // (the inner asymptotic is shallower than h^-1 and the outer is correspondingly steeper).
    // If that's not the case, modify the amplitudes of the first/last basis function.

    // values and derivatives B_i, B'_i of basis functions at the leftmost node
    // (actually for B-splines val[0] = 1, der[0] = -der[1], and the other terms should be zero)
    double val[N+1], der[N+1];
    double x = interp.xmin();
    int ind  = // index of the leftmost basis function out of the N+1 computed ones (should be zero)
    interp.nonzeroComponents(x, 0, val);
    interp.nonzeroComponents(x, 1, der);
    assert(ind == 0);
    // the value f and derivative df/dx at x=xmin are given by
    // f(xmin) = \sum_{k=0}^N f_k B_k(xmin), df/dx = \sum_{k=0}^M f_k B'_k,
    // where f_k are the amplitudes of the basis functions.
    // The log-slope w.r.t. unscaled variable h is df/dx dx/dh h/f, and it should be larger than...
    const double SLOPEMIN = -0.32;  // safe value even in the presence of a black hole
    double C = DHDSCALEH(x) / UNSCALEH(x) * SLOPEMIN;
    double L = C * val[0] - der[0], R = 0.;
    for(int k=1; k<=N; k++)
        R += result[k+ind] * (der[k] - C * val[k]);
    // we need to satisfy the inequality  f_0 L <= R, where R is guaranteed to be nonnegative
    if( result[0] * L > R )
        result[0] = R / L;

    // now the same exercise for the amplitude of the rightmost basis function
    x = interp.xmax();
    ind =
    interp.nonzeroComponents(x, 0, val);
    interp.nonzeroComponents(x, 1, der);
    assert(ind+N+1 == numAmpl);
    const double SLOPEMAX = -1.01;
    C = DHDSCALEH(x) / UNSCALEH(x) * SLOPEMAX;
    L = C * val[N] - der[N], R = 0.;
    for(int k=0; k<N; k++)
        R += result[k+ind] * (der[k] - C * val[k]);
    if( result[numAmpl-1] * L < R )
        result[numAmpl-1] = R / L;

    return result;
}

} // internal namespace


/** Abstract implementation of the spatial discretization scheme for the Fokker-Planck solver
    (the part that manages the evolution of the DF purely due to relaxation).
    The details of this implementation are opaque to the driver class FokkerPlanckSolver,
    and several different variants are provided by descendant classes.
    This class manages all tasks related to the discrete representation of functions of phase volume.
    The internal representation of DF and other functions uses a scaled variable instead of
    phase volume `h`, but this conversion is performed transparently to the outside code.

    Any function of `h` (including the DF) can be represented in the discretized form
    as an vector of B numbers. There are two associated tasks:

    - for a given smooth function `f(h)`, construct its discretized representation `f_j`, j=1..B
    - from the given array of numbers `f_b`, construct a smooth interpolating function
    which would be an approximation to the original function.

    The method `projVector(f)` computes the vector of projection operators applied to
    the function `f` - these are not the coefficients `f_j`, but a different array `P_i(f)`, i=1..B.
    These numbers are related to `f_i` via the linear equation system
    \f$  M_{ij} f_j = P_i ,  i=1..B \f$,
    where `M` is the so-called mass (or weight) matrix returned by the method `weightMatrix()`.
    This is a band matrix which can be cheaply inverted, but sometimes this is not even necessary
    (i.e. we do not need the coefficients `f_j` themselves, only the projections `P_i`).

    A slightly more complicated scenario involves a product of two functions, `f(h) v(h)`.
    Assume that we already have a discretized representation of `f(h)` in terms of the array `f_j`.
    Then for the given smooth function `v(h)`, the method `projMatrix(v)` returns a projection
    matrix `V_{ij}` such that
    \f$  V_{ij} f_j = P_i(f*v)  \f$.
    In other words, the vector  \f$  M^{-1} V f  \f$  provides the coefficients for reconstructing
    a smooth function which is an approximation of the product `f(h) v(h)`.
    In particular, if v=1, `V_{ij}` is identical to `M_{ij}`.

    Getting back to the Fokker-Planck equation, the evolution of the DF f, represented by its
    discretized form `f_j`, satisfies the followin linear equation system:
    \f$  M_{ij}  df_j/dt = (R_{ij} - V_{ij}) f_j + s_i,  i=1..B   \f$
    Here `M` is the mass matrix defined previously, `R` is the relaxation matrix,
    `V` is the projection matrix for the function `v(h)` describing the loss-cone draining rate,
    and `s` is the projection vector for the function `s(h)` that gives the star formation rate.
    The matrix `R` represents a discretized form of a differential operator, and is constructed
    using the method `relaxationMatrix()` from two other auxiliary arrays: advection and diffusion
    coefficients `A(h)` and `D(h)` collected at a pre-defined grid of points `h_k` 
    (their number  is different from the dimension B of vectors and matrices).
    The task of computing these coefficients lies on the driver class (FokkerPlanckSolver),
    only the location of these points is provided by the method `getGridForCoefs()`.

    This class does not store any data that evolves with time, and is not concerned with the time
    integration approach (this is done in the driver class).
*/
class FokkerPlanckImpl {
public:
    virtual ~FokkerPlanckImpl() {}

    /// construct the interpolated function `f(h)` from its discretized representation
    /// in terms of amplitudes `f_j`
    virtual math::PtrFunction getInterpolatedFunction(const std::vector<double>& amplitudes) const = 0;

    /// return the grid of points `h_k` at which the advection and diffusion coefficients
    /// `A(h), D(h)` need to be computed
    virtual std::vector<double> getGridForCoefs() const = 0;

    /// construct the relaxation matrix `R` that represents the differential operator
    /// in the Fokker-Planck equation, from the arrays of advection and diffusion coefficients
    /// collected at the predefined grid of points `h_k` by the calling code.
    virtual math::BandMatrix<double> relaxationMatrix(
        const std::vector<double>& gridAdv,
        const std::vector<double>& gridDif) const = 0;

    /// compute the vector of projections `P_i` of a given function `f(h)` on the basis elements;
    /// this vector can be used to find the amplitudes `f_j` from `M_{ij} f_j = P_i`.
    virtual std::vector<double> projVector(const math::IFunction& fnc) const = 0;

    /// return the weight/mass matrix `M_{ij}` that can be used to find the amplitudes of any
    /// function from its projection; equivalent to `projMatrix(I)` where I=1 identically.
    virtual math::BandMatrix<double> weightMatrix() const = 0;

    /// return the projection matrix `V_{ij}` for a given function `v(h)` that can be used
    /// to represent a discretized product of two functions.
    virtual math::BandMatrix<double> projMatrix(const math::IFunction& fnc) const = 0;
};


/** Variant of spatial discretization using the finite-element method with B-splines of degree N.
    The B-spline interpolator stores the grid that determines the shape of basis functions,
    and provides the interpolation of any function from its array of amplitudes.
    The size of the amplitudes array is K+N-1, where K is the number of nodes in the spatial grid.
    The FiniteElement1d class build on top of it additionally provides methods for computing vector
    and matrix projections of any function. The only non-trivial task left for this class is
    the coordinate transformation (it uses the globally defined macros SCALEH/UNSCALEH):
    all B-spline operations are performed in scaled coordinates, but interaction with the outside
    code uses unscaled ones.
    These projections are computed by numerical integration of the function multiplied by the basis
    functions or their derivatives; this integration uses the valus of the function at an auxiliary
    grid of points, also provided by the FiniteElement1d class in scaled coordinates and converted
    to unscaled ones.
    The same grid is used to collect the values of advection/diffusion coefficients.
    Due to coordinate transformation, the function values collected at the auxiliary grid
    are multiplied by the derivative of the scaling function, which are pre-computed in
    the constructor; the weight matrix M is essentially the projection matrix of the derivative
    of the scaling function.
*/
template<int N>
class FokkerPlanckImplFEM: public FokkerPlanckImpl {

    /// finite element B-spline interpolator in scaled variable
    const math::FiniteElement1d<N> fem;

    /// shortcut for the number of points in the auxiliary grid for computing the adv/dif coefs
    const unsigned int numPoints;

    /// the auxiliary grid for adv/dif coefs (in un-scaled variable, i.e. h)
    const std::vector<double> auxGridCoords;

    /// pre-computed values of weight function (derivative of the scaling transformation)
    /// at each point of the integration (auxiliary) grid
    const std::vector<double> auxGridWeights;

    /// pre-computed weight matrix
    const math::BandMatrix<double> weightMat;

    /// helper routine: collect the weighted function values at the nodes of integration grid
    std::vector<double> collectIntegrPoints(const math::IFunction& fnc) const
    {
        std::vector<double> fncValues(numPoints);
        for(unsigned int p=0; p<numPoints; p++)
            fncValues[p] = fnc(auxGridCoords[p]) * auxGridWeights[p];
        return fncValues;
    }

public:
    FokkerPlanckImplFEM(const std::vector<double>& gridh) :
        fem(math::BsplineInterpolator1d<N>(convert(gridh, SCALEH))),
        numPoints(fem.integrPoints().size()),
        auxGridCoords (convert(fem.integrPoints(), UNSCALEH)),
        auxGridWeights(convert(fem.integrPoints(), DHDSCALEH)),
        weightMat(fem.computeProjMatrix(auxGridWeights))
    {}

    virtual math::PtrFunction getInterpolatedFunction(const std::vector<double>& amplitudes) const;

    virtual std::vector<double> getGridForCoefs() const { return auxGridCoords; }

    virtual std::vector<double> projVector(const math::IFunction& fnc) const
    {
        return fem.computeProjVector(collectIntegrPoints(fnc));
    }

    virtual math::BandMatrix<double> projMatrix(const math::IFunction& fnc) const
    {
        return fem.computeProjMatrix(collectIntegrPoints(fnc));
    }

    virtual math::BandMatrix<double> weightMatrix() const { return weightMat; }

    virtual math::BandMatrix<double> relaxationMatrix(
        const std::vector<double>& gridAdv,
        const std::vector<double>& gridDif) const
    {
        assert(gridAdv.size() == numPoints && gridDif.size() == numPoints);
        // transform the diffusion coef: the original equation contains the term D df/dh,
        // and in the finite-element representation in terms of scaled variable x this reads
        // D' df/dx,  where D' = D / (dh/dx).
        // Additionally we change the sign due to integration by parts.
        std::vector<double> gridDifScaled(numPoints);
        for(unsigned int p=0; p<numPoints; p++)
            gridDifScaled[p] = -gridDif[p] / auxGridWeights[p];
        math::BandMatrix<double> matAdv = fem.computeProjMatrix(gridAdv, 1, 0);
        math::BandMatrix<double> matDif = fem.computeProjMatrix(gridDifScaled, 1, 1);
        math::blas_daxpy(-1., matAdv, matDif);
        return matDif;
    }
};

/// different implementations for N=1 (linear interpolator) and N=2,3 (cubic spline)
template<> math::PtrFunction FokkerPlanckImplFEM<1>::getInterpolatedFunction(
    const std::vector<double>& amplitudes) const
{
    return math::PtrFunction(new ScaledFunction(math::PtrFunction(new math::LinearInterpolator(
        fem.interp.xvalues(), regularizeAmplitudes(fem.interp, amplitudes)))));
}
template<int N> math::PtrFunction FokkerPlanckImplFEM<N>::getInterpolatedFunction(
    const std::vector<double>& amplitudes) const
{
    return math::PtrFunction(new ScaledFunction(math::PtrFunction(new math::CubicSpline(
        fem.interp.xvalues(), regularizeAmplitudes(fem.interp, amplitudes)))));
}


/** Variant of spatial discretization using the finite-difference approach combined with
    the Chang&Cooper(1970) prescription for weighted-upstream discretization of the advection term.
    In this approach the grid in the scaled spatial variable is used to represent a linearly
    interpolated function; the amplitudes simply correspond to the function values at grid nodes.
    Due to scaling transformation, the weight matrix is not a unit matrix, but it is still diagonal;
    computation of vector or matrix projections of any function simply amounts to collecting
    the function values at grid nodes and multiplying them by the derivative of the scaling function.
    By contrast, the advection/diffusion coefficients are collected at a different grid,
    namely at the cell centers of the original grid (i.e., if the function values are given at
    x_0, x_1, ... x_{N-1},  the cell centers are x_{i+1/2} = (x_i + x_{i+1}) / 2,  where x is
    the scaled spatial variable).
*/
class FokkerPlanckImplChangCooper: public FokkerPlanckImpl {

    /// grid in the phase volume `h` (unscaled!) that defines the discretized representation of a DF
    const std::vector<double> gridh;

    /// shortcut for gridh.size();
    /// this is the dimension of the amplitudes array, projection vectors and matrices
    const unsigned int gridSize;

    /// pre-computed values of weight function (derivative of the coordinate transformation
    /// multiplied by the width of the grid cell) at each point of the primary grid
    std::vector<double> gridWeights;

    /// pre-computed weight matrix (diagonal, with gridWeights on the main diaginal)
    math::BandMatrix<double> weightMat;

    /// auxiliary grid for collecting the advection/diffusion coefs: elements are put half-way
    /// between the nodes of the primary grid when expressed in the scaled coordinate,
    /// but this array contains the un-scaled coordinates that are provided to the calling code.
    std::vector<double> auxGridCoords;

    /// pre-computed array of coefficients that are multiplied with the value of diffusion coef
    /// at each point of the auxiliary grid
    std::vector<double> auxGridMult;

public:
    FokkerPlanckImplChangCooper(const std::vector<double>& _gridh) :
        gridh(_gridh),
        gridSize(_gridh.size()),
        gridWeights(gridSize),
        weightMat(gridSize, 1, 0.),
        auxGridCoords(gridSize-1),
        auxGridMult(gridSize-1)
    {
        std::vector<double> gridScaled = convert(gridh, SCALEH);  // the primary grid in scaled variable
        std::vector<double> auxGridScaled(gridSize-1);   // the auxiliary grid in scaled variable
        for(unsigned int i=0; i<gridSize-1; i++)
            auxGridScaled[i] = 0.5 * (gridScaled[i] + gridScaled[i+1]);
        for(unsigned int i=0; i<gridSize; i++) {
            double xleft   = i>0 ? auxGridScaled[i-1] : gridScaled[0];
            double xright  = i<gridSize-1 ? auxGridScaled[i] : gridScaled[gridSize-1];
            gridWeights[i] = DHDSCALEH(gridScaled[i]) * (xright-xleft);
            weightMat(i,i) = gridWeights[i];  // diagonal elements of the weight matrix
        }
        for(unsigned int i=0; i<gridSize-1; i++) {
            auxGridMult[i]   = 1 / (gridScaled[i+1] - gridScaled[i]) / DHDSCALEH(auxGridScaled[i]);
            auxGridCoords[i] = UNSCALEH(auxGridScaled[i]);
        }
    }

    virtual std::vector<double> projVector(const math::IFunction& fnc) const
    {
        std::vector<double> result(gridSize);
        for(unsigned int i=0; i<gridSize; i++)
            result[i] = fnc(gridh[i]) * gridWeights[i];
        return result;
    }

    virtual math::BandMatrix<double> projMatrix(const math::IFunction& fnc) const
    {
        math::BandMatrix<double> result(gridSize, 1, 0.);
        for(unsigned int i=0; i<gridSize; i++)
            result(i, i) = fnc(gridh[i]) * gridWeights[i];
        return result;
    }

    virtual math::PtrFunction getInterpolatedFunction(const std::vector<double>& gridf) const
    {
        return math::PtrFunction(new ScaledFunction(
            math::PtrFunction(new math::CubicSpline(convert(gridh, SCALEH), gridf, /*regularize*/true))));
    }

    virtual std::vector<double> getGridForCoefs() const
    {
        return auxGridCoords;
    }

    virtual math::BandMatrix<double> weightMatrix() const { return weightMat; }

    virtual math::BandMatrix<double> relaxationMatrix(
        const std::vector<double>& gridAdv,
        const std::vector<double>& gridDif) const
    {
        assert(gridAdv.size() == gridSize-1);
        math::BandMatrix<double> mat(gridSize, 1, 0.);  // matrix of the tridiagonal system
        double prevWminus = 0, prevWplus = 0, prevDifDx = 0;
        for(unsigned int i=0; i<gridSize; i++) {
            double nextWminus = 0, nextWplus = 0, nextDifDx = 0;
            if(i<gridSize-1) {
                // diffusion coef multiplied by the derivative of coordinate transformation
                // and by the distance between the nodes of the auxiliary grid
                nextDifDx  = gridDif[i] * auxGridMult[i];
                // the ratio of advection and diffusion coefficients [eq.27 of Park&Petrosian 1996]
                double w   = gridAdv[i] / nextDifDx;
                nextWminus = fabs(w) > 0.02 ?   // use a more accurate asymptotic expression for small w
                    w / (exp(w)-1) :
                    1 - 0.5 * w * (1 - (1./6) * w * (1 - (1./60) * w * w));
                nextWplus  = nextWminus + w;
                // Flux at the center of the grid cell [x_i to x_{i+1}] is
                // F_{i+1/2} = Dif * (Wplus f_{i+1} - Wminus f_i)
            }
            if(i>0)
                mat(i,i-1) = prevDifDx * prevWminus;  // below-diagonal element
            if(i<gridSize-1)
                mat(i,i+1) = nextDifDx * nextWplus;   // above-diagonal
            mat(i,i)  = -(prevDifDx * prevWplus + nextDifDx * nextWminus);  // diagonal
            prevDifDx =  nextDifDx;
            prevWplus =  nextWplus;
            prevWminus=  nextWminus;
        }
        return mat;
    }

};


/// The aggregate structure containing all internal data of FokkerPlanckSolver;
/// some of them stay fixed throughout the simulation while other evolve
class FokkerPlanckData {
public:

    /// number of species (fixed)
    const unsigned int numComp;

    /// Coulomb logarithm that enters the expressions for diffusion coefs (fixed)
    const double coulombLog;

    /// whether the density of evolving system contributes to the potential (fixed)
    const bool selfGravity;

    /// whether the stellar potential is updated in the course of simulation (fixed)
    const bool updatePotential;

    /// whether to use absorbing (true) or zero-flux (false) boundary condition at hmin (fixed)
    bool absorbingBoundaryCondition;

    /// if the boundary condition is absorbing, whether to use loss-cone draining at all energies (fixed)
    bool lossConeDrain;

    /// if set to nonzero, account for the energy loss due to GW emission (fixed)
    const double speedOfLight;

    /// array of masses of a single star in each species (fixed)
    std::vector<double> Mstar;

    /// array of capture radii in each species (may evolve as the central black hole mass changes)
    std::vector<double> captureRadius;

    /// power-law indices describing the evolution of capture radii with black hole mass
    std::vector<double> captureRadiusScalingExp;

    /// fractions of mass of disrupted stars of each species that is added to the black hole mass (fixed)
    std::vector<double> captureMassFraction;

    /// rate of mass added per unit time in each species accounting for star formation (fixed)
    std::vector<double> sourceRate;

    /// radii where the added mass in each species is deposited (fixed)
    std::vector<double> sourceRadius;

    /// total potential of all components plus the black hole (evolves or stays fixed)
    potential::PtrPotential currPot;

    /// total potential at the previous timestep (used to extrapolate its evolution)
    potential::PtrPotential prevPot;

    /// mapping between energy and phase volume for the total potential (evolves together with totalPot)
    potential::PtrPhaseVolume phasevol;

    /// mass of the central black hole (may evolve with time)
    double Mbh;

    /// mass of the black hole at the previous time when the capture radii were (re)computed (evolves)
    double prevMbh;

    /// total mass of each stellar component (evolves)
    std::vector<double> Mass;

    /// stellar potential at r=0 (evolves)
    double Phi0;

    /// total and kinetic energy of the entire system (evolves)
    double Etot, Ekin;

    /// total mass added due to star formation in each component (evolves)
    std::vector<double> sourceMass;

    /// the total energy change associated with star formation (sum for all components, evolves)
    double sourceEnergy;

    /// total change in mass (<0) resulting from the stars being captured by the black hole (evolves)
    std::vector<double> drainMass;

    /// associated change in energy (sum for all components, evolves)
    double drainEnergy;

    /// the rate of change of the total energy due to star formation (evolves)
    double sourceRateEnergy;

    /// conductive energy flux through the innermost boundary hmin (evolves)
    double drainRateEnergy;

    /// grid in phase volume h, (fixed)
    std::vector<double> gridh;

    /// arrays of amplitides that define the distribution function (one vector for each component, evolve)
    std::vector< std::vector<double> > gridf;

    /// auxiliary grid in phase volume where the advection/diffusion coefs are computed (fixed)
    std::vector<double> gridAdvDifCoefs;

    /// values of advection and diffusion coefficients collected at the nodes of an auxiliary grid,
    /// recomputed in `reinitAdvDifCoefs()` before each FP step (evolve)
    std::vector<double> gridAdv, gridDif;

    /// weight of each node in h in the total integral over the grid (fixed);
    /// the total mass of a DF component is a dot product of this vector and the array of amplitudes
    std::vector<double> gridMass;

    /// weight of each node in the weighted integral for energy (change as the potential evolves);
    /// the total energy of a DF component is a dot product of this vector and the array of amplitudes
    std::vector<double> gridEnergy;

    /// array of projections of the function that defines the source (star formation rate);
    /// evolves because the mapping between radius and phase volume changes with time
    std::vector< std::vector<double> > gridSourceRate;

    /// draining rate due to the angular-momentum flux into the loss cone of the central black hole
    /// (projection matrix computed from the loss rate, one per species, evolves)
    std::vector< math::BandMatrix<double> > drainMatrix;

    /// array of instantaneous drain times for each component (evolves)
    std::vector< std::vector<double> > drainTime;

    /// previously computed matrices R entering the matrix FP equation (one per species, evolves)
    std::vector< math::BandMatrix<double> > prevRelaxationMatrix;

    /// length of the previous timestep (evolves)
    double prevdeltat;

    /// number of times that the evolve() routine was called (evolves)
    int numSteps;

    /// check the input parameters and initialize all member variables
    FokkerPlanckData(const FokkerPlanckParams& params,
        const std::vector<FokkerPlanckComponent>& components) :
        numComp(components.size()),
        coulombLog(params.coulombLog),
        selfGravity(params.selfGravity),
        updatePotential(params.updatePotential),
        absorbingBoundaryCondition(false),
        lossConeDrain(params.lossConeDrain),
        speedOfLight(params.speedOfLight),
        Mstar(numComp, 0.),
        captureRadius(numComp, 0.),
        captureRadiusScalingExp(numComp, 0.),
        captureMassFraction(numComp, 0.),
        sourceRate(numComp, 0.),
        sourceRadius(numComp, 0.),
        Mbh(params.Mbh), prevMbh(params.Mbh),
        Mass(numComp, 0.), Etot(0.), Ekin(0.),
        sourceMass(numComp, 0.), sourceEnergy(0.), drainMass(numComp, 0.), drainEnergy(0.),
        sourceRateEnergy(0.), drainRateEnergy(0.),
        gridf(numComp),
        gridSourceRate(numComp),
        drainMatrix(numComp),
        drainTime(numComp),
        prevRelaxationMatrix(numComp),
        prevdeltat(0.), numSteps(0)
    {
        if(numComp == 0)
            throw std::runtime_error("FokkerPlanckSolver: empty component list");

        // assemble the total density of all components
        FncSum initDens(numComp);

        // check the validity of parameters for each component
        for(unsigned int c=0; c<numComp; c++){
            if(!components[c].initDensity)
                throw std::runtime_error("FokkerPlanckSolver: must provide initial density");
            if(components[c].captureMassFraction < 0. || components[c].captureMassFraction > 1.)
                throw std::runtime_error("FokkerPlanckSolver: capture mass faction be between 0 and 1");
            if(components[c].captureRadius < 0.)
                throw std::runtime_error("FokkerPlanckSolver: capture radius should be non-negative");
            if(components[c].captureRadiusScalingExp < 0. || components[c].captureRadiusScalingExp > 1.)
                throw std::runtime_error("FokkerPlanckSolver: "
                    "capture radius scaling exponent should be between 0 and 1");
            if(c==0)
                absorbingBoundaryCondition = components[c].captureRadius > 0.;
            else if(absorbingBoundaryCondition ^ (components[c].captureRadius > 0.))
                throw std::runtime_error("FokkerPlanckSolver: "
                    "if one capture radius is non-zero then all of them should be positive");
            if(components[c].Mstar <= 0.)
                throw std::runtime_error("FokkerPlanckSolver: stellar masses should be positive");
            if(components[c].sourceRate < 0.)
                throw std::runtime_error("FokkerPlanckSolver: source rate should be non-negative");
            if(components[c].sourceRadius < 0. ||
                (components[c].sourceRate > 0. && components[c].sourceRadius == 0.) )
                throw std::runtime_error("FokkerPlanckSolver: source radius should be non-negative");
            Mstar[c]                   = components[c].Mstar;
            captureRadius[c]           = components[c].captureRadius;
            captureRadiusScalingExp[c] = components[c].captureRadiusScalingExp;
            captureMassFraction[c]     = components[c].captureMassFraction;
            sourceRate[c]              = components[c].sourceRate;
            sourceRadius[c]            = components[c].sourceRadius;
            initDens.comps[c]          = components[c].initDensity;
        }
        if(!absorbingBoundaryCondition)
            lossConeDrain = false;   // if captureRadius is not set, there is no loss cone

        // initialize the potential and the phase volume
        currPot = computePotential(Mbh, selfGravity? &initDens : NULL,
            /*rmin-autodetect*/ 0., /*rmax*/ 0., /*diagnostic output*/ Phi0);
        phasevol.reset(new potential::PhaseVolume(
            potential::Sphericalized<potential::BasePotential>(*currPot)));
    }
};


// ------ the driver class for the Fokker-Planck solver ------ //

FokkerPlanckSolver::FokkerPlanckSolver(
    const FokkerPlanckParams& params,
    const std::vector<FokkerPlanckComponent>& components)
:
    data(new FokkerPlanckData(params, components))
{
    // set up the grid parameters
    size_t gridSize = params.gridSize ? params.gridSize : DEFAULT_GRID_SIZE;
    double hmin = params.hmin ? params.hmin : INFINITY, hmax = params.hmax ? params.hmax : 0.;
    bool fixmin = params.hmin!=0, fixmax = params.hmax!=0;

    // determine if we have a central black hole with a non-zero capture radius -
    // if yes, need to adjust the innermost boundary of phase volume.
    if(data->absorbingBoundaryCondition && data->Mbh>0.) {
        // compute the lowest possible energy corresponding to a circular orbit with
        // the angular momentum equal to the capture boundary, and its associated phase volume
        double rcapt = minElement(data->captureRadius);
        double rmin  = R_from_Lz(*data->currPot, sqrt(2 * data->Mbh * rcapt)), Phi;
        coord::GradCyl dPhi;
        data->currPot->eval(coord::PosCyl(rmin,0,0), &Phi, &dPhi);
        hmin = data->phasevol->value(Phi + 0.5 * rmin * dPhi.dR);
        fixmin = true;
    }

    // if necessary, set up the outer boundary expressed in terms of radius, not h
    if(params.rmax){
        coord::GradCyl dPhi;
        double Phi;
        data->currPot->eval(coord::PosCyl(params.rmax,0,0), &Phi, &dPhi);
        hmax = data->phasevol->value(Phi);
        fixmax = true;
    }

    // create the grid in phase volume (h) if its parameters were provided
    if(hmin>0. && hmax>hmin)
        data->gridh = math::createExpGrid(gridSize, hmin, hmax);

    // construct the initial distribution function(s)
    std::vector<math::PtrFunction> initDF(data->numComp);
    for(unsigned int comp=0; comp<data->numComp; comp++) {
        std::vector<double> gridh(data->gridh), initf;
        df::createSphericalIsotropicDF(
            *components[comp].initDensity,
            potential::Sphericalized<potential::BasePotential>(*data->currPot),
            /*input/output*/ gridh, /*output*/ initf);
        initDF[comp].reset(new math::LogLogSpline(gridh, initf));  // interpolated f(h)

        // if no input grid was provided, take the min/max values of h from the grid returned
        // by the Eddington routine (in case of several components, take the most extreme values)
        if(!fixmin)
            hmin = std::min(hmin, gridh.front());
        if(!fixmax)
            hmax = std::max(hmax, gridh.back());
    }

    // create a new grid, uniform in log(h)
    FILTERMSG(utils::VL_DEBUG, "FokkerPlanckSolver", "Grid in h=[" + 
        utils::toString(hmin) + ":" + utils::toString(hmax) + "], " + utils::toString(gridSize) + " nodes");
    data->gridh = math::createExpGrid(gridSize, hmin, hmax);

    // construct the appropriate implementation of the solver
    switch(params.method) {
        case FP_CHANGCOOPER: impl.reset(new FokkerPlanckImplChangCooper(data->gridh)); break;
        case FP_FEM1: impl.reset(new FokkerPlanckImplFEM<1>(data->gridh)); break;
        case FP_FEM2: impl.reset(new FokkerPlanckImplFEM<2>(data->gridh)); break;
        case FP_FEM3: impl.reset(new FokkerPlanckImplFEM<3>(data->gridh)); break;
        default: throw std::runtime_error("FokkerPlanckSolver: invalid choice of method");
    }

    // initialize the implementation-dependent representation of DF for each component
    // from the DF constructed by the Eddington routine
    for(unsigned int comp=0; comp<data->numComp; comp++) {
        data->gridf[comp] = math::solveBand(impl->weightMatrix(), impl->projVector(*initDF[comp]));
        if(data->absorbingBoundaryCondition)
            data->gridf[comp][0] = 0.;   // if using an absorbing boundary at hmin, set f(hmin) to zero
    }

    // allocate and assign various auxiliary arrays
    data->gridAdvDifCoefs = impl->getGridForCoefs();
    data->gridMass   = impl->projVector(FncMass());
    data->gridEnergy = impl->projVector(FncEnergy(*data->phasevol));

    // recompute the initial potential from the DF (as opposed to the initial density profile)
    if(params.updatePotential)
        reinitPotential(0.);
    // compute the initial advection/diffusion coefficients and 
    // assign the values of mass and energy from the combined DF of all components
    reinitAdvDifCoefs();
}

double FokkerPlanckSolver::Phi0()                        const { return data->Phi0; }
double FokkerPlanckSolver::Etot()                        const { return data->Etot; }
double FokkerPlanckSolver::Ekin()                        const { return data->Ekin; }
double FokkerPlanckSolver::Mbh ()                        const { return data->Mbh; }
double FokkerPlanckSolver::sourceEnergy()                const { return data->sourceEnergy; }
double FokkerPlanckSolver::drainEnergy()                 const { return data->drainEnergy; }
double FokkerPlanckSolver::Mass      (unsigned int comp) const { return data->Mass.at(comp); }
double FokkerPlanckSolver::sourceMass(unsigned int comp) const { return data->sourceMass.at(comp); }
double FokkerPlanckSolver::drainMass (unsigned int comp) const { return data->drainMass.at(comp); }
unsigned int FokkerPlanckSolver::numComp()               const { return data->numComp; }
std::vector<double> FokkerPlanckSolver::gridh()          const { return data->gridh; }
std::vector<double> FokkerPlanckSolver::drainTime(unsigned int comp) const {
    return data->drainTime.at(comp); }
math::PtrFunction   FokkerPlanckSolver::df(unsigned int comp) const {
    return impl->getInterpolatedFunction(data->gridf.at(comp)); }
math::PtrFunction FokkerPlanckSolver::potential() const {
    return math::PtrFunction(new potential::Sphericalized<potential::BasePotential>(*data->currPot)); }
potential::PtrPhaseVolume FokkerPlanckSolver::phaseVolume() const { return data->phasevol; }

void FokkerPlanckSolver::setMbh(double Mbh)
{
    if(data->Mbh == Mbh) return;
    data->Mbh = data->prevMbh = Mbh;
    // adiabatically modify the stellar density in response to the changed central black hole mass,
    // while keeping the DF fixed
    for(int i=0; i<10; i++)
        reinitPotential(0.);
}

double FokkerPlanckSolver::relaxationTime() const
{
    double maxf = 0;
    for(unsigned int c=0; c<data->numComp; c++)
        maxf += maxElement(data->gridf[c]) * data->Mstar[c];
    // the numerical factor corresponds to the standard definition of relaxation time
    // T = 0.34 sigma^3 / (rho m lnLambda)
    // if the DF is close to isothermal, in which case  f = rho / (sqrt(2pi) sigma)^3
    return 0.34 * pow(2*M_PI, -1.5) / maxf;
}


void FokkerPlanckSolver::reinitPotential(double deltat)
{
    // recompute the density profile of the model only if it contributes to the total potential
    if(data->selfGravity) {

        // construct the combined DF of all components
        math::PtrFunction df = impl->getInterpolatedFunction(sumVectors(data->gridf));

        // create an auxiliary grid in h and convert it to a grid in radius
        unsigned int gridRsize = std::min<unsigned int>(100, data->gridh.size()/2);
        std::vector<double> gridr = math::createExpGrid(gridRsize,
            R_max(*data->currPot, data->phasevol->E(data->gridh.front())),
            R_max(*data->currPot, data->phasevol->E(data->gridh.back())));

        // extrapolate the potential and phasevol mapping at the end of the timestep
        potential::PtrPotential nextPot;
        potential::PtrPhaseVolume nextPhasevol;
        if(!data->prevPot || deltat>=data->prevdeltat*2) {
            // do not extrapolate, use the current ones
            nextPot = data->currPot;
            nextPhasevol = data->phasevol;
        } else {
            // extrapolate linearly from the current potential (at the beginning of the timestep)
            // and the one at the previous timestep
            std::vector< std::vector<double> > Phi(1, std::vector<double>(gridRsize)), dPhi(Phi);
            for(unsigned int i=0; i<gridRsize; i++) {
                double currPhi, prevPhi;
                coord::GradCyl currGrad, prevGrad;
                data->prevPot->eval(coord::PosCyl(gridr[i],0,0), &prevPhi, &prevGrad);
                data->currPot->eval(coord::PosCyl(gridr[i],0,0), &currPhi, &currGrad);
                Phi [0][i] = currPhi + (currPhi-prevPhi) / data->prevdeltat * deltat;
                dPhi[0][i] = currGrad.dR + (currGrad.dR-prevGrad.dR) / data->prevdeltat * deltat;
            }
            nextPot.reset(new potential::Multipole(gridr, Phi, dPhi));
            // set up the phase volume mapping for the extrapolated potential
            nextPhasevol.reset(new potential::PhaseVolume(
                potential::Sphericalized<potential::BasePotential>(*nextPot)));
        }

        // convert the grid in radius into the grid in energy
        std::vector<double> gridPhi(gridRsize);
        for(unsigned int i=0; i<gridRsize; i++)
            nextPot->eval(coord::PosCyl(gridr[i],0,0), &gridPhi[i]);

        // compute the density profile by integrating the DF over velocity at each point of the energy grid
        std::vector<double> nextRho = computeDensity(*df, *nextPhasevol, gridPhi);

        // construct the density interpolator from the values computed on the radial grid
        math::LogLogSpline nextDensity(gridr, nextRho);

        // recompute the total potential by solving the Poisson equation
        data->prevPot = data->currPot;
        data->currPot = computePotential(data->Mbh, &nextDensity, gridr.front(), gridr.back(),
            /*diagnostic output*/ data->Phi0);
    } else {
        // no self-gravity, but may need to update potential as the black hole mass changes
        data->currPot = computePotential(data->Mbh, /*no stellar density*/ NULL,
            /*rmin-autodetect*/ 0, /*rmax*/ 0, /*diagnostic output, ignored*/ data->Phi0);
    }
    // reinit the mapping between energy and phase volume
    data->phasevol.reset(new potential::PhaseVolume(
        potential::Sphericalized<potential::BasePotential>(*data->currPot)));
    // recompute the energy associated with each basis function
    data->gridEnergy = impl->projVector(FncEnergy(*data->phasevol));
}

void FokkerPlanckSolver::reinitAdvDifCoefs()
{
    // recompute the angular-momentum draining rate once in a while only,
    // because this is a rather expensive operation, and being slightly off in estimating
    // the angular momentum diffusion is not a big deal
    bool computeDrainMatrix = data->absorbingBoundaryCondition && data->numSteps%16 == 0;
    const unsigned int
    numComp   = data->numComp,           // number of DF components
    gridSize  = data->gridh.size(),      // size of the grid in phase volume that defines the DF
    numPoints = data->gridAdvDifCoefs.size();   // size of the auxiliary grid for adv/dif coefs

    // total advection and diffusion coefs (sum over all species) at the nodes of the auxiliary grid
    data->gridAdv.assign(numPoints, 0.);
    data->gridDif.assign(numPoints, 0.);

    // prefactor for GW energy loss (for a circular orbit)
    double GWterm = data->Mbh>0 && data->speedOfLight>0 ?
        307.2 / pow_2(data->Mbh) / pow(data->speedOfLight, 5) : 0;

    // assemble two kinds of the composite DF:
    // f -- sum of DFs of all components
    // F -- sum of DFs multiplied by stellar mass of each component;
    // the former determines the advection, and the latter -- the diffusion coefficients.
    std::vector<double> gridf(data->gridf[0].size()), gridF(gridf.size());
    for(unsigned int comp = 0; comp < numComp; comp++) {
        math::blas_daxpy(1., data->gridf[comp], gridf);
        math::blas_daxpy(data->Mstar[comp], data->gridf[comp], gridF);
        data->Mass[comp] = math::blas_ddot(data->gridf[comp], data->gridMass);
    }
    math::PtrFunction f = impl->getInterpolatedFunction(gridf);
    math::PtrFunction F = impl->getInterpolatedFunction(gridF);

    // construct the integrals I0, Kg, Kh from the two composite DFs
    const SphericalIsotropicModel model(*data->phasevol, *f, *F, data->gridh);

    // store diagnostic quantities
    data->Etot = model.totalEnergy;
    data->Ekin = model.totalEkin;

    // constant multiplicative factor for advection/diffusion coefs
    const double GAMMA = 16*M_PI*M_PI * data->coulombLog;

    // compute the advection and diffusion coefficients
    // at the points of grid where these coefs are needed
    // (not necessarily the same grid that defines f)
    for(unsigned int p=0; p<numPoints; p++) {
        double h  = data->gridAdvDifCoefs[p], g;
        double I0 = model.I0(h);
        double Kg = model.Kg(h);
        double Kh = model.Kh(h);
        double E  = data->phasevol->E(h, &g);

        // advection coefficient D_h  without the pre-factor m_star
        data->gridAdv[p] += GAMMA * Kg;

        // diffusion coefficient D_hh
        data->gridDif[p] += GAMMA * g * (Kh + h * I0);

        // if the energy loss due to GW emission is considered, add the following term
        // to the advection coefficient (the loss is also proportional to the mass of a star)
        data->gridAdv[p] += GWterm * pow_2(E*E) * h;
    }

    // compute the energy conduction flux through the innermost boundary
    // (mass advection flux and associated energy flow will be computed later in the evolve() method)
    double h0 = data->gridh[0];
    data->drainRateEnergy =
        GAMMA * (-model.Kg(h0) * model.I0(h0) + (model.Kh(h0) + h0 * model.I0(h0)) * f->value(h0));

    // convert the sum of total energies of all stars into the total energy of the entire system
    data->Etot = 0.5 * (data->Etot + (data->Mbh!=0. ? data->Mbh * data->Phi0 : 0.) + data->Ekin);

    // if needed, compute the angular-momentum diffusion coefficient on a different grid in h
    // and initialize the loss-cone draining term in the matrix equation
    if(computeDrainMatrix && data->lossConeDrain) {
        // total angular-momentum diffusion coef
        std::vector<double> gridLC(gridSize);
        for(unsigned int i=0; i<gridSize; i++) {
            gridLC[i] = data->coulombLog * galaxymodel::difCoefLosscone(
                model, *data->currPot, data->phasevol->E(data->gridh[i]));
        }

        // construct interpolator for the angular-momentum diffusion coef
        math::LogLogSpline interpLC(data->gridh, gridLC);

        // compute the matrix elements for the draining rate, separately for each species
        for(unsigned int comp=0; comp<numComp; comp++) {
            // update the capture radii as the black hole mass changes
            data->captureRadius[comp] *=
                std::pow(data->Mbh / data->prevMbh, data->captureRadiusScalingExp[comp]);
            // the function describing instantaneous relative loss rate as a function of h
            FncDrainRate drainFnc(
                *data->currPot, *data->phasevol, 2 * data->Mbh * data->captureRadius[comp], interpLC);
            // store the drain time (for output purposes only)
            data->drainTime[comp].resize(gridSize);
            for(unsigned int i=0; i<gridSize; i++) {
                data->drainTime[comp][i] = -1 / drainFnc(data->gridh[i]);
                //GWterm * 8./3 * pow_2(pow_2(data->phasevol->E(data->gridh[i]))) * data->Mstar[comp]);
            }
            // compute the drain matrix - that's how this function is used in the computation itself
            math::BandMatrix<double> mat = impl->projMatrix(drainFnc);
            // if the loss cone occupies the entire range of angular momenta at a given energy,
            // the draining rate is -INFINITY, which is intended to make the DF instantly zero;
            // however, for this to work properly, we need to keep infinities on the diagonal,
            // but set all off-diagonal matrix elements to zero to avoid NANs in the solution.
            for(unsigned int r=0; r<mat.rows(); r++) {
                if(mat(r, r) == -INFINITY) {
                    for(unsigned int c = 1; c <= mat.bandwidth(); c++) {
                        if(r>=c)
                            mat(r, r-c) = 0.;
                        if(r+c<mat.rows())
                            mat(r, r+c) = 0.;
                    }
                }
            }
            data->drainMatrix[comp] = mat;
        }
        data->prevMbh = data->Mbh;
    }

    // initialize the source term in the matrix equation
    data->sourceRateEnergy = 0.;
    for(unsigned int comp=0; comp<numComp; comp++) {
        if(data->sourceRate[comp] == 0)
            continue;
        // translate the radius of the source term to phase volume
        double src_h = data->phasevol->value(
            data->currPot->value(coord::PosCyl(data->sourceRadius[comp], 0, 0)));
        // assign the source rate at grid nodes in h
        data->gridSourceRate[comp] = impl->projVector(LogNormal(src_h, SOURCE_WIDTH));
        // multiply by the actual rate
        math::blas_dmul(data->sourceRate[comp], data->gridSourceRate[comp]);
        // according to the boundary conditions, source rate must be zero at endpoints
        data->gridSourceRate[comp].front() = data->gridSourceRate[comp].back() = 0.;
        // keep track of the energy production rate due to the source term
        data->sourceRateEnergy += math::blas_ddot(data->gridEnergy,
            math::solveBand(impl->weightMatrix(), data->gridSourceRate[comp]));
    }
}

double FokkerPlanckSolver::evolve(double deltat)
{
    // use energy correction if the ratio of the current to the previous timesteps is not too extreme
    bool useCorrection  = deltat < data->prevdeltat*2;
    double accretedMass = 0;   // keep track of the change in Mbh

    // evolve the DF of each component
    double maxdeltaf = 0.;
    for(unsigned int comp = 0; comp < data->numComp; comp++) {
        // prepare the advection and diffusion coefficients for the current component
        std::vector<double> compAdv = data->gridAdv, compDif = data->gridDif;
        math::blas_dmul(data->Mstar[comp], compAdv);

        // weight matrix M and relaxation matrix R
        const math::BandMatrix<double> weightMatrix = impl->weightMatrix();
        const math::BandMatrix<double> relaxationMatrix = impl->relaxationMatrix(compAdv, compDif);
        const unsigned int bandwidth = weightMatrix.bandwidth();  // bandwidth of band matrices
        const unsigned int dim = data->gridf[comp].size();  // dimension of the linear system
        assert(relaxationMatrix.rows() == dim && weightMatrix.rows() == dim);

        // assemble the matrix equation  L f_new = R f_old + dt S,
        // where the lhs matrix L = weightMatrix - dt * (relaxationMatrix + drainMatrix),
        // and   the rhs matrix R = weightMatrix + dt * deltaRel
        // (the latter is the energy correction term), and S is the source matrix
        math::BandMatrix<double> lhsMatrix = weightMatrix;
        math::blas_daxpy(-deltat, relaxationMatrix, lhsMatrix);

        std::vector<double> rhs(dim);    // the r.h.s. of the above equation
        math::blas_dgemv(math::CblasNoTrans, 1., weightMatrix, data->gridf[comp], 0., rhs);

        // energy correction term
        if(useCorrection) {
            // estimate d relMatrix / d t = (relMatrix - prevRel) / prevdt,
            // and then extrapolate forward in time to estimate  deltaRel = (newRel - relMatrix) * deltat
            math::BandMatrix<double> deltaRel = data->prevRelaxationMatrix[comp];
            math::blas_daxpy(-1., relaxationMatrix, deltaRel);   // deltaRel := prevRel-relMatrix
            for(unsigned int b = 0; b <= deltaRel.bandwidth(); b++)
                deltaRel(0, b) = 0.;   // zero out the first row which sometimes leads to instability
            // add the correction term  "dt * deltaRel * f_old"  to the rhs
            math::blas_dgemv(math::CblasNoTrans, -pow_2(deltat) / data->prevdeltat, deltaRel,
                data->gridf[comp], 1., rhs);
        }
        data->prevRelaxationMatrix[comp] = relaxationMatrix;

        // sink term in the lhs:  lhsMatrix -= deltat * drainMatrix
        if(data->absorbingBoundaryCondition)
            math::blas_daxpy(-deltat, data->drainMatrix[comp], lhsMatrix);

        // source term in the rhs
        if(data->sourceRate[comp]>0.) {
            math::blas_daxpy(deltat, data->gridSourceRate[comp], rhs);
            // record the amount of created mass for this component
            data->sourceMass[comp] += deltat * sumElements(data->gridSourceRate[comp]);
        }

        // boundary conditions: zero-flux (Neumann) b/c does not need anything special,
        // while a constant-value (Dirichlet) b/c essentially eliminates the first/last row
        // of the matrix equation, or, rather, makes it trivial
        for(unsigned int b = 0; b <= bandwidth; b++) {
            lhsMatrix(dim-1, dim-1-b) = b==0 ? 1. : 0.;
            rhs[dim-1] = data->gridf[comp][dim-1];
            if(data->absorbingBoundaryCondition) {
                lhsMatrix(0, b) = b==0 ? 1. : 0.;
                rhs[0] = data->gridf[comp][0];
            }
        }

        // solve the matrix equation  L f_new = R f_old + dt S
        // newf := L^{-1} rhs
        std::vector<double> newf = math::solveBand(lhsMatrix, rhs);

        // keep track of the maximum relative change of DF over all grid points,
        // excluding those inside the loss cone (those will always be set to near-zero)
        for(unsigned int i=0; i<dim; i++) {
            if(data->gridf[comp][i] > 0. && newf[i] > 0.)
                maxdeltaf = fmax(maxdeltaf, fabs(log(newf[i] / data->gridf[comp][i])));
        }

        // reconstruct the mass flux through the boundary (in case of Dirichlet b/c) and
        if(data->absorbingBoundaryCondition) {
            double lostMass = 0.;
            // use the first row of the matrix equation that we have previously replaced with (1 0 0 ...)
            // now we take back the original matrices and compute the flux by summing up
            // M_{0j} (fnew_j - fold_j) - R_{0j} fnew_j dt.
            // note that this won't work if the absorbing boundary was effectively not at 0th element
            for(unsigned int b=1; b<=bandwidth; b++) {
                lostMass += (newf[b] - data->gridf[comp][b]) * weightMatrix(0, b) - 
                    newf[b] * relaxationMatrix(0, b) * deltat;
            }
            data->drainMass[comp] += lostMass;
            data->drainEnergy     += lostMass * data->phasevol->E(data->gridh[0]);
            accretedMass          -= lostMass * data->captureMassFraction[comp];
        }

        // keep track of mass and energy removed from the system through the loss cone
        if(data->absorbingBoundaryCondition) {
            // compute the change of DF resulting from the loss-cone term alone:
            // solve the same matrix equation L f_{new,LC} = R f_old, 
            // but with R = weightMatrix, L = weightMatrix - deltat * drainMatrix
            lhsMatrix = weightMatrix;
            math::blas_daxpy(-deltat, data->drainMatrix[comp], lhsMatrix);
            math::blas_dgemv(math::CblasNoTrans, 1., weightMatrix, data->gridf[comp], 0., rhs);
            std::vector<double> newfLC = math::solveBand(lhsMatrix, rhs);
            // compute deltaf:  f_{new,LC} -= f_old
            math::blas_daxpy(-1., data->gridf[comp], newfLC);
            // compute the change in total mass of this component
            double lostMass        = math::blas_ddot(newfLC, data->gridMass);
            data->drainMass[comp] += lostMass;
            data->drainEnergy     += math::blas_ddot(newfLC, data->gridEnergy);
            // a fraction of this mass will be contributed to the black hole mass
            accretedMass          -= lostMass * data->captureMassFraction[comp];
        }

        // overwrite the array of DF values at grid nodes with the new ones
        data->gridf[comp] = newf;
    }

    // keep track of the energy added to the system through the source term
    data->sourceEnergy += deltat * data->sourceRateEnergy;
    // same for the energy lost through the conduction across the inner boundary
    data->drainEnergy  += deltat * data->drainRateEnergy;
    if(data->Mbh) {
        // increase the black hole mass
        data->Mbh         += accretedMass;
        // and add a corresponding correction term to the energy lost from the system
        data->drainEnergy += accretedMass * data->Phi0;
    }

    // recompute the potential (if necessary) and the adv/dif coefs
    if(data->updatePotential)
        reinitPotential(deltat);

    reinitAdvDifCoefs();

    data->prevdeltat = deltat;
    data->numSteps++;

    return maxdeltaf;
}

}
