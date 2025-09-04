/** \file    math_sample.h
    \brief   sampling points from a probability distribution
    \date    2013-2025
    \author  Eugene Vasiliev
*/
#pragma once
#include "math_base.h"
#include "math_linalg.h"
#include "math_random.h"

namespace math{

/** Bit-field defining possible modes of operation for sampleNdim:
    0: return equal-weight samples, using adaptive refinement for the N-dim grid (default)
    1: return all collected samples, using adaptive refinement
    2: not allowed
    3: return all samples without any adaptive refinement
    4: same as 0, but use PRNG instead of QRNG
*/
enum SampleMethod {
    SM_RETURN_EQUAL_WEIGHT_SAMPLES = 0,  ///< return equally-weighted subset of all samples
    SM_RETURN_ALL_SAMPLES = 1,  ///< return all collected samples
    SM_DISABLE_REFINEMENT = 2,  ///< disable adaptive refinement of the N-dim grid (not used by itself)
    SM_RETURN_ALL_SAMPLES_DISABLE_REFINEMENT = SM_RETURN_ALL_SAMPLES | SM_DISABLE_REFINEMENT,
    ///< return all samples without performing any refinement, just sampling uniformly within root cell
    SM_USE_PRNG = 4,            ///< use pseudo-random instead of quasi-random number generators
    SM_DEFAULT = SM_RETURN_EQUAL_WEIGHT_SAMPLES
};

/** Sample points from the probability distribution function F in a N-dimensional hypercube.
    The routine uses the rejection sampling approach, constructing a piecewise-constant
    "envelope function" G, such that its value within each cell of an N-dimensional binary tree
    partitioning the entire hypercube is at least as high as the maximum value of F inside this cell.
    The number of internally collected sampling points in each cell is proportional to G,
    so that the weight of each sample does not exceed a maximum value w_max = I[F] / M,
    where I[F] is the integral of the function over the hypercube (which is estimated as a byproduct
    of the procedure) and M is the requested number of output samples.

    The routine has several different modes of operation, depending on the "method" bitfield.
    The general workflow consists of the following steps:
    1. Sample a certain initial number of points uniformly from the entire hypercube.
    This number is rounded down from numSamples to the nearest power of two, but with a lower limit
    equal to twice the minimum number of points in a subregion (a constant depending only on N).
    Compute the integral I[F] as the average of all function values times the volume of the hypercube.
    At this point, the tree structure representing the entire region consists of a single root cell.
    2. Unless the SM_DISABLE_REFINEMENT bit is set, perform one or more iterations of refinement:
    2a. For each cell in the tree, find the maximum value of F among all points belonging to this cell.
    2b. If the weight of any point in a cell exceeds w_max = I[F] / M, where I[F] is the current
    estimate of the integral over the entire hypercube, then this cell needs more points. Of course,
    adding them uniformly inside the whole cell would be a waste of resources, so we keep splitting
    the cell along a suitable dimension into two equal halves, until the number of points drops
    to a pre-determined minimum. Then this cell is scheduled for refinement, i.e. adding more points.
    2c. After the splitting procedure is finished and all cells that need refinement are allocated
    additional points, we collect the function values at the newly sampled points and recompute
    the total integral I[F]. Then the procedure is repeated from step 2a until the maximum weight
    of any point is below w_max.
    3. Output either all K collected samples (if SM_RETURN_ALL_SAMPLES bit is set) or a subset of
    M equally-weighted samples, selecting points in proportion to their weight.

    The coordinates of sampled points are assigned using either a pseudo-random number generator
    (if SM_USE_PRNG bit is set) or a quasi-random (low-discrepancy) Sobol sequence.
    The latter is the default mode of operation, since it greatly increases the accuracy
    of computing the integral I[F] (the error decreases roughly as M^-1 for smooth integrands,
    instead of M^-0.5 as in the case of PRNG).
    If the samples are subsequently used to estimate integrals I[Q] of some other derived quantities
    Q (e.g. F multiplied by some function of coordinates), then one needs to retrieve all collected
    samples, rather than just a subset of them, to retain the favourable M^-1 error scaling.
    Method=3 is a combination of disabling the refinement and returning all samples;
    in this case the number of collected samples is exactly M, but the weights of individual points
    may be considerably higher than w_max. This method is provided only because of a peculiar
    property of the Sobol sequence: when M is a power of two, the errors in I[F] and any derived
    quantity I[Q] may have an even better scaling as M^-3/2 (only for smooth integrands);
    however, for other values of M the errors in I[F] and I[Q] are higher than when using refinement
    (even when compared at the same number K of actually collected samples, which usually exceeds M
    by a factor of few), so this method has a rather niche applicability.
    When the SM_RETURN_ALL_SAMPLES bit is unset (method=0 or 4), the weights are set to I[F] / M
    for all returned samples, which are selected from all internally collected samples with
    probability proportional to their actual weights. In this case the density of returned points
    in the neighborhood of any location X is proportional to the value of F(X).
    In the opposite case (method=1 or 3), the weights returned by this routine are in general
    unequal, and the density of points is proportional to the envelope function G, i.e. is constant
    within each cell and changes by some power of two between cells.
    In all cases, the sum of all weights is equal to the estimate of the integral I[F],
    and the integrals of any derived quantities Q(x) can be estimated as the weighted sum
    I[Q] = \sum_i=1^{npoints} w_i Q(x_i) / F(x_i);
    in other words, the returned points and their weights provide the basis for importance sampling.

    The table below summarizes the differences between methods.

    method   #internal   #output   weights               RNG    error in I   error in Q
    0        K>=M           M        w_max, all equal    QRNG   ~M^-1        ~M^-1/2
    1        K>=M        K>=M      <=w_max, unequal      same    same        ~M^-1
    2   not allowed (cannot ensure that weights are all below w_max when not using refinement)
    3         K=M         K=M      can be anything       same   ~M^-3/2 if M=2^k, otherwise M^-1
    4        K>=M           M        w_max, all equal    PRNG   ~M^-1/2      ~M^-1/2
    >4  not allowed (when using PRNG, errors will not decrease faster than M^-1/2 anyway)

    \param[in]  F  is the probability distribution, the dimensionality N of the problem is given by
    F.numVars().  F must be non-negative in the hypercube, and the integral of F over this hypercube
    should exist; still better is if F is bounded from above everywhere in the hypercube.
    \param[in]  xlower  is the lower boundary of sampling volume (array of length N).
    \param[in]  xupper  is the upper boundary of sampling volume.
    \param[in]  numSamples  is the requested number of sampling points (M).
    The actual number of internally collected samples K is in general larger than M (except method=3).
    It is far more efficient to draw many samples in a single call rather than repeatedly sample
    the same function, because a significant amount of effort is spent on exploring the hypercube.
    \param[in]  method  controls the choice of sampling method and outputs, as described above.
    \param[out] samples  will be filled by coordinates of sampled points, i.e. contain the matrix
    with M rows (except method=1, when it has K rows) and N columns.
    \param[out] weights  witll be filled with corresponding weights of points, same length as samples.
    \param[out] relError  (optional) if not NULL, will store the error estimate of the integral I of
    the input function F over the entire hypercube; the value of this integral is the sum of weights.
    \param[out] numEval  (optional) if not NULL, will store the actual number of collected samples K.
    \param[in,out] state  (optional) is the seed for the pseudo-random number generator:
    even when the routine uses QRNG for assigning coordinates, the QRNG is initialized with a random
    scrambling of bits, so that when the routine is invoked with different seeds, it produces
    independent realizations of samples. If state is not provided (NULL), use the global state.
    \throw std::invalid_argument if F.numValues()>1, or N is too large, or xupper <= xlower,
    or method is invalid; std::runtime_error if F<0, or the refinement procedure did not converge,
    or in case of keyboard interrupt.
    \note OpenMP-parallelized loop over blocks of points, the provided function F is called from
    multiple threads with one block of 1024 points in each call.
*/
void sampleNdim(
    const IFunctionNdim& F, const double xlower[], const double xupper[], const size_t numSamples,
    const SampleMethod method,
    /*output*/ Matrix<double>& samples, std::vector<double>& weights,
    /*optional output*/ double* relError=NULL, size_t* numEval=NULL,
    /*optional input/output*/ PRNGState* state=NULL);

}  // namespace