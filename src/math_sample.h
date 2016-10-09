/** \file    math_sample.h
    \brief   sampling points from a probability distribution
    \date    2013-2015
    \author  Eugene Vasiliev
*/
#pragma once
#include "math_base.h"
#include "math_linalg.h"

namespace math{

/** Sample points from an N-dimensional probability distribution function F.
    F should be non-negative in the given region, and the integral of F over this region should exist;
    still better is if F is bounded from above everywhere in the region.
    The output consists of M sampling points from the given region, such that the density
    of points in the neighborhood of any location X is proportional to the value of F(X).

    \param[in]  F  is the probability distribution, the dimensionality N of the problem
                is given by F.numVars();
    \param[in]  xlower  is the lower boundary of sampling volume (array of length N);
    \param[in]  xupper  is the upper boundary of sampling volume;
    \param[in]  numSamples  is the required number of sampling points (M);
    \param[out] samples  will be filled by samples, i.e. contain the matrix of M rows and N columns;
    \param[out] numTrialPoints (optional) if not NULL, will store the actual number of function
                evaluations (so that the efficiency of sampling is estimated as the ratio
                numSamples/numTrialPoints);
    \param[out] integral (optional) if not NULL, will store the Monte Carlo estimate of the integral
                of F over the given region (this could be compared with the exact value, if known,
                to estimate the bias/error in sampling scheme);
    \param[out] interror (optional) if not NULL, will store the error estimate of the integral;
 */
void sampleNdim(const IFunctionNdim& F, const double xlower[], const double xupper[],
    const unsigned int numSamples,
    Matrix<double>& samples, int* numTrialPoints=NULL, double* integral=NULL, double* interror=NULL);

}  // namespace