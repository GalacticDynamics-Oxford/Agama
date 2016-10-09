/** \file   math_simple_cubature.h
    \brief  fixed-order rules for 2d and 3d integration
    \date   Aug 2016
    \author Eugene Vasiliev

    Computation of N-dimensional integrals using a sparse grid approach (Smolyak 1963),
    with nodes determined by the 1d nested Gauss-Patterson rule.
    This file contains only the data for the case of 2d and 3d quadrature rules
    with 7-point 1d grids, which is exact for polynomials up to a total degree of 9;
    the numbers are taken from Heiss&Winschel 2008, but they are apparently computed
    in single precision, thus the accuracy is ~1e-7 at best.
    For generic smooth integrands, relative error is of order 1e-3.
*/
#pragma once
#include "math_base.h"

namespace math{

/// nodes of Gauss-Patterson quadrature in one dimension
static const double GPnodes[7] = {.0197543656459898, .1127016653792583, .2828781253265987,
    .5, .7171218746734013, .8872983346207417, .9802456343540101};

/// indices of 1d nodes in each dimension for a 2d sparse grid
static const int NumNodes2d = 33;
static const int GPindex2d[NumNodes2d][2] = {
{0, 1}, {0, 3}, {0, 5}, {1, 0}, {1, 1}, {1, 2}, {1, 3}, {1, 4},
{1, 5}, {1, 6}, {2, 1}, {2, 3}, {2, 5}, {3, 0}, {3, 1}, {3, 2},
{3, 3}, {3, 4}, {3, 5}, {3, 6}, {4, 1}, {4, 3}, {4, 5}, {5, 0},
{5, 1}, {5, 2}, {5, 3}, {5, 4}, {5, 5}, {5, 6}, {6, 1}, {6, 3}, {6, 5} };

/// weights of nodes in a 2d sparse grid
static const double GPweight2d[NumNodes2d] = {
 .0145355859497385970, .023256933333333330, .0145355859497385970, .014535585949738597,
-.0025804927484386624, .055749648923824897,-.0011654708258483639, .055749648923824897,
-.0025804927484386624, .014535585949738597, .0557496489238248970, .089199422222222227,
 .0557496489238248970, .023256933333333337,-.0011654708258483084, .089199422222222227,
 .0028765530864176829, .089199422222222227,-.0011654708258483084, .023256933333333337,
 .0557496489238248970, .089199422222222227, .0557496489238248970, .014535585949738597,
-.0025804927484386624, .055749648923824897,-.0011654708258483639, .055749648923824897,
-.0025804927484386624, .014535585949738597, .0145355859497385970, .023256933333333330,
 .014535585949738597};

/// indices of 1d nodes for a 3d sparse grid
static const int NumNodes3d = 87;
static const int GPindex3d[NumNodes3d][4] = {
{0, 1, 3}, {0, 3, 1}, {0, 3, 3}, {0, 3, 5}, {0, 5, 3}, {1, 0, 3},
{1, 1, 1}, {1, 1, 3}, {1, 1, 5}, {1, 2, 3}, {1, 3, 0}, {1, 3, 1},
{1, 3, 2}, {1, 3, 3}, {1, 3, 4}, {1, 3, 5}, {1, 3, 6}, {1, 4, 3},
{1, 5, 1}, {1, 5, 3}, {1, 5, 5}, {1, 6, 3}, {2, 1, 3}, {2, 3, 1},
{2, 3, 3}, {2, 3, 5}, {2, 5, 3}, {3, 0, 1}, {3, 0, 3}, {3, 0, 5},
{3, 1, 0}, {3, 1, 1}, {3, 1, 2}, {3, 1, 3}, {3, 1, 4}, {3, 1, 5},
{3, 1, 6}, {3, 2, 1}, {3, 2, 3}, {3, 2, 5}, {3, 3, 0}, {3, 3, 1},
{3, 3, 2}, {3, 3, 3}, {3, 3, 4}, {3, 3, 5}, {3, 3, 6}, {3, 4, 1},
{3, 4, 3}, {3, 4, 5}, {3, 5, 0}, {3, 5, 1}, {3, 5, 2}, {3, 5, 3},
{3, 5, 4}, {3, 5, 5}, {3, 5, 6}, {3, 6, 1}, {3, 6, 3}, {3, 6, 5},
{4, 1, 3}, {4, 3, 1}, {4, 3, 3}, {4, 3, 5}, {4, 5, 3}, {5, 0, 3},
{5, 1, 1}, {5, 1, 3}, {5, 1, 5}, {5, 2, 3}, {5, 3, 0}, {5, 3, 1},
{5, 3, 2}, {5, 3, 3}, {5, 3, 4}, {5, 3, 5}, {5, 3, 6}, {5, 4, 3},
{5, 5, 1}, {5, 5, 3}, {5, 5, 5}, {5, 6, 3}, {6, 1, 3}, {6, 3, 1},
{6, 3, 3}, {6, 3, 5}, {6, 5, 3} };

/// weights of nodes in a 3d sparse grid
static const double GPweight3d[NumNodes3d] = {
 .014535585949738588, .014535585949738588,-.005814238566143851, .014535585949738588,
 .014535585949738588, .014535585949738588, .021433475651577910,-.045447444051594475,
 .021433475651577910, .055749648923824863, .014535585949738588,-.045447444051594475,
 .055749648923824863,-.050841052469786228, .055749648923824863,-.045447444051594475,
 .014535585949738588, .055749648923824863, .021433475651577910,-.045447444051594475,
 .021433475651577910, .014535585949738588, .055749648923824863, .055749648923824863,
-.022299875625427557, .055749648923824863, .055749648923824863, .014535585949738588,
-.005814238566143851, .014535585949738588, .014535585949738588,-.045447444051594475,
 .055749648923824863,-.050841052469786215, .055749648923824863,-.045447444051594475,
 .014535585949738588, .055749648923824863,-.022299875625427557, .055749648923824863,
-.005814238566143844,-.050841052469786416,-.022299875625427557, .160786886409133220,
-.022299875625427557,-.050841052469786416,-.005814238566143844, .055749648923824863,
-.022299875625427557, .055749648923824863, .014535585949738588,-.045447444051594475,
 .055749648923824863,-.050841052469786215, .055749648923824863,-.045447444051594475,
 .014535585949738588, .014535585949738588,-.005814238566143851, .014535585949738588,
 .055749648923824863, .055749648923824863,-.022299875625427557, .055749648923824863,
 .055749648923824863, .014535585949738588, .021433475651577910,-.045447444051594475,
 .021433475651577910, .055749648923824863, .014535585949738588,-.045447444051594475,
 .055749648923824863,-.050841052469786228, .055749648923824863,-.045447444051594475,
 .014535585949738588, .055749648923824863, .021433475651577910,-.045447444051594475,
 .021433475651577910, .014535585949738588, .014535585949738588, .014535585949738588,
-.005814238566143851, .014535585949738588, .014535585949738588};

/** Integration of a 2d function using a fixed-order rule with 33 points.
    \param[in]  fnc  is a function of 2 variables that returns 1 value;
    \param[in]  xlower  is the lower corner of integration cube;
    \param[in]  xupper  is the upper corner;
    \return     the value of integral (no error estimate).
*/
static inline double integrate2d(const math::IFunctionNdim &fnc, const double xlower[2], const double xupper[2])
{
    if(fnc.numVars() != 2 || fnc.numValues() != 1)
        return NAN;  // should rather throw an exception...
    double x[2], sum = 0, val;
    for(int i=0; i<NumNodes2d; i++) {
        for(int d=0; d<2; d++)
            x[d] = xlower[d] + (xupper[d]-xlower[d]) * GPnodes[GPindex2d[i][d]];
        fnc.eval(x, &val);
        sum += val * GPweight2d[i];
    }
    return sum * (xupper[0]-xlower[0]) * (xupper[1]-xlower[1]);
}

/** Integration of a 3d function using a fixed-order rule with 87 points.
    \param[in]  fnc  is a function of 3 variables that returns 1 value;
    \param[in]  xlower  is the lower corner of integration cube;
    \param[in]  xupper  is the upper corner;
    \return     the value of integral (no error estimate).
*/
static inline double integrate3d(const math::IFunctionNdim &fnc, const double xlower[3], const double xupper[3])
{
    if(fnc.numVars() != 3 || fnc.numValues() != 1)
        return NAN;
    double x[3], sum = 0, val;
    for(int i=0; i<NumNodes3d; i++) {
        for(int d=0; d<3; d++)
            x[d] = xlower[d] + (xupper[d]-xlower[d]) * GPnodes[GPindex3d[i][d]];
        fnc.eval(x, &val);
        sum += val * GPweight3d[i];
    }
    return sum * (xupper[0]-xlower[0]) * (xupper[1]-xlower[1]) * (xupper[2]-xlower[2]);
}

} // namespace