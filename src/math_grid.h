/** \file    math_grids.h
    \brief   utilities for manipulating 1d grids
    \author  Eugene Vasiliev
    \date    2011-2015

*/
#pragma once
#include <vector>

namespace mathutils {

/** generates a grid with exponentially growing spacing.
    x[k] = (exp(Z k) - 1)/(exp(Z) - 1),
    and the value of Z is computed so the the 1st element is at xmin and last at xmax.
    \param[in]  nnodes -- total number of grid points
    \param[in]  xmin, xmax -- location of the first and the last node
    \param[in]  zeroelem -- if true, 0th node is at zero (otherwise at xmin)
    \param[out] grid -- array of grid nodes created by this routine
*/
void createNonuniformGrid(size_t nnodes, double xmin, double xmax, bool zeroelem, std::vector<double>& grid);

/** creates an almost uniform grid so that each bin contains at least minbin points from input array.
    input points are in srcpoints array and MUST BE SORTED in ascending order (assumed but not cheched).
    \param[out] grid  is the array of grid nodes which will have length at most gridsize. 
    NB: in the present implementation, the algorithm is not very robust and works well only for gridsize*minbin << srcpoints.size,
    assuming that 'problematic' bins only are found close to endpoints but not in the middle of the grid.
*/
void createAlmostUniformGrid(const std::vector<double> &srcpoints, size_t minbin, size_t& gridsize, std::vector<double>& grid);

}  // namespace
