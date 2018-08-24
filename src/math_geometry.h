/** \file    math_geometry.h
    \brief   Tools for working with polygons and 2d finite-element grids
    \date    2017
    \author  Eugene Vasiliev

*/
#pragma once
#include "math_spline.h"

namespace math{

/// point in 2d cartesian coordinates
struct Point2d {
    double x;
    double y;
    Point2d() {}
    Point2d(double _x, double _y) : x(_x), y(_y) {}
    inline bool operator== (const Point2d& other) const { return x==other.x && y==other.y; }
    inline bool operator!= (const Point2d& other) const { return x!=other.x || y!=other.y; }
};

/// rectangle aligned with coordinate axes
struct Rectangle {
    Point2d lower;  ///< lower left corner
    Point2d upper;  ///< upper right corner
    Rectangle() {}
    Rectangle(const Point2d& _lower, const Point2d& _upper) : lower(_lower), upper(_upper) {}
    Rectangle(double xl, double yl,  double xu, double yu)  : lower(xl, yl), upper(xu, yu) {}
};

/// arbitrary closed polygon (last point needs not coincide with the first one)
typedef std::vector<Point2d> Polygon;

/// Compute the area of the polygon (positive if it is oriented counterclockwise)
double polygonArea(const Polygon& polygon);

/// Test if the point is inside the polygon
/// (assuming that it is oriented counterclockwise, otherwise need to negate the returned result
bool isPointInsidePolygon(double x, double y, const Polygon& polygon);

/** Compute the intersection of a polygon with a rectangular region.
    \param[in]  polygon  is the input polygon (not necessarily convex, but without self-intersections);
    \param[in]  rect     is the clipping rectangle;
    \return     the polygon resulting from the intersection of the two figures (could be empty)
*/
Polygon clipPolygonByRectangle(const Polygon& polygon, const Rectangle& rect);

/** Compute the integrals of 2d tensor-product B-spline basis functions over the region
    enclosed by a polygon.
    \param[in]  polygon  is the input polygon (not necessarily convex, but without self-intersections);
    \param[in]  bsplx, bsply  are two 1d B-spline basis sets of degree N, which form the 2d basis;
    \tparam     N is the degree of B-splines;
    \param[out] output will contain the integrals of all basis functions (their number is
    `bsplx.numValues() * bsply.numValues()`, and this array must contain enough room for them;
    the indexing scheme is: `output[iy * Nx + ix] = \int B_{ix}(x) B_{iy}(y)`.
    \return  true if the polygon extends beyond the 2d grid, false if it lies entirely within the grid.
*/
template<int N>
bool computeBsplineIntegralsOverPolygon(const Polygon& polygon,
    const BsplineInterpolator1d<N>& bsplx, const BsplineInterpolator1d<N>& bsply, double output[]);


}  // namespace