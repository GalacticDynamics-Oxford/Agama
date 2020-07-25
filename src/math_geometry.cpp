#include "math_geometry.h"
#include "math_core.h"
#include <cmath>

namespace math{

namespace{  // internal

/// integrate the tensor product of two B-spline functions over a rectangle;
/// store the integrals in the output array using the indexing scheme
/// `output[j * nx + i] = Bsplx_i(x) * Bsply_j(y)`
template<int N>
inline void integrateOverRectangle(const Rectangle& rect,
    const BsplineInterpolator1d<N>& bsplx, const BsplineInterpolator1d<N>& bsply,
    double output[])
{
    // integration of a product of two polynomial functions each of degree N in its coordinate
    // can be performed exactly using just one point (center of the rectangle) if N<=1,
    // or 4 points if N<=3 (two-point Gaussian quadrature in each dimension)
    double area = (rect.upper.x - rect.lower.x) * (rect.upper.y - rect.lower.y) * (N<=1 ? 1. : 0.25);
    double weightxa[N+1], weightya[N+1], weightxb[N+1], weightyb[N+1];
    const double X1 = 0.2113248654051871, X2 = 1-X1;  // nodes of 2-point Gaussian quadrature rule
    double xa = N<=1 ? 0.5 * (rect.lower.x + rect.upper.x) :
        X1 *  rect.lower.x + X2 * rect.upper.x;
    double ya = N<=1 ? 0.5 * (rect.lower.y + rect.upper.y) :
        X1 *  rect.lower.y + X2 * rect.upper.y;
    unsigned int indx = bsplx.nonzeroComponents(xa, 0, weightxa);
    unsigned int indy = bsply.nonzeroComponents(ya, 0, weightya);
    if(N>1) {
        bsplx.nonzeroComponents(X1 * rect.upper.x + X2 * rect.lower.x, 0, weightxb);
        bsply.nonzeroComponents(X1 * rect.upper.y + X2 * rect.lower.y, 0, weightyb);
    }
    const size_t nx = bsplx.numValues();
    // use one or four points depending on the order of B-splines
    for(int p=0; p < (N<=1 ? 1 : 4); p++) {
        const double* weightx = p%2==0 ? weightxa : weightxb;
        const double* weighty = p/2==0 ? weightya : weightyb;
        for(int i=0; i<=N; i++)
            for(int j=0; j<=N; j++)
                output[(j + indy) * nx + i + indx] += area * weightx[i] * weighty[j];
    }
}

/// integrate the tensor product of B-splines over a triangle defined by its three corners p1,p2,p3,
/// which is supposed to lie inside a single grid segment in both directions
template<int N>
inline void integrateOverTriangle(const Point2d& p1, const Point2d& p2, const Point2d& p3,
    const BsplineInterpolator1d<N>& bsplx, const BsplineInterpolator1d<N>& bsply,
    double output[])
{
    double area = 0.5 * (p1.x * (p2.y-p3.y) + p2.x * (p3.y-p1.y) + p3.x * (p1.y-p2.y));
    if(area == 0)
        return;
    // quadrature rules over a standard triangle: coord1, coord2, weight
    static const double rule1[1 * 3] = { 1./3, 1./3, 1. };
    static const double rule3[3 * 3] = { 1./6, 1./6, 1./3,  1./6, 2./3, 1./3,  2./3, 1./6, 1./3 };
    static const double rule6[6 * 3] = {  // exact for polynomials up to degree 4
        0.091576213509771, 0.091576213509771, 0.1099517436553219,
        0.091576213509771, 0.816847572980459, 0.1099517436553219,
        0.816847572980459, 0.091576213509771, 0.1099517436553219,
        0.445948490915965, 0.445948490915965, 0.2233815896780115,
        0.108103018168070, 0.445948490915965, 0.2233815896780115,
        0.445948490915965, 0.108103018168070, 0.2233815896780115};
    const int npoints = N==0 ? 1 : N==1 ? 3 : 6;  // for N=3 the last rule is not exact, but still okay
    const double* rule = N==0 ? rule1 : N==1 ? rule3 : rule6;
    const size_t nx = bsplx.numValues();
    for(int p=0; p<npoints; p++) {
        double x = p1.x * rule[p*3] + p2.x * rule[p*3+1] + p3.x * (1-rule[p*3]-rule[p*3+1]);
        double y = p1.y * rule[p*3] + p2.y * rule[p*3+1] + p3.y * (1-rule[p*3]-rule[p*3+1]);
        double weightx[N+1], weighty[N+1];
        unsigned int indx = bsplx.nonzeroComponents(x, 0, weightx);
        unsigned int indy = bsply.nonzeroComponents(y, 0, weighty);
        for(int i=0; i<=N; i++)
            for(int j=0; j<=N; j++)
                output[(j + indy) * nx + i + indx] += area * rule[p*3+2] * weightx[i] * weighty[j];
    }
}

/// integrate a tensor product of B-splines over an arbitrary polygon lying inside a single grid segment
template<int N>
void integrateOverPolygon(const Polygon& polygon,
    const BsplineInterpolator1d<N>& bsplx, const BsplineInterpolator1d<N>& bsply,
    double output[])
{
    const size_t numVertices = polygon.size();
    // check if the polygon is a regular rectangle aligned with coordinate axes
    if(numVertices == 4) {
        if( polygon[0].x == polygon[3].x && polygon[0].y == polygon[1].y &&
            polygon[1].x == polygon[2].x && polygon[2].y == polygon[3].y )
        {
            integrateOverRectangle(Rectangle(polygon[0], polygon[2]), bsplx, bsply, output);
            return;
        }
        if( polygon[1].x == polygon[0].x && polygon[1].y == polygon[2].y &&
            polygon[2].x == polygon[3].x && polygon[3].y == polygon[0].y )
        {
            integrateOverRectangle(Rectangle(polygon[1], polygon[3]), bsplx, bsply, output);
            return;
        }
    }
    // otherwise split a generic polygon into triangles
    for(size_t v=2; v<numVertices; v++)
        integrateOverTriangle(polygon[0], polygon[v-1], polygon[v], bsplx, bsply, output);
}

/// check if a point is in the half-plane determined by the boundary of a rectangle with index dir
inline bool isInside(const Point2d& point, const Rectangle& rect, int dir)
{
    switch(dir) {
        case 0:  return point.y >= rect.lower.y;  // bottom boundary
        case 1:  return point.x <= rect.upper.x;  // right
        case 2:  return point.y <= rect.upper.y;  // top
        default: return point.x >= rect.lower.x;  // left
    }
}

}  // internal ns

Polygon clipPolygonByRectangle(const Polygon& polygon, const Rectangle& rect)
{
    // implement the Sutherland-Hodgman algorithm:
    // clip the source polygon by four boundaries of the rectangle, one by one.
    // in doing so, we create 3 temporary intermediate polygons and one final,
    // but may re-use only two Polygon objects for this task (each arrow indicates one clipping):
    //   source   tmp   result
    //    (0) --> (1)
    //            (1) --> (2)
    //            (3) <-- (2)
    //            (3) --> (4)
    Polygon tmp, result;
    for(int dir=0; dir<4; dir++) {
        // source and destination objects for this clipping direction
        const Polygon& src = dir==0 ? polygon : dir==2 ? result : tmp;
        Polygon& dest = dir==0 || dir==2 ? tmp : result;
        dest.clear();
        const int numVertices = src.size();
        if(numVertices<1)
            continue;
        bool prevInside = isInside(src.back(), rect, dir);
        for(int curr=0, prev=numVertices-1; curr < numVertices; prev = curr++) {
            bool currInside = isInside(src[curr], rect, dir);
            if(prevInside ^ currInside) {
                // the two vertices lie on opposite sides of the line;
                // add a new vertex at the intersection
                switch(dir) {
                    case 0:  dest.push_back(Point2d(
                        linearInterp(rect.lower.y, src[prev].y, src[curr].y, src[prev].x, src[curr].x),
                        rect.lower.y));
                        break;
                    case 1:  dest.push_back(Point2d(
                        rect.upper.x, 
                        linearInterp(rect.upper.x, src[prev].x, src[curr].x, src[prev].y, src[curr].y)));
                        break;
                    case 2:  dest.push_back(Point2d(
                        linearInterp(rect.upper.y, src[prev].y, src[curr].y, src[prev].x, src[curr].x),
                        rect.upper.y));
                        break;
                    default: dest.push_back(Point2d(
                        rect.lower.x, 
                        linearInterp(rect.lower.x, src[prev].x, src[curr].x, src[prev].y, src[curr].y)));
                }
            }
            // also retain the current vertex if it was on the non-clipped side w.r.t. current direction
            if(currInside)
                dest.push_back(src[curr]);
            prevInside = currInside;
        }
    }
    return result;
}


template<int N>
bool computeBsplineIntegralsOverPolygon(const Polygon& polygon,
    const BsplineInterpolator1d<N>& bsplx, const BsplineInterpolator1d<N>& bsply, double output[])
{
    const std::vector<double> &gridx = bsplx.xvalues(), &gridy = bsply.xvalues();
    const size_t
        gridSizeX   = gridx.size(),  gridSizeY = gridy.size(),
        numVertices = polygon.size();
    if(numVertices <= 2)  // no further action needed
        return false;

    // determine the overall bounding box, area, and the orientation (counterclockwise is positive)
    double polygonArea = 0., xmin = INFINITY, xmax = -INFINITY, ymin = INFINITY, ymax = -INFINITY;
    bool outOfBounds = false;
    for(size_t i=0; i<numVertices; i++) {
        const Point2d& v = polygon[i], w = i<numVertices-1 ? polygon[i+1] : polygon[0];
        xmin = fmin(xmin, v.x);
        xmax = fmax(xmax, v.x);
        ymin = fmin(ymin, v.y);
        ymax = fmax(ymax, v.y);
        polygonArea += 0.5 * (v.x + w.x) * (w.y - v.y);
    }

    // determine the range of grid cells that enclose the bounding box
    int indXmin = binSearch(xmin,  &gridx.front(), gridSizeX);
    int indXmax = binSearch(xmax,  &gridx.front(), gridSizeX)+1;
    int indYmin = binSearch(ymin,  &gridy.front(), gridSizeY);
    int indYmax = binSearch(ymax,  &gridy.front(), gridSizeY)+1;
    if(indXmin < 0) { indXmin = 0; outOfBounds = true; }
    if(indYmin < 0) { indYmin = 0; outOfBounds = true; }
    if(indXmax > (int)gridSizeX-1) { indXmax = gridSizeX-1; outOfBounds = true; }
    if(indYmax > (int)gridSizeY-1) { indYmax = gridSizeY-1; outOfBounds = true; }

    // make sure that the polygon is oriented counterclockwise
    Polygon tmp;  // contains the reversed original polygon if necessary
    const Polygon& newpoly = polygonArea >= 0 ? polygon :     // use the original polygon or
        (tmp.assign(polygon.rbegin(), polygon.rend()), tmp);  // create and use the reversed copy

    // loop over the cells that potentially could lie within this polygon
    for(int iy = indYmin; iy < indYmax; iy++) {
        for(int ix = indXmin; ix < indXmax; ix++) {
            integrateOverPolygon(
                clipPolygonByRectangle(newpoly, Rectangle(gridx[ix], gridy[iy], gridx[ix+1], gridy[iy+1])),
                bsplx, bsply, output);
        }
    }
    return outOfBounds;
}

// template instantiations
template bool computeBsplineIntegralsOverPolygon(
    const Polygon&, const BsplineInterpolator1d<0>&, const BsplineInterpolator1d<0>&, double[]);
template bool computeBsplineIntegralsOverPolygon(
    const Polygon&, const BsplineInterpolator1d<1>&, const BsplineInterpolator1d<1>&, double[]);
template bool computeBsplineIntegralsOverPolygon(
    const Polygon&, const BsplineInterpolator1d<2>&, const BsplineInterpolator1d<2>&, double[]);
template bool computeBsplineIntegralsOverPolygon(
    const Polygon&, const BsplineInterpolator1d<3>&, const BsplineInterpolator1d<3>&, double[]);


double polygonArea(const Polygon& polygon)
{
    double result = 0.;
    const size_t numVertices = polygon.size();
    for(size_t i=0, j=numVertices-1; i<numVertices; j=i++)
        result += (polygon[i].x + polygon[j].x) * (polygon[i].y - polygon[j].y);
    return result * 0.5;
}

bool isPointInsidePolygon(double x, double y, const Polygon& polygon)
{
    if(polygon.empty())
        return false;
    int wind = 0;    // the winding number counter
    const unsigned int numVertices = polygon.size();
    for(unsigned int i=0; i<numVertices; i++) {
        const Point2d& v = polygon[i], w = i<numVertices-1 ? polygon[i+1] : polygon[0];
        double dx = w.x-v.x, dy = w.y-v.y,  sign = dx * (y-v.y) - dy * (x-v.x);
        if(v.y <= y) {
            if(w.y > y && sign > 0)
                ++wind;
        } else {
            if(w.y <= y && sign < 0)
                --wind;
        }
    }
    return wind != 0;
}

} // namespace
