#include "galaxymodel_densitygrid.h"
#include "galaxymodel_base.h"
#include "math_core.h"
#include "math_sphharm.h"
#include "utils.h"
#include <cmath>
#include <algorithm>
#include <cassert>
#include <stdexcept>
#ifndef _MSC_VER
#include <alloca.h>
#else
#include <malloc.h>
#endif

namespace galaxymodel{

namespace{

/// relative accuracy of 3d integration for computing the projections of density onto basis functions
static const double EPSREL_DENSITY_INT = 1e-3;

/// max number of density evaluations per basis function
static const unsigned int MAX_NUM_EVAL = 1000;

/// order of Gauss-Legendre integration in radial direction for optimized projection implementations
static const unsigned int GLORDER_RAD  = 6;

/// order of Gauss-Legendre integration in angular direction for the classical grid scheme
static const unsigned int GLORDER_ANG  = 4;

/// minimum order of spherical-harmonic or Fourier expansion for computing the density projection
static const int LMIN_SPHHARM = 16;

/// Helper class for 3-dimensional integration of a density multiplied by basis functions of the grid
class TargetDensityIntegrand: public math::IFunctionNdim {
    const potential::BaseDensity& dens;
    const BaseTargetDensity& grid;
    const unsigned int nval;
public:
    TargetDensityIntegrand(const potential::BaseDensity& _dens, const BaseTargetDensity& _grid) :
        dens(_dens), grid(_grid), nval(grid.numValues()) {}

    virtual void eval(const double vars[], double values[]) const {
        std::fill(values, values + nval, 0.);
        double val;
        const coord::PosCyl pcyl = potential::unscaleCoords(vars, &val /*jacobian of coord scaling*/);
        if(val!=0) val *= dens.density(pcyl);
        if(val!=0) {
            coord::PosCar pcar(toPosCar(pcyl));
            double xyz[3] = {pcar.x, pcar.y, pcar.z};
            grid.addPoint(xyz, val, values);
        }
    }
    virtual unsigned int numVars()   const { return 3; }
    virtual unsigned int numValues() const { return nval; }
};

// decode the index of the cell for TargetDensityClassic<0>,
// or its four corners for TargetDensityClassic<1>
template<int N> void getCornerIndicesClassic(
    /*input*/  int pane, int ind1, int ind2, int stripsPerPane,
    /*output*/ int& indll, int& indul, int& indlu, int& induu);

template<> inline void getCornerIndicesClassic<0>(
    /*input*/  int pane, int ind1, int ind2, int stripsPerPane,
    /*output*/ int& indCell, int&, int&, int&)
{
    indCell = ind1 + (ind2 + pane * stripsPerPane) * stripsPerPane;
}
template<> inline void getCornerIndicesClassic<1>(
    /*input*/  int pane, int ind1, int ind2, int stripsPerPane,
    /*output*/ int& indll, int& indul, int& indlu, int& induu)
{
    indll = ind1 + (ind2 + pane * stripsPerPane) * (stripsPerPane+1);
    indul = indll + 1;
    if(ind2 < stripsPerPane-1) {
        indlu = indll + stripsPerPane+1;
        induu = indlu + 1;
    } else {
        int adjpane = (pane+2) % 3;   // index of the adjacent pane
        indlu = (ind1 + adjpane * stripsPerPane + 1) * (stripsPerPane+1) - 1;
        if(ind1 < stripsPerPane-1)
            induu = indlu + stripsPerPane + 1;
        else
            induu = 3 * stripsPerPane * (stripsPerPane + 1);  // the central node where three panes join
    }
}

// decode the index of the basis element with the given index of angular harmonic m
// for TargetDensityCylindrical<0>, or the indices of four basis elements for TargetDensityCylindrical<1>
template<int N> void getCornerIndicesCylindrical(
    /*input*/  int m, int indR, int indz, int gridRsize, int gridzsize,
    /*output*/ int& indll, int& indul, int& indlu, int& induu);

template<> inline void getCornerIndicesCylindrical<0>(
    /*input*/  int m, int indR, int indz, int gridRsize, int gridzsize,
    /*output*/ int& indCell, int&, int&, int&)
{
    indCell = indR + (indz + m/2 * gridzsize) * gridRsize;
}
template<> inline void getCornerIndicesCylindrical<1>(
    /*input*/  int m, int indR, int indz, int gridRsize, int gridzsize,
    /*output*/ int& indll, int& indul, int& indlu, int& induu)
{
    int offsetm = 0;
    if(m>0) {
        offsetm = (1 + m/2 * gridRsize) * (gridzsize + 1);
        // higher harmonics do not have any basis functions at the grid node R=0
        gridRsize -= 1;
        indR -= 1;
    }
    indll = offsetm + indR + indz * (gridRsize + 1);
    indul = indll + 1;
    indlu = indll + gridRsize + 1;
    induu = indlu + 1;
}

} // internal ns


void BaseTargetDensity::computeDFProjection(const GalaxyModel& model, StorageNumT* output) const
{
    std::vector<double> result =
        // TODO: this routine should be OpenMP-parallelized in each of the descendant classes
        // update: DensityFromDF is itself parallelized when using evalmanyDensity***,
        // but in the current implementations of computeDensityProjection, the input density
        // is queried one point at a time - this needs to be replaced with evalmany...
        computeDensityProjection(DensityFromDF(model, EPSREL_DENSITY_INT, MAX_NUM_EVAL));
    for(size_t i=0; i<result.size(); i++)
        output[i] = static_cast<StorageNumT>(result[i]);
}

std::vector<double> BaseTargetDensity::computeDensityProjection(
    const potential::BaseDensity& density) const
{
    double xlower[3] = {0,0,0};   // boundaries of integration region in scaled coordinates
    double xupper[3] = {1,1,1};
    std::vector<double> result(numValues());
    math::integrateNdim(TargetDensityIntegrand(density, *this),
        xlower, xupper, EPSREL_DENSITY_INT, MAX_NUM_EVAL * numValues(), &result[0]);
    return result;
}


//----- Classic grid-based density representation -----//

template<int N>
TargetDensityClassic<N>::TargetDensityClassic(
    const unsigned int _stripsPerPane,
    const std::vector<double>& _gridr,
    const double axisYtoX, const double axisZtoX)
:
    stripsPerPane(_stripsPerPane),
    valuesPerShell(3 * stripsPerPane * (stripsPerPane + N) + N),
    // if the input grid starts from zero, skip this first array element, otherwise take the whole array
    shellRadii(_gridr.empty() || _gridr[0] > 0 ? _gridr.begin() : _gridr.begin()+1, _gridr.end()),
    axisX(1. / cbrt(axisYtoX*axisZtoX)),
    axisY(axisYtoX * axisX),
    axisZ(axisZtoX * axisX)   // the product axisX*axisY*axisZ is unity
{
    bool ok = shellRadii.size() >= 1 && shellRadii[0] > 0;
    for(unsigned int i=1; i<shellRadii.size(); i++)
        ok &= shellRadii[i] > shellRadii[i-1];
    if(!ok || stripsPerPane<1 || shellRadii.size()<1 || axisYtoX<=0 || axisZtoX<=0)
        throw std::invalid_argument("TargetDensityClassic: invalid grid parameters");
}

template<int N>
void TargetDensityClassic<N>::addPoint(const double point[3], const double mult, double values[]) const
{
    const int numShells = shellRadii.size();
    double X = fabs(point[0] / axisX), Y = fabs(point[1]) / axisY, Z = fabs(point[2]) / axisZ;
    double r = sqrt(X*X + Y*Y + Z*Z);
    int indShell = math::binSearch(r, &shellRadii[0], numShells) + 1;
    assert(indShell>=0);
    if(indShell >= numShells || mult == 0) {
        return;  // outside the grid
    }
    int pane;
    double coord0, coord1, coord2;
    if(X>=Y && X>Z) {
        pane=0;
        coord0=X;
        coord1=Y;
        coord2=Z;
    } else if(Y>=Z && Y>X) {
        pane=1;
        coord0=Y;
        coord1=Z;
        coord2=X;
    } else if(Z>=X && Z>=Y) {
        pane=2;
        coord0=Z;
        coord1=X;
        coord2=Y;
    } else
        throw std::runtime_error("TargetDensityClassic: cannot determine the cell index");
    // fractional index of the grid cell in both directions:
    // ratio1 between 0 and 1 - inside the first row, between 1 and 2 - inside the second row, etc.;
    // ratio2 between 0 and 1 - inside the first column, etc.
    double ratio1 = fmin(atan(coord1/coord0) * (4/M_PI), 1.) * stripsPerPane;
    double ratio2 = fmin(atan(coord2/coord0) * (4/M_PI), 1.) * stripsPerPane;
    int ind1 = std::min<int>(ratio1, stripsPerPane-1);
    int ind2 = std::min<int>(ratio2, stripsPerPane-1);

    // further details differ depending on the interpolation order of basis elements
    int indll, indul, indlu, induu;
    getCornerIndicesClassic<N>(pane, ind1, ind2, stripsPerPane, /*output*/indll, indul, indlu, induu);
    if(N == 0) {
        values[indll + indShell * valuesPerShell] += mult;
    } else if(N==1) {
        // convert ratio1,ratio2 and r into fractional coordinates within the current cell (between 0 and 1)
        ratio1 -= ind1;
        ratio2 -= ind2;
        double val = indShell == 0 ?
            mult *  r / shellRadii[0] :
            mult * (r - shellRadii[indShell-1]) / (shellRadii[indShell] - shellRadii[indShell-1]);
        // contribution to the basis functions at the upper end of the radial segment
        double* valOff = &values[indShell * valuesPerShell + 1];  // offset in the output array
        valOff[indll] += (1-ratio1) * (1-ratio2) * val;
        valOff[indul] +=    ratio1  * (1-ratio2) * val;
        valOff[indlu] += (1-ratio1) *    ratio2  * val;
        valOff[induu] +=    ratio1  *    ratio2  * val;
        // contributions to the lower end of the radial segment
        if(indShell   == 0) {          // if this is the innermost segment, then there is only
            values[0] += mult - val;   // a single basis function at origin
        } else {                       // otherwise a full set of four functions
            valOff -= valuesPerShell;  // another offset in the output array
            val = mult - val;          // remaining contribution of the input point
            valOff[indll] += (1-ratio1) * (1-ratio2) * val;
            valOff[indul] +=    ratio1  * (1-ratio2) * val;
            valOff[indlu] += (1-ratio1) *    ratio2  * val;
            valOff[induu] +=    ratio1  *    ratio2  * val;
        }
    } else
        assert(!"TargetDensityClassic: unimplemented N");
}

template<int N>
std::vector<double> TargetDensityClassic<N>::computeDensityProjection(
    const potential::BaseDensity& density) const
{
    const double *glnodesRad = math::GLPOINTS[GLORDER_RAD], *glweightsRad = math::GLWEIGHTS[GLORDER_RAD];
    const double *glnodesAng = math::GLPOINTS[GLORDER_ANG], *glweightsAng = math::GLWEIGHTS[GLORDER_ANG];
    // pre-compute nodes and weigths for all strips in the integration over angles
    std::vector<double> nodesAng(GLORDER_ANG * stripsPerPane), weightsAng(GLORDER_ANG * stripsPerPane);
    for(unsigned int iA=0; iA < GLORDER_ANG * stripsPerPane; iA++) {
        int strip = iA / GLORDER_ANG, glindex = iA % GLORDER_ANG;
        nodesAng  [iA] = tan(M_PI/4 * (strip + glnodesAng[glindex]) / stripsPerPane);
        weightsAng[iA] = glweightsAng[glindex] * (1 + pow_2(nodesAng[iA])) / stripsPerPane * M_PI/4;
    }
    std::vector<double> result(numValues());
    for(unsigned int iR=0; iR < shellRadii.size() * GLORDER_RAD; iR++) {
        int indShell = iR / GLORDER_RAD, offShell = iR % GLORDER_RAD;
        double
        radius1 = indShell == 0 ? 0. : shellRadii[indShell-1],
        radius2 = shellRadii[indShell],
        offsetR = glnodesRad[offShell],
        radius  = radius1 * (1-offsetR) + radius2 * offsetR,
        weightRad = 8/*octants*/ * pow_2(radius) * glweightsRad[offShell] * (radius2 - radius1);
        for(unsigned int i1 = 0; i1 < GLORDER_ANG * stripsPerPane; i1++) {
            for(unsigned int i2 = 0; i2 < GLORDER_ANG * stripsPerPane; i2++) {
                double
                offset1  = glnodesAng[i1 % GLORDER_ANG],  // fractional coords within the current cell
                offset2  = glnodesAng[i2 % GLORDER_ANG],
                denom    = 1. / sqrt(1 + pow_2(nodesAng[i1]) + pow_2(nodesAng[i2])),
                weight   = weightRad * weightsAng[i1] * weightsAng[i2] * pow_3(denom),
                coord[3] = {radius * denom, radius * denom * nodesAng[i1], radius * denom * nodesAng[i2]};
                int ind1 = i1 / GLORDER_ANG, ind2 = i2 / GLORDER_ANG;
                for(int pane=0; pane<3; pane++) {
                    double value = weight * density.density(coord::PosCar(
                        coord[(3-pane)%3] * axisX, coord[(4-pane)%3] * axisY, coord[(5-pane)%3] * axisZ));
                    int indll, indul, indlu, induu;
                    getCornerIndicesClassic<N>(pane, ind1, ind2, stripsPerPane,
                        /*output*/indll, indul, indlu, induu);
                    // further details depend on the shape of basis elements
                    if(N == 0) {
                        result[indll + indShell * valuesPerShell] += value;
                    } else if(N==1) {
                        double* resOff = &result[indShell * valuesPerShell + 1];
                        resOff[indll] += value * offsetR * (1-offset1) * (1-offset2);
                        resOff[indul] += value * offsetR *    offset1  * (1-offset2);
                        resOff[indlu] += value * offsetR * (1-offset1) *    offset2;
                        resOff[induu] += value * offsetR *    offset1  *    offset2;
                        if(indShell   == 0) {  // a single node at origin
                            result[0] += value * (1-offsetR);
                        } else {
                            resOff -= valuesPerShell;
                            resOff[indll] += value * (1-offsetR) * (1-offset1) * (1-offset2);
                            resOff[indul] += value * (1-offsetR) *    offset1  * (1-offset2);
                            resOff[indlu] += value * (1-offsetR) * (1-offset1) *    offset2;
                            resOff[induu] += value * (1-offsetR) *    offset1  *    offset2;
                        }
                    } else
                        assert(!"TargetDensityClassic: unimplemented N");
                }
            }
        }
    }
    return result;
}

template<int N>
std::string TargetDensityClassic<N>::coefName(unsigned int index) const
{
    if(index >= numValues())
        throw std::out_of_range("TargetDensityClassic: index out of range");
    double coord[3] = {1.}, radius;
    unsigned int pane;
    if(N==0) {
        unsigned int
        indShell = index / valuesPerShell,
        ind1     = index % stripsPerPane,
        ind2     = index / stripsPerPane % stripsPerPane;
        pane     = index % valuesPerShell / (stripsPerPane * stripsPerPane);
        coord[1] = tan(M_PI/4 * (ind1 + 0.5) / stripsPerPane);
        coord[2] = tan(M_PI/4 * (ind2 + 0.5) / stripsPerPane);
        radius   = 0.5 * (shellRadii.at(indShell) + (indShell>0 ? shellRadii[indShell-1] : 0.));
    } else if(N==1) {
        if(index==0) {  // the node at origin
            coord[1] = coord[2] = radius = 0.;
            pane = 0;
        } else {
            index--;
            radius = shellRadii.at(index / valuesPerShell);
            index %= valuesPerShell;
            if(index == valuesPerShell-1) {  // the central node
                pane  = 0;
                coord[1] = coord[2] = 1.;
            } else {  // ordinary nodes
                pane   = index / (stripsPerPane * (stripsPerPane+1));
                index %= stripsPerPane * (stripsPerPane+1);
                coord[1] = tan(M_PI/4 * (index % (stripsPerPane+1)) / stripsPerPane);
                coord[2] = tan(M_PI/4 * (index / (stripsPerPane+1)) / stripsPerPane);
            }
        }
    } else
        assert(!"TargetDensityClassic: unimplemented N");
    double denom = 1. / sqrt(pow_2(coord[0]) + pow_2(coord[1]) + pow_2(coord[2]));
    // pane=0:  coord = {x, y, z};  pane=1:  coord = {y, z, x};  pane=2:  coord = {z, x, y}
    return "x=" + utils::toString(radius * denom * coord[(3-pane)%3] * axisX) +
         ", y=" + utils::toString(radius * denom * coord[(4-pane)%3] * axisY) +
         ", z=" + utils::toString(radius * denom * coord[(5-pane)%3] * axisZ);
}

template<> const char* TargetDensityClassic<0>::name() const { return "DensityClassicTopHat"; }
template<> const char* TargetDensityClassic<1>::name() const { return "DensityClassicLinear"; }

template class TargetDensityClassic<0>;
template class TargetDensityClassic<1>;


//----- Spherical-harmonic density representation -----//

TargetDensitySphHarm::TargetDensitySphHarm(
    const int _lmax, const int _mmax, const std::vector<double>& _gridr)
:
    lmax(_lmax), mmax(_mmax),
    angularCoefs( (lmax/2+1) * (mmax/2+1) - mmax/2 * (mmax/2+1) / 2 ),
    // if the input grid starts from zero, skip this first array element, otherwise take the whole array
    gridr(_gridr.empty() || _gridr[0] > 0 ? _gridr.begin() : _gridr.begin()+1, _gridr.end())
{
    bool ok = lmax >= 0 && mmax >= 0 && mmax <= lmax && gridr.size() >= 1 && gridr[0] > 0;
    for(unsigned int i=1; i<gridr.size(); i++)
        ok &= gridr[i] > gridr[i-1];
    if(!ok)
        throw std::invalid_argument("TargetDensitySphHarm: invalid grid parameters");
}

void TargetDensitySphHarm::addPoint(const double point[3], double mult, double values[]) const
{
    const coord::PosCyl pcyl = toPosCyl(coord::PosCar(point[0], point[1], point[2]));
    double r   = sqrt(pow_2(pcyl.R) + pow_2(pcyl.z));
    double tau = pcyl.z / (r + pcyl.R);
    if(r==0) {
        values[0] += mult;
        return;
    }
    const int gridrsize = gridr.size();
    int indr = math::binSearch(r, &gridr[0], gridrsize) + 1;
    assert(indr>=0);
    if(indr >= gridrsize || mult == 0)
        return;  // outside the grid
    // convert r into a fractional offset within the shell [0..1]
    double offr = indr==0 ? r / gridr[0] : (r - gridr[indr-1]) / (gridr[indr] - gridr[indr-1]);

    // temporary array for storing the values of Legendre and trigonometric functions
    unsigned int size = 1 + lmax + mmax;
    double* leg = static_cast<double*>(alloca(size * sizeof(double))), *trig = leg + lmax+1;
    math::trigMultiAngle(pcyl.phi, mmax, false, trig);

    // storage scheme for the output: the radial variation of each harmonic coefficient
    // is represented by a continuous array of `gridrsize` numbers
    // (except l=0, which has one extra term at r=0); of these numbers, only two may be nonzero,
    // corresponding to the radial basis functions associated to the grid nodes 
    // gridr[indr] and gridr[indr+1], which enclose the radius of the input point.
    // The arrays for each harmonic term are stored one after another in the following order: 
    // first all terms with l=0,2,4,...,lmax and m=0,
    // then l=2,4,...,lmax and m=2, then l=4,...,lmax and m=4, up to mmax.
    // offset is the index of the coefficient with the given (l,m) in the output array.
    for(int m=0, offset=indr+1; m<=mmax; m+=2) {
        math::sphHarmArray(lmax, m, tau, leg);
        for(int l=m; l<=lmax; l+=2, offset+=gridrsize) {
            double val = mult * leg[l-m] * 2*M_SQRTPI * (m==0 ? 1. : M_SQRT2 * trig[m-1]);
            values[offset] += val * offr;
            if(indr>0 || l==0)
                values[offset-1] += val * (1-offr);
        }
    }
}

std::vector<double> TargetDensitySphHarm::computeDensityProjection(
    const potential::BaseDensity& density) const
{
    // the integration in radius follows the Gauss-Legendre rule
    const double *glnodesRad = math::GLPOINTS[GLORDER_RAD], *glweightsRad = math::GLWEIGHTS[GLORDER_RAD];

    // to improve accuracy of SH coefficient computation, we may increase the order of expansion
    // that determines the number of integration points in angles
    int lmax_tmp =     isSpherical(density) ? 0 : std::max(lmax, LMIN_SPHHARM);
    int mmax_tmp = isZRotSymmetric(density) ? 0 : std::max(mmax, LMIN_SPHHARM);
    math::SphHarmIndices ind(lmax_tmp, mmax_tmp, coord::ST_TRIAXIAL);
    math::SphHarmTransformForward trans(ind);
    unsigned int numSamplesAngles = trans.size();  // size of array of density values at each r
    std::vector<double> densValues(numSamplesAngles);
    std::vector<double> shcoefs(std::max<int>(ind.size(), pow_2(lmax+1)));
    std::vector<double> result(numValues());
    const int gridrsize = gridr.size();
    for(unsigned int ir=0; ir < gridrsize * GLORDER_RAD; ir++) {
        // 0. assign the radius of this point and its weigth in the total integral
        int indr = ir / GLORDER_RAD, subr = ir % GLORDER_RAD;
        double
        radius1 = indr == 0 ? 0. : gridr[indr-1],
        radius2 = gridr[indr],
        offr    = glnodesRad[subr],  // fractional [0..1] offset inside the current grid segment
        radius  = radius1 * (1-offr) + radius2 * offr,
        weight  = 4*M_PI * pow_2(radius) * glweightsRad[subr] * (radius2 - radius1);

        // 1. collect the density values at the prescribed set of points in angles
        for(unsigned int iA=0; iA<numSamplesAngles; iA++)  {
            double z   = radius * trans.costheta(iA);
            double R   = sqrt(pow_2(radius) - z*z);
            double phi = trans.phi(iA);
            densValues[iA] = density.density(coord::PosCyl(R, z, phi));
        }

        // 2. transform these values to spherical-harmonic expansion coefficients
        trans.transform(&densValues[0], &shcoefs[0]);

        // 3. store the contribution of each SH coef to the relevant radial basis functions
        // (which are simply triangular-shaped blocks, so that at most two of them are used per each l,m)
        for(int m=0, offset=indr+1; m<=mmax; m+=2) {
            for(int l=m; l<=lmax; l+=2, offset+=gridrsize) {
                double val = shcoefs[ind.index(l, m)] * weight;
                result.at(offset) += val * offr;
                if(indr>0 || l==0)  // for the grid node at origin, only one angular term (l=0) is used
                    result.at(offset-1) += val * (1-offr);
            }
        }
    }
    return result;
}

std::string TargetDensitySphHarm::coefName(unsigned int index) const
{
    if(index >= numValues())
        throw std::out_of_range("TargetDensitySphHarm: index out of range");
    if(index == 0)
        return "r=0, l=0, m=0";
    index--;
    unsigned int gridrsize = gridr.size(), m = 0;
    while(index >= gridrsize * (1 + lmax/2 - m/2)) {
        index -= gridrsize * (1 + lmax/2 - m/2);
        m += 2;
    }
    unsigned int l = index / gridrsize * 2 + m, indr = index % gridrsize;
    return "r=" + utils::toString(gridr[indr]) + ", l=" + utils::toString(l) + ", m=" + utils::toString(m);
}

const char* TargetDensitySphHarm::name() const { return "DensitySphHarm"; }


//----- Cylindrical grid + azimuthal Fourier density representation -----//

template<int N>
TargetDensityCylindrical<N>::TargetDensityCylindrical(const int _mmax,
    const std::vector<double>& _gridR, const std::vector<double>& _gridz)
:
    mmax(_mmax),
    // if the input grids start from zero, skip this first array element, otherwise take the whole array
    gridR(_gridR.empty() || _gridR[0] > 0 ? _gridR.begin() : _gridR.begin()+1, _gridR.end()),
    gridz(_gridz.empty() || _gridz[0] > 0 ? _gridz.begin() : _gridz.begin()+1, _gridz.end()),
    totalNumValues( (gridz.size() + N) * (gridR.size() * (mmax/2+1) + N) )
{
    bool ok = mmax >= 0 && gridR.size() >= 1 && gridz.size() >= 1 && gridR[0] > 0 && gridz[0] > 0;
    for(unsigned int i=1; i<gridR.size(); i++)
        ok &= gridR[i] > gridR[i-1];
    for(unsigned int i=1; i<gridz.size(); i++)
        ok &= gridz[i] > gridz[i-1];
    if(!ok)
        throw std::invalid_argument("TargetDensityCylindrical: invalid grid parameters");
}

template<int N>
void TargetDensityCylindrical<N>::addPoint(const double point[3], double mult, double values[]) const
{
    const coord::PosCyl pcyl = toPosCyl(coord::PosCar(point[0], point[1], fabs(point[2])));
    const int gridRsize = gridR.size(), gridzsize = gridz.size();
    int indR = math::binSearch(pcyl.R, &gridR[0], gridRsize) + 1;
    int indz = math::binSearch(pcyl.z, &gridz[0], gridzsize) + 1;
    assert(indR>=0 && indz>=0);
    if(indR >= gridRsize || indz >= gridzsize || mult == 0)
        return;  // outside the grid
    // convert R,z into fractional coordinates within the 2d cell [0..1]
    double prevR = indR>0 ? gridR[indR-1] : 0.;
    double prevz = indz>0 ? gridz[indz-1] : 0.;
    double offR  = (pcyl.R - prevR) / ( gridR[indR] - prevR );
    double offz  = (pcyl.z - prevz) / ( gridz[indz] - prevz );

    // temporary array for storing the values of trigonometric functions (cosines only)
    double* trig = static_cast<double*>(alloca(mmax * sizeof(double)));
    math::trigMultiAngle(pcyl.phi, mmax, false, trig);

    for(int m=0; m<=mmax; m+=2) {
        double val = mult * (m==0 ? 1. : 2*trig[m-1]);
        int indll, indul, indlu, induu;
        getCornerIndicesCylindrical<N>(m, indR, indz, gridRsize, gridzsize,
            /*output*/ indll, indul, indlu, induu);
        if(N==0) {  // only one term
            assert(indll < (int)totalNumValues);
            values[indll] += val;
        } else if(N==1) {  // up to four terms
            assert(induu < (int)totalNumValues);
            values[indul] += val * offR * (1-offz);
            values[induu] += val * offR *    offz;
            if(m==0 || indR>0) {
                values[indll] += val * (1-offR) * (1-offz);
                values[indlu] += val * (1-offR) *    offz;
            }   // otherwise there is no such term in the basis set
        } else
            assert(!"TargetDensityCylindrical: unimplemented N");
    }
}

template<int N>
std::vector<double> TargetDensityCylindrical<N>::computeDensityProjection(
    const potential::BaseDensity& density) const
{
    // the integration in R and z follows the Gauss-Legendre rule
    const double *glnodesRad = math::GLPOINTS[GLORDER_RAD], *glweightsRad = math::GLWEIGHTS[GLORDER_RAD];
    // select a sufficiently high order of integration in angles
    int mmax_tmp = isZRotSymmetric(density) ? 0 : std::max(mmax, LMIN_SPHHARM);
    math::FourierTransformForward trans(mmax_tmp, false/*no odd terms*/);
    unsigned int numSamplesAngles = trans.size();      // size of array of density values at each (R,z)
    std::vector<double> densValues(numSamplesAngles);  // temp.storage for density values
    std::vector<double> coefs(std::max<int>(trans.size(), mmax+1));  // temp.storage for transformed coefs
    std::vector<double> result(numValues());  // output array
    const int gridRsize = gridR.size(), gridzsize = gridz.size();
    for(unsigned int iR=0; iR < gridRsize * GLORDER_RAD; iR++) {
        int indR = iR / GLORDER_RAD, subR = iR % GLORDER_RAD;
        double
        R1      = indR == 0 ? 0. : gridR[indR-1],
        R2      = gridR[indR],
        offR    = glnodesRad[subR],  // fractional [0..1] offset in R inside the current grid segment
        R       = R1 * (1-offR) + R2 * offR,
        weightR = 2 * R * glweightsRad[subR] * (R2 - R1);

        for(unsigned int iz=0; iz < gridzsize * GLORDER_RAD; iz++) {
            int indz = iz / GLORDER_RAD, subz = iz % GLORDER_RAD;
            double
            z1     = indz == 0 ? 0. : gridz[indz-1],
            z2     = gridz[indz],
            offz   = glnodesRad[subz],  // fractional offset in z
            z      = z1 * (1-offz) + z2 * offz,
            weight = weightR * glweightsRad[subz] * (z2 - z1);

            // collect the density values and transform them into Fourier coefficients
            for(unsigned int iphi=0; iphi<numSamplesAngles; iphi++)
                densValues[iphi] = density.density(coord::PosCyl(R, z, trans.phi(iphi)));
            trans.transform(&densValues[0], &coefs[0]);

            // add the contribution to each basis function in the azimuthal plane
            for(int m=0; m<=mmax; m+=2) {
                double val = coefs[m] * weight * (1 + (m>0));
                int indll, indul, indlu, induu;
                getCornerIndicesCylindrical<N>(m, indR, indz, gridRsize, gridzsize,
                    /*output*/ indll, indul, indlu, induu);
                if(N==0) {  // only one term
                    result.at(indll) += val;
                } else if(N==1) {  // up to four terms
                    result.at(indul) += val * offR * (1-offz);
                    result.at(induu) += val * offR *    offz;
                    if(m==0 || indR>0) {
                        result.at(indll) += val * (1-offR) * (1-offz);
                        result.at(indlu) += val * (1-offR) *    offz;
                    }   // otherwise there is no such term in the basis set
                } else
                    assert(!"TargetDensityCylindrical: unimplemented N");
            }
        }
    }
    return result;
}

template<> std::string TargetDensityCylindrical<0>::coefName(unsigned int index) const
{
    if(index >= numValues())
        throw std::out_of_range("TargetDensityCylindrical: index out of range");
    unsigned int gridRsize = gridR.size(), gridzsize = gridz.size();
    unsigned int
    indR  = index % gridRsize,
    indmz = index / gridRsize,
    indz  = indmz % gridzsize,
    m     = indmz / gridzsize * 2;
    double R = 0.5 * (gridR[indR] + (indR>0 ? gridR[indR-1] : 0.));
    double z = 0.5 * (gridz[indz] + (indz>0 ? gridz[indz-1] : 0.));
    return "R=" + utils::toString(R) + ", z=" + utils::toString(z) + ", m=" + utils::toString(m);
}

template<> std::string TargetDensityCylindrical<1>::coefName(unsigned int index) const
{
    if(index >= numValues())
        throw std::out_of_range("TargetDensityCylindrical: index out of range");
    unsigned int gridRsize = gridR.size(), gridzsize = gridz.size();
    // # of meridional-plane elements for m=0 is larger than for higher m
    unsigned int
    m0size = (gridRsize+1) * (gridzsize+1),
    msize  =  gridRsize    * (gridzsize+1),
    m      = index < m0size ? 0 : ((index-m0size) / msize + 1) * 2,
    indz   = index < m0size ? index / (gridRsize+1) : (index-m0size) / gridRsize % (gridzsize+1),
    indR   = index < m0size ? index % (gridRsize+1) : (index-m0size) % gridRsize + 1;
    return "R=" + utils::toString(indR==0 ? 0. : gridR.at(indR-1)) +
         ", z=" + utils::toString(indz==0 ? 0. : gridz.at(indz-1)) +
         ", m=" + utils::toString(m);
}

template<> const char* TargetDensityCylindrical<0>::name() const { return "DensityCylindricalTopHat"; }
template<> const char* TargetDensityCylindrical<1>::name() const { return "DensityCylindricalLinear"; }

template class TargetDensityCylindrical<0>;
template class TargetDensityCylindrical<1>;

}  // namespace
