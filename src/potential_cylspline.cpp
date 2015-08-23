#include "potential_cylspline.h"
#include "math_core.h"
#include "math_fit.h"
#include "math_spline.h"
#include "math_specfunc.h"
#include "utils.h"
#include <cmath>
#include <cassert>
#include <stdexcept>

namespace potential {

const unsigned int CYLSPLINE_MIN_GRID_SIZE = 4;
const unsigned int CYLSPLINE_MAX_GRID_SIZE = 1024;
const unsigned int CYLSPLINE_MAX_ANGULAR_HARMONIC = 64;

/// max number of function evaluations in multidimensional integration
const unsigned int MAX_NUM_EVAL = 10000;

//------- Auxiliary math --------//

/** Compute the following integral for a fixed integer value of m>=0 and arbitrary a>=0, b>=0, c:
    \f$  \int_0^\infty J_m(a x) J_m(b x) \exp(-|c| x) dx  \f$,  where J_m are Bessel functions.
*/
class BesselIntegral {
public:
    /// setup the interpolation table for the fixed value of m
    BesselIntegral(const unsigned int m);
    double value(double a, double b, double c) const;
private:
    /// compute the hypergeometric function 2F1(1+mm/2, 1/2+mm/2; 3/2+mm; x), where mm=m-1/2
    double hypergeom(double x) const;
    unsigned int m;        ///< order of Bessel functions
    double multh0, multh1; ///< pre-computed coefs in hypergeometric function
    double multfactor;     ///< prefactor in Legendre Q function
    math::CubicSpline spl; ///< internal data for interpolation
};

BesselIntegral::BesselIntegral(const unsigned int _m) : m(_m)
{
    const double mm = m-0.5;
    // setup interpolation table for a faster evaluation of hypergeometric function 
    // 2F1(1+mm/2, 1/2+mm/2; 3/2+mm; x)  for 0<x<1
    // use the first term in asymptotic expansion (eq.15.3.10 in Abramowitz&Stegun) as 'approximation',
    // and interpolate the difference between the actual value and the 1st term via a cubic spline
    const size_t numpt = 16;
    std::vector<double> yval(numpt+1), fval(numpt+1);
    multh0 = math::gamma(1.5+mm)/math::gamma(1+mm/2)/math::gamma(0.5+mm/2);
    multh1 = 2*math::digamma(1) - math::digamma(1+mm/2) - math::digamma(0.5+mm/2);
    for(size_t i=0; i<numpt; i++) {
        double yy=i*1.0/numpt;
        double y=yy*(2-yy);
        double val = math::hypergeom2F1(1+mm/2, 0.5+mm/2, 1.5+mm, y);
        double appr= multh0*(multh1-log(1-y));
        yval[i] = y;
        fval[i] = val-appr;
    }
    yval.back() = 1.;
    fval.back() = 0.;
    spl = math::CubicSpline(yval, fval);
    multfactor = math::gamma(mm+1.) / math::gamma(mm+1.5) * M_SQRTPI;
}

double BesselIntegral::hypergeom(double x) const
{
    assert(x>0 && x<1);
    //math::hypergeom2F1(1+(m-0.5)/2, 0.5+(m-0.5)/2, 1.5+(m-0.5), x);
    return spl(x) + multh0*(multh1-log(1-x));
}

double BesselIntegral::value(double a, double b, double c) const
{
    assert(a>=0 && b>=0);
    if(fabs(a)<1e-10 || fabs(b)<1e-10)
        return m==0 ? 1/sqrt(a*a + b*b + c*c) : 0;
    else {
        /// Legendre function Q(m,x) expressed through Gauss hypergeometric function 
        double x = (a*a+b*b+c*c)/(2*a*b);
        double val = hypergeom(1/(x*x));
        return val * math::powInt(2*x, -m) / M_PI * multfactor / sqrt(2*x*a*b);
    }
}

//-------- Auxiliary direct-integration potential --------//

/** Direct computation of potential for any density profile, using double integration over space.
    Not suitable for orbit integration, as it does not provide expressions for forces;
    only used for computing potential harmonics (coefficients of Fourier expansion in 
    azimuthal angle phi) at any point in (R,z) plane, for initializing the Cylindrical Spline 
    potential approximation. 
    It can be used in either of the two modes: 
    1) with a smooth input density model, which may or may not be axisymmetric itself;
       in the latter case an intermediate representation of its angular(azimuthal) harmonics
       is created and interpolated on a 2d grid covering almost all of (R,z) plane,
       to speed up computation of potential integral.
    2) with a discrete point mass array, in which case the integral is evaluated by summing 
       the contribution from each particle
*/
class DirectPotential: public BasePotentialCyl
{
public:
    /// init potential from analytic mass model
    DirectPotential(const BaseDensity& _density, unsigned int mmax);

    /// init potential from N-body snapshot
    DirectPotential(const particles::PointMassArray<coord::PosCyl>& _points, 
        unsigned int mmax, SymmetryType sym);

    virtual ~DirectPotential() {};
    virtual const char* name() const { return myName(); };
    static const char* myName() { return "Direct"; };
    virtual SymmetryType symmetry() const { return mysymmetry; }

    /// compute m-th azimuthal harmonic of potential
    double Rho_m(double R, double z, int m) const;

    /// return m-th azimuthal harmonic of density, either by interpolating 
    /// the pre-computed 2d spline, or calculating it on-the-fly using computeRho_m()
    double Phi_m(double R, double z, int m) const;

    /// redefine the following two routines to count particles in the point-mass-set regime
    virtual double enclosedMass(const double radius) const;
    virtual double totalMass() const;

private:
    /// input density model (if provided)
    const BaseDensity* density;

    /// input discrete point mass set (if provided)
    const particles::PointMassArray<coord::PosCyl>* points;

    /// symmetry type of the input density model (axisymmetric or not)
    SymmetryType mysymmetry;

    /// interpolating splines for Fourier harmonics Rho_m(R,z), in case that the input density is not axisymmetric
    std::vector<math::CubicSpline2d> splines;

    /// objects that compute a particular integral involving Bessel fncs
    std::vector<BesselIntegral> besselInts;

    virtual void evalCyl(const coord::PosCyl& pos,
        double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const;

    virtual double densityCyl(const coord::PosCyl& pos) const;
};

DirectPotential::DirectPotential(const BaseDensity& _density, unsigned int _mmax) :
    density(&_density), points(NULL), mysymmetry(_density.symmetry())
{
    // initialize approximating spline for faster hypergeometric function evaluation
    for(unsigned int m=0; m<=_mmax; m++)
        besselInts.push_back(BesselIntegral(m));
    if((mysymmetry & ST_AXISYMMETRIC)==ST_AXISYMMETRIC)
        return;  // no further action necessary
    // otherwise prepare interpolating splines in (R,z) for Fourier expansion of density in angle phi
    int mmax=_mmax;
    splines.resize(mmax*2+1);
    // set up reasonable min/max values: if they are inappropriate, it only will slowdown the computation 
    // but not deteriorate its accuracy, because the interpolation is not used outside the grid
    double totalmass = density->totalMass();
    if(!math::isFinite(totalmass))
        throw std::invalid_argument("DirectPotential: source density model has infinite mass");
    double Rmin = getRadiusByMass(*density, totalmass*0.01)*0.02;
    double Rmax = getRadiusByMass(*density, totalmass*0.99)*50.0;
    double delta=0.05;  // relative difference between grid nodes = log(x[n+1]/x[n]) 
    unsigned int numNodes = static_cast<unsigned int>(log(Rmax/Rmin)/delta);
    std::vector<double> grid;
    math::createNonuniformGrid(numNodes, Rmin, Rmax, true, grid);
    std::vector<double> gridz(2*grid.size()-1);
    for(unsigned int i=0; i<grid.size(); i++) {
        gridz[grid.size()-1-i] =-grid[i];
        gridz[grid.size()-1+i] = grid[i];
    }
    math::Matrix<double> values(grid.size(), gridz.size());
    bool zsymmetry = (density->symmetry()&ST_PLANESYM)==ST_PLANESYM;      // whether densities at z and -z are different
    int mmin = (density->symmetry() & ST_PLANESYM)==ST_PLANESYM ? 0 :-1;  // if triaxial symmetry, do not use sine terms which correspond to m<0
    int mstep= (density->symmetry() & ST_PLANESYM)==ST_PLANESYM ? 2 : 1;  // if triaxial symmetry, use only even m
    for(int m=mmax*mmin; m<=mmax; m+=mstep) {
        for(unsigned int iR=0; iR<grid.size(); iR++)
            for(unsigned int iz=0; iz<grid.size(); iz++) {
                double val = computeRho_m(*density, grid[iR], grid[iz], m);
                if(!math::isFinite(val)) {
                    if(iR==0 && iz==0)  // may have a singularity at origin, substitute the infinite density with something reasonable
                        val = std::max<double>(computeRho_m(*density, grid[1], grid[0], m), 
                                               computeRho_m(*density, grid[0], grid[1], m));
                    else val=0;
                }
                values(iR, grid.size()-1+iz) = val;
                if(!zsymmetry && iz>0) {
                    val = computeRho_m(*density, grid[iR], -grid[iz], m);
                    if(!math::isFinite(val)) val=0;  // don't let rubbish in
                }
                values(iR, grid.size()-1-iz) = val;
            }
        splines[mmax+m] = math::CubicSpline2d(grid, gridz, values);
    }
}

DirectPotential::DirectPotential(const particles::PointMassArray<coord::PosCyl>& _points, 
    unsigned int _mmax, SymmetryType sym) :
    density(NULL), points(&_points), mysymmetry(sym) 
{
    if(points->size()==0)
        throw std::invalid_argument("DirectPotential: empty input array of particles");
    for(unsigned int m=0; m<=_mmax; m++)
        besselInts.push_back(BesselIntegral(m));
};

double DirectPotential::totalMass() const
{
    assert((density!=NULL) ^ (points!=NULL));  // either of the two regimes
    if(density!=NULL) 
        return density->totalMass();
    else {
        double mass=0;
        for(particles::PointMassArray<coord::PosCyl>::ArrayType::const_iterator pt=points->data.begin(); 
            pt!=points->data.end(); pt++) 
            mass+=pt->second;
        return mass;
    }
}

double DirectPotential::enclosedMass(const double r) const
{
    assert((density!=NULL) ^ (points!=NULL));  // either of the two regimes
    if(density!=NULL)
        return density->enclosedMass(r);
    else {
        double mass=0;
        for(particles::PointMassArray<coord::PosCyl>::ArrayType::const_iterator pt=points->data.begin(); 
            pt!=points->data.end(); pt++) 
        {
            if(pow_2(pt->first.R)+pow_2(pt->first.z) <= pow_2(r))
                mass+=pt->second;
        }
        return mass;
    }
}

double DirectPotential::densityCyl(const coord::PosCyl& pos) const
{
    assert(density!=NULL);  // not applicable in discrete point set mode
    if(splines.size()==0)   // no interpolation
        return density->density(pos); 
    else {
        double val=0;
        int mmax=splines.size()/2;
        for(int m=-mmax; m<=mmax; m++)
            val += Rho_m(pos.R, pos.z, m) * (m>=0 ? cos(m*pos.phi) : sin(-m*pos.phi));
        return std::max<double>(val, 0);
    }
}

double DirectPotential::Rho_m(double R, double z, int m) const
{
    if(splines.size()==0) {  // source density is axisymmetric
        assert(m==0 && density!=NULL);
        return density->density(coord::PosCyl(R, z, 0));
    }
    size_t mmax=splines.size()/2;
    if(splines[mmax+m].isEmpty())
        return 0;
    if( R<splines[mmax+m].xmin() || R>splines[mmax+m].xmax() || 
        z<splines[mmax+m].ymin() || z>splines[mmax+m].ymax() )
        return computeRho_m(*density, R, z, m);  // outside interpolating grid -- compute directly by integration
    else
        return splines[mmax+m].value(R, z);
}


/// integration for potential computation
class DirectPotentialIntegrand: public math::IFunctionNdim {
public:
    DirectPotentialIntegrand(const DirectPotential& _potential, const BesselIntegral& _besselInt, 
        double _R, double _z, int _m) :
        potential(_potential), besselInt(_besselInt), R(_R), z(_z), m(_m) {};
    // evaluate the function at a given (R1,z1) point (scaled)
    virtual void eval(const double Rz[], double values[]) const
    {
        if(Rz[0]>=1. || Rz[1]>=1.) {  // scaled coords point at infinity
            values[0] = 0;
            return;
        }
        const double R1 = Rz[0]/(1-Rz[0]);  // un-scale input coordinates
        const double z1 = Rz[1]/(1-Rz[1]);
        const double jac = pow_2((1-Rz[0])*(1-Rz[1]));  // jacobian of scaled coord transformation
        double result = 0;
        if(R1!=R || z1!=z)
            result = -2*M_PI*R1 * (m==0 ? 1 : 2) * (
            potential.Rho_m(R1, z1, m) * besselInt.value(R, R1, z-z1) +
            potential.Rho_m(R1,-z1, m) * besselInt.value(R, R1, z+z1) ) / jac;
        values[0] = result;
    }
    virtual unsigned int numVars() const { return 2; }
    virtual unsigned int numValues() const { return 1; }
private:
    const DirectPotential& potential;
    const BesselIntegral& besselInt;
    double R, z;
    int m;
};

double DirectPotential::Phi_m(double R, double Z, int m) const
{
    if(density==NULL) {  // invoked in the discrete point set mode
        assert(points->size()>0);
        double val=0;
        for(particles::PointMassArray<coord::PosCyl>::ArrayType::const_iterator pt=points->data.begin(); 
            pt!=points->data.end(); pt++) 
        {
            const coord::PosCyl& pc = pt->first;
            double val1 = besselInts[math::abs(m)].value(R, pc.R, Z-pc.z);
            if((mysymmetry & ST_PLANESYM)==ST_PLANESYM)   // add symmetric contribution from -Z
                val1 = (val1 + besselInts[math::abs(m)].value(R, pc.R, Z+pc.z))/2.;
            if(math::isFinite(val1))
                val += pt->second * val1 * (m==0 ? 1 : m>0 ? 2*cos(m*pc.phi) : 2*sin(-m*pc.phi) );
        }
        return -val;
    }
    // otherwise invoked in the smooth density profile mode
    int mmax = splines.size()/2;
    if(splines.size()>0 && splines[mmax+m].isEmpty())
        return 0;  // using splines for m-components of density but it is identically zero at this m
    DirectPotentialIntegrand fnc(*this, besselInts[math::abs(m)], R, Z, m);
    double Rzmin[2]={0.,0.}, Rzmax[2]={1.,1.}; // integration box in scaled coords
    double result, error;
    int numEval;
    math::integrateNdim(fnc, Rzmin, Rzmax, 
        EPSREL_POTENTIAL_INT, MAX_NUM_EVAL, &result, &error, &numEval);
    return result;
};

void DirectPotential::evalCyl(const coord::PosCyl& pos,
    double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const
{
    if(deriv!=NULL || deriv2!=NULL)
        throw std::invalid_argument("DirectPotential: derivatives not implemented");
    assert(potential!=NULL);
    *potential = 0;
    int mmax=splines.size()/2;
    for(int m=-mmax; m<=mmax; m++)
        *potential += Phi_m(pos.R, pos.z, m) * (m>=0 ? cos(m*pos.phi) : sin(-m*pos.phi));
}

//----------------------------------------------------------------------------//
// Cylindrical spline potential 

CylSplineExp::CylSplineExp(unsigned int Ncoefs_R, unsigned int Ncoefs_z, unsigned int Ncoefs_phi, 
    const BaseDensity& srcdensity, 
    double radius_min, double radius_max, double z_min, double z_max)
{
    DirectPotential pot_tmp(srcdensity, Ncoefs_phi);
    initPot(Ncoefs_R, Ncoefs_z, Ncoefs_phi, pot_tmp, radius_min, radius_max, z_min, z_max);
}

CylSplineExp::CylSplineExp(unsigned int Ncoefs_R, unsigned int Ncoefs_z, unsigned int Ncoefs_phi, 
    const BasePotential& potential, 
    double radius_min, double radius_max, double z_min, double z_max)
{
    initPot(Ncoefs_R, Ncoefs_z, Ncoefs_phi, potential, radius_min, radius_max, z_min, z_max);
}

CylSplineExp::CylSplineExp(unsigned int Ncoefs_R, unsigned int Ncoefs_z, unsigned int Ncoefs_phi, 
    const particles::PointMassArray<coord::PosCyl>& points, SymmetryType _sym, 
    double radius_min, double radius_max, double z_min, double z_max)
{
    mysymmetry=_sym;
    if(Ncoefs_phi==0)
        mysymmetry=(SymmetryType)(mysymmetry | ST_ZROTSYM);
    DirectPotential pot_tmp(points, Ncoefs_phi, mysymmetry);
    initPot(Ncoefs_R, Ncoefs_z, Ncoefs_phi, pot_tmp, radius_min, radius_max, z_min, z_max);
}

CylSplineExp::CylSplineExp(const std::vector<double> &gridR, const std::vector<double>& gridz, 
    const std::vector< std::vector<double> > &coefs)
{
    if( gridR.size()<CYLSPLINE_MIN_GRID_SIZE || gridz.size()<CYLSPLINE_MIN_GRID_SIZE || 
        gridR.size()>CYLSPLINE_MAX_GRID_SIZE || gridz.size()>CYLSPLINE_MAX_GRID_SIZE ||
        coefs.size()==0 || coefs.size()%2!=1 || 
        coefs[coefs.size()/2].size()!=gridR.size()*gridz.size()) {
        throw std::invalid_argument("CylSplineExp: Invalid parameters in the constructor");
    } else {
        grid_R=gridR;
        grid_z=gridz;
        initSplines(coefs);
        // check symmetry
        int mmax=static_cast<int>(splines.size()/2);
        mysymmetry=mmax==0 ? ST_AXISYMMETRIC : ST_TRIAXIAL;
        for(int m=-mmax; m<=mmax; m++)
            if(!splines[m+mmax].isEmpty()) {
                if(m<0 || (m>0 && m%2==1))
                    mysymmetry = ST_NONE;//(SymmetryType)(mysymmetry & ~ST_PLANESYM & ~ST_ZROTSYM);
            }
    }
}

class PotentialAzimuthalAverageIntegrand: public math::IFunctionNoDeriv {
public:
    PotentialAzimuthalAverageIntegrand(const BasePotential& _pot, double _R, double _z, int _m) :
    pot(_pot), R(_R), z(_z), m(_m) {};
    virtual double value(double phi) const {
        double val;
        pot.eval(coord::PosCyl(R, z, phi), &val);
        return val * (m==0 ? 1 : m>0 ? 2*cos(m*phi) : 2*sin(-m*phi));
    }
private:
    const BasePotential& pot;
    double R, z, m;
};

double CylSplineExp::computePhi_m(double R, double z, int m, const BasePotential& potential) const
{
    if(potential.name()==DirectPotential::myName()) {
        return dynamic_cast<const DirectPotential&>(potential).Phi_m(R, z, m);
    } else {  // compute azimuthal Fourier harmonic coefficient for the given m by averaging the input potential over phi
        if(R==0 && m!=0) return 0;
        double phimax=(potential.symmetry() & ST_PLANESYM)==ST_PLANESYM ? M_PI_2 : 2*M_PI;
        return math::integrate(PotentialAzimuthalAverageIntegrand(potential, R, z, m),
            0, phimax, EPSREL_POTENTIAL_INT) / phimax;
    }
}

void CylSplineExp::initPot(unsigned int _Ncoefs_R, unsigned int _Ncoefs_z, unsigned int _Ncoefs_phi, 
    const BasePotential& potential, double radius_min, double radius_max, double z_min, double z_max)
{
    if( _Ncoefs_R<CYLSPLINE_MIN_GRID_SIZE || _Ncoefs_z<CYLSPLINE_MIN_GRID_SIZE || 
        _Ncoefs_R>CYLSPLINE_MAX_GRID_SIZE || _Ncoefs_z>CYLSPLINE_MAX_GRID_SIZE || 
        _Ncoefs_phi>CYLSPLINE_MAX_ANGULAR_HARMONIC)
        throw std::invalid_argument("CylSplineExp: invalid grid size");
    mysymmetry = potential.symmetry();
    bool zsymmetry= (mysymmetry & ST_PLANESYM)==ST_PLANESYM;  // whether we need to compute potential at z<0 independently from z>0
    int mmax = (mysymmetry & ST_AXISYMMETRIC) == ST_AXISYMMETRIC ? 0 : _Ncoefs_phi;
    int mmin = (mysymmetry & ST_PLANESYM)==ST_PLANESYM ? 0 :-1;  // if triaxial symmetry, do not use sine terms which correspond to m<0
    int mstep= (mysymmetry & ST_PLANESYM)==ST_PLANESYM ? 2 : 1;  // if triaxial symmetry, use only even m
    if(radius_max==0 || radius_min==0) {
        double totalmass = potential.totalMass();
        if(!math::isFinite(totalmass))
            throw std::invalid_argument("CylSplineExp: source density model has infinite mass");
        if(radius_max==0)
            radius_max = getRadiusByMass(potential, totalmass*(1-1.0/(_Ncoefs_R*_Ncoefs_z)));
        if(!math::isFinite(radius_max)) 
            throw std::runtime_error("CylSplineExp: cannot determine outer radius for the grid");
        if(radius_min==0) 
            radius_min = std::min<double>(radius_max/_Ncoefs_R, 
                getRadiusByMass(potential, totalmass/(_Ncoefs_R*_Ncoefs_z)));
        if(!math::isFinite(radius_min)) 
            //radius_min = radius_max/_Ncoefs_R;
            throw std::runtime_error("CylSplineExp: cannot determine inner radius for the grid");
    }
    std::vector<double> splineRad;
    math::createNonuniformGrid(_Ncoefs_R, radius_min, radius_max, true, splineRad);
    grid_R = splineRad;
    if(z_max==0) z_max=radius_max;
    if(z_min==0) z_min=radius_min;
    z_min = std::min<double>(z_min, z_max/_Ncoefs_z);
    math::createNonuniformGrid(_Ncoefs_z, z_min, z_max, true, splineRad);
    grid_z.assign(2*_Ncoefs_z-1,0);
    for(size_t i=0; i<_Ncoefs_z; i++) {
        grid_z[_Ncoefs_z-1-i] =-splineRad[i];
        grid_z[_Ncoefs_z-1+i] = splineRad[i];
    }
    size_t Ncoefs_R=grid_R.size();
    size_t Ncoefs_z=grid_z.size();
    std::vector< std::vector<double> > coefs(2*mmax+1);
    for(int m=mmax*mmin; m<=mmax; m+=mstep) {
        coefs[mmax+m].assign(Ncoefs_R*Ncoefs_z,0);
    }
    // for some unknown reason, switching on the OpenMP parallelization makes the results 
    // of computation of m=4 coef irreproducible (adds a negligible random error). 
    // Assumed to be unimportant, thus OpenMP is enabled...
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int iR=0; iR<static_cast<int>(Ncoefs_R); iR++) {
        for(size_t iz=0; iz<=Ncoefs_z/2; iz++) {
            for(int m=mmax*mmin; m<=mmax; m+=mstep) {
                double val=computePhi_m(grid_R[iR], grid_z[Ncoefs_z/2+iz], m, potential);
                if(!math::isFinite(val)) {
                    throw std::runtime_error("CylSplineExp: error in computing potential at R=" +
                        utils::convertToString(grid_R[iR]) + ", z=" + 
                        utils::convertToString(grid_z[iz+Ncoefs_z/2])+", m=" + utils::convertToString(m));
                }
                coefs[mmax+m][(Ncoefs_z/2+iz)*Ncoefs_R+iR] = val;
                if(!zsymmetry && iz>0)   // no symmetry about x-y plane
                    val=computePhi_m(grid_R[iR], grid_z[Ncoefs_z/2-iz], m, potential);  // compute potential at -z independently
                coefs[mmax+m][(Ncoefs_z/2-iz)*Ncoefs_R+iR] = val;
            }
        }
    }
    initSplines(coefs);
}

void CylSplineExp::initSplines(const std::vector< std::vector<double> > &coefs)
{
    size_t Ncoefs_R=grid_R.size();
    size_t Ncoefs_z=grid_z.size();
    int mmax=coefs.size()/2;
    assert(coefs[mmax].size()==Ncoefs_R*Ncoefs_z);  // check that at least m=0 coefficients are present
    assert(Ncoefs_R>=CYLSPLINE_MIN_GRID_SIZE && Ncoefs_z>=CYLSPLINE_MIN_GRID_SIZE);
    // compute multipole coefficients for extrapolating the potential and forces beyond the grid,
    // by fitting them to the potential at the grid boundary
    C00=C20=C22=C40=0;
    bool fitm2=mmax>=2 && coefs[mmax+2].size()==Ncoefs_R*Ncoefs_z;  // whether to fit m=2
    size_t npointsboundary=2*(Ncoefs_R-1)+Ncoefs_z;
    math::Matrix<double> X0(npointsboundary, 3); // matrix of coefficients  for m=0
    std::vector<double> Y0(npointsboundary);     // vector of r.h.s. values for m=0
    std::vector<double> W0(npointsboundary);     // vector of weights
    std::vector<double> X2(npointsboundary);     // vector of coefficients  for m=2
    std::vector<double> Y2(npointsboundary);     // vector of r.h.s. values for m=2
    for(size_t i=0; i<npointsboundary; i++) {
        size_t iR=i<2*Ncoefs_R ? i/2 : Ncoefs_R-1;
        size_t iz=i<2*Ncoefs_R ? (i%2)*(Ncoefs_z-1) : i-2*Ncoefs_R+1;
        double R=grid_R[iR];
        double z=grid_z[iz];
        double oneoverr=1/sqrt(R*R+z*z);
        Y0[i] = coefs[mmax][iz*Ncoefs_R+iR];
        X0(i, 0) = oneoverr;
        X0(i, 1) = pow(oneoverr,5.0) * (2*z*z-R*R);
        X0(i, 2) = pow(oneoverr,9.0) * (8*pow(z,4.0)-24*z*z*R*R+3*pow(R,4.0));
        // weight proportionally to the value of potential itself (so that we minimize sum of squares of relative differences)
        W0[i] = 1.0/pow_2(coefs[mmax][iz*Ncoefs_R+iR]);
        if(fitm2) {
            X2[i] = R*R*pow(oneoverr,5.0);
            Y2[i] = coefs[mmax+2][iz*Ncoefs_R+iR];
        }
    }
    // fit m=0 by three parameters
    std::vector<double> fit;
    math::linearMultiFit(X0, Y0, &W0, fit);
    C00 = fit[0];  // C00 ~= -Mtotal
    C20 = fit[1];
    C40 = fit[2];
    // fit m=2 if necessary
    if(fitm2)
        C22 = math::linearFitZero(X2, Y2, NULL);
    // assign Rscale so that it approximately equals -Mtotal/Phi(r=0)
    Rscale=C00/coefs[mmax][(Ncoefs_z/2)*Ncoefs_R];
    if(Rscale<=0 || !math::isFinite(Rscale+C00+C20+C40+C22))
        throw std::runtime_error("CylSplineExp: cannot determine scaling factor");
        //Rscale=std::min<double>(grid_R.back(), grid_z.back())*0.5; // shouldn't occur?
#ifdef DEBUGPRINT
    my_message(FUNCNAME,  "Rscale="+convertToString(Rscale)+", C00="+convertToString(C00)+", C20="+convertToString(C20)+", C22="+convertToString(C22)+", C40="+convertToString(C40));
#endif

    std::vector<double> grid_Rscaled(Ncoefs_R);
    std::vector<double> grid_zscaled(Ncoefs_z);
    for(size_t i=0; i<Ncoefs_R; i++) {
        grid_Rscaled[i] = log(1+grid_R[i]/Rscale);
    }
    for(size_t i=0; i<Ncoefs_z; i++) {
        grid_zscaled[i] = log(1+fabs(grid_z[i])/Rscale)*(grid_z[i]>=0?1:-1);
    }
    splines.resize(coefs.size());
    math::Matrix<double> values(Ncoefs_R, Ncoefs_z);
    for(size_t m=0; m<coefs.size(); m++) {
        if(coefs[m].size() != Ncoefs_R*Ncoefs_z) 
            continue;
        bool allzero=true;
        for(size_t iR=0; iR<Ncoefs_R; iR++) {
            for(size_t iz=0; iz<Ncoefs_z; iz++) {
                double scaling = sqrt(pow_2(Rscale)+pow_2(grid_R[iR])+pow_2(grid_z[iz]));
                double val = coefs[m][iz*Ncoefs_R+iR] * scaling;
                values(iR, iz) = val;
                allzero &= (val==0);
            }
        }
        if(!allzero)
            splines[m] = math::CubicSpline2d(grid_Rscaled, grid_zscaled, values, 0, NAN, NAN, NAN);  // specify derivative at R=0 to be zero
    }
}

void CylSplineExp::getCoefs(std::vector<double>& gridR, std::vector<double>& gridz, std::vector< std::vector<double> >& coefs) const
{
    gridR = grid_R;
    gridz = grid_z;
    coefs.resize(splines.size());
    for(size_t m=0; m<splines.size(); m++)
        if(!splines[m].isEmpty())
            coefs[m].assign(grid_z.size()*grid_R.size(), 0);
    for(size_t iz=0; iz<grid_z.size(); iz++)
        for(size_t iR=0; iR<grid_R.size(); iR++) {
            double Rscaled = log(1 + grid_R[iR] / Rscale);
            double zscaled = log(1 + fabs(grid_z[iz]) / Rscale) * (grid_z[iz]>=0?1:-1);
            for(size_t m=0; m<splines.size(); m++)
                if(!splines[m].isEmpty()) {
                    double scaling = sqrt(pow_2(Rscale)+pow_2(grid_R[iR])+pow_2(grid_z[iz]));
                    coefs[m][iz*grid_R.size()+iR] = splines[m].value(Rscaled, zscaled) / scaling;
                }
        }
}

void CylSplineExp::evalCyl(const coord::PosCyl &pos,
    double* potential, coord::GradCyl* grad, coord::HessCyl* hess) const
{
    if(pos.R>=grid_R.back() || fabs(pos.z)>=grid_z.back()) 
    {   // fallback mechanism for extrapolation beyond grid definition region
        double Z2 = pow_2(pos.z), R2 = pow_2(pos.R), r2 = R2+Z2;
        double R2r2 = R2/r2, Z2r2 = Z2/r2;
        double cos2phi = cos(2*pos.phi);
        double sin2phi = sin(2*pos.phi);
        double oneoverr3 = 1 / (r2 * sqrt(r2));
        double mulC2 = (2 * Z2r2 - R2r2) * C20 + R2r2 * cos2phi * C22;
        double mulC4 = (35 * pow_2(Z2r2) - 30 * Z2r2 + 3) / r2 * C40;
        if(potential!=NULL)
            *potential = (C00 * r2 + mulC2 + mulC4) * oneoverr3;
        if(grad!=NULL) {
            double commonTerm = C00 + (5 * mulC2 + 9 * mulC4) / r2;
            grad->dR   = -pos.R * oneoverr3 *
                (commonTerm + (2*C20 - 2*C22*cos2phi + C40*(60*Z2r2-12)/r2) / r2);
            grad->dz   = -pos.z * oneoverr3 *
                (commonTerm + (-4*C20 + C40*(48-80*Z2r2)/r2) / r2);
            grad->dphi = -2*sin2phi * R2r2 * oneoverr3 * C22;
        }
        if(hess!=NULL) {
            double oneoverr4 = 1/pow_2(r2);
            double commonTerm1 = 3 * C00 + (35 * mulC2 + 99 * mulC4) / r2;
            double commonTerm2 = C00 + 5 * mulC2 / r2;
            hess->dR2  = oneoverr3 * ( commonTerm1 * R2r2 - commonTerm2
                + (20 * R2r2 - 2) * (C20 - C22 * cos2phi) / r2 
                + (-207 * pow_2(R2r2) + 1068 * R2r2 * Z2r2 - 120 * pow_2(Z2r2) ) * C40 * oneoverr4 );
            hess->dz2  = oneoverr3 * ( commonTerm1 * Z2r2 - commonTerm2
                + (-40 * Z2r2 + 4) * C20 / r2 
                + (-75 * pow_2(R2r2) + 1128 * R2r2 * Z2r2 - 552 * pow_2(Z2r2) ) * C40 * oneoverr4 );
            hess->dRdz = oneoverr3 * pos.R * pos.z / r2 * ( commonTerm1
                - 10 * (C20 + C22 * cos2phi) / r2
                + (228 * R2r2 + 48 * Z2r2) * C40 * oneoverr4 );
            double commonTerm3 = oneoverr3 / r2 * sin2phi * C22;
            hess->dRdphi = pos.R * commonTerm3 * (10 * R2r2 - 4);
            hess->dzdphi = pos.z * commonTerm3 *  10 * R2r2;
            hess->dphi2  = -4 * oneoverr3 * R2r2 * cos2phi * C22;
        }
        return;
    }
    double Rscaled = log(1+pos.R/Rscale);
    double zscaled = log(1+fabs(pos.z)/Rscale)*(pos.z>=0?1:-1);
    double Phi_tot = 0;
    coord::GradCyl sGrad;   // gradient in scaled coordinates
    sGrad.dR = sGrad.dz = sGrad.dphi = 0;
    coord::HessCyl sHess;   // hessian in scaled coordinates
    sHess.dR2 = sHess.dz2 = sHess.dphi2 = sHess.dRdz = sHess.dRdphi = sHess.dzdphi = 0;
    int mmax = splines.size()/2;
    for(int m=-mmax; m<=mmax; m++)
        if(!splines[m+mmax].isEmpty()) {
            double cosmphi = m>=0 ? cos(m*pos.phi) : sin(-m*pos.phi);
            double sinmphi = m>=0 ? sin(m*pos.phi) : cos(-m*pos.phi);
            double Phi_m, dPhi_m_dRscaled, dPhi_m_dzscaled, d2Phi_m_dRscaled2, d2Phi_m_dRscaleddzscaled, d2Phi_m_dzscaled2;
            splines[m+mmax].evalDeriv(Rscaled, zscaled, &Phi_m, 
                &dPhi_m_dRscaled, &dPhi_m_dzscaled, &d2Phi_m_dRscaled2, &d2Phi_m_dRscaleddzscaled, &d2Phi_m_dzscaled2);
            Phi_tot += Phi_m*cosmphi;
            if(grad!=NULL || hess!=NULL) {
                sGrad.dR   += dPhi_m_dRscaled*cosmphi;
                sGrad.dz   += dPhi_m_dzscaled*cosmphi;
                sGrad.dphi += Phi_m * -m*sinmphi;
            }
            if(hess!=NULL) {
                sHess.dR2    += d2Phi_m_dRscaled2 * cosmphi;
                sHess.dz2    += d2Phi_m_dzscaled2 * cosmphi;
                sHess.dphi2  += Phi_m * -m*m*cosmphi;
                sHess.dRdz   += d2Phi_m_dRscaleddzscaled * cosmphi;
                sHess.dRdphi += dPhi_m_dRscaled* -m*sinmphi;
                sHess.dzdphi += dPhi_m_dzscaled* -m*sinmphi;
            }
        }
    /*if(pos.z==0 && (mysymmetry & ST_PLANESYM)==ST_PLANESYM) { // symmetric about z -> -z
        sGrad.dz=0;
        sHess.dzdphi=0;
        sHess.dRdz=0;
    }*/
    double r2 = pow_2(pos.R) + pow_2(pos.z);
    double S  = 1/sqrt(pow_2(Rscale)+r2);  // scaling
    double dSdr_over_r = -S*S*S;
    double dRscaleddR = 1/(Rscale+pos.R);
    double dzscaleddz = 1/(Rscale+fabs(pos.z));
    if(potential!=NULL)
        *potential = S * Phi_tot;
    if(grad!=NULL) {
        grad->dR   = S * sGrad.dR * dRscaleddR + Phi_tot * pos.R * dSdr_over_r;
        grad->dz   = S * sGrad.dz * dzscaleddz + Phi_tot * pos.z * dSdr_over_r;
        grad->dphi = S * sGrad.dphi;
    }
    if(hess!=NULL)
    {
        double d2RscaleddR2 = -pow_2(dRscaleddR);
        double d2zscaleddz2 = -pow_2(dzscaleddz) * (pos.z>=0?1:-1);
        double d2Sdr2 = (pow_2(Rscale) - 2 * r2) * dSdr_over_r * S * S;
        hess->dR2 =
            (pow_2(pos.R) * d2Sdr2 + pow_2(pos.z) * dSdr_over_r) / r2 * Phi_tot + 
            dSdr_over_r * 2 * pos.R * dRscaleddR * sGrad.dR +
            S * (sHess.dR2*pow_2(dRscaleddR) + sGrad.dR*d2RscaleddR2);
        hess->dz2 =
            (pow_2(pos.z) * d2Sdr2 + pow_2(pos.R) * dSdr_over_r) / r2 * Phi_tot +
            dSdr_over_r * 2 * pos.z * dzscaleddz * sGrad.dz +
            S * (sHess.dz2 * pow_2(dzscaleddz) + sGrad.dz * d2zscaleddz2);
        hess->dRdz =
            (d2Sdr2 - dSdr_over_r) * pos.R * pos.z / r2 * Phi_tot +
            dSdr_over_r * (pos.z * dRscaleddR * sGrad.dR + pos.R * dzscaleddz * sGrad.dz) +
            S * sHess.dRdz * dRscaleddR * dzscaleddz;
        hess->dRdphi =
            dSdr_over_r * pos.R * sGrad.dphi +
            S * dRscaleddR * sHess.dRdphi;
        hess->dzdphi =
            dSdr_over_r * pos.z * sGrad.dphi +
            S * dzscaleddz * sHess.dzdphi;
        hess->dphi2 = S * sHess.dphi2;
    }
}

}; // namespace
