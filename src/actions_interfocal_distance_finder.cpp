#include "actions_interfocal_distance_finder.h"
#include "math_core.h"
#include "math_ode.h"
#include <cassert>
#include <stdexcept>
#include <cmath>

namespace actions{

/// number of sampling points for a shell orbit (equally spaced in time)
static const unsigned int NUM_STEPS_TRAJ = 16;
/// accuracy of root-finding and orbit integration for the functions in this module
static const double ACCURACY = 1e-6;
/// upper limit on the number of timesteps in ODE solver (should be enough to track half of the orbit)
static const unsigned int MAX_NUM_STEPS_ODE = 100;


// estimate IFD for a series of points in R-z plane
template<typename PointT>
double estimateInterfocalDistancePoints(
    const potential::BasePotential& potential, const std::vector<PointT>& traj)
{
    if(traj.size()==0)
        throw std::invalid_argument("Error in finding interfocal distance: empty array of points");
    std::vector<double> x(traj.size()), y(traj.size());
    double sumsq = 0;
    double minr2 = INFINITY;
    for(unsigned int i=0; i<traj.size(); i++) {
        const coord::PosCyl p = coord::toPosCyl(traj[i]);
        minr2 = fmin(minr2, p.R*p.R+p.z*p.z);
        coord::GradCyl grad;
        coord::HessCyl hess;
        potential.eval(p, NULL, &grad, &hess);
        x[i] = hess.dRdz;
        y[i] = 3*p.z * grad.dR - 3*p.R * grad.dz + p.R*p.z * (hess.dR2-hess.dz2)
             + (p.z*p.z - p.R*p.R) * hess.dRdz;
        sumsq += pow_2(x[i]);
    }
    double result = sumsq>0 ? math::linearFitZero(x, y, NULL) : 0;
    return sqrt( fmax( result, minr2*1e-4) );  // ensure that the computed value is positive
}

template double estimateInterfocalDistancePoints(
    const potential::BasePotential& potential, const std::vector<coord::PosCar>& traj);
template double estimateInterfocalDistancePoints(
    const potential::BasePotential& potential, const std::vector<coord::PosVelCar>& traj);
template double estimateInterfocalDistancePoints(
    const potential::BasePotential& potential, const std::vector<coord::PosCyl>& traj);
template double estimateInterfocalDistancePoints(
    const potential::BasePotential& potential, const std::vector<coord::PosVelCyl>& traj);
template double estimateInterfocalDistancePoints(
    const potential::BasePotential& potential, const std::vector<coord::PosSph>& traj);
template double estimateInterfocalDistancePoints(
    const potential::BasePotential& potential, const std::vector<coord::PosVelSph>& traj);


/** find the best-fit value of interfocal distance for a shell orbit.
    \param[in] traj  contains the trajectory of this orbit in R-z plane,
    \return  the parameter of a prolate spheroidal coordinate system which minimizes
    the variation of `lambda` coordinate for this orbit.
    If the best-fit value is negative, it is replaced with a small positive quantity.
*/
static double fitInterfocalDistanceShellOrbit(const std::vector<coord::PosCyl>& traj)
{
    if(traj.size()==0)
        throw std::invalid_argument("Error in finding interfocal distance for a shell orbit: empty array");
    math::Matrix<double> coefs(traj.size(), 2);
    std::vector<double> rhs(traj.size());
    std::vector<double> result;  // regression parameters:  lambda/(lambda-delta), lambda
    double minr2 = INFINITY;
    for(unsigned int i=0; i<traj.size(); i++) {
        minr2 = fmin(minr2, pow_2(traj[i].R) + pow_2(traj[i].z));
        coefs(i, 0) = -pow_2(traj[i].R);
        coefs(i, 1) = 1.;
        rhs[i] = pow_2(traj[i].z);
    }
    math::linearMultiFit(coefs, rhs, NULL, result);
    return sqrt( fmax( result[1] * (1 - 1/result[0]), minr2*1e-4) );
}


/** Helper function for finding the roots of (effective) potential in either R or z direction */
class OrbitSizeFunction: public math::IFunction {
public:
    const potential::BasePotential& potential;
    double R;
    double phi;
    double E;
    double Lz2;
    enum { FIND_RMIN, FIND_RMAX, FIND_ZMAX, FIND_JR, FIND_JZ } mode;
    explicit OrbitSizeFunction(const potential::BasePotential& p) : potential(p), mode(FIND_RMAX) {};
    virtual unsigned int numDerivs() const { return 2; }
    /** This function is used in the root-finder to determine the turnaround points of an orbit:
        in the radial direction, it returns -(1/2) v_R^2, and in the vertical direction -(1/2) v_z^2 .
        Moreover, to compute the location of pericenter this is multiplied by R^2 to curb the sharp rise 
        of effective potential at zero, which is problematic for root-finder. */
    virtual void evalDeriv(const double x, 
        double* val=0, double* deriv=0, double* deriv2=0) const
    {
        double Phi=0;
        coord::GradCyl grad;
        coord::HessCyl hess;
        if(math::isFinite(x)) {
            if(mode == FIND_ZMAX)
                potential.eval(coord::PosCyl(R, x, phi), &Phi, deriv? &grad : NULL, deriv2? &hess: NULL);
            else
                potential.eval(coord::PosCyl(x, 0, phi), &Phi, deriv? &grad : NULL, deriv2? &hess: NULL);
        } else {
            if(deriv) 
                grad.dR = NAN;
            if(deriv2)
                hess.dR2 = NAN;
        }
        double result = E-Phi;
        if(mode == FIND_RMIN) {    // f(R) = (1/2) v_R^2 * R^2
            result = result*x*x - Lz2/2;
            if(deriv) 
                *deriv = 2*x*(E-Phi) - x*x*grad.dR;
            if(deriv2)
                *deriv2 = 2*(E-Phi) - 4*x*grad.dR - x*x*hess.dR2;
        } else if(mode == FIND_RMAX) {  // f(R) = (1/2) v_R^2 = E - Phi(R) - Lz^2/(2 R^2)
            if(Lz2>0)
                result -= Lz2/(2*x*x);
            if(deriv)
                *deriv = -grad.dR + (Lz2>0 ? Lz2/(x*x*x) : 0);
            if(deriv2)
                *deriv2 = -hess.dR2 - (Lz2>0 ? 3*Lz2/(x*x*x*x) : 0);
        } else if(mode == FIND_ZMAX) {  // f(z) = (1/2) v_z^2
            if(deriv)
                *deriv = -grad.dz;
            if(deriv2)
                *deriv2= -hess.dz2;
        } else if(mode == FIND_JR) {  // f(R) = v_R
            result = sqrt(fmax(0, 2*result - (Lz2>0 ? Lz2/(x*x) : 0) ) );
        } else if(mode == FIND_JZ) {  // f(R) = v_z
            result = sqrt(fmax(0, 2*result) );
        } else
            assert("Invalid operation mode in OrbitSizeFunction"==0);
        if(val)
            *val = result;
    }
};

void findPlanarOrbitExtent(const potential::BasePotential& poten, double E, double Lz, 
    double& Rmin, double& Rmax, double* Jr)
{
    OrbitSizeFunction fnc(poten);
    fnc.Lz2 = Lz*Lz;
    fnc.R   = R_from_Lz(poten, Lz);
    fnc.phi = 0;
    fnc.E   = E;
    math::PointNeighborhood nh(fnc, fnc.R);
    double dR_to_zero = nh.dxToNearestRoot();
    Rmin = Rmax = fnc.R;
    double maxPeri = fnc.R, minApo = fnc.R;    // endpoints of interval for locating peri/apocenter radii
    if(fabs(dR_to_zero) < fnc.R*ACCURACY) {    // we are already near peri- or apocenter radius
        if(dR_to_zero > 0) {
            minApo  = NAN;
            maxPeri = fnc.R + nh.dxToPositive();
        } else {
            maxPeri = NAN;
            minApo  = fnc.R + nh.dxToPositive();
        }
    }
    if(fnc.Lz2>0) {
        if(math::isFinite(maxPeri)) {
            fnc.mode = OrbitSizeFunction::FIND_RMIN;
            Rmin = math::findRoot(fnc, 0., maxPeri, ACCURACY);
            // ensure that E-Phi(Rmin) >= 0
            // (due to finite accuracy in root-finding, a small adjustment may be needed)
            Rmin += math::PointNeighborhood(fnc, Rmin).dxToPositive();
        }
    } else  // angular momentum is zero
        Rmin = 0;
    if(math::isFinite(minApo)) {
        fnc.mode = OrbitSizeFunction::FIND_RMAX;
        Rmax = math::findRoot(fnc, minApo, INFINITY, ACCURACY);
        Rmax += math::PointNeighborhood(fnc, Rmax).dxToPositive();  // ensure that E>=Phi(Rmax)
    }   // else Rmax=absR
    if(Jr!=NULL) {  // compute radial action
        fnc.mode = OrbitSizeFunction::FIND_JR;
        *Jr = math::integrateGL(fnc, Rmin, Rmax, 10) / M_PI;
    }
}


/// function to use in ODE integrator
class OrbitIntegratorMeridionalPlane: public math::IOdeSystem {
public:
    OrbitIntegratorMeridionalPlane(const potential::BasePotential& p, double Lz) :
        poten(p), Lz2(Lz*Lz) {};
    
    /** apply the equations of motion in R,z plane without tracking the azimuthal motion */
    virtual void eval(const double /*t*/, const math::OdeStateType& y, math::OdeStateType& dydt) const
    {
        coord::GradCyl grad;
        poten.eval(coord::PosCyl(y[0], y[1], 0), NULL, &grad);
        dydt[0] = y[2];
        dydt[1] = y[3];
        dydt[2] = -grad.dR + (Lz2>0 ? Lz2/pow_3(y[0]) : 0);
        dydt[3] = -grad.dz;
    }
    
    /** return the size of ODE system: R, z, vR, vz */
    virtual unsigned int size() const { return 4;}
private:
    const potential::BasePotential& poten;
    const double Lz2;
};

/// function to use in locating the exact time of the x-y plane crossing
class FindCrossingPointZequal0: public math::IFunction {
public:
    FindCrossingPointZequal0(const math::BaseOdeSolver& _solver) :
        solver(_solver) {};
    /** used in root-finder to locate the root z(t)=0 */
    virtual void evalDeriv(const double time, 
        double* val=0, double* der=0, double* /*der2*/=0) const {
        if(val)
            *val = solver.value(time, 1);  // z
        if(der)
            *der = solver.value(time, 3);  // vz
    }
    virtual unsigned int numDerivs() const { return 1; }
private:
    const math::BaseOdeSolver& solver;
};

/** launch an orbit perpendicularly to x-y plane from radius R with vz>0,
    and record the radius at which it crosses this plane downward (vz<0).
    \param[out] timeCross stores the time required to complete the half-oscillation in z;
    \param[out] traj stores the trajectory recorded at equal intervals of time;
    \param[out] Jz stores the vertical action computed for this trajectory;
    \return  the crossing radius
*/
static double findCrossingPointR(
    const potential::BasePotential& poten, double E, double Lz, double R,
    double* timeCross, std::vector<coord::PosCyl>* traj, double* Jz)
{
    double vz = sqrt(fmax( 2 * (E-poten.value(coord::PosCyl(R, 0, 0))) - (Lz>0 ? pow_2(Lz/R) : 0), R*R*1e-16));
    OrbitIntegratorMeridionalPlane odeSystem(poten, Lz);
    math::OdeStateType vars(odeSystem.size());
    vars[0] = R;
    vars[1] = 0;
    vars[2] = 0;
    vars[3] = vz;
    math::OdeSolverDOP853 solver(odeSystem, 0, ACCURACY);
    solver.init(vars);
    bool finished = false;
    unsigned int numStepsODE = 0;
    double timePrev = 0;
    double timeCurr = 0;
    double timeTraj = 0;
    const double timeStepTraj = timeCross!=NULL ? *timeCross*0.5/(NUM_STEPS_TRAJ-1) : INFINITY;
    if(traj!=NULL)
        traj->clear();
    if(Jz!=NULL)
        *Jz = 0;
    while(!finished) {
        if(solver.step() <= 0 || numStepsODE >= MAX_NUM_STEPS_ODE)  // signal of error
            finished = true;
        else {
            numStepsODE++;
            timePrev = timeCurr;
            timeCurr = solver.getTime();
            if(timeStepTraj!=INFINITY && traj!=NULL)
            {   // store trajectory
                while(timeTraj <= timeCurr && traj->size() < NUM_STEPS_TRAJ) {
                    traj->push_back(coord::PosCyl(  // store R and z at equal intervals of time
                        fabs(solver.value(timeTraj, 0)), solver.value(timeTraj, 1), 0)); 
                    timeTraj += timeStepTraj;
                }
            }
            if(solver.value(timeCurr, 1) < 0) {  // z<0 - we're done
                finished = true;
                timeCurr = math::findRoot(FindCrossingPointZequal0(solver),
                    timePrev, timeCurr, ACCURACY);
            }
            if(Jz!=NULL)
            {   // compute vertical action  (very crude approximation! one integration point per timestep)
                *Jz +=
                 (  solver.value((timePrev+timeCurr)/2, 2) *   // vR at the mid-timestep
                   (solver.value(timeCurr, 0) - solver.value(timePrev, 0))  // delta R over timestep
                  + solver.value((timePrev+timeCurr)/2, 3) *   // vz at the mid-timestep
                   (solver.value(timeCurr, 1) - solver.value(timePrev, 1))  // delta z over timestep
                  ) / M_PI;
            }
        }
    }
    if(timeCross!=NULL)
        *timeCross = timeCurr;
    return fabs(solver.value(timeCurr, 0));   // value of R at the moment of crossing x-y plane
}

/// function to be used in root-finder for locating the thin orbit in R-z plane
class FindClosedOrbitRZplane: public math::IFunctionNoDeriv {
public:
    FindClosedOrbitRZplane(const potential::BasePotential& p, 
        double _E, double _Lz, double _Rmin, double _Rmax,
        double* _timeCross, std::vector<coord::PosCyl>* _traj, double* _Jz) :
        poten(p), E(_E), Lz(_Lz), Rmin(_Rmin), Rmax(_Rmax), 
        timeCross(_timeCross), traj(_traj), Jz(_Jz) {};
    /// report the difference in R between starting point (R, z=0, vz>0) and return point (R1, z=0, vz<0)
    virtual double value(const double R) const {
        // first two calls in root-finder are for the boundary points, we already know the answer
        if(R==Rmin)
            return R-Rmax;
        if(R==Rmax)
            return R-Rmin;
        double R1 = findCrossingPointR(poten, E, Lz, R, timeCross, traj, Jz);
        return R-R1;
    }
private:
    const potential::BasePotential& poten;
    const double E, Lz;               ///< parameters of motion in the R-z plane
    const double Rmin, Rmax;          ///< boundaries of interval in R (to skip the first two calls)
    double* timeCross;                ///< keep track of time required to complete orbit
    std::vector<coord::PosCyl>* traj; ///< store the trajectory
    double* Jz;                       ///< store the estimated value of vertical action
};


double estimateInterfocalDistanceShellOrbit(
    const potential::BasePotential& poten, double E, double Lz, 
    double* R, double* Jz)
{
    double Rmin, Rmax;
    findPlanarOrbitExtent(poten, E, Lz, Rmin, Rmax);
    double timeCross = INFINITY;
    std::vector<coord::PosCyl> traj;
    FindClosedOrbitRZplane fnc(poten, E, Lz, Rmin, Rmax, &timeCross, &traj, Jz);
    // locate the radius of thin orbit;
    // as a by-product, store the orbit in 'traj' and the vertical action in Jz (if necessary)
    double Rthin = math::findRoot(fnc, Rmin, Rmax, ACCURACY);
    if(R!=NULL)
        *R=Rthin;
    if(Rthin!=Rthin || traj.size()==0)
        return Rmin;  // anything
    // now find the best-fit value of delta for this orbit
    return fitInterfocalDistanceShellOrbit(traj);
}


// ----------- Interpolation of interfocal distance in E,Lz plane ------------ //
InterfocalDistanceFinder::InterfocalDistanceFinder(
    const potential::BasePotential& _potential, const unsigned int gridSizeE) :
    potential(_potential)
{
    if((potential.symmetry() & potential::ST_ZROTSYM) != potential::ST_ZROTSYM)
        throw std::invalid_argument("Potential is not axisymmetric, "
            "interfocal distance estimator is not suitable for this case");
    
    if(gridSizeE<10 || gridSizeE>500)
        throw std::invalid_argument("InterfocalDistanceFinder: incorrect grid size");
    
    // find out characteristic energy values
    double E0 = potential.value(coord::PosCar(0, 0, 0));
    double Ehalf = E0*0.5;
    double totalMass = potential.totalMass();
    if(math::isFinite(totalMass)) {
        double halfMassRadius = getRadiusByMass(potential, 0.5*totalMass);
        Ehalf = potential.value(coord::PosCyl(halfMassRadius, 0, 0));
    }
    double Einfinity = 0;
    if((!math::isFinite(E0) && E0!=-INFINITY) || !math::isFinite(Ehalf) || 
        E0>=Ehalf || Ehalf>=Einfinity)
        throw std::runtime_error("InterfocalDistanceFinder: weird behaviour of potential");

    // create a somewhat non-uniform grid in energy
    const double minBin = 0.5/gridSizeE;
    std::vector<double> energyBins;
    math::createNonuniformGrid((gridSizeE+1)/2, minBin, 1-1./gridSizeE, false, energyBins);
    std::vector<double> gridE(gridSizeE);
    for(unsigned int i=0; i<(gridSizeE+1)/2; i++) {
        // inner part of the model
        gridE[i] = math::isFinite(E0) ? E0 + (Ehalf-E0)*energyBins[i] : Ehalf/energyBins[i];
        // outer part of the model
        gridE[gridSizeE-1-i] = Einfinity - (Einfinity-Ehalf)*energyBins[i];
    }

    // fill a 1d interpolator for Lcirc(E)
    Lscale = potential::L_circ(potential, gridE[gridSizeE/2]);
    std::vector<double> gridLcirc(gridSizeE);
    for(unsigned int i=0; i<gridSizeE; i++) {
        double Lc = potential::L_circ(potential, gridE[i]);
        gridLcirc[i] = Lc / (Lc + Lscale);
    }
    xLcirc = math::CubicSpline(gridE, gridLcirc);

    // create a uniform grid in Lz/Lcirc(E)
    const unsigned int gridSizeLzrel = gridSizeE<80 ? gridSizeE/4 : 20;
    std::vector<double> gridLzrel(gridSizeLzrel);
    for(unsigned int i=0; i<gridSizeLzrel; i++)
        gridLzrel[i] = (i+0.01) / (gridSizeLzrel-0.98);

    // fill a 2d grid in (E, Lz/Lcirc(E) )
    math::Matrix<double> grid2d(gridE.size(), gridLzrel.size());
    for(unsigned int iE=0; iE<gridE.size(); iE++) {
        const double x  = xLcirc(gridE[iE]);
        const double Lc = Lscale * x / (1-x);
        for(unsigned int iL=0; iL<gridLzrel.size(); iL++) {
            double Lz = gridLzrel[iL] * Lc;
            grid2d(iE, iL) = estimateInterfocalDistanceShellOrbit(potential, gridE[iE], Lz);
        }
    }
    
    // create a 2d interpolator
    interp = math::LinearInterpolator2d(gridE, gridLzrel, grid2d);
}

double InterfocalDistanceFinder::value(const coord::PosVelCyl& point) const
{
    double E  = totalEnergy(potential, point);
    E = fmin(fmax(E, interp.xmin()), interp.xmax());
    double Lz = fabs(point.R*point.vphi);
    double x  = xLcirc(E);
    double Lc = Lscale * x / (1-x);
    double Lzrel = fmin(fmax(Lz/Lc, interp.ymin()), interp.ymax());
    return interp.value(E, Lzrel);
}

}  // namespace actions
