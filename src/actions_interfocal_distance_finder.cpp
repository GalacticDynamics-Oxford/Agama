#include "actions_interfocal_distance_finder.h"
#include "math_core.h"
#include "math_ode.h"
#include <cassert>
#include <stdexcept>
#include <cmath>

#include <iostream>
#include "utils.h"
#include "actions_staeckel.h"
using utils::pp;

namespace actions{

// ------ Estimation of interfocal distance -------

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

static bool estimateOrbitExtent(const potential::BasePotential& potential, const coord::PosVelCyl& point,
    double& Rmin, double& Rmax, double& zmaxRmin, double& zmaxRmax)
{
    const double toler = 1e-4;  // relative tolerance in root-finder
    double absz = fabs(point.z), absR = fabs(point.R);
    OrbitSizeFunction fnc(potential);
    fnc.Lz2 = pow_2(point.R*point.vphi);
    fnc.R   = absR;
    fnc.phi = point.phi;

    // examine the behavior of effective potential near R_0:
    // call the effective potential function to store the potential and its derivatives,
    // then compute the total energy and subtract it from the preliminary value of the function at R_0.
    // in this way we call the potential evaluation function only once.
    fnc.E   = 0;                                       // assign a temporary value
    math::PointNeighborhood nh(fnc, absR);             // compute the potential and its derivatives at point
    double Phi_R_0 = nh.f0 - 0.5*pow_2(point.vphi);    // nh.f0 contained  Phi(R_0) + Lz^2 / (2 R_0^2)
    fnc.E   = nh.f0 + 0.5*pow_2(point.vR);             // compute the correct value of energy of in-plane motion (excluding v_z)
    nh.f0   = Phi_R_0 - fnc.E + 0.5*pow_2(point.vphi); // and store the correct value of Phi_eff(R_0) - E
    
    // estimate radial extent
    Rmin = Rmax = absR;
    double dR_to_zero = nh.dxToNearestRoot();
    double maxPeri = absR, minApo = absR;  // endpoints of interval for locating peri/apocenter radii
    if(fabs(dR_to_zero) < absR*toler) {    // we are already near peri- or apocenter radius
        if(dR_to_zero > 0) {
            minApo  = NAN;
            maxPeri = absR + nh.dxToPositive();
        } else {
            maxPeri = NAN;
            minApo  = absR + nh.dxToPositive();
        }
    }
    if(fnc.Lz2>0) {
        if(math::isFinite(maxPeri)) {
            fnc.mode = OrbitSizeFunction::FIND_RMIN;
            Rmin = math::findRoot(fnc, 0., maxPeri, toler);
        }
    } else  // angular momentum is zero
        Rmin = 0;
    if(math::isFinite(minApo)) {
        fnc.mode = OrbitSizeFunction::FIND_RMAX;
        Rmax = math::findRoot(fnc, minApo, INFINITY, toler);
    }   // else Rmax=absR

    if(!math::isFinite(Rmin+Rmax))
        return false;  // likely reason: energy is positive

    // estimate vertical extent at R=R_0
    double zmax = absz;
    double Phi_R_z;  // potential at the initial position
    if(point.z != 0)
        potential.eval(point, &Phi_R_z);
    else
        Phi_R_z = Phi_R_0;
    fnc.E = Phi_R_z + pow_2(point.vz)/2;  // "vertical energy"
    if(point.vz != 0) {
        fnc.mode = OrbitSizeFunction::FIND_ZMAX;
        zmax = math::findRoot(fnc, absz, INFINITY, toler);
        if(!math::isFinite(zmax))
            return false;
    }
    zmaxRmin=zmaxRmax=zmax;
    if(zmax>0 && absR>Rmin*1.2) {
        // a first-order correction for vertical extent
        fnc.E -= Phi_R_0;  // energy in vertical oscillation at R_0, equals to Phi(R_0,zmax)-Phi(R_0,0)
        double Phi_Rmin_0, Phi_Rmin_zmax;
        potential.eval(coord::PosCyl(Rmin, 0, point.phi), &Phi_Rmin_0);
        potential.eval(coord::PosCyl(Rmin, zmax, point.phi), &Phi_Rmin_zmax);
        // assuming that the potential varies quadratically with z, estimate corrected zmax at Rmin
        double corr=fnc.E/(Phi_Rmin_zmax-Phi_Rmin_0);
        if(corr>0.1 && corr<10)
            zmaxRmin = zmax*sqrt(corr);
    }
    if(zmax>0 && absR<Rmax*0.8) {
        // same at Rmax
        double Phi_Rmax_0, Phi_Rmax_zmax;
        potential.eval(coord::PosCyl(Rmax, 0, point.phi), &Phi_Rmax_0);
        potential.eval(coord::PosCyl(Rmax, zmax, point.phi), &Phi_Rmax_zmax);
        double corr = fnc.E/(Phi_Rmax_zmax-Phi_Rmax_0);
        if(corr>0.1 && corr<10)
            zmaxRmax = zmax*sqrt(corr);
    }
    return true;
}

// estimate IFD for a series of points in R-z plane
template<typename PointT>
double estimateSquaredInterfocalDistancePoints(
    const potential::BasePotential& potential, const std::vector<PointT>& traj)
{
    std::vector<double> x(traj.size()), y(traj.size());
    double sumsq=0;
    for(unsigned int i=0; i<traj.size(); i++) {
        const coord::PosCyl p = coord::toPosCyl(traj[i]);
        coord::GradCyl grad;
        coord::HessCyl hess;
        potential.eval(p, NULL, &grad, &hess);
        x[i] = hess.dRdz;
        y[i] = 3*p.z * grad.dR - 3*p.R * grad.dz + p.R*p.z * (hess.dR2-hess.dz2)
             + (p.z*p.z - p.R*p.R) * hess.dRdz;
        sumsq += pow_2(x[i]);
    }
    return sumsq>0 ? math::linearFitZero(x, y) : 0;
}

template double estimateSquaredInterfocalDistancePoints(
    const potential::BasePotential& potential, const std::vector<coord::PosCar>& traj);
template double estimateSquaredInterfocalDistancePoints(
    const potential::BasePotential& potential, const std::vector<coord::PosVelCar>& traj);
template double estimateSquaredInterfocalDistancePoints(
    const potential::BasePotential& potential, const std::vector<coord::PosCyl>& traj);
template double estimateSquaredInterfocalDistancePoints(
    const potential::BasePotential& potential, const std::vector<coord::PosVelCyl>& traj);
template double estimateSquaredInterfocalDistancePoints(
    const potential::BasePotential& potential, const std::vector<coord::PosSph>& traj);
template double estimateSquaredInterfocalDistancePoints(
    const potential::BasePotential& potential, const std::vector<coord::PosVelSph>& traj);

template<typename PointT>
double estimateSquaredInterfocalDistanceThinOrbit(const std::vector<PointT>& traj)
{
    math::Matrix<double> coefs(traj.size(), 2);
    std::vector<double> rhs(traj.size());
    std::vector<double> result;  // regression parameters:  lambda/(lambda-delta), lambda
    for(unsigned int i=0; i<traj.size(); i++) {
        coefs(i, 0) = -pow_2(traj[i].R);
        coefs(i, 1) = 1.;
        rhs[i] = pow_2(traj[i].z);
    }
    math::linearMultiFit(coefs, rhs, result);
    return result[1] * (1 - 1/result[0]);
}

static double estimateInterfocalDistance(
    const potential::BasePotential& potential, const coord::PosVelCyl& point)
{
    double R1, R2, z1, z2;
    if(!estimateOrbitExtent(potential, point, R1, R2, z1, z2)) {
        R1=R2=point.R; z1=z2=point.z;
    }
    if(z1+z2<=(R1+R2)*1e-8)   // orbit in x-y plane, any (non-zero) result will go
        return (R1+R2)/2;
    const int nR=4, nz=2, numpoints=nR*nz;
    std::vector<coord::PosCyl> points(numpoints);
    const double r1=sqrt(R1*R1+z1*z1), r2=sqrt(R2*R2+z2*z2);
    const double a1=atan2(z1, R1), a2=atan2(z2, R2);
    for(int iR=0; iR<nR; iR++) {
        double r=r1+(r2-r1)*iR/(nR-1);
        for(int iz=0; iz<nz; iz++) {
            double a=(iz+1.)/nz * (a1+(a2-a1)*iR/(nR-1));
            points[iR*nz+iz] = coord::PosCyl(r*cos(a), r*sin(a), 0);
        }
    }
    double result = estimateSquaredInterfocalDistancePoints(potential, points);
    // ensure that the distance is positive
    return sqrt(fmax(result, fmin(R1*R1+z1*z1, R2*R2+z2*z2)*0.0001) );
}

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

class FindCrossingPointZequal0: public math::IFunction {
public:
    FindCrossingPointZequal0(const math::BaseOdeSolver& _solver) :
        solver(_solver) {};
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

static const unsigned int NUM_STEPS_TRAJ = 16;
static const double ACCURACY = 1e-6;
static const unsigned int MAX_NUM_STEPS_ODE = 100;

/// launch an orbit perpendicularly to x-y plane from radius R with vz>0,
/// and record the radius at which it crosses this plane downward (vz<0)
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
            {   // compute vertical action
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

class FindClosedOrbitRZplane: public math::IFunctionNoDeriv {
public:
    FindClosedOrbitRZplane(const potential::BasePotential& p, 
        double _E, double _Lz, double _Rmin, double _Rmax,
        double* _timeCross, std::vector<coord::PosCyl>* _traj, double* _Jz) :
        poten(p), E(_E), Lz(_Lz), Rmin(_Rmin), Rmax(_Rmax), 
        timeCross(_timeCross), traj(_traj), Jz(_Jz) {};
    /// report the difference in R between starting point (R, z=0, vz>0) and return point (R1, z=0, vz<0)
    virtual double value(const double R) const {
        if(R==Rmin)
            return R-Rmax;
        if(R==Rmax)
            return R-Rmin;
        double R1 = findCrossingPointR(poten, E, Lz, R, timeCross, traj, Jz);
        return R-R1;
    }
private:
    const potential::BasePotential& poten;
    const double E, Lz;
    const double Rmin, Rmax;
    double* timeCross;
    std::vector<coord::PosCyl>* traj;
    double* Jz;
};

void findPeriApocenter(const potential::BasePotential& poten, double E, double Lz, 
    double& Rmin, double& Rmax, double* Jr=0)
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
            // ensure that E-Phi(Rmin) >= 0  (due to finite accuracy in root-finding, a small adjustment may be needed)
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

static double findZmax(const potential::BasePotential& poten, double R, double Jz)
{
    coord::HessCyl hess;
    poten.eval(coord::PosCyl(R, 0, 0), NULL, NULL, &hess);
    return sqrt(fmax( Jz/M_PI/sqrt(hess.dz2), 0) );
}

void findClosedOrbitRZplane(const potential::BasePotential& poten, double E, double Lz, 
    double &Rthin, double& Jr, double& Jz, double &IFD)
{
    double Rmin, Rmax;
    findPeriApocenter(poten, E, Lz, Rmin, Rmax, &Jr);
    double timeCross = INFINITY;
    std::vector<coord::PosCyl> traj;
    FindClosedOrbitRZplane fnc(poten, E, Lz, Rmin, Rmax, &timeCross, &traj, &Jz);
    double R = math::findRoot(fnc, Rmin, Rmax, ACCURACY);
    Rthin = R;
    double vz = sqrt(2 * (E-poten.value(coord::PosCyl(R, 0, 0))) - (Lz>0 ? pow_2(Lz/R) : 0));
    double ifd= estimateSquaredInterfocalDistancePoints(poten, traj);
    double ift= estimateSquaredInterfocalDistanceThinOrbit(traj);
    traj.clear();
    traj.push_back(coord::PosCyl(Rmin, findZmax(poten, Rmin, Jr*1e-4), 0));
    traj.push_back(coord::PosCyl((Rmin+Rmax)/2, findZmax(poten, (Rmin+Rmax)/2, Jr*1e-4), 0));
    traj.push_back(coord::PosCyl(Rmax, findZmax(poten, Rmax, Jr*1e-4), 0));
    double iff= estimateSquaredInterfocalDistancePoints(poten, traj);
    
    /*std::cout << pp(E,7) <<"\t"<< pp(Lz,7) <<"\t"<< pp(Jr,7) <<"\t"<< pp(Jz,7) <<"\t"<< 
        pp(Rmin,7) <<"\t"<< pp(R,7) <<"\t"<< pp(Rmax,7) <<"\t"<< pp(vz,7) <<"\t"<<
        pp(sqrt(ifd),7) <<"\t"<< pp(sqrt(ift),7) <<"\t"<< pp(sqrt(iff),7) <<"\t";*/
    ift = sqrt(fmax( ift, 1e-6*Rmax*Rmax) );
    IFD = ift;
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
    std::vector< std::vector<double> > grid2d(gridE.size());
    std::vector< std::vector<double> > grid2dJr(gridE.size());
    std::vector< std::vector<double> > grid2dJz(gridE.size());
    std::vector< std::vector<double> > grid2dRt(gridE.size());
    
    for(unsigned int iE=0; iE<gridE.size(); iE++) {
        grid2d[iE].resize(gridLzrel.size());
        grid2dJr[iE].resize(gridLzrel.size());
        grid2dJz[iE].resize(gridLzrel.size());
        grid2dRt[iE].resize(gridLzrel.size());
        const double x  = xLcirc(gridE[iE]);
        const double Lc = Lscale * x / (1-x);
        for(unsigned int iL=0; iL<gridLzrel.size(); iL++) {
            double Lz = gridLzrel[iL] * Lc;
            findClosedOrbitRZplane(potential, gridE[iE], Lz, 
                grid2d[iE][iL], grid2dJr[iE][iL], grid2dJz[iE][iL], grid2dRt[iE][iL]);
            grid2dJr[iE][iL] /= Lc;
            grid2dJz[iE][iL] /= Lc;
            //std::cout << Lc<<"\n";
#if 0
            double rad= R_from_Lz(potential, Lz);
            double v2 = 2 * (gridE[iE] - potential.value(coord::PosCyl(rad, 0, 0)) ) 
                - (Lz>0 ? pow_2(Lz/rad) : 0);
            double vmer = sqrt(fmax(v2, 0));  // velocity component in meridional plane
            if(!math::isFinite(vmer))
                throw std::runtime_error("InterfocalDistanceFinder: error in creating interpolation table");
            const int NUM_ANGLES = 5;
            std::vector<double> ifdvalues(NUM_ANGLES);
            std::cout << "E="<<gridE[iE] << ", Lz="<<Lz<<", IFD = "<<IFD;
            for(int ia=0; ia<NUM_ANGLES; ia++) {
                double angle = (ia+0.5)/NUM_ANGLES * M_PI/2;  // direction of meridional velocity in (R,z) plane
                coord::PosVelCyl point(rad, 0, 0, vmer*cos(angle), vmer*sin(angle), Lz>0 ? Lz/rad : 0);
                double ifd = estimateInterfocalDistance(potential, point);
                ifdvalues[ia]= ifd;
                std::cout <<", "<<ifd;
            }
            std::cout << "\n";
            // find median
            std::sort(ifdvalues.begin(), ifdvalues.end());
            grid2d[iE][iL] = ifdvalues[NUM_ANGLES/2];
#endif
        }
    }
    
    // create a 2d interpolator
    interp = math::LinearInterpolator2d(gridE, gridLzrel, grid2d);
    interpJr = math::LinearInterpolator2d(gridE, gridLzrel, grid2dJr);
    interpJz = math::LinearInterpolator2d(gridE, gridLzrel, grid2dJz);
    interpRt = math::LinearInterpolator2d(gridE, gridLzrel, grid2dRt);
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

void InterfocalDistanceFinder::params(double E, double Lz,
    double& maxJr, double& maxJz, double& Rthin) const
{
    E = fmin(fmax(E, interp.xmin()), interp.xmax());
    double x  = xLcirc(E);
    double Lc = Lscale * x / (1-x);
    double Lzrel = fmin(fmax(Lz/Lc, interp.ymin()), interp.ymax());
    maxJr = interpJr.value(E, Lzrel) * Lc;
    maxJz = interpJz.value(E, Lzrel) * Lc;
    Rthin = interpRt.value(E, Lzrel);
}

}  // namespace actions
