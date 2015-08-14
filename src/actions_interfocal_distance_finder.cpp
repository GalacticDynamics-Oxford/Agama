#include "actions_interfocal_distance_finder.h"
#include "math_core.h"
#include "math_ode.h"
#include <stdexcept>
#include <cmath>
#include <algorithm>

#include <iostream>
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
    enum { FIND_RMIN, FIND_RMAX, FIND_ZMAX } mode;
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
        double result=Phi-E;
        if(mode == FIND_RMIN) {    // f(R) = -(1/2) v_R^2 * R^2
            result = result*x*x + Lz2/2;
            if(deriv) 
                *deriv = 2*x*(Phi-E) + x*x*grad.dR;
            if(deriv2)
                *deriv2 = 2*(Phi-E) + 4*x*grad.dR + x*x*hess.dR2;
        } else if(mode == FIND_RMAX) {  // f(R) = -(1/2) v_R^2 = Phi(R) - E + Lz^2/(2 R^2)
            if(Lz2>0)
                result += Lz2/(2*x*x);
            if(deriv)
                *deriv = grad.dR - (Lz2>0 ? Lz2/(x*x*x) : 0);
            if(deriv2)
                *deriv2 = hess.dR2 + (Lz2>0 ? 3*Lz2/(x*x*x*x) : 0);
        } else {  // FIND_ZMAX:   f(z) = -(1/2) v_z^2
            if(deriv)
                *deriv = grad.dz;
            if(deriv2)
                *deriv2= hess.dz2;
        }
        if(val)
            *val = result;
    }
};

bool estimateOrbitExtent(const potential::BasePotential& potential, const coord::PosVelCyl& point,
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
            maxPeri = absR + nh.dxToNegative();
        } else {
            maxPeri = NAN;
            minApo  = absR + nh.dxToNegative();
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

double estimateInterfocalDistanceBox(const potential::BasePotential& potential, 
    double R1, double R2, double z1, double z2)
{
    if(z1+z2<=(R1+R2)*1e-8)   // orbit in x-y plane, any (non-zero) result will go
        return (R1+R2)/2;
    const int nR=4, nz=2, numpoints=nR*nz;
    double x[numpoints], y[numpoints];
    const double r1=sqrt(R1*R1+z1*z1), r2=sqrt(R2*R2+z2*z2);
    const double a1=atan2(z1, R1), a2=atan2(z2, R2);
    double sumsq=0;
    for(int iR=0; iR<nR; iR++) {
        double r=r1+(r2-r1)*iR/(nR-1);
        for(int iz=0; iz<nz; iz++) {
            const int ind=iR*nz+iz;
            double a=(iz+1.)/nz * (a1+(a2-a1)*iR/(nR-1));
            coord::GradCyl grad;
            coord::HessCyl hess;
            coord::PosCyl pos(r*cos(a), r*sin(a), 0);
            potential.eval(pos, NULL, &grad, &hess);
            x[ind] = hess.dRdz;
            y[ind] = 3*pos.z*grad.dR - 3*pos.R*grad.dz + pos.R*pos.z*(hess.dR2-hess.dz2)
                   + (pos.z*pos.z-pos.R*pos.R) * hess.dRdz;
            sumsq += pow_2(x[ind]);
        }
    }
    double coef = sumsq>0 ? math::linearFitZero(numpoints, x, y) : 0;
    coef = fmax(coef, fmin(R1*R1+z1*z1,R2*R2+z2*z2)*0.0001);  // prevent it from going below or around zero
    return sqrt(coef);
}

double estimateInterfocalDistance(
    const potential::BasePotential& potential, const coord::PosVelCyl& point)
{
    double R1, R2, z1, z2;
    if(!estimateOrbitExtent(potential, point, R1, R2, z1, z2)) {
        R1=R2=point.R; z1=z2=point.z;
    }
    return estimateInterfocalDistanceBox(potential, R1, R2, z1, z2);
}

// estimate IFD for a series of points in R-z plane
double estimateInterfocalDistanceLine(
    const potential::BasePotential& potential, const std::vector< std::pair<double, double> >& traj)
{
    std::vector<double> x(traj.size()), y(traj.size());
    double sumsq=0;
    for(unsigned int i=0; i<traj.size(); i++) {
        const double R = traj[i].first;
        const double z = traj[i].second;
        coord::GradCyl grad;
        coord::HessCyl hess;
        potential.eval(coord::PosCyl(R, z, 0), NULL, &grad, &hess);
        x[i] = hess.dRdz;
        y[i] = 3*z * grad.dR - 3*R * grad.dz + R*z * (hess.dR2-hess.dz2) + (z*z-R*R) * hess.dRdz;
        sumsq += pow_2(x[i]);
    }
    double coef = sumsq>0 ? math::linearFitZero(traj.size(), &x.front(), &y.front()) : 0;
    coef = fmax(coef, pow_2(traj[0].first) * 0.0001);  // prevent it from going below or around zero
    return sqrt(coef);
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

static const unsigned int NUM_STEPS_TRAJ = 10;
static const double ACCURACY = 1e-6;
static const unsigned int MAX_NUM_STEPS_ODE = 100;

/// launch an orbit perpendicularly to x-y plane from radius R with vz>0,
/// and record the radius at which it crosses this plane downward (vz<0)
static double findCrossingPointR(
    const potential::BasePotential& poten, double E, double Lz, double R,
    double& timeCross, std::vector<std::pair<double, double> > &traj)
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
    const double timeStepTraj = timeCross*0.5/(NUM_STEPS_TRAJ-1);
    traj.clear();
    while(!finished) {
        if(solver.step() <= 0 || numStepsODE >= MAX_NUM_STEPS_ODE)  // signal of error
            finished = true;
        else {
            numStepsODE++;
            timePrev = timeCurr;
            timeCurr = solver.getTime();
            finished = solver.value(timeCurr, 1) < 0;  // z<0
            // store trajectory
            while(timeTraj <= timeCurr && traj.size() < NUM_STEPS_TRAJ) {
                traj.push_back(std::pair<double, double>  // store R and z at equal intervals of time
                    (fabs(solver.value(timeTraj, 0)), solver.value(timeTraj, 1))); 
                timeTraj += timeStepTraj;
            }
        }
    }
    timeCross = math::findRoot(FindCrossingPointZequal0(solver),
        timePrev, timeCurr, ACCURACY);
    if(timeCross >= timePrev && timeCross <= timeCurr)
        return fabs(solver.value(timeCross, 0));   // value of R at the moment of crossing x-y plane
    else
        return NAN;
}

class FindClosedOrbitRZplane: public math::IFunctionNoDeriv {
public:
    FindClosedOrbitRZplane(const potential::BasePotential& p, 
        double _E, double _Lz, double _Rmin, double _Rmax,
        double& _timeCross, std::vector<std::pair<double, double> > &_traj) :
        poten(p), E(_E), Lz(_Lz), Rmin(_Rmin), Rmax(_Rmax), timeCross(_timeCross), traj(_traj) {};
    /// report the difference in R between starting point (R, z=0, vz>0) and return point (R1, z=0, vz<0)
    virtual double value(const double R) const {
        if(R==Rmin)
            return R-Rmax;
        if(R==Rmax)
            return R-Rmin;
        double R1 = findCrossingPointR(poten, E, Lz, R, timeCross, traj);
        return R-R1;
    }
private:
    const potential::BasePotential& poten;
    const double E, Lz;
    const double Rmin, Rmax;
    double& timeCross;
    std::vector<std::pair<double, double> > &traj;
};

void findPeriApocenter(const potential::BasePotential& poten, double E, double Lz, 
    double& Rmin, double& Rmax)
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
            maxPeri = fnc.R + nh.dxToNegative();
        } else {
            maxPeri = NAN;
            minApo  = fnc.R + nh.dxToNegative();
        }
    }
    if(fnc.Lz2>0) {
        if(math::isFinite(maxPeri)) {
            fnc.mode = OrbitSizeFunction::FIND_RMIN;
            Rmin = math::findRoot(fnc, 0., maxPeri, ACCURACY);
            // ensure that E-Phi(Rmin) >= 0  (due to finite accuracy in root-finding, a small adjustment may be needed)
            Rmin += math::PointNeighborhood(fnc, Rmin).dxToNegative();
        }
    } else  // angular momentum is zero
        Rmin = 0;
    if(math::isFinite(minApo)) {
        fnc.mode = OrbitSizeFunction::FIND_RMAX;
        Rmax = math::findRoot(fnc, minApo, INFINITY, ACCURACY);
        Rmax += math::PointNeighborhood(fnc, Rmax).dxToNegative();  // ensure that E>=Phi(Rmax)
    }   // else Rmax=absR
}

double findClosedOrbitRZplane(const potential::BasePotential& poten, double E, double Lz)
{
    double Rmin, Rmax;
    findPeriApocenter(poten, E, Lz, Rmin, Rmax);
    double timeCross = INFINITY;
    std::vector<std::pair<double, double> > traj;
    FindClosedOrbitRZplane fnc(poten, E, Lz, Rmin, Rmax, timeCross, traj);
    double R = math::findRoot(fnc, Rmin, Rmax, ACCURACY);
    //double vz = sqrt(2 * (E-poten.value(coord::PosCyl(R, 0, 0))) - (Lz>0 ? pow_2(Lz/R) : 0));
    //std::cout << "E="<<E<<", Lz="<<Lz<<", "<<Rmin<<"< R="<<R<<" <"<<Rmax<<", vz="<<vz<<", vphi="<<(Lz/R)<<"\n";
    return estimateInterfocalDistanceLine(poten, traj);
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
        gridLzrel[i] = (i+0.) / gridSizeLzrel;

    // fill a 2d grid in (E, Lz/Lcirc(E) )
    std::vector< std::vector<double> > grid2d(gridE.size());
    
    for(unsigned int iE=0; iE<gridE.size(); iE++) {
        grid2d[iE].resize(gridLzrel.size());
        const double x  = xLcirc(gridE[iE]);
        const double Lc = Lscale * x / (1-x);
        for(unsigned int iL=0; iL<gridLzrel.size(); iL++) {
            double Lz = gridLzrel[iL] * Lc;
            double IFD = findClosedOrbitRZplane(potential, gridE[iE], Lz);
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
#else
            grid2d[iE][iL] = IFD;
#endif
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
