#include "actions_focal_distance_finder.h"
#include "math_core.h"
#include "math_fit.h"
#include "math_ode.h"
#include "utils.h"
#include <cassert>
#include <stdexcept>
#include <cmath>
// debugging
#include <fstream>

namespace actions{

/// number of sampling points for a shell orbit (equally spaced in time)
static const unsigned int NUM_STEPS_TRAJ = 16;
/// accuracy of root-finding for the radius of thin (shell) orbit
static const double ACCURACY_RTHIN = 1e-6;
/// accuracy of orbit integration for shell orbit
static const double ACCURACY_INTEGR = 1e-6;
/// upper limit on the number of timesteps in ODE solver (should be enough to track half of the orbit)
static const unsigned int MAX_NUM_STEPS_ODE = 100;


// estimate IFD for a series of points in R-z plane
template<typename PointT>
double estimateFocalDistancePoints(
    const potential::BasePotential& potential, const std::vector<PointT>& traj)
{
    if(traj.size()==0)
        throw std::invalid_argument("Error in finding focal distance: empty array of points");
    std::vector<double> x(traj.size()), y(traj.size());
    double sumsq = 0;
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
    double result = sumsq>0 ? math::linearFitZero(x, y, NULL) : 0;
    return sqrt( fmax( result, 0) );  // ensure that the computed value is non-negative
}

template double estimateFocalDistancePoints(
    const potential::BasePotential& potential, const std::vector<coord::PosCar>& traj);
template double estimateFocalDistancePoints(
    const potential::BasePotential& potential, const std::vector<coord::PosVelCar>& traj);
template double estimateFocalDistancePoints(
    const potential::BasePotential& potential, const std::vector<coord::PosCyl>& traj);
template double estimateFocalDistancePoints(
    const potential::BasePotential& potential, const std::vector<coord::PosVelCyl>& traj);
template double estimateFocalDistancePoints(
    const potential::BasePotential& potential, const std::vector<coord::PosSph>& traj);
template double estimateFocalDistancePoints(
    const potential::BasePotential& potential, const std::vector<coord::PosVelSph>& traj);


namespace{
/** find the best-fit value of focal distance for a shell orbit.
    \param[in] traj  contains the trajectory of this orbit in R-z plane,
    \return  the parameter of a prolate spheroidal coordinate system which minimizes
    the variation of `lambda` coordinate for this orbit.
    If the best-fit value is negative, it is replaced with zero.
*/
double fitFocalDistanceShellOrbit(const std::vector<coord::PosCyl>& traj)
{
    if(traj.size()==0)
        throw std::invalid_argument("Error in finding focal distance for a shell orbit: empty array");
    math::Matrix<double> coefs(traj.size(), 2);
    std::vector<double> rhs(traj.size());
    std::vector<double> result;  // regression parameters:  lambda/(lambda-delta), lambda
    for(unsigned int i=0; i<traj.size(); i++) {
        coefs(i, 0) = -pow_2(traj[i].R);
        coefs(i, 1) = 1.;
        rhs[i] = pow_2(traj[i].z);
    }
    math::linearMultiFit(coefs, rhs, NULL, result);
    return sqrt( fmax( result[1] * (1 - 1/result[0]), 0) );
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
        double sign = y[0]>=0 ? 1 : -1;
        poten.eval(coord::PosCyl(fabs(y[0]), y[1], 0), NULL, &grad);
        dydt[0] = y[2];
        dydt[1] = y[3];
        dydt[2] = -grad.dR*sign + (Lz2>0 ? Lz2/pow_3(y[0]) : 0);
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
    virtual void evalDeriv(const double time, double* val, double* der, double*) const
    {
        double vars[4];
        solver.getSol(time, vars);
        if(val)
            *val = vars[1];  // z
        if(der)
            *der = vars[3];  // vz
    }
    virtual unsigned int numDerivs() const { return 1; }
private:
    const math::BaseOdeSolver& solver;
};

/** launch an orbit perpendicularly to x-y plane from radius R with vz>0,
    and record the radius at which it crosses this plane downward (vz<0).
    \param[out] timeCross stores the time required to complete the half-oscillation in z;
    \param[out] traj stores the trajectory recorded at equal intervals of time;
    \return  the crossing radius
*/
double findCrossingPointR(
    const potential::BasePotential& poten, double E, double Lz, double R,
    double* timeCross, std::vector<coord::PosCyl>* traj)
{
    double vz = sqrt(fmax( 2 * (E-poten.value(coord::PosCyl(R, 0, 0))) - (Lz>0 ? pow_2(Lz/R) : 0), R*R*1e-16));
    OrbitIntegratorMeridionalPlane odeSystem(poten, Lz);
    math::OdeStateType vars(odeSystem.size());
    vars[0] = R;
    vars[1] = 0;
    vars[2] = 0;
    vars[3] = vz;
    math::OdeSolverDOP853 solver(odeSystem, ACCURACY_INTEGR);
    solver.init(vars);
    bool finished = false;
    unsigned int numStepsODE = 0;
    double timePrev = 0;
    double timeCurr = 0;
    double timeTraj = 0;
    const double timeStepTraj = timeCross!=NULL ? *timeCross*0.5/(NUM_STEPS_TRAJ-1) : INFINITY;
    if(traj!=NULL)
        traj->clear();
    while(!finished) {
        if(solver.doStep() <= 0 || numStepsODE >= MAX_NUM_STEPS_ODE)  // signal of error
            finished = true;
        else {
            numStepsODE++;
            timePrev = timeCurr;
            timeCurr = solver.getTime();
            if(timeStepTraj!=INFINITY && traj!=NULL)
            {   // store trajectory
                while(timeTraj <= timeCurr && traj->size() < NUM_STEPS_TRAJ) {
                    // store R and z at equal intervals of time
                    double vtraj[4];
                    solver.getSol(timeTraj, vtraj);
                    traj->push_back(coord::PosCyl(fabs(vtraj[0]), vtraj[1], 0));
                    timeTraj += timeStepTraj;
                }
            }
            double vcurr[4];
            solver.getSol(timeCurr, vcurr);
            if(vcurr[1] < 0) {  // z<0 - we're done
                finished = true;
                timeCurr = math::findRoot(FindCrossingPointZequal0(solver),
                    timePrev, timeCurr, ACCURACY_RTHIN);
            }
        }
    }
    if(timeCross!=NULL)
        *timeCross = timeCurr;
    double vroot[4];
    solver.getSol(timeCurr, vroot);    
    return fabs(vroot[0]);   // value of R at the moment of crossing x-y plane
}

/// function to be used in root-finder for locating the thin orbit in R-z plane
class FindClosedOrbitRZplane: public math::IFunctionNoDeriv {
public:
    FindClosedOrbitRZplane(const potential::BasePotential& p, 
        double _E, double _Lz, double _Rmin, double _Rmax,
        double* _timeCross, std::vector<coord::PosCyl>* _traj) :
        poten(p), E(_E), Lz(_Lz), Rmin(_Rmin), Rmax(_Rmax), 
        timeCross(_timeCross), traj(_traj) {};
    /// report the difference in R between starting point (R, z=0, vz>0) and return point (R1, z=0, vz<0)
    virtual double value(const double R) const {
        // first two calls in root-finder are for the boundary points, we already know the answer
        if(R==Rmin)
            return R-Rmax;
        if(R==Rmax)
            return R-Rmin;
        double R1 = findCrossingPointR(poten, E, Lz, R, timeCross, traj);
        return R-R1;
    }
private:
    const potential::BasePotential& poten;
    const double E, Lz;               ///< parameters of motion in the R-z plane
    const double Rmin, Rmax;          ///< boundaries of interval in R (to skip the first two calls)
    double* timeCross;                ///< keep track of time required to complete orbit
    std::vector<coord::PosCyl>* traj; ///< store the trajectory
};
}  // namespace

double estimateFocalDistanceShellOrbit(
    const potential::BasePotential& poten, double E, double Lz, 
    double* R)
{
    double Rmin, Rmax;
    findPlanarOrbitExtent(poten, E, Lz, Rmin, Rmax);
    double timeCross = INFINITY;
    std::vector<coord::PosCyl> traj;
    FindClosedOrbitRZplane fnc(poten, E, Lz, Rmin, Rmax, &timeCross, &traj);
    // locate the radius of thin orbit;
    // as a by-product, store the orbit in 'traj'
    double Rthin = math::findRoot(fnc, Rmin, Rmax, ACCURACY_RTHIN);
    if(R!=NULL)
        *R=Rthin;
    if(!isFinite(Rthin) || traj.size()==0) {
        utils::msg(utils::VL_WARNING, FUNCNAME,
            "Could not find a thin orbit for E="+utils::toString(E)+", Lz="+utils::toString(Lz)+
            " - returning "+utils::toString(Rmin));
        return Rmin;  // anything
    }
    // now find the best-fit value of delta for this orbit
    return fitFocalDistanceShellOrbit(traj);
}

}  // namespace actions
