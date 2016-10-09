#include "actions_interfocal_distance_finder.h"
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

using potential::BasePotential;

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
double estimateInterfocalDistancePoints(
    const BasePotential& potential, const std::vector<PointT>& traj)
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
    const BasePotential& potential, const std::vector<coord::PosCar>& traj);
template double estimateInterfocalDistancePoints(
    const BasePotential& potential, const std::vector<coord::PosVelCar>& traj);
template double estimateInterfocalDistancePoints(
    const BasePotential& potential, const std::vector<coord::PosCyl>& traj);
template double estimateInterfocalDistancePoints(
    const BasePotential& potential, const std::vector<coord::PosVelCyl>& traj);
template double estimateInterfocalDistancePoints(
    const BasePotential& potential, const std::vector<coord::PosSph>& traj);
template double estimateInterfocalDistancePoints(
    const BasePotential& potential, const std::vector<coord::PosVelSph>& traj);


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


/// function to use in ODE integrator
class OrbitIntegratorMeridionalPlane: public math::IOdeSystem {
public:
    OrbitIntegratorMeridionalPlane(const BasePotential& p, double Lz) :
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
    const BasePotential& poten;
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
    \param[out] Jz stores the vertical action computed for this trajectory;
    \return  the crossing radius
*/
static double findCrossingPointR(
    const BasePotential& poten, double E, double Lz, double R,
    double* timeCross, std::vector<coord::PosCyl>* traj, double* Jz)
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
    if(Jz!=NULL)
        *Jz = 0;
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
                    traj->push_back(coord::PosCyl(vtraj[0], vtraj[1], 0));
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
            if(Jz!=NULL)
            {   // compute vertical action  (very crude approximation! one integration point per timestep)
                double vprev[4], vmid[4];
                solver.getSol(timePrev, vprev);
                solver.getSol((timePrev+timeCurr)/2, vmid);
                *Jz += 1 / M_PI * (
                    vmid[2] * (vcurr[0]-vprev[0]) +  // vR at the mid-timestep * delta R over timestep
                    vmid[3] * (vcurr[1]-vprev[1]) ); // vz at the mid-timestep * delta z over timestep
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
    FindClosedOrbitRZplane(const BasePotential& p, 
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
    const BasePotential& poten;
    const double E, Lz;               ///< parameters of motion in the R-z plane
    const double Rmin, Rmax;          ///< boundaries of interval in R (to skip the first two calls)
    double* timeCross;                ///< keep track of time required to complete orbit
    std::vector<coord::PosCyl>* traj; ///< store the trajectory
    double* Jz;                       ///< store the estimated value of vertical action
};


double estimateInterfocalDistanceShellOrbit(
    const BasePotential& poten, double E, double Lz, 
    double* R, double* Jz)
{
    double Rmin, Rmax;
    findPlanarOrbitExtent(poten, E, Lz, Rmin, Rmax);
    double timeCross = INFINITY;
    std::vector<coord::PosCyl> traj;
    FindClosedOrbitRZplane fnc(poten, E, Lz, Rmin, Rmax, &timeCross, &traj, Jz);
    // locate the radius of thin orbit;
    // as a by-product, store the orbit in 'traj' and the vertical action in Jz (if necessary)
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
    return fitInterfocalDistanceShellOrbit(traj);
}


// ----------- Interpolation of interfocal distance in E,Lz plane ------------ //
InterfocalDistanceFinder::InterfocalDistanceFinder(
    const BasePotential& potential, const unsigned int gridSizeE) :
    interpLcirc(potential)
{
    if(!isAxisymmetric(potential))
        throw std::invalid_argument("Potential is not axisymmetric, "
            "interfocal distance estimator is not suitable for this case");

    if(gridSizeE<10 || gridSizeE>500)
        throw std::invalid_argument("InterfocalDistanceFinder: incorrect grid size");

    // find out characteristic energy values
    double Ein  = potential.value(coord::PosCar(0, 0, 0));
    double Eout = 0;  // default assumption for Phi(r=infinity)
    if(!isFinite(Ein) || Ein>=Eout)
        throw std::runtime_error("InterfocalDistanceFinder: weird behaviour of potential");

    // create a grid in energy
    Ein *= 1-0.5/gridSizeE;  // slightly offset from zero
    std::vector<double> gridE(gridSizeE);
    for(unsigned int i=0; i<gridSizeE; i++) 
        gridE[i] = Ein + i*(Eout-Ein)/gridSizeE;

    // create a uniform grid in Lz/Lcirc(E)
    const unsigned int gridSizeLzrel = gridSizeE<80 ? gridSizeE/4 : 20;
    std::vector<double> gridLzrel(gridSizeLzrel);
    for(unsigned int i=0; i<gridSizeLzrel; i++)
        gridLzrel[i] = (i+0.01) / (gridSizeLzrel-0.98);

    // fill a 2d grid in (E, Lz/Lcirc(E) )
    math::Matrix<double> grid2dD(gridSizeE, gridSizeLzrel);  // 2d grid for interfocal distance
    math::Matrix<double> grid2dR(gridSizeE, gridSizeLzrel);  // 2d grid for Rthin
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for(int iE=0; iE<(int)gridSizeE; iE++) {
        double Rc = R_circ(potential, gridE[iE]);
        double Lc = Rc * v_circ(potential, Rc);  // almost the same as interpLcirc(gridE[iE]);
        for(unsigned int iL=0; iL<gridSizeLzrel; iL++) {
            double Lz = gridLzrel[iL] * Lc;
            double Rthin;
            grid2dD(iE, iL) = estimateInterfocalDistanceShellOrbit(potential, gridE[iE], Lz, &Rthin);
            grid2dR(iE, iL) = Rthin / Rc;
        }
    }

    // debugging output
    if(utils::verbosityLevel >= utils::VL_VERBOSE) {
        std::ofstream strm("ifd");
        for(unsigned int iE=0; iE<gridE.size(); iE++) {
            double Rc = R_circ(potential, gridE[iE]);
            for(unsigned int iL=0; iL<gridLzrel.size(); iL++) {
                strm << Rc << '\t' << gridLzrel[iL] << '\t' <<
                grid2dD(iE, iL) << '\t' << grid2dR(iE, iL) << '\n';
            }
            strm << '\n';
        }
    }

    // create 2d interpolators
    interpD = math::LinearInterpolator2d(gridE, gridLzrel, grid2dD);
    interpR = math::LinearInterpolator2d(gridE, gridLzrel, grid2dR);
}

double InterfocalDistanceFinder::value(double E, double Lz) const
{
    E = fmin(fmax(E, interpD.xmin()), interpD.xmax());
    double Lc = interpLcirc.L_circ(E);
    double Lzrel = fmin(fmax(fabs(Lz)/Lc, interpD.ymin()), interpD.ymax());
    return interpD.value(E, Lzrel);
}

double InterfocalDistanceFinder::Rthin(double E, double Lz) const
{
    E = fmin(fmax(E, interpR.xmin()), interpR.xmax());
    double Lc = interpLcirc.L_circ(E);
    double Rc = interpLcirc.R_circ(E);
    double Lzrel = fmin(fmax(fabs(Lz)/Lc, interpR.ymin()), interpR.ymax());
    return fmax(interpR.value(E, Lzrel), 0) * Rc;
}

}  // namespace actions
