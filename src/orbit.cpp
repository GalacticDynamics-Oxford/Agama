#include "orbit.h"
#include "potential_base.h"
#include "utils.h"
#include "math_core.h"
#include <stdexcept>
#include <cmath>

namespace orbit{

namespace{

/// roundoff tolerance in the trajectory sampling routine
const double ROUNDOFF = 1e-15;

// normalize the position-velocity returned by orbit integrator in case of r<0 or R<0
template<typename CoordT>
inline coord::PosVelT<CoordT> getPosVel(const double data[6]) { return coord::PosVelT<CoordT>(data); }

template<>
inline coord::PosVelCyl getPosVel(const double data[6])
{
    if(data[0] >= 0)
        return coord::PosVelCyl(data);
    else
        return coord::PosVelCyl(-data[0], data[1], data[2]+M_PI, -data[3], data[4], -data[5]);
}

template<>
inline coord::PosVelSph getPosVel(const double data[6])
{
    double r = data[0];
    double phi = data[2];
    int signr = 1, signt = 1;
    double theta = fmod(data[1], 2*M_PI);
    // normalize theta to the range 0..pi, and r to >=0
    if(theta<-M_PI) {
        theta += 2*M_PI;
    } else if(theta<0) {
        theta = -theta;
        signt = -1;
    } else if(theta>M_PI) {
        theta = 2*M_PI-theta;
        signt = -1;
    }
    if(r<0) {
        r = -r;
        theta = M_PI-theta;
        signr = -1;
    }
    if((signr == -1) ^ (signt == -1))
        phi += M_PI;
    phi = math::wrapAngle(phi);
    return coord::PosVelSph(r, theta, phi, data[3] * signr, data[4] * signt, data[5] * signr * signt);
}
}

template<typename CoordT>
StepResult RuntimeTrajectory<CoordT>::processTimestep(
    const math::BaseOdeSolver& solver, const double tbegin, const double tend, double vars[])
{
    if(t0!=t0)
        t0 = tbegin;   // record the first ever moment of time
    if(samplingInterval == INFINITY) {
        // store just one point from the trajectory, which always contains the current orbital state and time
        if(trajectory.empty())
            trajectory.resize(1);
        trajectory[0].first  = getPosVel<CoordT>(vars);
        trajectory[0].second = tend;
    } else if(samplingInterval > 0) {
        // store trajectory at regular intervals of time
        ptrdiff_t ibegin = static_cast<ptrdiff_t>((tbegin-t0) / samplingInterval);
        ptrdiff_t iend   = static_cast<ptrdiff_t>((tend-t0)   / samplingInterval + ROUNDOFF);
        double dtroundoff = ROUNDOFF * fmax(fabs(tend), fabs(tbegin));
        trajectory.resize(iend + 1);
        for(ptrdiff_t iout=ibegin; iout<=iend; iout++) {
            double tout = samplingInterval * iout + t0;
            if(tout >= tbegin-dtroundoff && tout <= tend+dtroundoff) {
                double data[6];
                for(int d=0; d<6; d++)
                    data[d] = solver.getSol(tout, d);
                trajectory[iout].first  = getPosVel<CoordT>(data);
                trajectory[iout].second = tout;
            }
        }
    } else {
        // store trajectory at every integration timestep
        if(trajectory.empty()) {
            // add the initial point
            double data[6];
            for(int d=0; d<6; d++)
                data[d] = solver.getSol(tbegin, d);
            trajectory.push_back(
                std::pair<coord::PosVelT<CoordT>, double>(getPosVel<CoordT>(data), tbegin));
        }
        // add the current point (at the end of the timestep)
        trajectory.push_back(
            std::pair<coord::PosVelT<CoordT>, double>(getPosVel<CoordT>(vars), tend));
    }
    return SR_CONTINUE;
}


void OrbitIntegratorRot::eval(const double time, const double x[], double dxdt[]) const
{
    coord::GradCar grad;
    potential.eval(coord::PosCar(x[0], x[1], x[2]), NULL, &grad, NULL, time);
    // time derivative of position
    dxdt[0] = x[3] + Omega * x[1];
    dxdt[1] = x[4] - Omega * x[0];
    dxdt[2] = x[5];
    // time derivative of velocity
    dxdt[3] = -grad.dx + Omega * x[4];
    dxdt[4] = -grad.dy - Omega * x[3];
    dxdt[5] = -grad.dz;
}

double OrbitIntegratorRot::getAccuracyFactor(const double time, const double x[]) const
{
    double Epot = potential.value(coord::PosCar(x[0], x[1], x[2]), time);
    double Ekin = 0.5 * (x[3]*x[3] + x[4]*x[4] + x[5]*x[5]);
    return fmin(1, fabs(Epot + Ekin) / fmax(fabs(Epot), Ekin));
}

template<>
void OrbitIntegrator<coord::Car>::eval(const double time, const double x[], double dxdt[]) const
{
    coord::GradCar grad;
    potential.eval(coord::PosCar(x[0], x[1], x[2]), NULL, &grad, NULL, time);
    // time derivative of position
    dxdt[0] = x[3];
    dxdt[1] = x[4];
    dxdt[2] = x[5];
    // time derivative of velocity
    dxdt[3] = -grad.dx;
    dxdt[4] = -grad.dy;
    dxdt[5] = -grad.dz;
}

template<>
void OrbitIntegrator<coord::Cyl>::eval(const double time, const double x[], double dxdt[]) const
{
    coord::PosVelCyl p(x);
    if(x[0]<0) {    // R<0
        p.R = -p.R; // apply reflection
        p.phi += M_PI;
    }
    coord::GradCyl grad;
    potential.eval(p, NULL, &grad, NULL, time);
    double Rinv = p.R!=0 ? 1/p.R : 0;  // avoid NAN in degenerate cases
    dxdt[0] = p.vR;
    dxdt[1] = p.vz;
    dxdt[2] = p.vphi * Rinv;
    dxdt[3] = (x[0]<0 ? grad.dR : -grad.dR) + pow_2(p.vphi) * Rinv;
    dxdt[4] = -grad.dz;
    dxdt[5] = -(grad.dphi + p.vR*p.vphi) * Rinv;
}

template<>
void OrbitIntegrator<coord::Sph>::eval(const double time, const double x[], double dxdt[]) const
{
    double r = x[0];
    double phi = x[2];
    int signr = 1, signt = 1;
    double theta = fmod(x[1], 2*M_PI);
    // normalize theta to the range 0..pi, and r to >=0
    if(theta<-M_PI) {
        theta += 2*M_PI;
    } else if(theta<0) {
        theta = -theta;
        signt = -1;
    } else if(theta>M_PI) {
        theta = 2*M_PI-theta;
        signt = -1;
    }
    if(r<0) {
        r = -r;
        theta = M_PI-theta;
        signr = -1;
    }
    if((signr == -1) ^ (signt == -1))
        phi += M_PI;
    const coord::PosVelSph p(r, theta, phi, x[3], x[4], x[5]);
    coord::GradSph grad;
    potential.eval(p, NULL, &grad, NULL, time);
    double rinv = r!=0 ? 1/r : 0, sintheta, costheta;
    math::sincos(theta, sintheta, costheta);
    double sinthinv = sintheta!=0 ? 1./sintheta : 0;
    double cottheta = costheta * sinthinv;
    dxdt[0] = p.vr;
    dxdt[1] = p.vtheta * rinv;
    dxdt[2] = p.vphi * rinv * sinthinv;
    dxdt[3] = -grad.dr*signr + (pow_2(p.vtheta) + pow_2(p.vphi)) * rinv;
    dxdt[4] = (-grad.dtheta*signt + pow_2(p.vphi)*cottheta - p.vr*p.vtheta) * rinv;
    dxdt[5] = (-grad.dphi * sinthinv - (p.vr+p.vtheta*cottheta) * p.vphi) * rinv;
}

template<typename CoordT>
double OrbitIntegrator<CoordT>::getAccuracyFactor(const double time, const double x[]) const
{
    double Epot = potential.value(coord::PosT<CoordT>(x[0], x[1], x[2]), time);
    double Ekin = 0.5 * (x[3]*x[3] + x[4]*x[4] + x[5]*x[5]);
    return fmin(1, fabs(Epot + Ekin) / fmax(fabs(Epot), Ekin));
}


template<typename CoordT>
coord::PosVelT<CoordT> integrate(
    const coord::PosVelT<CoordT>& initialConditions,
    const double totalTime,
    const math::IOdeSystem& orbitIntegrator,
    const RuntimeFncArray& runtimeFncs,
    const OrbitIntParams& params,
    const double startTime)
{
    math::OdeSolverDOP853 solver(orbitIntegrator, params.accuracy);
    int NDIM = orbitIntegrator.size();
    if(NDIM < 6)
        throw std::runtime_error("orbit::integrate() needs at least 6 variables");
    std::vector<double> state(NDIM);
    double* vars = &state.front();
    initialConditions.unpack_to(vars);  // first 6 variables are always position/velocity
    solver.init(vars, startTime);
    size_t numSteps = 0;
    double currentTime = startTime, endTime = totalTime + startTime;
    while(currentTime < endTime) {
        if(!(solver.doStep() > 0.)) {
            // signal of error
            utils::msg(utils::VL_WARNING,
                "orbit::integrate", "terminated at t="+utils::toString(currentTime));
            break;
        }
        double prevTime = currentTime;
        currentTime = fmin(solver.getTime(), endTime);
        for(int d=0; d<NDIM; d++)
            vars[d] = solver.getSol(currentTime, d);
        bool reinit = false, finish = currentTime >= endTime;
        for(size_t i=0; i<runtimeFncs.size(); i++) {
            switch(runtimeFncs[i]->processTimestep(solver, prevTime, currentTime, vars))
            {
                case orbit::SR_TERMINATE: finish = true; break;
                case orbit::SR_REINIT:    reinit = true; break;
                default: /*nothing*/;
            }
        }
        if(reinit)
            solver.init(vars);
        if(finish || ++numSteps >= params.maxNumSteps)
            break;
    }
    return coord::PosVelT<CoordT>(vars);
}

// explicit template instantiations to make sure all of them get compiled
template class OrbitIntegrator<coord::Car>;
template class OrbitIntegrator<coord::Cyl>;
template class OrbitIntegrator<coord::Sph>;
template class RuntimeTrajectory<coord::Car>;
template class RuntimeTrajectory<coord::Cyl>;
template class RuntimeTrajectory<coord::Sph>;
template coord::PosVelCar integrate(const coord::PosVelCar&,
    double, const math::IOdeSystem&, const RuntimeFncArray&, const OrbitIntParams&, double);
template coord::PosVelCyl integrate(const coord::PosVelCyl&,
    double, const math::IOdeSystem&, const RuntimeFncArray&, const OrbitIntParams&, double);
template coord::PosVelSph integrate(const coord::PosVelSph&,
    double, const math::IOdeSystem&, const RuntimeFncArray&, const OrbitIntParams&, double);

}  // namespace orbit