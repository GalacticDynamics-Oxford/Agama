#include "orbit.h"
#include "potential_base.h"
#include "utils.h"
#include "math_core.h"
#include <stdexcept>
#include <cmath>

namespace orbit{

namespace{
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
    int tmp;
    double r = data[0];
    double phi = data[2];
    double theta = remquo(data[1], 2*M_PI, &tmp);  // reduce the range of theta to -pi..pi
    int signr = r<0 ? -1 : 1, signt = theta<0 ? -1 : 1;
    if(theta<0) {  // happens also if pi < theta < 2pi, which is flipped to -pi..0
        theta = -theta;
        phi += M_PI;
    }
    if(r<0) {
        r = -r;
        theta = M_PI-theta;
        phi += M_PI;
    }
    phi = math::wrapAngle(phi);
    return coord::PosVelSph(r, theta, phi, data[3] * signr, data[4] * signt, data[5] * signr * signt);
}
}

template<typename CoordT>
StepResult RuntimeTrajectory<CoordT>::processTimestep(
    const math::BaseOdeSolver& solver, const double tbegin, const double tend, double vars[])
{
    if(samplingInterval > 0) {
        // store trajectory at regular intervals of time
        while(samplingInterval * trajectory.size() <= tend) {
            double tsamp = samplingInterval * trajectory.size();
            double data[6];
            for(int d=0; d<6; d++)
                data[d] = solver.getSol(tsamp, d);
            trajectory.push_back(
                std::pair<coord::PosVelT<CoordT>, double>(getPosVel<CoordT>(data), tsamp));
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


void OrbitIntegratorRot::eval(const double /*t*/, const double x[], double dxdt[]) const
{
    coord::GradCar grad;
    potential.eval(coord::PosCar(x[0], x[1], x[2]), NULL, &grad);
    // time derivative of position
    dxdt[0] = x[3] + Omega * x[1];
    dxdt[1] = x[4] - Omega * x[0];
    dxdt[2] = x[5];
    // time derivative of velocity
    dxdt[3] = -grad.dx + Omega * x[4];
    dxdt[4] = -grad.dy - Omega * x[3];
    dxdt[5] = -grad.dz;
}

double OrbitIntegratorRot::getAccuracyFactor(const double /*t*/, const double x[]) const
{
    double Epot = potential.value(coord::PosCar(x[0], x[1], x[2]));
    double Ekin = 0.5 * (x[3]*x[3] + x[4]*x[4] + x[5]*x[5]);
    return fmin(1, fabs(Epot + Ekin) / fmax(fabs(Epot), Ekin));
}

template<>
void OrbitIntegrator<coord::Car>::eval(const double /*t*/, const double x[], double dxdt[]) const
{
    coord::GradCar grad;
    potential.eval(coord::PosCar(x[0], x[1], x[2]), NULL, &grad);
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
void OrbitIntegrator<coord::Cyl>::eval(const double /*t*/, const double x[], double dxdt[]) const
{
    coord::PosVelCyl p(x);
    if(x[0]<0) {    // R<0
        p.R = -p.R; // apply reflection
        p.phi += M_PI;
    }
    coord::GradCyl grad;
    potential.eval(p, NULL, &grad);
    double Rinv = p.R!=0 ? 1/p.R : 0;  // avoid NAN in degenerate cases
    dxdt[0] = p.vR;
    dxdt[1] = p.vz;
    dxdt[2] = p.vphi * Rinv;
    dxdt[3] = (x[0]<0 ? grad.dR : -grad.dR) + pow_2(p.vphi) * Rinv;
    dxdt[4] = -grad.dz;
    dxdt[5] = -(grad.dphi + p.vR*p.vphi) * Rinv;
}

template<>
void OrbitIntegrator<coord::Sph>::eval(const double /*t*/, const double x[], double dxdt[]) const
{
    int tmp;
    double r = x[0];
    double phi = x[2];
    double theta = remquo(x[1], 2*M_PI, &tmp);  // reduce the range of theta to -pi..pi
    int signr = r<0 ? -1 : 1, signt = theta<0 ? -1 : 1;
    if(theta<0) {  // happens also if pi < theta < 2pi, which is flipped to -pi..0
        theta = -theta;
        phi += M_PI;
    }
    if(r<0) {
        r = -r;
        theta = M_PI-theta;
        phi += M_PI;
    }

    const coord::PosVelSph p(r, theta, phi, x[3], x[4], x[5]);
    coord::GradSph grad;
    potential.eval(p, NULL, &grad);
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
double OrbitIntegrator<CoordT>::getAccuracyFactor(const double /*t*/, const double x[]) const
{
    double Epot = potential.value(coord::PosT<CoordT>(x[0], x[1], x[2]));
    double Ekin = 0.5 * (x[3]*x[3] + x[4]*x[4] + x[5]*x[5]);
    return fmin(1, fabs(Epot + Ekin) / fmax(fabs(Epot), Ekin));
}


template<typename CoordT>
coord::PosVelT<CoordT> integrate(
    const coord::PosVelT<CoordT>& initialConditions,
    const double totalTime,
    const math::IOdeSystem& orbitIntegrator,
    const RuntimeFncArray& runtimeFncs,
    const OrbitIntParams& params)
{
    math::OdeSolverDOP853 solver(orbitIntegrator, params.accuracy);
    int NDIM = orbitIntegrator.size();
    if(NDIM < 6)
        throw std::runtime_error("orbit::integrate() needs at least 6 variables");
    std::vector<double> state(NDIM);
    double* vars = &state.front();
    initialConditions.unpack_to(vars);  // first 6 variables are always position/velocity
    solver.init(vars);
    size_t numSteps = 0;
    double timeCurr = 0;
    while(timeCurr < totalTime) {
        if(!(solver.doStep() > 0.)) {
            // signal of error
            utils::msg(utils::VL_WARNING, "orbit::integrate", "terminated at t="+utils::toString(timeCurr));
            break;
        }
        double timePrev = timeCurr;
        timeCurr = std::min(solver.getTime(), totalTime);
        for(int d=0; d<NDIM; d++)
            vars[d] = solver.getSol(timeCurr, d);
        bool reinit = false, finish = timeCurr >= totalTime;
        for(size_t i=0; i<runtimeFncs.size(); i++) {
            switch(runtimeFncs[i]->processTimestep(solver, timePrev, timeCurr, vars))
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
    const double, const math::IOdeSystem&, const RuntimeFncArray&, const OrbitIntParams&);
template coord::PosVelCyl integrate(const coord::PosVelCyl&,
    const double, const math::IOdeSystem&, const RuntimeFncArray&, const OrbitIntParams&);
template coord::PosVelSph integrate(const coord::PosVelSph&,
    const double, const math::IOdeSystem&, const RuntimeFncArray&, const OrbitIntParams&);

}  // namespace orbit