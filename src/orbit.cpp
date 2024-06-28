#include "orbit.h"
#include "potential_base.h"
#include "utils.h"
#include "math_core.h"
#include <stdexcept>
#include <cmath>

namespace orbit{

/// roundoff tolerance in the trajectory sampling routine
static const double ROUNDOFF = 10*DBL_EPSILON;

//---- RuntimeTrajectory ----//

bool RuntimeTrajectory::processTimestep(const double tbegin, const double tend)
{
    if(t0!=t0)
        t0 = tbegin;   // record the first ever moment of time
    if(samplingInterval == INFINITY) {
        // store just one point from the trajectory,
        // which always contains the current orbital state and time
        if(trajectory.empty())
            trajectory.resize(1);
        trajectory[0].first = orbint.getSol(tend);
        trajectory[0].second = tend;
    } else if(samplingInterval > 0) {
        // store trajectory at regular intervals of time
        double sign = tend>=tbegin ? +1 : -1;  // integrating forward or backward in time
        double dtroundoff = ROUNDOFF * fmax(fmax(fabs(tend), fabs(tbegin)), fabs(t0));
        ptrdiff_t ibegin = static_cast<ptrdiff_t>((sign * (tbegin-t0) - dtroundoff) / samplingInterval);
        ptrdiff_t iend   = static_cast<ptrdiff_t>((sign * (tend-t0)   + dtroundoff) / samplingInterval);
        trajectory.resize(iend + 1);
        for(ptrdiff_t iout=ibegin; iout<=iend; iout++) {
            double tout = sign * samplingInterval * iout + t0;
            if(sign * tout >= sign * tbegin - dtroundoff && sign * tout <= sign * tend + dtroundoff)
                trajectory[iout] = Trajectory::value_type(orbint.getSol(tout), tout);
        }
    } else {
        // store trajectory at every integration timestep
        if(trajectory.empty())  // add the initial point
            trajectory.push_back(Trajectory::value_type(orbint.getSol(tbegin), tbegin));
        // add the current point (at the end of the timestep)
        trajectory.push_back(Trajectory::value_type(orbint.getSol(tend), tend));
    }
    return true;
}


//---- OrbitIntegrator ----//

coord::PosVelCar BaseOrbitIntegrator::run(const double totalTime)
{
    if(totalTime==0 || !isFinite(totalTime))  // don't bother
        return getSol(solver->getTime());
    size_t numSteps = 0;
    double sign = totalTime>0 ? +1 : -1;   // integrate forward (+1) or backward (-1) in time
    double currentTime = solver->getTime(), endTime = totalTime + currentTime;
    while(true) {
        if(!(solver->doStep(sign>0 ? +0.0 : -0.0) * sign > 0.)) {
            // signal of error
            FILTERMSG(utils::VL_WARNING,
                "OrbitIntegrator", "terminated at t="+utils::toString(currentTime));
            break;
        }
        double prevTime = currentTime;
        currentTime = fmin(solver->getTime()*sign, endTime*sign) * sign;
        bool contin = true;
        for(size_t i=0; contin && i<fncs.size(); i++)
            contin &= fncs[i]->processTimestep(prevTime, currentTime);
        if(!contin || currentTime*sign >= endTime*sign || ++numSteps >= maxNumSteps)
            break;
    }
    return getSol(currentTime);
}

template<typename CoordT>
void OrbitIntegrator<CoordT>::init(const coord::PosVelCar& ic, double time)
{
    double posvel[6];
    coord::toPosVel<coord::Car, CoordT>(ic).unpack_to(posvel);
    solver->init(posvel, time);
}

// a different implementation for the cartesian case, where the integration is performed
// in the inertial frame, while the input/output coords are provided in the rotating frame
template<>
void OrbitIntegrator<coord::Car>::init(const coord::PosVelCar& ic, double time)
{
    double posvel[6];
    if(Omega) {
        double ca=1, sa=0, t = time==time ? time : solver->getTime();
        math::sincos(Omega * t, sa, ca);
        posvel[0] = ic.x *ca - ic.y *sa;
        posvel[1] = ic.y *ca + ic.x *sa;
        posvel[2] = ic.z;
        posvel[3] = ic.vx*ca - ic.vy*sa;
        posvel[4] = ic.vy*ca + ic.vx*sa;
        posvel[5] = ic.vz;
    }
    else
        ic.unpack_to(posvel);
    solver->init(posvel, time);
}

template<>
coord::PosVelCar OrbitIntegrator<coord::Car>::getSolNative(double time) const
{
    double data[6];
    for(int i=0; i<6; i++)
        data[i] = solver->getSol(time, i);
    if(Omega) {
        // integration is performed in the inertial frame; transform output to the rotating frame
        double ca=1, sa=0;
        math::sincos(Omega * time, sa, ca);
        return coord::PosVelCar(data[0]*ca + data[1]*sa, data[1]*ca - data[0]*sa, data[2],
                                data[3]*ca + data[4]*sa, data[4]*ca - data[3]*sa, data[5]);
    } else
        return coord::PosVelCar(data);
}

// retrieve and normalize the position-velocity from the ODE integrator [rectify cases of r<0 or R<0]
template<>
coord::PosVelCyl OrbitIntegrator<coord::Cyl>::getSolNative(double time) const
{
    double data[6];
    for(int i=0; i<6; i++)
        data[i] = solver->getSol(time, i);
    if(data[0] < 0) {
        data[2] += M_PI;
        data[0]  = -data[0];
        data[3]  = -data[3];
        data[5]  = -data[5];
    }
    return coord::PosVelCyl(data);
}

template<>
coord::PosVelSph OrbitIntegrator<coord::Sph>::getSolNative(double time) const
{
    double data[6];
    for(int i=0; i<6; i++)
        data[i] = solver->getSol(time, i);
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

template<>
void OrbitIntegrator<coord::Car>::eval(const double time, const double x[], double dxdt[]) const
{
    double ca=1, sa=0;
    if(Omega)
        math::sincos(Omega * time, sa, ca);
    // it appears to be more efficient to perform the integration in an inertial frame,
    // and rotate the potential instead (same as adding the Rotating modifier to the potential,
    // but storing the resulting orbit in the rotating frame)
    coord::GradCar grad;
    potential.eval(coord::PosCar(x[0]*ca + x[1]*sa, x[1]*ca - x[0]*sa, x[2]), NULL, &grad, NULL, time);
    // time derivative of position
    dxdt[0] = x[3];
    dxdt[1] = x[4];
    dxdt[2] = x[5];
    // time derivative of velocity
    dxdt[3] = -grad.dx*ca + grad.dy*sa;
    dxdt[4] = -grad.dy*ca - grad.dx*sa;
    dxdt[5] = -grad.dz;
}

template<typename CoordT>
void OrbitIntegrator<CoordT>::eval(const double, const double[], double[], double[]) const
{
    throw std::runtime_error("Hermite integration in non-Cartesian coordinates is not implemented");
}

template<>
void OrbitIntegrator<coord::Car>::eval(const double time, const double xv[],
    double d2xdt2[], double d3xdt3[]) const
{
    if(Omega)
        throw std::runtime_error("Hermite integration in the rotating frame is not implemented");
    coord::GradCar grad;
    coord::HessCar hess;
    potential.eval(coord::PosCar(xv[0], xv[1], xv[2]), NULL, &grad, &hess, time);
    // time derivative of velocity
    d2xdt2[0] = -grad.dx;
    d2xdt2[1] = -grad.dy;
    d2xdt2[2] = -grad.dz;
    // time derivative of acceleration; NB: does not work in the rotating frame!!
    d3xdt3[0] = -hess.dx2  * xv[3] - hess.dxdy * xv[4] - hess.dxdz * xv[5];
    d3xdt3[1] = -hess.dxdy * xv[3] - hess.dy2  * xv[4] - hess.dydz * xv[5];
    d3xdt3[2] = -hess.dxdz * xv[3] - hess.dydz * xv[4] - hess.dz2  * xv[5];
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
    if(x[0]<0)
        grad.dR = -grad.dR;
    dxdt[0] = p.vR;
    dxdt[1] = p.vz;
    dxdt[2] = p.vphi * Rinv - Omega;
    dxdt[3] = -grad.dR + pow_2(p.vphi) * Rinv;
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
    dxdt[2] = p.vphi * rinv * sinthinv - Omega;
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


// explicit template instantiations to make sure all of them get compiled
template class OrbitIntegrator<coord::Car>;
template class OrbitIntegrator<coord::Cyl>;
template class OrbitIntegrator<coord::Sph>;

}  // namespace orbit

