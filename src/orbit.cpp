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

bool RuntimeTrajectory::processTimestep(const double tbegin, const double timestep)
{
    if(t0!=t0)
        t0 = tbegin;   // record the first ever moment of time
    if(samplingInterval == INFINITY) {
        // store just one point from the trajectory,
        // which always contains the current orbital state and time
        if(trajectory.empty())
            trajectory.resize(1);
        trajectory[0].first = orbint.getSol(timestep);  // solution at the end of this timestep
        trajectory[0].second = tbegin + timestep;
    } else if(samplingInterval > 0) {
        // store trajectory at regular intervals of time
        double sign = timestep>=0 ? +1 : -1;  // integrating forward or backward in time
        double tend = tbegin + timestep;
        double dtroundoff = ROUNDOFF * fmax(fmax(fabs(tend), fabs(tbegin)), fabs(t0));
        ptrdiff_t iout = static_cast<ptrdiff_t>(ceil( sign * (tbegin-t0) / samplingInterval));
        ptrdiff_t iend = static_cast<ptrdiff_t>((sign * (tend-t0) + dtroundoff) / samplingInterval);
        trajectory.resize(iend + 1);
        for(; iout <= iend; iout++) {
            double timeout = sign * samplingInterval * iout + t0;
            double offsetout = math::clip(sign * (timeout - tbegin), 0.0, sign * timestep) * sign;
            trajectory[iout] = Trajectory::value_type(orbint.getSol(offsetout), timeout);
        }
    } else {
        // store trajectory at every integration timestep
        if(trajectory.empty())  // add the initial point
            trajectory.push_back(Trajectory::value_type(orbint.getSol(0), tbegin));
        // add the current point (at the end of the timestep)
        trajectory.push_back(Trajectory::value_type(orbint.getSol(timestep), tbegin+timestep));
    }
    return true;
}


//---- OrbitIntegrator ----//

coord::PosVelCar BaseOrbitIntegrator::run(const double timeTotal)
{
    double timeEnd = timeBegin + timeTotal, timeStep = 0;
    if(timeTotal==0 || !isFinite(timeEnd)) {  // don't bother - return the current state
        bool keepGoing = true;
        for(size_t i=0; keepGoing && i<fncs.size(); i++)
            keepGoing &= fncs[i]->processTimestep(timeBegin, timeStep);
        return getSol(timeStep);
    }
    size_t numSteps = 0;
    double sign = timeTotal>0 ? +1 : -1;   // integrate forward (+1) or backward (-1) in time
    // on entry, timeBegin contains the initial time set by init(),
    // and it will be updated after every timestep is completed and processed by runtime fncs.
    while(timeBegin != timeEnd) {
        double timeRemaining = timeEnd - timeBegin;
        // actual length of the step taken is smaller than timeRemaining until the very last step
        timeStep = stepper->doStep(timeRemaining);
        if(!(timeStep * sign > 0.)) {
            // signal of error
            FILTERMSG(utils::VL_WARNING, "OrbitIntegrator",
                "terminated at t=" + utils::toString(timeBegin) +
                ", timestep=" + utils::toString(timeStep));
            break;
        }
        bool keepGoing = true;
        for(size_t i=0; keepGoing && i<fncs.size(); i++)
            keepGoing &= fncs[i]->processTimestep(timeBegin, timeStep);
        timeBegin = timeStep == timeRemaining ? timeEnd : timeBegin + timeStep;
        if(!keepGoing || ++numSteps >= maxNumSteps)
            break;
    }
    return getSol(timeStep);  // last point on the trajectory
}

template<typename CoordT>
void OrbitIntegrator<CoordT>::init(const coord::PosVelCar& ic, double newTime)
{
    if(newTime == newTime)
        timeBegin = newTime;
    double posvel[6];
    coord::toPosVel<coord::Car, CoordT>(ic).unpack_to(posvel);
    stepper->init(posvel);
}

// a different implementation for the cartesian case, where the integration is performed
// in the inertial frame, while the input/output coords are provided in the rotating frame
template<>
void OrbitIntegrator<coord::Car>::init(const coord::PosVelCar& ic, double newTime)
{
    if(newTime == newTime)
        timeBegin = newTime;
    double posvel[6];
    if(Omega) {
        double ca=1, sa=0;
        math::sincos(Omega * timeBegin, sa, ca);
        posvel[0] = ic.x *ca - ic.y *sa;
        posvel[1] = ic.y *ca + ic.x *sa;
        posvel[2] = ic.z;
        posvel[3] = ic.vx*ca - ic.vy*sa;
        posvel[4] = ic.vy*ca + ic.vx*sa;
        posvel[5] = ic.vz;
    }
    else
        ic.unpack_to(posvel);
    stepper->init(posvel);
}

template<>
coord::PosVelCar OrbitIntegrator<coord::Car>::getSolNative(double timeOffset) const
{
    double data[6];
    for(int i=0; i<6; i++)
        data[i] = stepper->getSol(timeOffset, i);
    if(Omega) {
        // integration is performed in the inertial frame; transform output to the rotating frame
        double ca=1, sa=0;
        math::sincos(Omega * (timeOffset + timeBegin), sa, ca);
        return coord::PosVelCar(data[0]*ca + data[1]*sa, data[1]*ca - data[0]*sa, data[2],
                                data[3]*ca + data[4]*sa, data[4]*ca - data[3]*sa, data[5]);
    } else
        return coord::PosVelCar(data);
}

// retrieve and normalize the position-velocity from the ODE integrator [rectify cases of r<0 or R<0]
template<>
coord::PosVelCyl OrbitIntegrator<coord::Cyl>::getSolNative(double timeOffset) const
{
    double data[6];
    for(int i=0; i<6; i++)
        data[i] = stepper->getSol(timeOffset, i);
    if(data[0] < 0) {
        data[2] += M_PI;
        data[0]  = -data[0];
        data[3]  = -data[3];
        data[5]  = -data[5];
    }
    return coord::PosVelCyl(data);
}

template<>
coord::PosVelSph OrbitIntegrator<coord::Sph>::getSolNative(double timeOffset) const
{
    double data[6];
    for(int i=0; i<6; i++)
        data[i] = stepper->getSol(timeOffset, i);
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
void OrbitIntegrator<coord::Car>::eval(const double timeOffset, const double xv[],
    /*output*/ double dxdt[], double* accFac) const
{
    double ca=1, sa=0;
    if(Omega)
        math::sincos(Omega * (timeBegin + timeOffset), sa, ca);
    // it appears to be more efficient to perform the integration in an inertial frame,
    // and rotate the potential instead (same as adding the Rotating modifier to the potential,
    // but storing the resulting orbit in the rotating frame)
    double Epot;
    coord::GradCar grad;
    potential.eval(coord::PosCar(xv[0]*ca + xv[1]*sa, xv[1]*ca - xv[0]*sa, xv[2]),
        accFac ? &Epot : NULL, &grad, NULL, timeBegin + timeOffset);
    // time derivative of position
    dxdt[0] = xv[3];
    dxdt[1] = xv[4];
    dxdt[2] = xv[5];
    // time derivative of velocity
    dxdt[3] = -grad.dx*ca + grad.dy*sa;
    dxdt[4] = -grad.dy*ca - grad.dx*sa;
    dxdt[5] = -grad.dz;
    if(accFac) {
        double Ekin = 0.5 * (pow_2(xv[3]) + pow_2(xv[4]) + pow_2(xv[5]));
        *accFac = fmin(1, fabs(Epot + Ekin) / fmax(fabs(Epot), Ekin));
    }
}

template<typename CoordT>
void OrbitIntegrator<CoordT>::eval2(const double, const double[], double[], double[], double*) const
{
    throw std::runtime_error("2nd order ODE integration in non-Cartesian coordinates is not implemented");
}

template<>
void OrbitIntegrator<coord::Car>::eval2(const double timeOffset, const double xv[],
    /*output*/ double d2xdt2[], double d3xdt3[], double* accFac) const
{
    double ca=1, sa=0;
    if(Omega)
        math::sincos(Omega * (timeBegin + timeOffset), sa, ca);
    double Epot;
    coord::GradCar grad;
    coord::HessCar hess;
    potential.eval(coord::PosCar(xv[0]*ca + xv[1]*sa, xv[1]*ca - xv[0]*sa, xv[2]),
        accFac ? &Epot : NULL,
        &grad,
        /*only for Hermite integrator*/ d3xdt3 ? &hess : NULL,
        timeBegin + timeOffset);
    // time derivative of velocity
    d2xdt2[0] = -grad.dx*ca + grad.dy*sa;
    d2xdt2[1] = -grad.dy*ca - grad.dx*sa;
    d2xdt2[2] = -grad.dz;
    if(d3xdt3) {
        if(Omega)
            throw std::runtime_error("Hermite integration in the rotating frame is not implemented");
        // time derivative of acceleration;
        // NB: although tedious, it can be (but is not) implemented for a rotating potential;
        // however, for a time-dependent potential it cannot be implemented without knowing dPhi/dt
        // (there is no way to check whether the potential is time-dependent!)
        d3xdt3[0] = -hess.dx2  * xv[3] - hess.dxdy * xv[4] - hess.dxdz * xv[5];
        d3xdt3[1] = -hess.dxdy * xv[3] - hess.dy2  * xv[4] - hess.dydz * xv[5];
        d3xdt3[2] = -hess.dxdz * xv[3] - hess.dydz * xv[4] - hess.dz2  * xv[5];
    }
    if(accFac) {
        double Ekin = 0.5 * (pow_2(xv[3]) + pow_2(xv[4]) + pow_2(xv[5]));
        *accFac = fmin(1, fabs(Epot + Ekin) / fmax(fabs(Epot), Ekin));
    }
}

template<>
void OrbitIntegrator<coord::Cyl>::eval(const double timeOffset, const double xv[],
    /*output*/ double dxdt[], double* accFac) const
{
    coord::PosVelCyl p(xv);
    if(xv[0]<0) {    // R<0
        p.R = -p.R; // apply reflection
        p.phi += M_PI;
    }
    double Epot;
    coord::GradCyl grad;
    potential.eval(p, accFac ? &Epot : NULL, &grad, NULL, timeBegin + timeOffset);
    double Rinv = p.R!=0 ? 1/p.R : 0;  // avoid NAN in degenerate cases
    if(xv[0]<0)
        grad.dR = -grad.dR;
    dxdt[0] = p.vR;
    dxdt[1] = p.vz;
    dxdt[2] = p.vphi * Rinv - Omega;
    dxdt[3] = -grad.dR + pow_2(p.vphi) * Rinv;
    dxdt[4] = -grad.dz;
    dxdt[5] = -(grad.dphi + p.vR*p.vphi) * Rinv;
    if(accFac) {
        double Ekin = 0.5 * (pow_2(p.vR) + pow_2(p.vz) + pow_2(p.vphi));
        *accFac = fmin(1, fabs(Epot + Ekin) / fmax(fabs(Epot), Ekin));
    }
}

template<>
void OrbitIntegrator<coord::Sph>::eval(const double timeOffset, const double xv[],
    /*output*/ double dxdt[], double* accFac) const
{
    double r = xv[0];
    double phi = xv[2];
    int signr = 1, signt = 1;
    double theta = fmod(xv[1], 2*M_PI);
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
    const coord::PosVelSph p(r, theta, phi, xv[3], xv[4], xv[5]);
    double Epot;
    coord::GradSph grad;
    potential.eval(p, accFac ? &Epot : NULL, &grad, NULL, timeBegin + timeOffset);
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
    if(accFac) {
        double Ekin = 0.5 * (pow_2(p.vr) + pow_2(p.vtheta) + pow_2(p.vphi));
        *accFac = fmin(1, fabs(Epot + Ekin) / fmax(fabs(Epot), Ekin));
    }
}


// explicit template instantiations to make sure all of them get compiled
template class OrbitIntegrator<coord::Car>;
template class OrbitIntegrator<coord::Cyl>;
template class OrbitIntegrator<coord::Sph>;

}  // namespace orbit

