/** \file    test_potential_modifiers.cpp
    \date    2022
    \author  Eugene Vasiliev

    Test potential modifiers by integrating and comparing orbits in variously modified potentials
*/
#include "orbit.h"
#include "potential_factory.h"
#include "utils.h"
#include <iostream>
#include <fstream>
#include <cmath>

bool testLess(const char* msg, double val, double limit)
{
    std::cout << msg << val;
    if(!(val<limit)) {
        std::cout << "\033[1;31m **\033[0m\n";
        return false;
    } else {
        std::cout << "\n";
        return true;
    }
}

std::string printPoint(const coord::PosVelCar& point)
{
    char buf[256];
    snprintf(buf, 256, "\t%15.10f %15.10f %15.10f %15.10f %15.10f %15.10f",
        point.x, point.y, point.z, point.vx, point.vy, point.vz);
    return std::string(buf);
}

inline double difPosVel(const coord::PosVelCar& a, const coord::PosVelCar& b)
{
    return sqrt(
        pow_2(a.x -b.x ) + pow_2(a.y -b.y ) + pow_2(a.z -b.z ) +
        pow_2(a.vx-b.vx) + pow_2(a.vy-b.vy) + pow_2(a.vz-b.vz));
}

// compare the hessian returned by the potential with a finite-difference estimate from gradients
template<typename CoordT>
double difHess(const potential::BasePotential& pot, const coord::PosT<CoordT>& pos, double time);
const double delta = 1e-8;  // finite offset in each coordinate

template<>
double difHess(const potential::BasePotential& pot, const coord::PosCar& pos, double time)
{
    coord::GradCar g0, gx, gy, gz;
    coord::HessCar h0;
    pot.eval(pos, NULL, &g0, &h0, time);
    pot.eval(coord::PosCar(pos.x+delta, pos.y, pos.z), NULL, &gx, NULL, time);
    pot.eval(coord::PosCar(pos.x, pos.y+delta, pos.z), NULL, &gy, NULL, time);
    pot.eval(coord::PosCar(pos.x, pos.y, pos.z+delta), NULL, &gz, NULL, time);
    return
    fabs(h0.dx2  - (gx.dx-g0.dx) / delta) +
    fabs(h0.dy2  - (gy.dy-g0.dy) / delta) +
    fabs(h0.dz2  - (gz.dz-g0.dz) / delta) +
    fabs(h0.dxdy - (gx.dy-g0.dy + gy.dx-g0.dx) / delta * 0.5) +
    fabs(h0.dxdz - (gx.dz-g0.dz + gz.dx-g0.dx) / delta * 0.5) +
    fabs(h0.dydz - (gy.dz-g0.dz + gz.dy-g0.dy) / delta * 0.5);
}

template<>
double difHess(const potential::BasePotential& pot, const coord::PosCyl& pos, double time)
{
    coord::GradCyl g0, gR, gz, gp;
    coord::HessCyl h0;
    pot.eval(pos, NULL, &g0, &h0, time);
    pot.eval(coord::PosCyl(pos.R+delta, pos.z, pos.phi), NULL, &gR, NULL, time);
    pot.eval(coord::PosCyl(pos.R, pos.z+delta, pos.phi), NULL, &gz, NULL, time);
    pot.eval(coord::PosCyl(pos.R, pos.z, pos.phi+delta), NULL, &gp, NULL, time);
    return
    fabs(h0.dR2    - (gR.dR  -g0.dR  ) / delta) +
    fabs(h0.dz2    - (gz.dz  -g0.dz  ) / delta) +
    fabs(h0.dphi2  - (gp.dphi-g0.dphi) / delta) +
    fabs(h0.dRdz   - (gR.dz  -g0.dz   + gz.dR-g0.dR) / delta * 0.5) +
    fabs(h0.dRdphi - (gR.dphi-g0.dphi + gp.dR-g0.dR) / delta * 0.5) +
    fabs(h0.dzdphi - (gz.dphi-g0.dphi + gp.dz-g0.dz) / delta * 0.5);
}

template<>
double difHess(const potential::BasePotential& pot, const coord::PosSph& pos, double time)
{
    coord::GradSph g0, gr, gt, gp;
    coord::HessSph h0;
    pot.eval(pos, NULL, &g0, &h0, time);
    pot.eval(coord::PosSph(pos.r+delta, pos.theta, pos.phi), NULL, &gr, NULL, time);
    pot.eval(coord::PosSph(pos.r, pos.theta+delta, pos.phi), NULL, &gt, NULL, time);
    pot.eval(coord::PosSph(pos.r, pos.theta, pos.phi+delta), NULL, &gp, NULL, time);
    return
    fabs(h0.dr2       - (gr.dr    -g0.dr    ) / delta) +
    fabs(h0.dtheta2   - (gt.dtheta-g0.dtheta) / delta) +
    fabs(h0.dphi2     - (gp.dphi  -g0.dphi  ) / delta) +
    fabs(h0.drdtheta  - (gr.dtheta-g0.dtheta + gt.dr    -g0.dr    ) / delta * 0.5) +
    fabs(h0.drdphi    - (gr.dphi  -g0.dphi   + gp.dr    -g0.dr    ) / delta * 0.5) +
    fabs(h0.dthetadphi- (gt.dphi  -g0.dphi   + gp.dtheta-g0.dtheta) / delta * 0.5);
}

// compare the density returned by a Potential and a corresponding Density instance,
// both for a single point or using the batch evaluation for many points (just one, anyway),
// and also with the density computed from the Hessian
template<typename CoordT>
double difDens(const potential::BasePotential& pot, const potential::BaseDensity& den,
    const coord::PosT<CoordT>& pos, double time);

template<>
double difDens(const potential::BasePotential& pot, const potential::BaseDensity& den,
    const coord::PosCar& pos, double time)
{
    coord::GradCar g0;
    coord::HessCar h0;
    pot.eval(pos, NULL, &g0, &h0, time);
    double dh1 = (h0.dx2 + h0.dy2 + h0.dz2) / (4*M_PI);  // density from the Hessian
    double dp1 = pot.density(pos, time), dd1 = den.density(pos, time), dpm, ddm;
    pot.evalmanyDensityCar(1, &pos, &dpm, time);
    den.evalmanyDensityCar(1, &pos, &ddm, time);
    return fmax(fmax(fabs(dp1/dd1-1), fabs(dp1/dh1-1)), fmax(fabs(dp1/dpm-1), fabs(dd1/ddm-1)));
}

template<>
double difDens(const potential::BasePotential& pot, const potential::BaseDensity& den,
    const coord::PosCyl& pos, double time)
{
    coord::GradCyl g0;
    coord::HessCyl h0;
    pot.eval(pos, NULL, &g0, &h0, time);
    double dh1 = (h0.dR2 + g0.dR / pos.R + h0.dz2 + h0.dphi2 / pow_2(pos.R)) / (4*M_PI);
    double dp1 = pot.density(pos, time), dd1 = den.density(pos, time), dpm, ddm;
    pot.evalmanyDensityCyl(1, &pos, &dpm, time);
    den.evalmanyDensityCyl(1, &pos, &ddm, time);
    return fmax(fmax(fabs(dp1/dd1-1), fabs(dp1/dh1-1)), fmax(fabs(dp1/dpm-1), fabs(dd1/ddm-1)));
}

template<>
double difDens(const potential::BasePotential& pot, const potential::BaseDensity& den,
    const coord::PosSph& pos, double time)
{
    coord::GradSph g0;
    coord::HessSph h0;
    pot.eval(pos, NULL, &g0, &h0, time);
    double sth = sin(pos.theta), cth = cos(pos.theta);
    double dh1 = (h0.dr2 + 2*g0.dr / pos.r +
        (h0.dtheta2 + g0.dtheta * cth / sth + h0.dphi2 / pow_2(sth) ) / pow_2(pos.r)) / (4*M_PI);
    double dp1 = pot.density(pos, time), dd1 = den.density(pos, time), dpm, ddm;
    pot.evalmanyDensitySph(1, &pos, &dpm, time);
    den.evalmanyDensitySph(1, &pos, &ddm, time);
    return fmax(fmax(fabs(dp1/dd1-1), fabs(dp1/dh1-1)), fmax(fabs(dp1/dpm-1), fabs(dd1/ddm-1)));
}

// equivalent to orbit::integrateTraj, but can use orbit integrators in different coordinate systems
template<typename CoordT>
inline std::vector<std::pair<coord::PosVelCar, double> > integrateTraj(
    const coord::PosVelCar& initialConditions,
    const double totalTime,
    const double samplingInterval,
    const potential::BasePotential& potential,
    const double Omega = 0)
{
    std::vector<std::pair<coord::PosVelCar, double> > output;
    if(samplingInterval > 0)
        // reserve space for the trajectory, including one extra point for the final state
        output.reserve((totalTime>=0 ? totalTime : -totalTime) * (1+1e-15) / samplingInterval + 1);
    orbit::OrbitIntegrator<CoordT> orbint(potential, Omega);
    orbint.addRuntimeFnc(orbit::PtrRuntimeFnc(
        new orbit::RuntimeTrajectory(orbint, samplingInterval, output)));
    orbint.init(initialConditions);
    orbint.run(totalTime);
    return output;
}

// helper class for computing the position/velocity at any time on a cubic trajectory
// with the given initial position, velocity, acceleration and jerk
class CubicTrajectory {
public:
    const double posx, posy, posz, velx, vely, velz, accx, accy, accz, jerx, jery, jerz;
    CubicTrajectory(
        double _posx, double _posy, double _posz, double _velx, double _vely, double _velz,
        double _accx, double _accy, double _accz, double _jerx, double _jery, double _jerz) :
        posx(_posx), posy(_posy), posz(_posz), velx(_velx), vely(_vely), velz(_velz),
        accx(_accx), accy(_accy), accz(_accz), jerx(_jerx), jery(_jery), jerz(_jerz) {}
    coord::PosVelCar operator() (double time) const {
        return coord::PosVelCar(
            posx + time * (velx + 0.5 * time * (accx + 1./3 * time * jerx)),
            posy + time * (vely + 0.5 * time * (accy + 1./3 * time * jery)),
            posz + time * (velz + 0.5 * time * (accz + 1./3 * time * jerz)),
            velx + time * (accx + 0.5 * time *  jerx),
            vely + time * (accy + 0.5 * time *  jery),
            velz + time * (accz + 0.5 * time *  jerz) );
    }
};

// test the equivalence of a Rotating modifier with a constant rotation angle
// and a Tilted modifier with beta=0 (i.e. rotated around the z axis by the same angle)
bool testTiltedRotating()
{
    utils::KeyValueMap
        potParams("type=Ferrers, scaleRadius=2, axisRatioY=0.8, axisRatioZ=0.6"),
        potParamsTilted = potParams,
        potParamsRotating = potParams;
    double alpha=0.3, gamma=0.4;  // two angles will effectively add up, since beta=0
    potParamsTilted.set("orientation", utils::toString(alpha)+",0,"+utils::toString(gamma));
    potParamsRotating.set("rotation", utils::toString(alpha+gamma));
    potential::PtrPotential potOrig     = potential::createPotential(potParams);
    potential::PtrPotential potTilted   = potential::createPotential(potParamsTilted);
    potential::PtrPotential potRotating = potential::createPotential(potParamsRotating);
    coord::PosCyl  pos(1.0, 0.5, 0.3);  // fiducial point for the original potential..
    coord::PosCyl rpos(pos.R, pos.z, pos.phi + alpha+gamma);  // and a rotated point for the other two
    double valOrig, valTilted, valRotating;
    coord::GradCyl gradOrig, gradTilted, gradRotating;
    coord::HessCyl hessOrig, hessTilted, hessRotating;
    potOrig    ->eval( pos, &valOrig,     &gradOrig,     &hessOrig);
    potTilted  ->eval(rpos, &valTilted,   &gradTilted,   &hessTilted);
    potRotating->eval(rpos, &valRotating, &gradRotating, &hessRotating);
    double difVal  = fabs(valOrig-valTilted) + fabs(valOrig-valRotating);
    double difGrad =
        fabs(gradOrig.dR  -gradTilted.dR  ) + fabs(gradOrig.dR  -gradRotating.dR) +
        fabs(gradOrig.dz  -gradTilted.dz  ) + fabs(gradOrig.dz  -gradRotating.dz) +
        fabs(gradOrig.dphi-gradTilted.dphi) + fabs(gradOrig.dphi-gradRotating.dphi);
    double difHess =
        fabs(hessOrig.dR2   -hessTilted.dR2   ) + fabs(hessOrig.dR2   -hessRotating.dR2)   +
        fabs(hessOrig.dz2   -hessTilted.dz2   ) + fabs(hessOrig.dz2   -hessRotating.dz2)   +
        fabs(hessOrig.dphi2 -hessTilted.dphi2 ) + fabs(hessOrig.dphi2 -hessRotating.dphi2) +
        fabs(hessOrig.dRdz  -hessTilted.dRdz  ) + fabs(hessOrig.dRdz  -hessRotating.dRdz)  +
        fabs(hessOrig.dRdphi-hessTilted.dRdphi) + fabs(hessOrig.dRdphi-hessRotating.dRdphi)+
        fabs(hessOrig.dzdphi-hessTilted.dzdphi) + fabs(hessOrig.dzdphi-hessRotating.dzdphi);
    if(difVal > 1e-14 || difGrad > 1e-14 || difHess > 1e-14) {
        std::cout << "rotating and tilted potentials inconsistent\033[1;31m **\033[0m\n";
        return false;
    }
    return true;
}

const double totalTime = 100.0, timeStep = 0.005 * totalTime;
const double Omega = -0.1;  // rotation frequency (arbitrary)

// the main test suite comparing the orbits integrated in different coordinate systems
// for variously modified potentials, verifying the gradients and hessians in different coordinates
template<typename CoordT> bool test(
    const coord::PosVelCar& initCond,
    const coord::PosVelCar& initTilt,
    const coord::PosVelCar& initMove,
    const coord::Orientation& orientation,
    const CubicTrajectory& trajCenter,
    potential::PtrPotential& potOrig,
    potential::PtrPotential& potTilt,
    potential::PtrPotential& potMove,
    potential::PtrPotential& potSpin,
    potential::PtrPotential& potScal,
    potential::PtrPotential& potEvol,
    potential::PtrDensity& denTilt,
    potential::PtrDensity& denMove,
    potential::PtrDensity& denSpin,
    potential::PtrDensity& denScal)
{
    std::cout << "\033[1;37m" << CoordT::name() << "\033[0m\n";

    // integrate the orbit in different potentials with correspondingly modified initial conditions
    std::vector< std::pair<coord::PosVelCar, double> >
    trajOrig = integrateTraj<CoordT>(initCond, totalTime, timeStep, *potOrig),
    trajTilt = integrateTraj<CoordT>(initTilt, totalTime, timeStep, *potTilt),
    trajMove = integrateTraj<CoordT>(initMove, totalTime, timeStep, *potMove),
    trajSpin = integrateTraj<CoordT>(initCond, totalTime, timeStep, *potSpin),
    trajRotF = integrateTraj<CoordT>(initCond, totalTime, timeStep, *potOrig, Omega),
    trajEvol = integrateTraj<CoordT>(initCond, totalTime, timeStep, *potEvol),
    trajScal = integrateTraj<CoordT>(initCond, totalTime, timeStep, *potScal);

    if( trajOrig.size() != trajTilt.size() || trajOrig.size() != trajMove.size() ||
        trajOrig.size() != trajSpin.size() || trajOrig.size() != trajRotF.size() ||
        trajOrig.size() != trajEvol.size() || trajOrig.size() != trajScal.size() )
    {
        std::cout << "Trajectory sizes differ!\n";
        return false;
    }

    const bool output = utils::verbosityLevel >= utils::VL_VERBOSE;
    std::ofstream strm;
    if(output)
        strm.open(("test_potential_modifiers_"+std::string(CoordT::name())+".txt").c_str());

    double
        difTiltTraj = 0, difMoveTraj = 0, difSpinTraj = 0, difScalTraj = 0,
        difTiltHess = 0, difMoveHess = 0, difSpinHess = 0, difScalHess = 0, difEvolHess = 0,
        difTiltDens = 0, difMoveDens = 0, difSpinDens = 0, difScalDens = 0;
    for(size_t i=0; i<trajOrig.size(); i++) {
        double time = trajOrig[i].second, cosa = cos(Omega * time), sina = sin(Omega * time);

        // orbit integrated in a tilted potential converted back to un-tilted reference frame
        coord::PosVelCar pointTilt = orientation.toRotated(trajTilt[i].first);

        // orbit integrated in a moving potential converted into the comoving reference frame
        coord::PosVelCar center = trajCenter(time);  // origin of the moving potential at this time
        coord::PosVelCar pointMove(
            trajMove[i].first.x  - center.x,
            trajMove[i].first.y  - center.y,
            trajMove[i].first.z  - center.z,
            trajMove[i].first.vx - center.vx,
            trajMove[i].first.vy - center.vy,
            trajMove[i].first.vz - center.vz);

        // orbit integrated in the rotating frame converted back to inertial frame
        // (to be compared with an orbit integrated in the inertial frame using a rotating potential)
        coord::PosVelCar pointIner(
            trajRotF[i].first.x * cosa - trajRotF[i].first.y * sina,
            trajRotF[i].first.y * cosa + trajRotF[i].first.x * sina,
            trajRotF[i].first.z,
            trajRotF[i].first.vx * cosa - trajRotF[i].first.vy * sina,
            trajRotF[i].first.vy * cosa + trajRotF[i].first.vx * sina,
            trajRotF[i].first.vz);
        difTiltTraj = fmax(difTiltTraj, difPosVel(trajOrig[i].first, pointTilt));
        difMoveTraj = fmax(difMoveTraj, difPosVel(trajOrig[i].first, pointMove));
        difSpinTraj = fmax(difSpinTraj, difPosVel(trajSpin[i].first, pointIner));
        difScalTraj = fmax(difScalTraj, difPosVel(trajScal[i].first, trajEvol[i].first));

        // compare finite-difference and analytic hessians
        difTiltHess = fmax(difTiltHess,
            difHess(*potTilt, coord::toPos<coord::Car, CoordT>(trajTilt[i].first), time));
        difMoveHess = fmax(difMoveHess,
            difHess(*potMove, coord::toPos<coord::Car, CoordT>(trajMove[i].first), time));
        difSpinHess = fmax(difSpinHess,
            difHess(*potSpin, coord::toPos<coord::Car, CoordT>(trajSpin[i].first), time));
        difScalHess = fmax(difScalHess,
            difHess(*potScal, coord::toPos<coord::Car, CoordT>(trajScal[i].first), time));
        difEvolHess = fmax(difEvolHess,
            difHess(*potEvol, coord::toPos<coord::Car, CoordT>(trajEvol[i].first), time));

        // compare corresponding potential and density modifiers (these are separate classes)
        difTiltDens = fmax(difTiltDens,
            difDens(*potTilt, *denTilt, coord::toPos<coord::Car, CoordT>(trajTilt[i].first), time));
        difMoveDens = fmax(difMoveDens,
            difDens(*potMove, *denMove, coord::toPos<coord::Car, CoordT>(trajMove[i].first), time));
        difSpinDens = fmax(difSpinDens,
            difDens(*potSpin, *denSpin, coord::toPos<coord::Car, CoordT>(trajSpin[i].first), time));
        difScalDens = fmax(difScalDens,
            difDens(*potScal, *denScal, coord::toPos<coord::Car, CoordT>(trajScal[i].first), time));

        if(output) {
            strm << utils::pp(time, 7) << printPoint(trajOrig[i].first) <<
                printPoint(trajTilt[i].first) << printPoint(pointTilt) <<
                printPoint(trajMove[i].first) << printPoint(pointMove) <<
                printPoint(trajSpin[i].first) << printPoint(pointIner) <<
                printPoint(trajScal[i].first) << printPoint(trajEvol[i].first) <<
            "\n";
        }
    }
    if(output)
        strm.close();

    bool allok = true;
    allok &= testLess("orbits in the original and tilted potentials:        ", difTiltTraj, 1e-5);
    allok &= testLess("original and shifted potentials:                     ", difMoveTraj, 1e-5);
    allok &= testLess("rotating potential and orbit in the rotating frame:  ", difSpinTraj, 1e-5);
    allok &= testLess("evolving and scaled potentials:                      ", difScalTraj, 1e-5);
    allok &= testLess("finite-difference hessian in the tilted potential:   ", difTiltHess, 1e-5);
    allok &= testLess("finite-difference hessian in the shifted potential:  ", difMoveHess, 1e-5);
    allok &= testLess("finite-difference hessian in the rotating potential: ", difSpinHess, 1e-5);
    allok &= testLess("finite-difference hessian in the scaled potential:   ", difScalHess, 1e-5);
    allok &= testLess("finite-difference hessian in the evolving potential: ", difEvolHess, 1e-5);
    allok &= testLess("density and hessian for the tilted potential:        ", difTiltDens, 1e-12);
    allok &= testLess("density and hessian for the shifted potential:       ", difMoveDens, 1e-12);
    allok &= testLess("density and hessian for the rotating potential:      ", difSpinDens, 1e-12);
    allok &= testLess("density and hessian for the scaled potential:        ", difScalDens, 1e-12);
    return allok;
}

int main()
{
    bool allok = testTiltedRotating();

    const double v0 = 1.0;  // amplitude of the circular velocity for the logarithmic potential
    utils::KeyValueMap potParams("type=Logarithmic, scaleRadius=0, axisRatioY=0.8, axisRatioZ=0.6");
    potParams.set("v0", v0);
    const coord::PosVelCar initCond(0.5, 0.6, 0.7,-0.4, 0.5,-0.3);
    const double alpha=0.4, beta=0.5, gamma=0.6;  // fiducial tilt orientation
    const coord::Orientation orientation(alpha, beta, gamma);
    utils::KeyValueMap potParamsTilt = potParams;
    potParamsTilt.set("orientation",
        utils::toString(alpha) + "," + utils::toString(beta) + "," + utils::toString(gamma));
    const coord::PosVelCar initTilt = orientation.fromRotated(initCond);

    // put the potential center on a curved trajectory,
    // where each coordinate is a cubic function of time,
    // and add a spatially uniform, time-dependent acceleration (linearly changing with time)
    // designed to compensate the curvilinear motion of the potential center:
    // the orbit integration will be carried in the non-inertial frame where the potential is moving,
    // but the orbit relative to the origin of the moving potential should be equivalent
    // to an orbit in the inertial frame where the potential is static.
    CubicTrajectory trajCenter(
        /*pos*/  1.0,   0.5,  -1.0,
        /*vel*/ -0.05,  0.03, +0.01,
        /*acc*/  5e-4,-15e-4, 11e-4,
        /*jerk*/ 9e-6,  3e-5, -3e-5);
    // the trajectory is specified as a clamped cubic spline,
    // providing the position and velocity of the center at both endpoints of the time interval
    const char tmpcenter[] = "tmpcenter.txt";
    std::ofstream strm(tmpcenter);
    strm << "0 "      <<        printPoint(trajCenter(0.0))       << "\n";
    strm << totalTime << " " << printPoint(trajCenter(totalTime)) << "\n";
    strm.close();
    // corresponding acceleration is a linear function of time, also provided at both endpoints
    const char tmpaccel[] = "tmpaccel.txt";
    strm.open(tmpaccel);
    strm << "0 " << trajCenter.accx << " " << trajCenter.accy << " " << trajCenter.accz << "\n";
    strm << totalTime << " " <<
        (trajCenter.accx + trajCenter.jerx * totalTime) << " " <<
        (trajCenter.accy + trajCenter.jery * totalTime) << " " <<
        (trajCenter.accz + trajCenter.jerz * totalTime) << "\n";
    strm.close();
    std::vector<utils::KeyValueMap> potParamsMove(2);    // two-component composite potential:
    potParamsMove[0] = potParams;                        // the original potential
    potParamsMove[0].set("center", tmpcenter);           // decorated with a Shifted modifier,
    potParamsMove[1].set("type", "UniformAcceleration"); // and a uniform acceleration
    potParamsMove[1].set("file", tmpaccel);              // compensating for the non-inertial frame
    const coord::PosVelCar initMove(
        initCond.x  + trajCenter.posx, initCond.y  + trajCenter.posy, initCond.z  + trajCenter.posz,
        initCond.vx + trajCenter.velx, initCond.vy + trajCenter.vely, initCond.vz + trajCenter.velz);

    // steady rotation about the z axis imparted on the potential
    // (compare with orbit integration in rotating frame)
    const char tmprotation[] = "tmprotation.txt";
    strm.open(tmprotation);
    strm << "0 0\n1 " << Omega << "\n";
    strm.close();
    utils::KeyValueMap potParamsSpin = potParams;
    potParamsSpin.set("rotation", tmprotation);

    // potential growing in size and mass, represented in two variants:
    // (a) a linearly interpolated Evolving potential, changing from pot_init to pot_final,
    // (b) as a single instance of the initial potential, suitably scaled in size and mass.
    // Since "Evolving" can only linearly interpolate between two potentials,
    // we must use a scale-free model (Logarithmic) and change the amplitude of the potential
    // linearly with time (from 1 at the initial moment to finalPotScale at the final moment):
    // Phi ~ 1 + (finalPotScale-1) * (t/timeTotal) .
    // The "Scaled" potential modifier, on the other hand, can vary both mass and length scales
    // as cubic splines in time.
    // Although the potential remains scale-free, we can still formally modify its length scale,
    // making it linearly increasing with time:
    // L ~ 1 + (finalLengthScale-1) * (t/totalTime) .
    // The mass normalization then becomes a quadratic function of time:
    // M ~ (1 + (finalPotScale-1) * (t/totalTime)) * (1 + (finalLengthScale-1) * (t/totalTime)),
    // and its time derivative is thus
    // dM/dt = 1 + (finalPotScale-1 + finalLengthScale-1) / totalTime
    //           + (finalPotScale-1)*(finalLengthScale-1) * 2 * t / totalTime^2 .
    // This can be represented by a clamped cubic spline
    // (explicitly providing time derivatives at both endpoints of the interval).
    const double finalPotScale = 1.44, finalLengthScale = 1.25;
    // prepare the temporary ini files for the Evolving potential
    utils::KeyValueMap potParamsFinal = potParams;
    potParamsFinal.set("v0", v0 * sqrt(finalPotScale));
    const char tmppotinit[] = "tmppotinit.ini", tmppotfinal[] = "tmppotfinal.ini";
    strm.open(tmppotinit);
    strm << "[Potential init]\n";
    std::vector<std::string> lines = potParams.dumpLines();
    for(size_t i=0; i<lines.size(); i++)
        strm << lines[i] << '\n';
    strm.close();
    strm.open(tmppotfinal);
    strm << "[Potential final]\n";
    lines = potParamsFinal.dumpLines();
    for(size_t i=0; i<lines.size(); i++)
        strm << lines[i] << '\n';
    strm.close();
    utils::KeyValueMap potParamsEvol("type=Evolving interpLinear=true");
    potParamsEvol.set("", "Timestamps");
    potParamsEvol.set("", "0 " + std::string(tmppotinit));
    potParamsEvol.set("", utils::toString(totalTime) + " " + std::string(tmppotfinal));
    // prepare the text file describing the time variation of mass and length scales
    const char tmpscaled[] = "tmpscaled.txt";
    strm.open(tmpscaled);
    double dLdt = (finalLengthScale-1) / totalTime;    // same at both endpoints
    double dMdt_init = (finalPotScale-1 + finalLengthScale-1) / totalTime;
    double dMdt_final= (finalPotScale-1)*(finalLengthScale-1) / totalTime * 2 + dMdt_init;
    strm << "0 1 1 " << dMdt_init << " " << dLdt << "\n";
    strm << totalTime << " " << (finalPotScale * finalLengthScale) << " " << finalLengthScale <<
        " " << dMdt_final << " " << dLdt << "\n";
    strm.close();
    utils::KeyValueMap potParamsScal = potParams;
    potParamsScal.set("scale", tmpscaled);

    // create the variously modified potentials and correspondingly modified density instances
    // (these are independent classes, but should behave identically when evaluating the density)
    potential::PtrPotential potOrig = potential::createPotential(potParams);
    potential::PtrPotential potTilt = potential::createPotential(potParamsTilt);
    potential::PtrPotential potMove = potential::createPotential(potParamsMove);
    potential::PtrPotential potSpin = potential::createPotential(potParamsSpin);
    potential::PtrPotential potScal = potential::createPotential(potParamsScal);
    potential::PtrPotential potEvol = potential::createPotential(potParamsEvol);
    potential::PtrDensity   denTilt = potential::createDensity  (potParamsTilt);
    potential::PtrDensity   denMove = potential::createDensity  (potParamsMove[0]);
    potential::PtrDensity   denSpin = potential::createDensity  (potParamsSpin);
    potential::PtrDensity   denScal = potential::createDensity  (potParamsScal);
    // cleanup temporary text files used to initialize the above potentials
    std::remove(tmpcenter);
    std::remove(tmpaccel);
    std::remove(tmprotation);
    std::remove(tmpscaled);
    std::remove(tmppotinit);
    std::remove(tmppotfinal);

    // check the equivalence of Evolving and Scaled potentials (or rather, forces and densities,
    // since the potential zero-point changes differently in these two cases, but it doesn't matter)
    coord::GradCar gradOrig, gradScal, gradEvol;
    potOrig->eval(initCond, NULL, &gradOrig, NULL, /*t*/ 0);
    double densOrig = potOrig->density(initCond, /*t*/ 0);
    double difEvolGrad = 0, difEvolDens = 0;
    for(int i=0, N=100; i<=N; i++) {
        double t = totalTime*i/N;
        potEvol->eval(initCond, NULL, &gradEvol, NULL, t);
        potScal->eval(initCond, NULL, &gradScal, NULL, t);
        double densEvol = potEvol->density(initCond, t), densScal = potScal->density(initCond, t);
        // the potential gradient and the density are expected to scale linearly with time
        double mult = 1 + (finalPotScale-1) * i/N;
        difEvolGrad = fmax(difEvolGrad, fmax(
            fabs(gradEvol.dx / gradOrig.dx - mult),
            fabs(gradScal.dx / gradOrig.dx - mult) ) );
        difEvolDens = fmax(difEvolDens, fmax(
            fabs(densEvol / densOrig - mult),
            fabs(densScal / densOrig - mult) ) );
    }
    if(! (difEvolGrad < 1e-14 && difEvolDens < 1e-14) ) {
        std::cout << "evolving and scaled potentials inconsistent\033[1;31m **\033[0m\n";
        allok = false;
    }

    // run the orbit integrations in three different coordinate systems for all these potentials
    allok &= test<coord::Car>(initCond, initTilt, initMove, orientation, trajCenter,
        potOrig, potTilt, potMove, potSpin, potScal, potEvol, denTilt, denMove, denSpin, denScal);
    allok &= test<coord::Cyl>(initCond, initTilt, initMove, orientation, trajCenter,
        potOrig, potTilt, potMove, potSpin, potScal, potEvol, denTilt, denMove, denSpin, denScal);
    allok &= test<coord::Sph>(initCond, initTilt, initMove, orientation, trajCenter,
        potOrig, potTilt, potMove, potSpin, potScal, potEvol, denTilt, denMove, denSpin, denScal);
    if(allok)
        std::cout << "\033[1;32mALL TESTS PASSED\033[0m\n";
    else
        std::cout << "\033[1;31mSOME TESTS FAILED\033[0m\n";
    return 0;
}
