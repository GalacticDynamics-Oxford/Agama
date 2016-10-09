/** \file    debug_utils.h 
    \brief   Auxiliary routines for comparing and printing data types from coord.h and actions_base.h
    \author  Eugene Vasiliev
    \date    2015
*/ 
#pragma once
#include "coord.h"
#include "actions_base.h"
#include "math_core.h"
#include "math_fit.h"
#include "math_linalg.h"
#include <cmath>
#include <ostream>

// Coordinate tools
namespace coord {
/// comparison functions for positions, gradients and hessians

inline bool equalPos(const PosCar& p1, const PosCar& p2, const double eps) {
    return fabs(p1.x-p2.x)<eps*(1+fabs(p1.x))
        && fabs(p1.y-p2.y)<eps*(1+fabs(p1.y))
        && fabs(p1.z-p2.z)<eps*(1+fabs(p1.z)); }
inline bool equalPos(const PosCyl& p1, const PosCyl& p2, const double eps) {
    return fabs(p1.R-p2.R)<eps*(1+fabs(p1.R))
        && fabs(p1.z-p2.z)<eps*(1+fabs(p1.z))
        && fabs(p1.phi-p2.phi)<eps*(1+fabs(p1.phi)); }
inline bool equalPos(const PosSph& p1, const PosSph& p2, const double eps) {
    return fabs(p1.r-p2.r)<eps*(1+fabs(p1.r))
        && fabs(p1.theta-p2.theta)<eps*(1+fabs(p1.theta))
        && fabs(p1.phi-p2.phi)<eps*(1+fabs(p1.phi)); }

inline bool equalPosVel(const PosVelCar& p1, const PosVelCar& p2, const double eps) {
    return fabs(p1.x-p2.x)<eps*(1+fabs(p1.x))
        && fabs(p1.y-p2.y)<eps*(1+fabs(p1.y))
        && fabs(p1.z-p2.z)<eps*(1+fabs(p1.z))
        && fabs(p2.vx-p2.vx)<eps*(1+fabs(p1.vx))
        && fabs(p1.vy-p2.vy)<eps*(1+fabs(p1.vy))
        && fabs(p1.vz-p2.vz)<eps*(1+fabs(p1.vz)); }
inline bool equalPosVel(const PosVelCyl& p1, const PosVelCyl& p2, const double eps) {
    return fabs(p1.R-p2.R)<eps*(1+fabs(p1.R))
        && fabs(p1.z-p2.z)<eps*(1+fabs(p1.z))
        && fabs(p1.phi-p2.phi)<eps*(1+fabs(p1.phi))
        && fabs(p2.vR-p2.vR)<eps*(1+fabs(p1.vR))
        && fabs(p1.vz-p2.vz)<eps*(1+fabs(p1.vz))
        && fabs(p1.vphi-p2.vphi)<eps*(1+fabs(p1.vphi)); }
inline bool equalPosVel(const PosVelSph& p1, const PosVelSph& p2, const double eps) {
    return fabs(p1.r-p2.r)<eps*(1+fabs(p1.r))
        && fabs(p1.theta-p2.theta)<eps*(1+fabs(p1.theta))
        && fabs(p1.phi-p2.phi)<eps*(1+fabs(p1.phi))
        && fabs(p2.vr-p2.vr)<eps*(1+fabs(p1.vr))
        && fabs(p1.vtheta-p2.vtheta)<eps*(1+fabs(p1.vtheta))
        && fabs(p1.vphi-p2.vphi)<eps*(1+fabs(p1.vphi)); }
inline bool equalPosVel(const PosVelSphMod& p1, const PosVelSphMod& p2, const double eps) {
    return fabs(p1.r-p2.r)<eps*(1+fabs(p1.r))
        && fabs(p1.tau-p2.tau)<eps*(1+fabs(p1.tau))
        && fabs(p1.phi-p2.phi)<eps*(1+fabs(p1.phi))
        && fabs(p2.pr-p2.pr)<eps*(1+fabs(p1.pr))
        && fabs(p1.ptau-p2.ptau)<eps*(1+fabs(p1.ptau))
        && fabs(p1.pphi-p2.pphi)<eps*(1+fabs(p1.pphi)); }

inline bool equalGrad(const GradCar& g1, const GradCar& g2, const double eps) {
    return fabs(g1.dx-g2.dx)<eps && fabs(g1.dy-g2.dy)<eps && fabs(g1.dz-g2.dz)<eps; }
inline bool equalGrad(const GradCyl& g1, const GradCyl& g2, const double eps) {
    return fabs(g1.dR-g2.dR)<eps && fabs(g1.dphi-g2.dphi)<eps && fabs(g1.dz-g2.dz)<eps; }
inline bool equalGrad(const GradSph& g1, const GradSph& g2, const double eps) {
    return fabs(g1.dr-g2.dr)<eps && fabs(g1.dtheta-g2.dtheta)<eps && fabs(g1.dphi-g2.dphi)<eps; }

inline bool equalHess(const HessCar& h1, const HessCar& h2, const double eps) {
    return fabs(h1.dx2-h2.dx2)<eps && fabs(h1.dy2-h2.dy2)<eps && fabs(h1.dz2-h2.dz2)<eps &&
        fabs(h1.dxdy-h2.dxdy)<eps && fabs(h1.dydz-h2.dydz)<eps && fabs(h1.dxdz-h2.dxdz)<eps; }
inline bool equalHess(const HessCyl& h1, const HessCyl& h2, const double eps) {
    return fabs(h1.dR2-h2.dR2)<eps && fabs(h1.dphi2-h2.dphi2)<eps && fabs(h1.dz2-h2.dz2)<eps &&
        fabs(h1.dRdphi-h2.dRdphi)<eps && fabs(h1.dzdphi-h2.dzdphi)<eps && fabs(h1.dRdz-h2.dRdz)<eps; }
inline bool equalHess(const HessSph& h1, const HessSph& h2, const double eps) {
    return fabs(h1.dr2-h2.dr2)<eps && fabs(h1.dtheta2-h2.dtheta2)<eps && fabs(h1.dphi2-h2.dphi2)<eps &&
        fabs(h1.drdtheta-h2.drdtheta)<eps && fabs(h1.drdphi-h2.drdphi)<eps && fabs(h1.dthetadphi-h2.dthetadphi)<eps; }
}  // namespace

/// printout functions - outside the namespace
inline std::ostream& operator<< (std::ostream& s, const coord::PosCar& p) {
    s << "x: "<<p.x <<"  y: "<<p.y <<"  z: "<<p.z<< "   ";
    return s;
}
inline std::ostream& operator<< (std::ostream& s, const coord::PosCyl& p) {
    s << "R: "<<p.R <<"  z: "<<p.z <<"  phi: "<<p.phi<< "   ";
    return s;
}
inline std::ostream& operator<< (std::ostream& s, const coord::PosSph& p) {
    s << "r: "<<p.r <<"  theta: "<<p.theta <<"  phi: "<<p.phi<< "   ";
    return s;
}
inline std::ostream& operator<< (std::ostream& s, const coord::PosSphMod& p) {
    s << "r: "<<p.r <<"  theta: "<<p.tau <<"  phi: "<<p.phi<< "   ";
    return s;
}

inline std::ostream& operator<< (std::ostream& s, const coord::PosVelCar& p) {
    s << "x: "<< p.x << "  y: "<< p.y << "  z: "<< p.z<< "  "
        "vx: "<<p.vx <<"  vy: "<<p.vy <<"  vz: "<<p.vz<< "   ";
    return s;
}
inline std::ostream& operator<< (std::ostream& s, const coord::PosVelCyl& p) {
    s << "R: " <<p.R << "  z: "<< p.z << "  phi: " <<p.phi<< "  "
        "vR: "<<p.vR <<"  vz: "<<p.vz <<"  vphi: "<<p.vphi<< "   ";
    return s;
}
inline std::ostream& operator<< (std::ostream& s, const coord::PosVelSph& p) {
    s << "r: "<< p.r << "  theta: "<< p.theta << "  phi: "<< p.phi<< "  "
        "vr: "<<p.vr <<"  vtheta: "<<p.vtheta <<"  vphi: "<<p.vphi<< "   ";
    return s;
}
inline std::ostream& operator<< (std::ostream& s, const coord::PosVelSphMod& p) {
    s << "r: "<< p.r << "  tau: "<< p.tau << "  phi: "<< p.phi<< "  "
        "pr: "<<p.pr <<"  ptau: "<<p.ptau <<"  pphi: "<<p.pphi<< "   ";
    return s;
}

inline std::ostream& operator<< (std::ostream& s, const coord::GradCar& p) {
    s << "dx: "<<p.dx <<"  dy: "<<p.dy <<"  dz: "<<p.dz<< "   ";
    return s;
}
inline std::ostream& operator<< (std::ostream& s, const coord::GradCyl& p) {
    s << "dR: "<<p.dR <<"  dz: "<<p.dz <<"  dphi: "<<p.dphi<< "   ";
    return s;
}
inline std::ostream& operator<< (std::ostream& s, const coord::GradSph& p) {
    s << "dr: "<<p.dr <<"  dtheta: "<<p.dtheta <<"  dphi: "<<p.dphi<< "   ";
    return s;
}

inline std::ostream& operator<< (std::ostream& s, const coord::HessCar& p) {
    s << "dx2: "<< p.dx2 << "  dy2: "<< p.dy2 << "  dz2: "<< p.dz2<< "  "
        "dxdy: "<<p.dxdy <<"  dxdz: "<<p.dxdz <<"  dydz: "<<p.dydz<< "   ";
    return s;
}
inline std::ostream& operator<< (std::ostream& s, const coord::HessCyl& p) {
    s << "dR2: "<< p.dR2 << "  dz2: "<< p.dz2 << "  dphi2: "<< p.dphi2<< "  "
        "dRdz: "<< p.dRdz<< "  dRdphi: "<< p.dRdphi <<"  dzdphi: "<< p.dzdphi << "   ";
    return s;
}
inline std::ostream& operator<< (std::ostream& s, const coord::HessSph& p) {
    s << "dr2: "<<p.dr2 <<"  dtheta2: "<<p.dtheta2 <<"  dphi2: "<<p.dphi2<< "  "
        "drdtheta: "<< p.drdtheta << "  drdphi: "<< p.drdphi << "  dthetaphi: "<< p.dthetadphi<< "   ";
    return s;
}

// Action tools
namespace actions{

/// Helper class to compute scatter in actions
class ActionStat{
public:
    math::Averager Jr, Jz, Jphi;
    actions::Actions avg, rms;
    void add(const actions::Actions& act) {
        Jr.add(act.Jr);
        Jz.add(act.Jz);
        Jphi.add(act.Jphi);
    }
    void finish() {
        avg.Jr=Jr.mean(); rms.Jr=sqrt(Jr.disp());
        avg.Jz=Jz.mean(); rms.Jz=sqrt(Jz.disp());
        avg.Jphi=Jphi.mean(); rms.Jphi=sqrt(Jphi.disp());
    }
};

inline void add_unwrap(const double val, std::vector<double>& vec)
{
    if(vec.size()==0)
        vec.push_back(val);
    else
        vec.push_back(math::unwrapAngle(val, vec.back()));
}

/// Helper class to check linearity of angles evolution
class AngleStat{
public:
    std::vector<double> thetar, thetaz, thetaphi, time;
    double freqr, freqz, freqphi;
    double dispr, dispz, dispphi;
    void add(double t, const actions::Angles& a) {
        time.push_back(t);
        add_unwrap(a.thetar, thetar);
        add_unwrap(a.thetaz, thetaz);
        add_unwrap(a.thetaphi, thetaphi);
    }
    void finish() {
        double bla;
        math::linearFit(time, thetar, NULL, freqr, bla, &dispr);
        math::linearFit(time, thetaz, NULL, freqz, bla, &dispz);
        math::linearFit(time, thetaphi, NULL, freqphi, bla, &dispphi);
    }
};
}  // namespace

// printout functions are declared outside the namespace
inline std::ostream& operator<< (std::ostream& s, const actions::Actions& a) {
    s << "Jr: "<< a.Jr <<"  Jz: "<< a.Jz <<"  Jphi: "<< a.Jphi <<"  ";
    return s;
}
inline std::ostream& operator<< (std::ostream& s, const actions::Angles& a) {
    s << "thetar: "<< a.thetar <<"  thetaz: "<< a.thetaz <<"  thetaphi: "<< a.thetaphi <<"  ";
    return s;
}
inline std::ostream& operator<< (std::ostream& s, const actions::ActionAngles& a) {
    s << "Jr: "<< a.Jr <<"  Jz: "<< a.Jz <<"  Jphi: "<< a.Jphi <<"  "<<
         "thetar: "<< a.thetar <<"  thetaz: "<< a.thetaz <<"  thetaphi: "<< a.thetaphi <<"  ";
    return s;
}
inline std::ostream& operator<< (std::ostream& s, const actions::Frequencies& f) {
    s << "Omegar: "<< f.Omegar <<"  Omegaz: "<< f.Omegaz <<"  Omegaphi: "<< f.Omegaphi <<"  ";
    return s;
}

inline std::ostream& operator<< (std::ostream& s, const std::vector<double>& v) {
    for(unsigned int i=0; i<v.size(); i++)
        s << v[i] << '\n';
    return s;
}
inline std::ostream& operator<< (std::ostream& s, const math::Matrix<double>& m) {
    for(unsigned int i=0; i<m.rows(); i++) {
        for(unsigned int j=0; j<m.cols(); j++)
            s << m(i,j) << ' ';
        s << '\n';
    }
    return s;
}
