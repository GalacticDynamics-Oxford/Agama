//
//  main.cpp
//  Torus2
//
//  Created by Brian Jia Jiunn Khor on 07/08/2015.
//  Copyright (c) 2015 Brian Jia Jiunn Khor. All rights reserved.
//

#include <iostream>
#include <iomanip>
#include <fstream>
#include "torus/Torus.h"
#include "potential_factory.h"
#include "debug_utils.h"

/// Auxiliary class for using any of BasePotential-derived potentials with Torus code
class TorusPotentialWrapper: public Torus::Potential{
public:
    TorusPotentialWrapper(const potential::BasePotential& _poten) : poten(_poten) {};
    virtual ~TorusPotentialWrapper() {};
    virtual double operator()(const double R, const double z) const {
        return poten.value(coord::PosCyl(R, z, 0));
    }
    virtual double operator()(const double R, const double z, double& dPhidR, double& dPhidz) const {
        double val;
        coord::GradCyl grad;
        poten.eval(coord::PosCyl(R, z, 0), &val, &grad);
        dPhidR = grad.dR;
        dPhidz = grad.dz;
        return val;
    }
    virtual double RfromLc(double Lz, double* =0) const {
        return R_from_Lz(poten, Lz);
    }
    virtual double LfromRc(double R, double* ) const {
        return v_circ(poten, R) * R;
    }
    virtual Torus::Frequencies KapNuOm(double R) const {
        Torus::Frequencies freq;
        epicycleFreqs(poten, R, freq[0], freq[1], freq[2]);
        return freq;
    }
private:
    const potential::BasePotential& poten;
};

void test(bool useNewAngleMapping, Torus::Potential *Phi, 
    coord::PosVelCyl& outPoint, actions::Frequencies& outFreqs)
{
    Torus::Actions J;
    Torus::Angles theta;
    J[0]     = 0.01; // actions in whatever units
    J[1]     = 2.;
    J[2]     = 3.;
    theta[0] = 1.; // angles in radians
    theta[1] = 2.;
    theta[2] = 3.;
    Torus::Torus T(useNewAngleMapping);
    T.AutoFit(J,Phi,0.001,700,300,15,5,24,200,24,0);
    Torus::PSPT P     = T.Map3D(theta);  // returns (R,z,phi, vR,vz,vphi) given (Tr, Tt, Tphi)
    outPoint          = coord::PosVelCyl(P[0], P[1], P[2], P[3], P[4], P[5]);
    outFreqs.Omegar   = T.omega(0);
    outFreqs.Omegaz   = T.omega(1);
    outFreqs.Omegaphi = T.omega(2);
}

int main() {
    const potential::BasePotential* poten = 
        potential::readGalaxyPotential("../temp/GSM_potential.pot", units::galactic_Myr);
    TorusPotentialWrapper Phi(*poten);
    coord::PosVelCyl oldP, newP;
    actions::Frequencies oldF, newF;
    test(false, &Phi, oldP, oldF);   // old method
    std::cout << "Old method gives "<<oldP<<
        "and frequencies Or: "<<oldF.Omegar<<"  Oz: "<<oldF.Omegaz<<"  Ophi: "<<oldF.Omegaphi<<std::endl;
    test(true,  &Phi, newP, newF);   // new angle determination method
    std::cout << "New method gives "<<newP<<
        "and frequencies Or: "<<newF.Omegar<<"  Oz: "<<newF.Omegaz<<"  Ophi: "<<newF.Omegaphi<<std::endl;
    if(coord::equalPosVel(oldP, newP, 0.05) && 
        math::fcmp(oldF.Omegar, newF.Omegar, 1e-4)==0 &&
        math::fcmp(oldF.Omegar, newF.Omegar, 1e-4)==0 &&
        math::fcmp(oldF.Omegar, newF.Omegar, 1e-4)==0)
        std::cout << "ALL TESTS PASSED\n";
    return 0;
}
