#include "actions_torus.h"
#include "math_core.h"
#include "potential_utils.h"
#include "utils.h"
#include "torus/Torus.h"
#include "torus/Potential.h"
#include <stdexcept>

namespace actions{

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
    virtual double RfromLc(double Lz, double* dRdLz=0) const {
        if(dRdLz!=0)
            throw std::runtime_error("dR/dLz not implemented");
        return R_from_Lz(poten, Lz);
    }
    virtual double LfromRc(double R, double* dLcdR) const {
        if(dLcdR!=0)
            throw std::runtime_error("dLc/dR not implemented");
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

ActionMapperTorus::ActionMapperTorus(const potential::BasePotential& poten, const Actions& acts, double tol)
{
    if(!isAxisymmetric(poten))
        throw std::invalid_argument("ActionMapperTorus only works for axisymmetric potentials");
    torus = Torus::PtrTorus(new Torus::Torus(true));
    // the actual potential is used only during torus fitting, but not required 
    // later in angle mapping - so we create a temporary object
    TorusPotentialWrapper potwrap(poten);
    Torus::Actions act;
    act[0] = acts.Jr;
    act[1] = acts.Jz;
    act[2] = acts.Jphi;
    int result = torus->AutoFit(act, &potwrap, tol, 600, 150, 12, 3, 16, 200, 12,
        (int)utils::verbosityLevel);
    if(result!=0) {
        utils::msg(utils::VL_WARNING, "Torus", "Not converged: "+utils::toString(result));
        //torus->show(std::cout);
    }
}

coord::PosVelCyl ActionMapperTorus::map(const ActionAngles& actAng, Frequencies* freq) const
{
    // make sure that the input actions are the same as in the Torus object
    if( math::fcmp(actAng.Jr,   torus->action(0)) != 0 ||
        math::fcmp(actAng.Jz,   torus->action(1)) != 0 ||
        math::fcmp(actAng.Jphi, torus->action(2)) != 0 )
        throw std::invalid_argument("ActionMapperTorus: "
            "values of actions are different from those provided to the constructor");
    // frequencies are constant for a given torus (depend only on actions, not on angles)
    if(freq!=NULL) {
        Torus::Frequencies tfreq = torus->omega();
        freq->Omegar   = tfreq[0];
        freq->Omegaz   = tfreq[1];
        freq->Omegaphi = tfreq[2];
    }
    Torus::Angles ang;
    ang[0] = actAng.thetar;
    ang[1] = actAng.thetaz;
    ang[2] = actAng.thetaphi;
    Torus::PSPT xv = torus->Map3D(ang);
    return coord::PosVelCyl(xv[0], xv[1], xv[2], xv[3], xv[4], xv[5]);
}

}  // namespace actions
