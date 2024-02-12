#include "actions_torus.h"
#include "math_core.h"
#include "math_random.h"
#include "potential_utils.h"
#include "utils.h"
#include "torus/Torus.h"
#include "torus/Potential.h"
#include <stdexcept>

#if __cplusplus >= 201103L
// with C++11 use unordered map as it is faster
#include <unordered_map>
#include "math_random.h"
namespace{
struct ActionsHash {
    size_t operator() (const actions::Actions& a) const {
        return math::hash((const void*)(&a), 3);
    }
};

struct ActionsEqual {
    bool operator() (const actions::Actions& lhs, const actions::Actions& rhs) const {
        return lhs.Jr == rhs.Jr && lhs.Jz == rhs.Jz && lhs.Jphi == rhs.Jphi;
    }
};
typedef std::unordered_map<actions::Actions, shared_ptr<torus::Torus>, ActionsHash, ActionsEqual>
    TorusCache;
}
#else
// use ordinary map pre-C++11
#include <map>
namespace {
struct ActionsLess {
    bool operator() (const actions::Actions& lhs, const actions::Actions& rhs) const {
        if(lhs.Jr < rhs.Jr) return true;
        if(lhs.Jr > rhs.Jr) return false;
        if(lhs.Jz < rhs.Jz) return true;
        if(lhs.Jz > rhs.Jz) return false;
        return lhs.Jphi < rhs.Jphi;
    }
};

typedef std::map<actions::Actions, shared_ptr<torus::Torus>, ActionsLess> TorusCache;
}
#endif

namespace actions{

/// Auxiliary class for using any of BasePotential-derived potentials with Torus code
class TorusPotentialWrapper: public torus::Potential{
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
    virtual torus::Frequencies KapNuOm(double R) const {
        torus::Frequencies freq;
        epicycleFreqs(poten, R, freq[0], freq[1], freq[2]);
        return freq;
    }
private:
    const potential::BasePotential& poten;
};

/// Auxiliary class implementing on-the-fly creation of Torus instances and their caching
class ActionMapperTorus::Impl {
public:
    Impl(const potential::PtrPotential _pot, const double _tol) :
        pot(_pot), tol(_tol)
    {
        if(!isAxisymmetric(*pot))
            throw std::invalid_argument("ActionMapperTorus only works for axisymmetric potentials");
    }

    /// construct a new Torus if necessary, or retrieve an existing one from the cache
    torus::Torus* getTorus(const Actions& act)
    {
        if(!(act.Jr>=0 && act.Jz>=0 && isFinite(act.Jr+act.Jz+act.Jphi)))
            return NULL;  // invalid actions => no torus

        // check if a Torus object for the given triplet of actions has been constructed before
        TorusCache::iterator it = cache.find(act);
        if(it != cache.end())
            return it->second.get();

        // not found: create a new Torus
        shared_ptr<torus::Torus> newTorus;
        try{
            newTorus.reset(new torus::Torus(true));
            torus::Actions acts;
            acts[0] = act.Jr;
            acts[1] = act.Jz;
            acts[2] = act.Jphi;
            // the actual potential is used only during torus fitting, but not required
            // later in angle mapping - so we create a temporary object
            TorusPotentialWrapper potwrap(*pot);
            int result = newTorus->AutoFit(acts, &potwrap,
                tol, 600, 150, 12, 3, 16, 200, 12, utils::verbosityLevel >= utils::VL_VERBOSE);
            if(result!=0) {
                FILTERMSG(utils::VL_WARNING, "Torus", "Not converged: "+utils::toString(result));
                // delete the faulty torus, returning NAN is better than returning nonsense
                newTorus.reset();
            }
        }
        catch(std::exception& ex) {
            FILTERMSG(utils::VL_WARNING, "Torus", "Exception: "+std::string(ex.what()));
            newTorus.reset();  // assign a null pointer to this torus
        }

        // add the newly created torus (even if it is NULL) to the cache
        cache.insert(std::make_pair(act, newTorus));
        return newTorus.get();
    }

    const potential::PtrPotential pot;  ///< the potential used to create new tori
    const double tol;                   ///< accuracy parameter for torus construction
    TorusCache cache;                   ///< cache for previously constructed tori
};

ActionMapperTorus::ActionMapperTorus(const potential::PtrPotential& pot, double tol) :
    impl(new Impl(pot, tol))
{}

ActionMapperTorus::~ActionMapperTorus()
{
    delete impl;
}

coord::PosVelCyl ActionMapperTorus::map(const ActionAngles& actAng, Frequencies* freq) const
{
    torus::Torus* torus = impl->getTorus(actAng);
    // "torus" contains an instance of Torus corresponding to the given actions,
    // or NULL in case that torus construction failed or actions were invalid
    if(torus) {
        if(freq!=NULL) {
            torus::Frequencies tfreq = torus->omega();
            freq->Omegar   = tfreq[0];
            freq->Omegaz   = tfreq[1];
            freq->Omegaphi = tfreq[2];
        }
        torus::Angles ang;
        ang[0] = actAng.thetar;
        ang[1] = actAng.thetaz;
        ang[2] = actAng.thetaphi;
        torus::PSPT xv = torus->Map3D(ang);
        return coord::PosVelCyl(xv[0], xv[1], xv[2], xv[3], xv[4], xv[5]);
    } else {
        if(freq)
            freq->Omegar = freq->Omegaz = freq->Omegaphi = NAN;
        return coord::PosVelCyl(NAN, NAN, NAN, NAN, NAN, NAN);
    }
}

std::string ActionMapperTorus::name() const
{
    return "Torus(potential=" + impl->pot->name() +
        ", num_cached_tori=" + utils::toString(impl->cache.size()) + ")";
}

}  // namespace actions
