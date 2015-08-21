#include "galaxymodel.h"
#include "math_core.h"
#include "actions_torus.h"
#include <cmath>
#include <stdexcept>

namespace galaxymodel{

/// helper class for integrating distribution function
class DFMomentsIntegrandNdim: public math::IFunctionNdim {
public:
    /// mode of operation:
    enum MODE {
        ZEROTH_MOMENT,  ///< 0th moment only (i.e., density)
        FIRST_MOMENT,   ///< 0th and 1st moments (density and three velocity components)
        SECOND_MOMENT   ///< 0th, 1st and 2nd moments (density, velocity and outer product of velocity components)
    };

    DFMomentsIntegrandNdim(
        const actions::BaseActionFinder& _actFinder,
        const df::BaseDistributionFunction& _distrFunc,
        const coord::PosCyl& _point, const MODE _mode) :
        actFinder(_actFinder), distrFunc(_distrFunc), point(_point), mode(_mode) {};

    /** compute one or more moments of distribution function.
        Input array of length 3 defines the velocity: [ magnitude, cos(theta), phi ],
        where magnitude ranges from 0 to the escape velocity,
        and the two angles define the orientation of velocity vector
        in spherical coordinates centered at a given point.
        Output array contains the value of distribution function (f),
        multiplied by various combinations of velocity components (depending on MODE):
        {f, [f*vR, f*vz, f*vphi, [f*vR^2, f*vz^2, f*vphi^2, f*vR*vz, f*vR*vphi, f*vz*vphi] ] }.
    */
    virtual void eval(const double vars[], double values[]) const
    {
        // 1. get the components of velocity in cylindrical coordinates
        const double sintheta = sqrt(1-pow_2(vars[1]));
        coord::VelCyl vel(
            vars[0] * sintheta * cos(vars[2]),
            vars[0] * sintheta * sin(vars[2]),
            vars[0] * vars[1]);

        // 2. determine the actions
        actions::Actions acts = actFinder.actions(coord::PosVelCyl(point, vel));

        // 3. compute the value of distribution function
        // additional factor v^2 comes from the jacobian of velocity transformation in spherical coords
        double dfval = distrFunc.value(acts) * pow_2(vars[0]);

        // 4. output the value, optionally multiplied by various combinations of velocity components
        values[0] = dfval;
        if(mode == FIRST_MOMENT || mode == SECOND_MOMENT) {
            values[1] = dfval * vel.vR;
            values[2] = dfval * vel.vz;
            values[3] = dfval * vel.vphi;
        }
        if(mode == SECOND_MOMENT) {
            values[4] = dfval * vel.vR   * vel.vR;
            values[5] = dfval * vel.vz   * vel.vz;
            values[6] = dfval * vel.vphi * vel.vphi;
            values[7] = dfval * vel.vR   * vel.vz;
            values[8] = dfval * vel.vR   * vel.vphi;
            values[9] = dfval * vel.vz   * vel.vphi;
        }
    }

    /// number of variables (3 velocity components)
    virtual unsigned int numVars()   const { return 3; }

    /// number of values to compute
    virtual unsigned int numValues() const {
        return mode == SECOND_MOMENT ? 10 :
               mode == FIRST_MOMENT ? 4 : 1;
    }
private:
    const actions::BaseActionFinder& actFinder;
    const df::BaseDistributionFunction& distrFunc;
    const coord::PosCyl& point;
    const MODE mode;
};

void GalaxyModel::computeMoments(const coord::PosCyl& point,
    const double reqRelError, const int maxNumEval,
    double* density, coord::VelCyl* velocityFirstMoment, coord::Vel2Cyl* velocitySecondMoment,
    double* densityErr, coord::VelCyl* velocityFirstMomentErr, coord::Vel2Cyl* velocitySecondMomentErr) const
{
    DFMomentsIntegrandNdim::MODE  mode =
        velocitySecondMoment!=NULL ? DFMomentsIntegrandNdim::SECOND_MOMENT :
        velocityFirstMoment !=NULL ? DFMomentsIntegrandNdim::FIRST_MOMENT : DFMomentsIntegrandNdim::ZEROTH_MOMENT;
    DFMomentsIntegrandNdim fnc(actFinder, distrFunc, point, mode);

    // compute the escape velocity
    double Phi_inf = poten.value(coord::PosCar(INFINITY, 0, 0));
    if(!math::isFinite(Phi_inf))
        Phi_inf = 0;  // default assumption
    double V_esc = sqrt(2. * (Phi_inf - poten.value(point)));
    if(!math::isFinite(V_esc))
        throw std::invalid_argument("Error in computing moments: escape velocity is undetermined");

    // define the integration region
    double xlower[3] = {0,    -1, 0};  // boundaries of integration region in velocity (magnitude, cos(theta), phi)
    double xupper[3] = {V_esc, 1, 2*M_PI};
    double result[10], error[10];  // the values of integrals and their error estimates
    int numEval; // actual number of evaluations

    // perform the multidimensional integration
    math::integrateNdim(fnc, xlower, xupper, reqRelError, maxNumEval, result, error, &numEval);

    // store the results
    if(density!=NULL) {
        *density = result[0];
        if(densityErr!=NULL)
            *densityErr = error[0];
    }
    double densRelErr2 = pow_2(error[0]/result[0]);
    if(velocityFirstMoment!=NULL) {
        *velocityFirstMoment = coord::VelCyl(result[1]/result[0], result[2]/result[0], result[3]/result[0]);
        if(velocityFirstMomentErr!=NULL) {
            // relative errors in moments are summed in quadrature from errors in rho and rho*v
            velocityFirstMomentErr->vR = 
                sqrt(pow_2(error[1]/result[1]) + densRelErr2) * fabs(velocityFirstMoment->vR);
            velocityFirstMomentErr->vz =
                sqrt(pow_2(error[2]/result[2]) + densRelErr2) * fabs(velocityFirstMoment->vz);
            velocityFirstMomentErr->vphi =
                sqrt(pow_2(error[3]/result[3]) + densRelErr2) * fabs(velocityFirstMoment->vphi);
        }
    }
    if(velocitySecondMoment!=NULL) {
        velocitySecondMoment->vR2    = result[4]/result[0];
        velocitySecondMoment->vz2    = result[5]/result[0];
        velocitySecondMoment->vphi2  = result[6]/result[0];
        velocitySecondMoment->vRvz   = result[7]/result[0];
        velocitySecondMoment->vRvphi = result[8]/result[0];
        velocitySecondMoment->vzvphi = result[9]/result[0];
        if(velocitySecondMomentErr!=NULL) {
            velocitySecondMomentErr->vR2 =
                sqrt(pow_2(error[4]/result[4]) + densRelErr2) * fabs(velocitySecondMoment->vR2);
            velocitySecondMomentErr->vz2 =
                sqrt(pow_2(error[5]/result[5]) + densRelErr2) * fabs(velocitySecondMoment->vz2);
            velocitySecondMomentErr->vphi2 =
                sqrt(pow_2(error[6]/result[6]) + densRelErr2) * fabs(velocitySecondMoment->vphi2);
            velocitySecondMomentErr->vRvz =
                sqrt(pow_2(error[7]/result[7]) + densRelErr2) * fabs(velocitySecondMoment->vRvz);
            velocitySecondMomentErr->vRvphi =
                sqrt(pow_2(error[8]/result[8]) + densRelErr2) * fabs(velocitySecondMoment->vRvphi);
            velocitySecondMomentErr->vzvphi =
                sqrt(pow_2(error[9]/result[9]) + densRelErr2) * fabs(velocitySecondMoment->vzvphi);
        }
    }
}

void GalaxyModel::computeActionSamples(const unsigned int nSamp, particles::PointMassArrayCar &points,
    std::vector<actions::Actions>* actsOutput) const
{
    // first sample points from the action space:
    // we use nAct << nSamp  distinct values for actions, and construct tori for these actions;
    // then each torus is sampled with nAng = nSamp/nAct  distinct values of angles,
    // and the action/angles are converted to position/velocity points
    unsigned int nAng = std::min<unsigned int>(nSamp/100+1, 16);   // number of sample angles per torus
    unsigned int nAct = nSamp / nAng + 1;
    std::vector<actions::Actions> actions;
    df::sampleActions(distrFunc, nAct, actions);   // do the sampling in actions space
    nAct = actions.size();                         // could be different from requested?
    // sampling procedure does not provide the total mass, need to compute it via deterministic integration
    double totalMass = distrFunc.totalMass();
    double pointMass = totalMass / (nAct*nAng);
    
    // next sample angles from each torus
    if(actsOutput!=NULL) {
        actsOutput->clear();
        actsOutput->reserve(nAct*nAng);
    }
    for(unsigned int t=0; t<nAct; t++) {
        actions::ActionMapperTorus torus(poten, actions[t]);
        for(unsigned int a=0; a<nAng; a++) {
            actions::Angles ang;
            ang.thetar   = 2*M_PI*math::random();
            ang.thetaz   = 2*M_PI*math::random();
            ang.thetaphi = 2*M_PI*math::random();
            points.add( coord::toPosVelCar(
                torus.map(actions::ActionAngles(actions[t], ang)) ), pointMass );
            if(actsOutput!=NULL)
                actsOutput->push_back(actions[t]);
        }
    }
}

}  // namespace