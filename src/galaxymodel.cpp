#include "galaxymodel.h"
#include "math_core.h"
#include "actions_torus.h"
#include "math_sample.h"
#include <cmath>
#include <stdexcept>

#include "debug_utils.h"
#include <iostream>

namespace galaxymodel{

/** convert from scaled velocity variables to the actual velocity.
    \param[in]  vars are the scaled variables: |v|/vmag, cos(theta), phi,
    where the latter two quantities specify the orientation of velocity vector 
    in spherical coordinates centered at a given point, and
    \param[in]  velmag is the magnutude of velocity.
    \param[out] jac (optional) if not NULL, output the jacobian of transformation.
    \return  three components of velocity in cylindrical coordinates
*/
static coord::VelCyl unscaleVelocity(const double vars[], const double velmag, double* jac=0)
{
    const double costheta = vars[1]*2 - 1;
    const double sintheta = sqrt(1-pow_2(costheta));
    const double vel = vars[0]*velmag;
    if(jac)
        *jac = 4*M_PI * vel*vel * velmag;
    return coord::VelCyl(
        vel * sintheta * cos(2*M_PI * vars[2]),
        vel * sintheta * sin(2*M_PI * vars[2]),
        vel * costheta);
}

/** compute the escape velocity at a given position in the given ponential */
static double escapeVel(const coord::PosCyl& pos, const potential::BasePotential& poten)
{
    const double Phi_inf = 0;   // assume that the potential is zero at infinity
    const double vesc = sqrt(2. * (Phi_inf - poten.value(pos)));
    if(!math::isFinite(vesc))
        throw std::invalid_argument("Error in computing moments: escape velocity is undetermined");
    return vesc;
}

/** convert from scaled position/velocity coordinates to the real ones.
    The coordinates in cylindrical system are scaled in the same way as for 
    the density integration; the velocity magnitude is scaled with local escape velocity.
    If needed, also provide the jacobian of transformation.
*/
static coord::PosVelCyl unscalePosVel(const double vars[], 
    const potential::BasePotential& poten, double* jac=0)
{
    // 1. determine the position from the first three scaled variables
    double jacPos=0;
    const coord::PosCyl pos = potential::unscaleCoords(vars, jac==NULL ? NULL : &jacPos);
    // 2. determine the velocity from the second three scaled vars
    const double velmag = escapeVel(pos, poten);
    const coord::VelCyl vel = unscaleVelocity(vars+3, velmag, jac);
    if(jac!=NULL)
        *jac *= jacPos;
    return coord::PosVelCyl(pos, vel);
}
    
/// helper class for integrating distribution function over 3d velocity or 6d position/velocity space
class DFIntegrandNdim: public math::IFunctionNdim {
public:
    /// mode of operation - first three options are for 3d integration over velocities at a fixed position
    enum MODE {
        ZEROTH_MOMENT,  ///< 0th moment only (i.e., density)
        FIRST_MOMENT,   ///< 0th and 1st moments (density and three velocity components)
        SECOND_MOMENT,  ///< 0th, 1st and 2nd moments (density, velocity and outer product of velocity components)
        DF_VALUE        ///< same as ZEROTH_MOMENT, but integrating over the entire 6d phase volume
    };

    /// prepare for integration in 3d velocity space for a fixed position
    DFIntegrandNdim(const GalaxyModel& _model, const coord::PosCyl& _point, const MODE _mode) :
        model(_model), point(_point), mode(_mode), velmag(escapeVel(_point, _model.potential)) {}

    /// prepare for integration in 6d position/velocity space
    explicit DFIntegrandNdim(const GalaxyModel& _model) :
        model(_model), point(0,0,0), mode(DF_VALUE), velmag(0) {}

    /** compute one or more moments of distribution function.
        Input array defines 3 components of velocity (if computing moments of DF at a fixed position),
        or 6 components of position+velocity (if sampling the DF over the entire phase space).
        In both cases these components are obtained by a suitable 
        scaling transformation from the input array.
        Output array contains the value of distribution function (f),
        multiplied by various combinations of velocity components (depending on MODE):
        {f, [f*vR, f*vz, f*vphi, [f*vR^2, f*vz^2, f*vphi^2, f*vR*vz, f*vR*vphi, f*vz*vphi] ] }.
    */
    virtual void eval(const double vars[], double values[]) const
    {
        // 1. get the position/velocity components in cylindrical coordinates
        double jac;                // jacobian of variable transformation
        coord::PosVelCyl posvel =
            (mode == DF_VALUE) ?   // we are integrating over 6d position/velocity space
            unscalePosVel(vars, model.potential, &jac)
            :   // we are integrating over 3d velocity at a fixed position
            coord::PosVelCyl(point, unscaleVelocity(vars, velmag, &jac));
    
        // 2. determine the actions
        double dfval=0;
        try{
            actions::Actions acts;
            if(math::withinReasonableRange(posvel.R+fabs(posvel.z)))
                acts = model.actFinder.actions(posvel);
            else {
                jac = 0;
                acts.Jr = acts.Jz = acts.Jphi = 1;  // doesn't matter since jac=0
            }

            // 3. compute the value of distribution function times the jacobian
            dfval = model.distrFunc.value(acts) * jac;
        }
        catch(std::exception& e) {
            dfval=0;
            std::cout << posvel << e.what() <<std::endl;
        }

        // 4. output the value, optionally multiplied by various combinations of velocity components
        values[0] = dfval;
        if(mode == FIRST_MOMENT || mode == SECOND_MOMENT) {
            values[1] = dfval * posvel.vR;
            values[2] = dfval * posvel.vz;
            values[3] = dfval * posvel.vphi;
        }
        if(mode == SECOND_MOMENT) {
            values[4] = dfval * posvel.vR   * posvel.vR;
            values[5] = dfval * posvel.vz   * posvel.vz;
            values[6] = dfval * posvel.vphi * posvel.vphi;
            values[7] = dfval * posvel.vR   * posvel.vz;
            values[8] = dfval * posvel.vR   * posvel.vphi;
            values[9] = dfval * posvel.vz   * posvel.vphi;
        }
    }

    /// number of variables (3 velocity components or 6 position/velocity components)
    virtual unsigned int numVars()   const { return mode == DF_VALUE ? 6 : 3; }

    /// number of values to compute
    virtual unsigned int numValues() const {
        return mode == SECOND_MOMENT ? 10 :
               mode == FIRST_MOMENT ? 4 : 1;
    }
private:
    const GalaxyModel& model;
    const coord::PosCyl point;
    const MODE mode;
    const double velmag;
};

void computeMoments(const GalaxyModel& model,
    const coord::PosCyl& point, const double reqRelError, const int maxNumEval,
    double* density, coord::VelCyl* velocityFirstMoment, coord::Vel2Cyl* velocitySecondMoment,
    double* densityErr, coord::VelCyl* velocityFirstMomentErr, coord::Vel2Cyl* velocitySecondMomentErr)
{
    DFIntegrandNdim::MODE  mode =
        velocitySecondMoment!=NULL ? DFIntegrandNdim::SECOND_MOMENT :
        velocityFirstMoment !=NULL ? DFIntegrandNdim::FIRST_MOMENT : DFIntegrandNdim::ZEROTH_MOMENT;
    DFIntegrandNdim fnc(model, point, mode);

    // define the integration region in scaled velocities
    double xlower[3] = {0, 0, 0};
    double xupper[3] = {1, 1, 1};
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

void generateActionSamples(const GalaxyModel& model, const unsigned int nSamp,
    particles::PointMassArrayCar &points, std::vector<actions::Actions>* actsOutput)
{
    // first sample points from the action space:
    // we use nAct << nSamp  distinct values for actions, and construct tori for these actions;
    // then each torus is sampled with nAng = nSamp/nAct  distinct values of angles,
    // and the action/angles are converted to position/velocity points
    unsigned int nAng = std::min<unsigned int>(nSamp/100+1, 16);   // number of sample angles per torus
    unsigned int nAct = nSamp / nAng + 1;
    std::vector<actions::Actions> actions;
    
    // do the sampling in actions space
    double totalMass, totalMassErr;
    df::sampleActions(model.distrFunc, nAct, actions, &totalMass, &totalMassErr);
    nAct = actions.size();   // could be different from requested?
    //double totalMass = distrFunc.totalMass();
    double pointMass = totalMass / (nAct*nAng);
    
    // next sample angles from each torus
    points.data.clear();
    if(actsOutput!=NULL)
        actsOutput->clear();
    for(unsigned int t=0; t<nAct && points.size()<nSamp; t++) {
        actions::ActionMapperTorus torus(model.potential, actions[t]);
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

void generatePosVelSamples(const GalaxyModel& model, const unsigned int numSamples, 
    particles::PointMassArrayCar &points)
{
    DFIntegrandNdim fnc(model);
    math::Matrix<double> result;      // sampled scaled coordinates/velocities
    double totalMass, errorMass;      // total normalization of the distribution function and its estimated error
    double xlower[6] = {0,0,0,0,0,0}; // boundaries of sampling region in scaled coordinates
    double xupper[6] = {1,1,1,1,1,1};
    // determine optimal binning scheme: 6 dimensions is too much to let it go uniformly
    unsigned int NB = static_cast<unsigned int>(pow(numSamples*0.1, 1./3))+1;  // # of bins per dimension
    // use adaptive binning in R, z and |v| dimensions only, and leave the other three unbinned
    unsigned int numBins[6] = {NB, NB, 1, NB, 1, 1};
    math::sampleNdim(fnc, xlower, xupper, numSamples, numBins, result, NULL, &totalMass, &errorMass);
    const double pointMass = totalMass / result.numRows();
    for(unsigned int i=0; i<result.numRows(); i++) {
        // transform from scaled vars (array of 6 numbers) to real pos/vel
        const coord::PosVelCyl pt = unscalePosVel(&result(i,0), model.potential);
        points.add(coord::toPosVelCar(pt), pointMass);
    }
}

}  // namespace