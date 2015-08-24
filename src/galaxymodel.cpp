#include "galaxymodel.h"
#include "math_core.h"
#include "actions_torus.h"
#include "math_sample.h"
#include <cmath>
#include <stdexcept>

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


/** Base helper class for integrating the distribution function 
    over 3d velocity or 6d position/velocity space.
    The integration is carried over in scaled coordinates which range from 0 to 1;
    the task for converting them to position/velocity point lies on the derived classes.
    The actions corresponding to the given point are computed with the action finder object
    from the GalaxyModel, and the distribution function value for these actions is provided
    by the eponimous member object from the GalaxyModel.
    The output may consist of one or more values, determined by the derived classes.
*/
class DFIntegrandNdim: public math::IFunctionNdim {
public:
    explicit DFIntegrandNdim(const GalaxyModel& _model) :
        model(_model) {}

    /** compute one or more moments of distribution function. */
    virtual void eval(const double vars[], double values[]) const
    {
        // 1. get the position/velocity components in cylindrical coordinates
        double jac;   // jacobian of variable transformation
        coord::PosVelCyl posvel = unscaleVars(vars, &jac);
        if(jac == 0) {  // we can't compute actions, but pretend that DF*jac is zero
            outputValues(posvel, 0, values);
            return;
        }
        
        // 2. determine the actions
        actions::Actions acts = model.actFinder.actions(posvel);

        // 3. compute the value of distribution function times the jacobian
        double dfval = model.distrFunc.value(acts) * jac;
        
        // 4. output the value(s) to the integration routine
        outputValues(posvel, dfval, values);
    }

    /** convert from scaled variables used in the integration routine 
        to the actual position/velocity point.
        \param[in]  vars  is the array of scaled variables;
        \param[out] jac (optional)  is the jacobian of transformation, if NULL it is not computed;
        \return  the position and velocity in cylindrical coordinates.
    */
    virtual coord::PosVelCyl unscaleVars(const double vars[], double* jac=0) const = 0;

protected:
    /** output the value(s) computed at a given point to the integration routine.
        \param[in]  point  is the position/velocity point;
        \param[in]  dfval  is the value of distribution function at this point;
        \param[out] values is the array of one or more values that are computed
    */
    virtual void outputValues(const coord::PosVelCyl& point, const double dfval, 
        double values[]) const = 0;

    const GalaxyModel& model;  ///< reference to the galaxy model to work with
};


/** helper class for integrating the distribution function over velocity at a fixed position */
class DFIntegrandAtPoint: public DFIntegrandNdim {
public:
    DFIntegrandAtPoint(const GalaxyModel& _model, const coord::PosCyl& _point) :
        DFIntegrandNdim(_model), point(_point), v_esc(escapeVel(_point, _model.potential)) {}

    /// input variables define 3 components of velocity, suitably scaled
    virtual coord::PosVelCyl unscaleVars(const double vars[], double* jac=0) const { 
        return coord::PosVelCyl(point, unscaleVelocity(vars, v_esc, jac));
    }
    
protected:
    /// dimension of the input array (3 scaled velocity components)
    virtual unsigned int numVars()   const { return 3; }

    /// dimension of the output array
    virtual unsigned int numValues() const { return 1; }

    /// output array contains one element - the value of DF
    virtual void outputValues(const coord::PosVelCyl& , const double dfval, 
        double values[]) const {
        values[0] = dfval;
    }
    
    const coord::PosCyl point;  ///< fixed position
    const double v_esc ;        ///< escape velocity at this position
};


/** helper class for integrating the distribution function and its first moments
    over velocity at a fixed position */
class DFIntegrandAtPointFirstMoment: public DFIntegrandAtPoint {
public:
    DFIntegrandAtPointFirstMoment(const GalaxyModel& _model, const coord::PosCyl& _point) :
        DFIntegrandAtPoint(_model, _point) {}

protected:
    virtual unsigned int numValues() const { return 4; }

    /// output array contains four elements - the value of DF 
    /// itself and multiplied by three components of velocity
    virtual void outputValues(const coord::PosVelCyl& pt, const double dfval, 
        double values[]) const {
        DFIntegrandAtPoint::outputValues(pt, dfval, values);
        values[1] = dfval * pt.vR;
        values[2] = dfval * pt.vz;
        values[3] = dfval * pt.vphi;
    }
};

/** helper class for integrating the distribution function and its first 
    and second moments over velocity at a fixed position */
class DFIntegrandAtPointFirstAndSecondMoment: public DFIntegrandAtPointFirstMoment {
public:
    DFIntegrandAtPointFirstAndSecondMoment(const GalaxyModel& _model, const coord::PosCyl& _point) :
        DFIntegrandAtPointFirstMoment(_model, _point) {}
        
protected:
    virtual unsigned int numValues() const { return 10; }

    /** output array contains ten elements - the value of DF 
        itself, multiplied by various combinations of velocity components:
        {f, f*vR, f*vz, f*vphi, f*vR^2, f*vz^2, f*vphi^2, f*vR*vz, f*vR*vphi, f*vz*vphi }.  */
    virtual void outputValues(const coord::PosVelCyl& pt, const double dfval, 
        double values[]) const {
        DFIntegrandAtPointFirstMoment::outputValues(pt, dfval, values);
        values[4] = dfval * pt.vR   * pt.vR;
        values[5] = dfval * pt.vz   * pt.vz;
        values[6] = dfval * pt.vphi * pt.vphi;
        values[7] = dfval * pt.vR   * pt.vz;
        values[8] = dfval * pt.vR   * pt.vphi;
        values[9] = dfval * pt.vz   * pt.vphi;
    }
};

/** helper class for integrating the distribution function over the entire 6d phase space */
class DFIntegrand6dim: public DFIntegrandNdim {
public:
    DFIntegrand6dim(const GalaxyModel& _model) :
        DFIntegrandNdim(_model) {}
    
    /// input variables define 6 components of position and velocity, suitably scaled
    virtual coord::PosVelCyl unscaleVars(const double vars[], double* jac=0) const { 
        return unscalePosVel(vars, model.potential, jac);
    }
    
protected:
    virtual unsigned int numVars()   const { return 6; }
    virtual unsigned int numValues() const { return 1; }
    
    /// output array contains one element - the value of DF
    virtual void outputValues(const coord::PosVelCyl& , const double dfval, 
        double values[]) const {
        values[0] = dfval;
    }
};


void computeMoments(const GalaxyModel& model,
    const coord::PosCyl& point, const double reqRelError, const int maxNumEval,
    double* density, coord::VelCyl* velocityFirstMoment, coord::Vel2Cyl* velocitySecondMoment,
    double* densityErr, coord::VelCyl* velocityFirstMomentErr, coord::Vel2Cyl* velocitySecondMomentErr)
{
    // define the integration region in scaled velocities
    double xlower[3] = {0, 0, 0};
    double xupper[3] = {1, 1, 1};
    double result[10], error[10];  // the values of integrals and their error estimates
    int numEval; // actual number of evaluations

    // perform the multidimensional integration using a suitable helper function, 
    // depending on the requested combination of output arguments
    const DFIntegrandNdim* fnc = 
        velocitySecondMoment!=NULL ? new DFIntegrandAtPointFirstAndSecondMoment(model, point) :
        velocityFirstMoment !=NULL ? new DFIntegrandAtPointFirstMoment(model, point) :
        new DFIntegrandAtPoint(model, point);
    math::integrateNdim(*fnc, xlower, xupper, reqRelError, maxNumEval, result, error, &numEval);

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
            const coord::PosVelCyl pt = torus.map(actions::ActionAngles(actions[t], ang));
            points.add(coord::toPosVelCar(pt), pointMass);
            if(actsOutput!=NULL)
                actsOutput->push_back(actions[t]);
        }
    }
}


void generatePosVelSamples(const GalaxyModel& model, const unsigned int numSamples, 
    particles::PointMassArrayCar &points)
{
    DFIntegrand6dim fnc(model);
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
    points.data.clear();
    for(unsigned int i=0; i<result.numRows(); i++) {
        double scaledvars[6] = {result(i,0), result(i,1), result(i,2),
            result(i,3), result(i,4), result(i,5)};
        // transform from scaled vars (array of 6 numbers) to real pos/vel
        const coord::PosVelCyl pt = fnc.unscaleVars(scaledvars);
        points.add(coord::toPosVelCar(pt), pointMass);
    }
}


void generateDensitySamples(const potential::BaseDensity& dens, const unsigned int numPoints,
    particles::PointMassArray<coord::PosCyl>& points)
{
    const bool axisym = isAxisymmetric(dens);
    potential::DensityIntegrandNdim fnc(dens);
    math::Matrix<double> result;      // sampled scaled coordinates
    double totalMass, errorMass;      // total mass and its estimated error
    double xlower[3] = {0,0,0};       // boundaries of sampling region in scaled coordinates
    double xupper[3] = {1,1,1};
    math::sampleNdim(fnc, xlower, xupper, numPoints, NULL, result, NULL, &totalMass, &errorMass);
    const double pointMass = totalMass / result.numRows();
    points.data.clear();
    for(unsigned int i=0; i<result.numRows(); i++) {
        // if the system is axisymmetric, phi is not provided by the sampling routine
        double scaledvars[3] = {result(i,0), result(i,1), 
            axisym ? math::random() : result(i,2)};
        // transform from scaled coordinates to the real ones, and store the point into the array
        points.add(fnc.unscaleVars(scaledvars), pointMass);
    }
}

}  // namespace