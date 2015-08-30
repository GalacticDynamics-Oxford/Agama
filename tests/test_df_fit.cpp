/** \file    test_df_fit.cpp
    \author  Eugene Vasiliev
    \date    August 2015

    This example demonstrates how to find best-fit parameters of an action-based
    distribution function that matches the given N-body snapshot.

    The N-body model itself corresponds to a spherically-symmetric isotropic
    Hernquist profile, and we fit it with a double-power-law distribution function
    of Posti et al.2015. We use the exact potential (i.e., do not compute it
    from the N-body model itself, nor try to vary its parameters, although
    both options are possible), and compute actions for all particles only once.
    Then we scan the parameter space of DF, finding the maximum of the likelihood
    function with a multidimensional minimization algorithm.
    This takes a few hundred iterations to converge.
*/
#include <iostream>
#include "potential_dehnen.h"
#include "actions_staeckel.h"
#include "df_halo.h"
#include "particles_io.h"
#include "math_fit.h"
#include <cmath>
#include <stdexcept>

typedef std::vector<actions::Actions> ActionArray;

/// compute log-likelihood of DF with given params against an array of points
double modelLikelihood(const df::DoublePowerLawParam& params, const ActionArray& points)
{
    std::cout << "J0="<<params.j0<<", Jcore="<<params.jcore<<
        ", alpha="<<params.alpha<<", ar="<<params.ar<<", az="<<params.az<<", aphi="<<params.aphi<<
        ", beta=" <<params.beta <</*", br="<<params.br<<", bz="<<params.bz<<", bphi="<<params.bphi<<*/": ";
    double sumlog = 0;
    try{
        df::DoublePowerLaw dpl(params);
        double norm = dpl.totalMass(1e-4, 100000);
        for(unsigned int i=0; i<points.size(); i++)
            sumlog += log(dpl.value(points[i])/norm);
        std::cout << "LogL="<<sumlog<<", norm="<<norm<< std::endl;
        return sumlog;
    }
    catch(std::invalid_argument& e) {
        std::cout << "Exception "<<e.what()<<"\n";
        return -1000.*points.size();
    }
}

/// convert from parameter space to DF params: note that we apply
/// some non-trivial scaling to make the life easier for the minimizer
df::DoublePowerLawParam dfparams(const double vars[])
{
    double jcore = exp(vars[4]);  // will always stay positive
    double alpha = vars[0];
    double beta  = vars[1];
    double j0    = vars[2];
    double ar    = 3./(2+vars[3])*vars[3];
    double az    = 3./(2+vars[3]);
    double aphi  = 3./(2+vars[3]);  // ensure that sum of ar*Jr+az*Jz+aphi*Jphi doesn't depend on vars[3]
    double br    = 1.;
    double bz    = 1.;
    double bphi  = 1.;
    double norm  = 1.;
    df::DoublePowerLawParam params = {norm,j0,jcore,alpha,beta,ar,az,aphi,br,bz,bphi};
    return params;
}

/// function to be minimized
class ModelSearchFnc: public math::IFunctionNdim{
public:
    ModelSearchFnc(const ActionArray& _points) : points(_points) {};
    virtual void eval(const double vars[], double values[]) const
    {
        values[0] = -modelLikelihood(dfparams(vars), points);
    }
    virtual unsigned int numVars() const { return 5; }
    virtual unsigned int numValues() const { return 1; }
private:
    const ActionArray& points;
};

int main(){
    const potential::Dehnen pot(1., 1., 1., 1., 1.);
    const actions::ActionFinderAxisymFudge actf(pot);
    particles::PointMassArrayCar particles;
    readSnapshot("../temp/hernquist.dat", units::ExternalUnits(), particles);
    ActionArray particleActions(particles.size());
    for(unsigned int i=0; i<particles.size(); i++) {
        try{
            particleActions[i] = actf.actions(toPosVelCyl(particles.point(i)));
        }
        catch(std::exception& e) {
            std::cout << e.what() << std::endl;
            particleActions[i] = particleActions[i>0?i-1:0];  // put something reasonable (unless i==0)
        }
    }

    // do a parameter search to find best-fit distribution function describing these particles
    ModelSearchFnc fnc(particleActions);
    const double initparams[] = {2.0, 4.0, 1.0, 1.0,-9.0};
    const double stepsizes[]  = {0.1, 0.1, 0.1, 0.1, 0.1};
    const int maxNumIter = 1000;
    const double toler   = 1e-4;
    double bestparams[7];
    int numIter = math::findMinNdim(fnc, initparams, stepsizes, toler, maxNumIter, bestparams);
    std::cout << numIter << " iterations\n";

    return 0;
}
