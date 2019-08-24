/** \file    example_df_fit.cpp
    \author  Eugene Vasiliev
    \date    2015-2016

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

    The Python counterpart of this example program additionally explores
    the uncertainties in DF parameters around their best-fit values,
    using the MCMC approach.
*/
#include "potential_dehnen.h"
#include "actions_spherical.h"
#include "df_halo.h"
#include "particles_base.h"
#include "math_fit.h"
#include "math_core.h"
#include "math_random.h"
#include "utils.h"
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <iostream>

const unsigned int NPARAMS = 5;
typedef std::vector<actions::Actions> ActionArray;

/// convert from parameter space to DF params
df::DoublePowerLawParam dfparams(const double vars[])
{
    df::DoublePowerLawParam params;
    params.slopeIn   = vars[0];
    params.slopeOut  = vars[1];
    params.steepness = vars[2];
    params.coefJrIn  = vars[3];
    params.coefJzIn  = (3-vars[3])/2; // fix g_z=g_phi taking into account that g_r+g_z+g_phi=3
    params.J0        = exp(vars[4]);
    params.norm      = 1.;
    return params;
}

/// compute log-likelihood of DF with given params against an array of points
double modelLikelihood(const df::DoublePowerLawParam& params, const ActionArray& points)
{
    std::cout <<
        "J0="          << utils::pp(params.J0,       7) <<
        ", slopeIn="   << utils::pp(params.slopeIn,  7) <<
        ", slopeOut="  << utils::pp(params.slopeOut, 7) <<
        ", steepness=" << utils::pp(params.steepness,7) <<
        ", coefJrIn="  << utils::pp(params.coefJrIn, 7) << ": ";
    double sumlog = 0;
    try{
        df::DoublePowerLaw dpl(params);
        double norm = dpl.totalMass();
        for(unsigned int i=0; i<points.size(); i++)
            sumlog += log(dpl.value(points[i])/norm);
        std::cout << "LogL=" << utils::pp(sumlog,10) << std::endl;
        return sumlog;
    }
    catch(std::invalid_argument& e) {
        std::cout << "Exception "<<e.what()<<"\n";
        return -1000.*points.size();
    }
}

/// function to be minimized
class ModelSearchFnc: public math::IFunctionNdim{
public:
    ModelSearchFnc(const ActionArray& _points) : points(_points) {};
    virtual void eval(const double vars[], double values[]) const
    {
        values[0] = -modelLikelihood(dfparams(vars), points);
    }
    virtual unsigned int numVars() const { return NPARAMS; }
    virtual unsigned int numValues() const { return 1; }
private:
    const ActionArray& points;
};

/// analytic expression for the ergodic distribution function f(E)
/// in a Hernquist model with mass M, scale radius a, at energy E.
double dfHernquist(double M, double a, double E)
{
    double q = sqrt(-E*a/M);
    return M / (4 * pow(2 * M * a * M_PI*M_PI, 1.5) ) * pow(1-q*q, -2.5) *
        (3*asin(q) + q * sqrt(1-q*q) * (1-2*q*q) * (8*q*q*q*q - 8*q*q - 3) );
}

/// create an N-body representation of Hernquist model
particles::ParticleArrayCyl createHernquistModel(unsigned int nbody)
{
    particles::ParticleArrayCyl points;
    for(unsigned int i=0; i<nbody; i++) {
        // 1. choose position
        double f = math::random();   // fraction of enclosed mass chosen at random
        double r = 1/(1/sqrt(f)-1);  // and converted to radius, using the known inversion of M(r)
        double costheta = math::random()*2 - 1;
        double sintheta = sqrt(1-pow_2(costheta));
        double phi  = math::random()*2*M_PI;
        // 2. assign velocity
        double pot  = -1./(r+1.);
        double fmax = 0.025/(r*r*(r+3.));  // magic number
        double E, fE;
        do{ // rejection algorithm
            E = math::random() * pot;
            f = math::random() * fmax;
            fE= dfHernquist(1., 1., E) * sqrt(E-pot);
            assert(fE<fmax);  // we must have selected a safe upper bound on f(E)*sqrt(E-Phi)
        } while(f > fE);
        double v = sqrt(2*(E-pot));
        double vcostheta = math::random()*2 - 1;
        double vsintheta = sqrt(1-pow_2(vcostheta));
        double vphi = math::random()*2*M_PI;
        points.add(coord::PosVelCyl(r*sintheta, r*costheta, phi,
            v*vsintheta*cos(vphi), v*vsintheta*sin(vphi), v*vcostheta), 1./nbody);
    }
    return points;
}

int main(){
    potential::Dehnen pot(1., 1., 1., 1., 1.);
    const actions::ActionFinderSpherical actf(pot);
    particles::ParticleArrayCyl particles(createHernquistModel(100000));
    ActionArray particleActions(particles.size());
    for(unsigned int i=0; i<particles.size(); i++)
        particleActions[i] = actf.actions(particles.point(i));

    // do a parameter search to find best-fit distribution function describing these particles
    const double initparams[NPARAMS] = {2.0, 4.0, 1.0, 1.0, 0.0};
    const double stepsizes [NPARAMS] = {0.1, 0.1, 0.1, 0.1, 0.1};
    const int maxNumIter = 1000;
    const double toler   = 1e-4;
    double bestparams[NPARAMS];
    ModelSearchFnc fnc(particleActions);
    int numIter = math::findMinNdim(fnc, initparams, stepsizes, toler, maxNumIter, bestparams);
    std::cout << numIter << " iterations\n";
}
