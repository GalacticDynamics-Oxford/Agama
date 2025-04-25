#include "math_specfunc.h"
#include "math_ode.h"
#include "math_core.h"
#include "potential_multipole.h"
#include "utils.h"
#include <stdexcept>
#include <cmath>

namespace potential {

namespace {  // internal

/// accuracy parameter for the ODE integrator
const double ACCURACY_INTEGR = 1e-8;

/// accuracy parameter for locating the truncation radius
const double ACCURACY_ROOT   = 1e-8;

/// upper limit on the number of steps in the ODE integrator
const int MAX_NUM_STEPS_ODE  = 200;


/// RHS of the differential equation for the (dimensionless) potential
class KingPotentialIntegrator: public math::IOdeSystem {
    const double W0;     ///< dimensionless potential at origin: W0 = [Phi(rtrunc) - Phi(0)] / sigma^2
    const double trunc;  ///< truncation strength parameter (0 - Woolley, 1 - King, 2 - Wilson, etc.)
    const double gamma;  ///< value of Gamma function for trunc+3/2
    const double norm;   ///< normalization constant for the density
    const double phicrit;///< smallest value below which we use analytic approximation for gammainc
    const double& rbegin;///< radius at the beginning of the current step, updated by the calling code
public:
    KingPotentialIntegrator(double _W0, double _trunc, const double& _rbegin) :
        W0(_W0),
        trunc(_trunc),
        gamma(math::gamma(trunc+1.5)),
        norm(exp(-W0) / (gamma - math::gammainc(trunc+1.5, W0))),
        phicrit(1e-3 * pow_3(trunc+0.1)),
        rbegin(_rbegin)
    {}

    /// dimensionless density as a function of dimensionless potential
    double rho(double phi) const
    {
        return phi<=0 ? 0 :
            phi>phicrit ? norm * exp(phi) * (gamma - math::gammainc(trunc+1.5, phi)):
            // the above expression suffers from cancellation at small phi, replace with series expansion
            norm * pow(phi, trunc+1.5) / (trunc+1.5) * (1 + phi / (trunc+2.5));
    }

    virtual void eval(const double deltar, const double y[], double dydr[], double* /*ignored*/) const
    {
        double r = rbegin + deltar, phi = y[0], dphidr = y[1];
        dydr[0] = dphidr;
        dydr[1] = r==0 ? -3 * rho(phi) : -9 * rho(phi) - 2/r * dphidr;
    }

    virtual unsigned int size() const { return 2; }  // two variables: phi, dphi/dr
};


/// helper function for locating the radius where phi(r)=0
class FindRadiusFromPhi: public math::IFunction {
public:
    FindRadiusFromPhi(const math::BaseOdeStepper& _stepper, double _phi0) :
        stepper(_stepper), phi0(_phi0) {};
    /** used in root-finder to locate the root phi(rprev+deltar)=phi0 */
    virtual void evalDeriv(const double deltar, double* val, double* der, double*) const
    {
        if(val)
            *val = stepper.getSol(deltar, 0) - phi0;  // phi-phi0
        if(der)
            *der = stepper.getSol(deltar, 1);  // dphi/dr
    }
    virtual unsigned int numDerivs() const { return 1; }
private:
    const math::BaseOdeStepper& stepper;
    const double phi0;
};

// construct both density and potential profiles by integrating an ODE
void createKingModel(double mass, double scaleRadius, double W0, double trunc,
    /*output arrays will be filled by this routine*/
    std::vector<double>& radii,  std::vector<double>& phi,
    std::vector<double>& dphidr, std::vector<double>& rho)
{
    if(!(mass > 0 && scaleRadius > 0))
        throw std::invalid_argument("createKingModel: mass and scale radius must be positive");
    if(!(W0 > 0))
        throw std::invalid_argument("createKingModel: dimensionless potential W0 must be positive");
    if(!(trunc >= 0 && trunc < 3.5))
        throw std::invalid_argument("createKingModel: truncation parameter must be between 0 and 3.5");

    // compute the potential in dimensionless units by solving an ODE
    double vars[2] = {W0, 0};
    double rcurr = 0, phicurr = W0, phitrans = NAN;
    KingPotentialIntegrator odeSystem(W0, trunc,
        /*variable containing the radius at the beginning of the current step*/ rcurr);
    math::OdeStepperDOP853 stepper(odeSystem, ACCURACY_INTEGR);
    stepper.init(vars);
    bool finished = false;
    int numStepsODE = 0;
    while(!finished) {
        double deltar = stepper.doStep(INFINITY);
        if(deltar <= 0 || ++numStepsODE >= MAX_NUM_STEPS_ODE) {
            finished = true;
        } else {
            double phiprev = phicurr, rprev = rcurr;
            rcurr = rprev + deltar;
            phicurr = stepper.getSol(/*offset from rprev at the end of this step*/ deltar, 0);
            // when we approach the outer boundary (phi=0), output a more densely spaced radial grid,
            // to improve the accuracy of interpolation of density profile.
            // Instead of placing one point at the end of each radial integration step,
            // allocate points at a pre-defined "quadratic" grid in phi, which is denser around phi=0,
            // and store the interpolated phi, dphi/dr and rho inside the current integration step.
            if(phicurr < phiprev * 0.75 || phitrans > 0) {
                if(phitrans != phitrans)
                    phitrans = phiprev;  // switch to the predefined grid mode
                const int NGRID=8;
                for(int igrid=NGRID-1; igrid>=0; igrid--) {
                    double phiinter = phitrans * pow_2(igrid*1./NGRID);  // quadratically spaced grid
                    if(phiinter < phiprev && phiinter >= phicurr) {
                        // the predefined grid is specified for phi, convert it to radius;
                        // deltarinter is the offset in radius from rprev (between 0 and deltar)
                        double deltarinter = math::findRoot(
                            FindRadiusFromPhi(stepper, phiinter), 0, deltar, ACCURACY_ROOT);
                        radii. push_back(rprev + deltarinter);
                        phi.   push_back(phiinter);
                        dphidr.push_back(stepper.getSol(deltarinter, 1));
                        rho.   push_back(odeSystem.rho(phiinter));
                        if(phiinter == 0)    // the last grid point is exactly at the truncation radius
                            finished = true;
                    }
                }
            } else {
                // store the values of phi, dphi/dr and rho at the end of the current radial step
                radii. push_back(rcurr);
                phi.   push_back(phicurr);
                dphidr.push_back(stepper.getSol(/*offset from rprev*/ deltar, 1));
                rho.   push_back(odeSystem.rho(phicurr));
            }
        }
    }
    if(radii.empty() || phi.back()!=0)
        throw std::runtime_error("createKingModel: failed to construct model");

    // shift the potential by its value at r_trunc, which is mtotal/rtrunc
    double totalMass = -dphidr.back() * pow_2(radii.back());  // total mass in dimensionless units
    double phiadd = totalMass / radii.back();

    FILTERMSG(utils::VL_DEBUG, "createKingModel",
           "W0=" + utils::toString(W0) + ", g=" + utils::toString(trunc) +
           " => concentration c=log10(rtrunc/rscale)=" + utils::toString(log10(radii.back())) +
           ", sigma=" + utils::toString(sqrt(phi[0] * mass / scaleRadius / totalMass / W0)) );
    // scale the model to the given total mass and scale radius, inverting the sign of potential
    for(size_t i=0; i<radii.size(); i++) {
        phi[i]     = -(phi[i] + phiadd) * mass / scaleRadius / totalMass;
        dphidr[i] *= -mass / pow_2(scaleRadius) / totalMass;
        radii[i]  *= scaleRadius;
        rho[i]    *= 9./4/M_PI * mass / pow_3(scaleRadius) / totalMass;
    }
}

}  // internal ns

// driver functions
PtrDensity createKingDensity(double mass, double scaleRadius, double W0, double trunc)
{
    std::vector<double> radii, phi, dphidr, rho;
    createKingModel(mass, scaleRadius, W0, trunc, /*output*/ radii, phi, dphidr, rho);

    // create the interpolated density
    return PtrDensity(new DensitySphericalHarmonic(radii,
        std::vector<std::vector<double> >(1, rho)));
}

PtrPotential createKingPotential(double mass, double scaleRadius, double W0, double trunc)
{
    std::vector<double> radii, phi, dphidr, rho;
    createKingModel(mass, scaleRadius, W0, trunc, /*output*/ radii, phi, dphidr, rho);

    // create the multipole potential
    return PtrPotential(new Multipole(radii,
        std::vector<std::vector<double> >(1, phi),
        std::vector<std::vector<double> >(1, dphidr)));
}

}  // namespace potential
