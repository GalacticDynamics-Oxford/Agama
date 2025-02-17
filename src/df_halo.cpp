#include "df_halo.h"
#include "math_core.h"
#include <cmath>
#include <stdexcept>

namespace df{

namespace {  // internal ns

/// helper class used in the root-finder to determine the auxiliary coefficient beta for a cored halo
class BetaFinder: public math::IFunctionNoDeriv{
    const DoublePowerLawParam& par;

    // return the difference between the non-modified and modified DF as a function of beta and
    // the appropriately scaled action variable (t -> hJ), weighted by d(hJ)/dt for the integration in t
    double deltaf(const double t, const double beta) const
    {
        // integration is performed in a scaled variable t, ranging from 0 to 1,
        // which is remapped to hJ ranging from 0 to infinity as follows:
        double hJ    = par.Jcore * t*t*(3-2*t) / pow_2(1-t) / (1+2*t); 
        double dhJdt = par.Jcore * 6*t / pow_3(1-t) / pow_2(1+2*t);
        return hJ * hJ * dhJdt *
            math::pow(1 + math::pow(par.J0 / hJ, par.steepness),  par.slopeIn  / par.steepness) *
            math::pow(1 + math::pow(hJ / par.J0, par.steepness), -par.slopeOut / par.steepness) *
            (math::pow(1 + par.Jcore/hJ * (par.Jcore/hJ - beta), -0.5*par.slopeIn) - 1);
    }

public:
    BetaFinder(const DoublePowerLawParam& _par) : par(_par) {}

    virtual double value(const double beta) const
    {
        double result = 0;
        // use a fixed-order GL quadrature to compute the integrated difference in normalization between
        // unmodified and core-modified DF, which is sought to be zero by choosing an appropriate beta
        static const int GLORDER = 20;  // should be even, to avoid singularity in the integrand at t=0.5
        for(int i=0; i<GLORDER; i++)
            result += math::GLWEIGHTS[GLORDER][i] * deltaf(math::GLPOINTS[GLORDER][i], beta);
        return result;
    }
};

// helper function to compute the auxiliary coefficient beta in the case of a central core
double computeBeta(const DoublePowerLawParam &par)
{
    if(par.Jcore<=0)
        return 0;
    return math::findRoot(BetaFinder(par), 0.0, 2.0, /*root-finder tolerance*/ SQRT_DBL_EPSILON);
}

}  // internal ns

DoublePowerLaw::DoublePowerLaw(const DoublePowerLawParam &inparams) :
    par(inparams), beta(computeBeta(par))
{
    // sanity checks on parameters
    if(!(par.norm>0))
        throw std::invalid_argument("DoublePowerLaw: normalization must be positive");
    if(!(par.J0>0))
        throw std::invalid_argument("DoublePowerLaw: break action J0 must be positive");
    if(!(par.Jcore>=0 && beta>=0))
        throw std::invalid_argument("DoublePowerLaw: core action Jcore is invalid");
    if(!(par.Jcutoff>=0))
        throw std::invalid_argument("DoublePowerLaw: truncation action Jcutoff must be non-negative");
    if(!(par.slopeOut>3) && (par.Jcutoff==0 || par.Jcutoff==INFINITY))
        throw std::invalid_argument(
            "DoublePowerLaw: mass diverges at large J (outer slope must be > 3)");
    if(!(par.slopeIn<3))
        throw std::invalid_argument(
            "DoublePowerLaw: mass diverges at J->0 (inner slope must be < 3)");
    if(!(par.steepness>0))
        throw std::invalid_argument("DoublePowerLaw: transition steepness parameter must be positive");
    if(!(par.cutoffStrength>0))
        throw std::invalid_argument("DoublePowerLaw: cutoff strength parameter must be positive");
    if(!(par.coefJrIn>0 && par.coefJzIn >0 && par.coefJrIn +par.coefJzIn <3 &&
        par.coefJrOut>0 && par.coefJzOut>0 && par.coefJrOut+par.coefJzOut<3) )
        throw std::invalid_argument(
            "DoublePowerLaw: invalid weights in the linear combination of actions");
    if(!(fabs(par.rotFrac)<=1))
        throw std::invalid_argument("DoublePowerLaw: rotFrac must be between -1 and 1");
}

void DoublePowerLaw::evalDeriv(const actions::Actions &J,
    double *value, DerivByActions *deriv) const
{
    double
    signJphi    = J.Jphi>=0 ? 1 : -1,
    coefJphiIn  = (3-par.coefJrIn -par.coefJzIn)  * signJphi,
    coefJphiOut = (3-par.coefJrOut-par.coefJzOut) * signJphi,
    // linear combination of actions in the inner part of the model (for J<~J0)
    h = par.coefJrIn * J.Jr + par.coefJzIn * J.Jz + coefJphiIn * J.Jphi,
    // linear combination of actions in the outer part of the model (for J>~J0)
    g = par.coefJrOut* J.Jr + par.coefJzOut* J.Jz + coefJphiOut* J.Jphi,
    J0he = math::pow(par.J0 / h, par.steepness),
    gJ0e = math::pow(g / par.J0, par.steepness),
    gJcz = par.Jcutoff>0 && par.Jcutoff!=INFINITY ? math::pow(g / par.Jcutoff, par.cutoffStrength) : 0;
    *value = par.norm / pow_3(2*M_PI * par.J0) *
        math::pow(1 + J0he,  par.slopeIn  / par.steepness) *  // H(h)
        math::pow(1 + gJ0e, -par.slopeOut / par.steepness);   // G(g)
    if(par.Jcutoff>0)   // exponential cutoff at large J
        *value *= exp(-gJcz);
    if(par.Jcore>0) {   // central core of nearly-constant f(J) at small J
        if(h==0)
            *value = par.norm / pow_3(2*M_PI * par.J0);
        *value *= math::pow(1 + par.Jcore/h * (par.Jcore/h - beta), -0.5*par.slopeIn);
    }
    // add the odd part if necessary
    double rot = par.rotFrac!=0 && par.Jphi0!=INFINITY ? par.rotFrac * tanh(J.Jphi / par.Jphi0) : 0;
    *value *= (1+rot);

    if(deriv) {
        double dlogHdh = -par.slopeIn *
            (J0he + par.Jcore/h * (0.5 * beta * (1 - J0he) - par.Jcore/h)) /
            (h * (1 + J0he) * (1 + par.Jcore/h * (par.Jcore/h - beta)));
        double dlogGdg = -(par.slopeOut * gJ0e + 
            (par.Jcutoff>0 && par.Jcutoff!=INFINITY ? par.cutoffStrength * gJcz * (1 + gJ0e) : 0)) /
            (g * (1 + gJ0e));
        deriv->dbyJr   = *value * (par.coefJrIn * dlogHdh + par.coefJrOut * dlogGdg);
        deriv->dbyJz   = *value * (par.coefJzIn * dlogHdh + par.coefJzOut * dlogGdg);
        deriv->dbyJphi = *value * (  coefJphiIn * dlogHdh +   coefJphiOut * dlogGdg);
        if(par.Jphi0!=0 && par.rotFrac!=0)
            deriv->dbyJphi += *value * par.rotFrac * (1 - pow_2(rot / par.rotFrac)) / (1+rot) / par.Jphi0;
    }
}

}  // namespace df
