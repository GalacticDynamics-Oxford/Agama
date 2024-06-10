/** \name   test_df_spherical.cpp
    \author Eugene Vasiliev
    \date   2016-2019

    This program tests the accuracy of computation of phase volume 'h(E)', density of states 'g(E)',
    distribution function 'f(h)' or 'f(E,L)', and diffusion coefficients,
    for Plummer and Hernquist spherical isotropic and anisotropic models.
*/
#include "potential_analytic.h"
#include "potential_dehnen.h"
#include "potential_factory.h"
#include "potential_utils.h"
#include "math_core.h"
#include "math_specfunc.h"
#include "galaxymodel_spherical.h"
#include "galaxymodel_jeans.h"
#include "df_spherical.h"
#include "utils.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <stdexcept>

// two classes of models with analytic expressions are tested
struct Plummer{};
struct Hernquist{};

template<typename Model> class Rmax;     // provides r(Phi)
template<typename Model> class Phasevol; // provides h(E), g(E)
template<typename Model> class DF;       // provides f(E,L)

/// whether to produce output files
const bool output = utils::verbosityLevel >= utils::VL_VERBOSE;


/// analytic expressions for rmax(E) in the Plummer and Hernquist models
template<> class Rmax<Plummer>: public math::IFunction {
public:
    Rmax(){}
    virtual void evalDeriv(double E, double* Rmax, double* dRmaxdE, double* =NULL) const {
        if(Rmax)
            *Rmax = -sqrt(1-E*E)/E;
        if(dRmaxdE)
            *dRmaxdE = 1/(E*E*sqrt(1-E*E));
    }
    virtual unsigned int numDerivs() const { return 1; }
};

template<> class Rmax<Hernquist>: public math::IFunction {
public:
    Rmax(){}
    virtual void evalDeriv(double E, double* Rmax, double* dRmaxdE, double* =NULL) const {
        if(Rmax)
            *Rmax = -1/E-1;
        if(dRmaxdE)
            *dRmaxdE = 1/pow_2(E);
    }
    virtual unsigned int numDerivs() const { return 1; }
};


/// analytic expressions for phase volume and density of states in the Plummer model
template<> class Phasevol<Plummer>: public math::IFunction {
public:
    Phasevol(){}
    virtual void evalDeriv(double E, double* h, double* g, double* =NULL) const {
        if(E<-0.99) {  // asymptotic expressions for E -> -1
            double x = E+1;
            if(h)
                *h = pow_3(M_PI * x) *
                    (4./3 + x * (15./8 + x * ( 287./128 + x * ( 5187./2048 + x * 182193./65536))));
            if(g)
                *g = pow_2(M_PI * x) * M_PI *
                    (4.   + x * (15./2 + x * (1435./128 + x * (15561./1024 + x *1275351./65536))));
            return;
        }
        double x=sqrt((E+1)/2);
        double elE=math::ellintE(x);
        double elK=math::ellintK(x);
        double elP=math::ellintP(M_PI/2, x, -1-E);
        if(h)
            *h = 4*M_PI*M_PI/(9*E) *  ( elK * (3-34*E-8*E*E) + elE * (16*E*E-6) - elP * (36*E*E+3) );
        if(g)
            *g = 2*M_PI*M_PI/(3*E*E) * ( elK * (2*E-3-8*E*E) + elE * (16*E*E+6) - elP * (12*E*E-3) );
    }
    virtual unsigned int numDerivs() const { return 1; }
};

/// analytic expressions for phase volume and density of states in the Hernquist model
template<> class Phasevol<Hernquist>: public math::IFunction {
public:
    Phasevol(){}
    virtual void evalDeriv(double E, double* h, double* g, double* =NULL) const {
        if(E<-0.975) {  // asymptotic expressions for E -> -1
            double x = E+1;
            if(h)
                *h = M_PI*M_PI*M_SQRT2*512./945 * sqrt(x) * pow_2(x*x) *
                (1 + x * 24./11 * (1 + x * 20./13 * (1 + x * 4. /3 * (1 + x *21./17* (1 + x * 112./95)))));
            if(g)
                *g = M_PI*M_PI*M_SQRT2*256./105 * sqrt(x) * pow_3(x) *
                (1 + x * 8. /3  * (1 + x * 20./11 * (1 + x * 20./13* (1 + x * 7./5 * (1 + x * 112./85)))));
            return;
        }
        double sqE = sqrt(-E), x = sqrt(-(E+1)/E), phi = atan(x);
        if(h)
            *h = 4*M_SQRT2*M_PI*M_PI/9 * ( phi * (72*E-36-3/E) + x * (8*E*E-94*E+3) ) / sqE;
        if(g)
            *g = 2*M_SQRT2*M_PI*M_PI/3 * (-phi * (24*E+12+3/E) - x * (8*E*E-10*E-3) ) / pow_3(sqE);
    }
    virtual unsigned int numDerivs() const { return 1; }
};


/// DF for the Plummer model with the Cuddeford-Osipkov-Merritt anisotropy profile
template<> class DF<Plummer>: public df::QuasiSpherical {
public:
    const double beta, an;

    DF(double _beta=0, double r_a=INFINITY) :
        QuasiSpherical(potential::Sphericalized<potential::BasePotential>(potential::Plummer(1,1))),
        beta(_beta), an(1/r_a/r_a)
    {}

    // the energy-dependent part of DF
    inline double dfE(double E, double beta, double an) const {
        if(E<=-1 || E>=0)
            return NAN;
        double E2=E*E;
        if(beta==0)  // the classical isotropic case, generalized by Merritt(1985)
            return M_SQRT2*24/7/pow_3(M_PI) * pow(-E, 3.5) * (1 + (7./16/E2 - 1) * an);
        if(beta==-0.5)  // tangentially anisotropic case from Cuddeford(1991)
            return 3./8/pow_3(M_PI) * pow(1-E2, -2.5) / sqrt(1 + (1/E2-1) * an) * (
                (30 -  E2 * (47 - 20*E2) ) * E2 * E2 +
                (33 - 40*E2) * pow_2(1-E2) * E2 * an +
                ( 6 - 20*E2) * pow_3(1-E2) * an * an);
        return NAN;  // no simple expression for an arbitrary beta and r_a (see An&Evans 2006)
    }

    // interface for anisotropic DF as a function of (E,L)
    virtual void evalDeriv(const df::ClassicalIntegrals& ints,
        double *value, df::DerivByClassicalIntegrals* /*deriv*/ =NULL) const
    {
        *value = dfE(ints.E + 0.5 * pow_2(ints.L)*an, beta, an) *
            (beta==0 ? 1 : math::pow(ints.L, -2*beta));
    }
};


/// DF for the Hernquist model with a constant anisotropy (Baes&Dejonghe 2002)
template<> class DF<Hernquist>: public df::QuasiSpherical {
public:
    const double beta, prefact;

    DF(double _beta=0, double /*r_a*/=INFINITY) :
        QuasiSpherical(potential::Sphericalized<potential::BasePotential>(potential::Dehnen(1,1,1,1,1))),
        beta(_beta),
        prefact(pow(2., beta-2.5) * math::gamma(5-2*beta) /
            (M_PI*M_PI*M_SQRTPI * math::gamma(1-beta) * math::gamma(3.5-beta)) )
    {}

    // the energy-dependent part of DF
    inline double dfE(double E, double beta) const {
        if(E<=-1 || E>=0)
            return NAN;
        if(beta==0) {   // special case of isotropic model
            double q = sqrt(-E), sq1pE = sqrt(1+E);
            if(E>-1e-5)
                return 8./5*M_SQRT2/pow_3(M_PI) * math::pow(q, 5) * (1 - 10./7*E + 40./21*E*E);
            if(E<-1+1e-5)
                return 3./32*M_SQRT2/pow_2(M_PI) * (math::pow(sq1pE, -5) - 256./(15*M_PI));
            return 1/M_SQRT2/pow_3(2*M_PI) / ( pow_2(1+E) * sq1pE ) *
                (3*asin(q) + q * sq1pE * (1+2*E) * (8*E*E + 8*E - 3) );
        } else if(beta==0.5)
            return 0.75/pow_3(M_PI) * E*E;
        else if(beta==-0.5)
            return 0.5/pow_3(M_PI) * (10 + 10*E + 3*E*E) * pow_3(-E) / pow_2(pow_2(E+1));
        return prefact * math::hypergeom2F1(5-2*beta, 1-2*beta, 3.5-beta, -E) * pow(-E, 2.5-beta);
    }

    // interface for anisotropic DF as a function of (E,L)
    virtual void evalDeriv(const df::ClassicalIntegrals& ints,
        double *value, df::DerivByClassicalIntegrals* /*deriv*/ =NULL) const
    {
        *value = dfE(ints.E, beta) * (beta==0 ? 1 : math::pow(ints.L, -2*beta));
    }
};


/// bind together the true DF f(E) and the phase volume h(E)<->E(h) to form a function f(h)
class DFWrapper: public math::IFunctionNoDeriv {
    const potential::PhaseVolume& phasevol;
    const df::QuasiSpherical& df;
public:
    DFWrapper(const potential::PhaseVolume& _phasevol, const df::QuasiSpherical& _df) :
        phasevol(_phasevol), df(_df) {}

    virtual double value(const double h) const {
        double value;
        df::ClassicalIntegrals ints;
        ints.E = phasevol.E(h);
        ints.L = ints.Lz = 0;
        df.evalDeriv(ints, &value);
        return value;
    }
};


/// analytic drift and diffusion coefficients for energy in the case of Plummer model
void difCoefsPlummer(const double Phi, const double E, double &dvpar, double &dv2par, double &dv2per)
{
    double E0 = fmin(E, 0);
    double I0 = pow(-E0, 4.5);
    double J0 = pow(-Phi, 4.5) - I0;
    double J1, J3;
    if(E==Phi) {
        J1 = J0 * 2./3;
        J3 = J0 * 2./5;
    } else if(E>=0) {
        J1 = J0 * M_PI*63/512  * sqrt(-Phi/(E-Phi));
        J3 = J0 * M_PI*63/2048 * pow (-Phi/(E-Phi), 1.5);
    } else {
        double y  = E0/Phi, sqy = sqrt(y), z = sqrt(1-y);
        J1 = J0 * 3 / (2560 * (pow_3(pow_3(sqy)) - 1)) * 
        (105 * (asin(2*y-1) - M_PI/2) / z - y / sqy * (210 + y * (140 + y * (112 + y * (96 - y * 768)))) );
        J3 = J0 * 0.75 / (1 - pow_3(pow_3(1/sqy))) + J1 * 0.25 / (1-y);
    }
    double mult = 32*M_PI*M_PI/3 * 2*8*M_SQRT2/21/pow_3(M_PI);
    dvpar  = -mult * J1 * 3;
    dv2par =  mult * (I0 + J3);
    dv2per =  mult * (I0 * 2 + J1 * 3 - J3);
}


/// construct an interpolator for f(h)
math::LogLogSpline createInterpolatedDF(const math::IFunction& trueDF)
{
    std::vector<double> gridh = math::createExpGrid(200, 1e-25, 1e15);
    std::vector<double> gridf(gridh.size());
    for(unsigned int i=0; i<gridh.size(); i++)
        gridf[i] = trueDF(gridh[i]);
    return math::LogLogSpline(gridh, gridf);
}


/// construct an interpolated density from a cumulative mass profile sampled at discrete values of radii
/// (test the routine `densityFromCumulativeMass`)
math::LogLogSpline createInterpolatedDensity(const potential::BasePotential& pot)
{
    std::vector<double> gridr = math::createUniformGrid(101, -1., +1.), gridm(gridr.size());
    for(unsigned int i=0; i<gridr.size(); i++) {
        // logarithmic grid in r from 1e-5 to 1e5 with a denser spacing around r=1
        gridr[i] = pow(10., gridr[i] * (2 + 3*pow_2(gridr[i])));
        coord::GradCyl grad;
        pot.eval(coord::PosCyl(gridr[i], 0, 0), NULL, &grad);
        gridm[i] = pow_2(gridr[i]) * grad.dR;
    }
    return math::LogLogSpline(gridr, potential::densityFromCumulativeMass(gridr, gridm));
}


/// integrand for Rmax(Phi)
class RmaxIntegrand: public math::IFunctionNoDeriv {
    const potential::PhaseVolume& phasevol;
    const double loghmax;
public:
    RmaxIntegrand(const potential::PhaseVolume& _phasevol, double _hmax) :
    phasevol(_phasevol), loghmax(log(_hmax)) {}
    virtual double value(const double h) const {
        double PhiminusE = phasevol.deltaE(loghmax, log(h));
        double dgdh;
        phasevol.E(h, NULL, &dgdh);
        return (3/TWO_PI_CUBE/M_SQRT2) * dgdh / sqrt(PhiminusE);
    }
};


/// check the accuracy and print a warning if it's not within the required tolerance
std::string checkLess(double val, double max, bool &ok)
{
    if(!(val<max))
        ok = false;
    return utils::pp(val, 7) + (val<max ? "" : "\033[1;31m ** \033[0m");
}


template<typename Model>
bool test(const potential::BasePotential& pot, double beta=0, double r_a=INFINITY)
{
    bool isotropic= beta==0 && r_a==INFINITY;  // some tests are only applicable to isotropic models
    bool ok=true;

    const Rmax<Model> trueRmax;
    const Phasevol<Model> truePhasevol;
    const DF<Model> trueDF(beta, r_a);
    const df::QuasiSphericalCOM comDF(
        (potential::Sphericalized<potential::BaseDensity>(pot)),
        (potential::Sphericalized<potential::BasePotential>(pot)), beta, r_a);
    double mass = comDF.totalMass();

    const potential::Interpolator interp(pot);
    const potential::PhaseVolume phasevol((potential::Sphericalized<potential::BasePotential>(pot)));
    const math::LogLogSpline intRho= createInterpolatedDensity(pot);
    const math::LogLogSpline velDisp = galaxymodel::createJeansSphModel(
        potential::Sphericalized<potential::BaseDensity>(pot),
        potential::Sphericalized<potential::BasePotential>(pot), beta);

    math::PtrFunction splDF, fitDF1, fitDF2;
    shared_ptr<const galaxymodel::SphericalIsotropicModelLocal> model;
    double sumw=0, errRc=0, errRm=0, erriRm=0, errPhi=0, errdPhi=0, errdens=0, errg=0, errh=0,
        sumwf=0, errcomf=0, errsphf=0, errfitf1=0, errfitf2=0;
    std::vector<double> gridr = math::createExpGrid(321, 1e-7, 1e9);
    const double dlogr = log(gridr[1]/gridr[0]);
    std::vector<double> gridPhi(gridr.size()), gridRhoDF(gridr.size());
    for(unsigned int i=0; i<gridr.size(); i++)
        gridPhi[i] = pot.value(coord::PosCyl(gridr[i], 0, 0));

    if(isotropic) {
        // only consider the classes and routines dealing with f(h) if the model is isotropic
        DFWrapper dfw(phasevol, trueDF);
        splDF.reset(new math::LogLogSpline(createInterpolatedDF(dfw)));
        model.reset(new galaxymodel::SphericalIsotropicModelLocal(phasevol, dfw, dfw));
        gridRhoDF = galaxymodel::computeDensity(dfw, phasevol, gridPhi);   // rho(Phi) from DF

        // draw samples from the DF
        const unsigned int npoints = 100000;
        const particles::ParticleArraySph particles =
            galaxymodel::samplePosVel(interp, dfw, npoints);

        // convert position/velocity samples back into h samples
        std::vector<double> particle_h(npoints), particle_m(npoints);
        for(unsigned int i=0; i<npoints; i++) {
            particle_h[i] = phasevol(totalEnergy(pot, particles[i].first));
            particle_m[i] = particles[i].second;
        }
        fitDF1.reset(new math::LogLogSpline(df::fitSphericalIsotropicDF(particle_h, particle_m, 25)));

        // now reassign particle velocities using a different sampling procedure
        for(unsigned int i=0; i<npoints; i++) {
            double Phi = pot.value(particles[i].first);
            double v = model->sampleVelocity(Phi);
            particle_h[i] = phasevol(Phi + 0.5*v*v);
        }
        fitDF2.reset(new math::LogLogSpline(df::fitSphericalIsotropicDF(particle_h, particle_m, 25)));
    }

    std::ofstream strm, strmd;
    if(output && isotropic) {
        std::string filename = std::string("test_pot_") + pot.name() +
            "_beta" + utils::toString(beta) + "_ra" + utils::toString(r_a);
        // gnuplot script for plotting the results
        strm.open((filename+".plt").c_str());
        strm << "set term pdf enhanced size 15cm,10cm\nset output '"+filename+".pdf'\n"
        "set logscale\nset xrange [1e-7:1e9]\nset yrange [1e-16:1e-4]\n"
        "set format x '10^{%T}'\nset format y '10^{%T}'\nset multiplot layout 2,2\n"
        "plot '"+filename+".dat' u 2:(abs(1-$3/$2)) w l title 'Rcirc(E),root', \\\n"
        "  '' u 2:(abs(1-$4 /$2))  w l title 'Rcirc(E),interp', \\\n"
        "  '' u 2:(abs(1-$6 /$5))  w l title 'Lcirc(E),root', \\\n"
        "  '' u 2:(abs(1-$7 /$5))  w l title 'Lcirc(E),interp'\n"
        "p '' u 2:(abs(1-$26/$25)) w l title 'h(E),interp', \\\n"
        "  '' u 2:(abs(1-$28/$27)) w l title 'g(E),interp'\n"
        "p '' u 2:(abs(1-$9 /$8))  w l title 'Rmax(E),root', \\\n"
        "  '' u 2:(abs(1-$10/$8))  w l title 'Rmax(E),interp', \\\n"
        "  '' u 2:(abs(1-$13/$12)) w l title 'dRmax/dE,interp'\n"
        "p '' u 2:(abs(1-$15/$14)) w l title 'Phi(r),interp', \\\n"
        "  '' u 2:(abs(1-$17/$16)) w l title 'dPhi/dr,interp', \\\n"
        "  '' u 2:(abs(1-$19/$18)) w l title 'rho(r),interp', \\\n"
        "  '' u 2:(abs(1-$20/$18)) w l title 'rho(r) from M(r)', \\\n"
        "  '' u 2:(abs(1-$21/$18)) w l title 'rho(r) from DFisotr', \\\n"
        "  '' u 2:(abs(1-$22/$18)) w l title 'rho(r) from sph.model'\n";
        strm.close();
        strm. open((filename+".dat").c_str());
        strm << std::setprecision(16) <<
        "E                  \t"
        "Rcirc(E),true       "
        "Rcirc(E),root       "
        "Rcirc(E),interp    \t"
        "Lcirc(E),true       "
        "Lcirc(E),root       "
        "Lcirc(E),interp    \t"
        "Rmax(E),true        "
        "Rmax(E),root        "
        "Rmax(E),interp      "
        "Rmax(h),AbelInverse\t"
        "dRmax(E)/dE,true    "
        "dRmax(E)/dE,interp \t"
        "Phi(R=Rcirc),true   "
        "Phi(R),interp      \t"
        "dPhi/dR,true        "
        "dPhi/dR,interp     \t"
        "rho(R),true         "
        "rho(R),interp       "
        "rho(R),LogLogSpline "
        "rho(R),fromSphDF    "
        "rho(R),SphModel    \t"
        "sigma_r(R),Jeans    "
        "sigma,SphModel     \t"
        "h(E),true           "
        "h(E),PhaseVolume    "
        "g(E),true           "
        "g(E),PhaseVolume   \t"
        "f(E),true           "
        "f(E),SphAnisotrCOM  "
        "f(E),LogLogSpline   "
        "f(E),SphModel       "
        "f(E),fitToSampled1  "
        "f(E),fitToSampled2 \t"
        "DiffusionCoefE      "
        "DiffusionCoefEE    \n";
    }

    for(unsigned int i=0; i<gridr.size(); i++) {
        double r = gridr[i];
        double truePhi;
        coord::GradCyl grad;
        coord::HessCyl hess;
        pot.eval(coord::PosCyl(r,0,0), &truePhi, &grad, &hess);
        double E      = truePhi + 0.5*pow_2(v_circ(pot, r));
        double trueg, trueh;  // exact phase volume and density of states
        truePhasevol.evalDeriv(E, &trueh, &trueg);
        double intg, inth;  // interpolated h and g
        phasevol.evalDeriv(E, &inth, &intg);
        double trueLc = v_circ(pot, r) * r;
        double trueRc = r;
        double rootRc = R_circ(pot, E);
        double intRc  = interp.R_circ(E);
        double rootLc = L_circ(pot, E);
        double intLc  = interp.L_circ(E);
        double rootRm = R_max(pot, E);
        double trueRm, truedRmdE, intdRmdE;
        trueRmax.evalDeriv(E, &trueRm, &truedRmdE);
        double intRm  = interp.R_max(E, &intdRmdE);
        double invRm    = cbrt( math::integrateAdaptive(  // Abel inversion for r(h)
            math::ScaledIntegrand<math::ScalingCub>(math::ScalingCub(0, inth),
            RmaxIntegrand(phasevol, inth)), 0, 1, 1e-6) );
        double intPhi, intdPhi, intd2Phi;
        interp.evalDeriv(r, &intPhi, &intdPhi, &intd2Phi);
        double truedens = pot.density(coord::PosCyl(r,0,0));
        double intdens  = (intd2Phi + 2*intdPhi/r) / (4*M_PI);
        double spldens  = intRho(gridr[i]);
        double jeansdisp= velDisp(gridr[i]);
        // since f(E,L=0) is infinite when beta>0, we need to evaluate the DF at some finite value of L
        // and then extract the energy-dependent part of DF
        df::ClassicalIntegrals ints;
        ints.E = E;
        ints.L = 1e-100;
        ints.Lz = 0;
        double truef, comf;
        trueDF.evalDeriv(ints, &truef);
        comDF .evalDeriv(ints, &comf);
        truef *= pow(ints.L, 2*beta);
        comf  *= pow(ints.L, 2*beta);
        double splf=0, sphf=0, fitf1=0, fitf2=0, dfdens=0, sphdens=0, sphdisp=0, difE=0, difEE=0;
        if(isotropic) {
            splf   = splDF ->value(trueh);
            model->I0.evalDeriv(trueh, NULL, &sphf);  // DF in the spherical model is -g dI0/dh
            sphf  *= -trueg;
            fitf1  = fitDF1->value(trueh);
            fitf2  = fitDF2->value(trueh);
            dfdens = gridRhoDF[i];
            sphdens= model->density(truePhi);
            sphdisp= model->velDisp(truePhi);
            galaxymodel::difCoefEnergy(*model, E, /*m*/ 1., /*output*/ difE, difEE);
        }

        // density-weighted error: integrate |1-x/x_true|^2  \rho(r) r^3  d\log(r)
        double weight = truedens * 4*M_PI * pow_3(r) * dlogr;
        if(weight>0) {
            // accuracy of various quantities provided by potential::Interpolator and PhaseVolume
            sumw    += weight;
            errRc   += weight * pow_2(1 - intRc   / trueRc);
            errRm   += weight * pow_2(1 - intRm   / trueRm);
            erriRm  += weight * pow_2(1 - invRm   / trueRm);
            errPhi  += weight * pow_2(1 - intPhi  / truePhi);
            errdPhi += weight * pow_2(1 - intdPhi / grad.dR);
            errdens += weight * pow_2(1 - intdens / (truedens + fabs(intdens)) * 2);
            errh    += weight * pow_2(1 - inth    / trueh);
            errg    += weight * pow_2(1 - intg    / trueg);
        }
        // df-weighted error: integrate |1-f/f_true|^2  f(E)  G(E)  dE/dr  r  d\log(r),
        // where G(E) is the density of states in anisotropic models (equal to g in isotropic case)
        weight = truef * (1.5*grad.dR + 0.5*r*hess.dR2) * r * dlogr * trueg *
            pow(trueLc, -2*beta) / (1-beta) *  // multiplicative factor for constant-beta case
            pow(1 + pow_2(r/r_a), beta-1);     // approximate fudge factor for OM models
        if(weight) {
            sumwf   += weight;
            errcomf += weight * fabs(1 - comf  / truef);
            errsphf += weight * fabs(1 - sphf  / truef);
            errfitf1+= weight * fabs(1 - fitf1 / truef);
            errfitf2+= weight * fabs(1 - fitf2 / truef);
        }
        strm <<
            utils::pp(E,        19) + '\t'+
            utils::pp(trueRc,   19) + ' ' +
            utils::pp(rootRc,   19) + ' ' +
            utils::pp(intRc,    19) + '\t'+
            utils::pp(trueLc,   19) + ' ' +
            utils::pp(rootLc,   19) + ' ' +
            utils::pp(intLc,    19) + '\t'+
            utils::pp(trueRm,   19) + ' ' + 
            utils::pp(rootRm,   19) + ' ' +
            utils::pp(intRm,    19) + ' ' + 
            utils::pp(invRm,    19) + '\t'+
            utils::pp(truedRmdE,19) + ' ' +
            utils::pp(intdRmdE, 19) + '\t'+
            utils::pp(truePhi,  19) + ' ' +
            utils::pp(intPhi,   19) + '\t'+ 
            utils::pp(grad.dR,  19) + ' ' +
            utils::pp(intdPhi,  19) + '\t'+
            utils::pp(truedens, 19) + ' ' +
            utils::pp(intdens,  19) + ' ' +
            utils::pp(spldens,  19) + ' ' +
            utils::pp(dfdens,   19) + ' ' +
            utils::pp(sphdens,  19) + '\t'+
            utils::pp(jeansdisp,19) + ' ' +
            utils::pp(sphdisp,  19) + '\t'+
            utils::pp(trueh,    19) + ' ' +
            utils::pp(inth,     19) + ' ' +
            utils::pp(trueg,    19) + ' ' +
            utils::pp(intg,     19) + '\t'+
            utils::pp(truef,    19) + ' ' +
            utils::pp(comf,     19) + ' ' +
            utils::pp(splf,     19) + ' ' +
            utils::pp(sphf,     19) + ' ' +
            utils::pp(fitf1,    19) + ' ' +
            utils::pp(fitf2,    19) + '\t'+
            utils::pp(difE,     19) + ' ' +
            utils::pp(difEE,    19) + '\n';

        for(double vrel=0; isotropic && vrel<1.25; vrel+=0.03125) {
            double E = (1-pow_2(vrel)) * truePhi;
            double intdvpar, intdv2par, intdv2per;
            double truedvpar=0, truedv2par=0, truedv2per=0;
            model->evalLocal(truePhi, E, 1., intdvpar, intdv2par, intdv2per);
            // note: no analytic expressions for the Hernquist model
            if(pot.name() == potential::Plummer::myName())
                difCoefsPlummer(truePhi, E, truedvpar, truedv2par, truedv2per);
            strmd << log(phasevol(truePhi)) << ' ' << log(phasevol(E)) << ' ' <<
                truePhi << ' ' << E << '\t' <<
                truedvpar << ' ' << truedv2par << ' ' << truedv2per << '\t' <<
                intdvpar  << ' ' <<  intdv2par << ' ' <<  intdv2per << '\n';
        }
        strmd << '\n';
    }
    errRc   = sqrt(errRc/sumw);
    errRm   = sqrt(errRm/sumw);
    erriRm  = sqrt(erriRm/sumw);
    errPhi  = sqrt(errPhi/sumw);
    errdPhi = sqrt(errdPhi/sumw);
    errdens = sqrt(errdens/sumw);
    errh    = sqrt(errh/sumw);
    errg    = sqrt(errg/sumw);
    errcomf = errcomf/sumwf;
    errsphf = errsphf/sumwf;
    errfitf1= errfitf1/sumwf;
    errfitf2= errfitf2/sumwf;
    std::cout << "\033[1;33m " << pot.name() << " beta=" << beta << " r_a=" << r_a <<
        " \033[0m: weighted RMS error in";
    if(model) {  // check most quantities only once, for the beta=0 and r_a=0 variant of each model
        std::cout << 
        "  Rcirc="  + checkLess(errRc,  2e-09, ok) +
        ", Rmax(E)="+ checkLess(errRm,  1e-10, ok) +
        ", Rmax(h)="+ checkLess(erriRm, 2e-08, ok) +
        ", Phi="    + checkLess(errPhi, 1e-10, ok) +
        ", dPhi/dr="+ checkLess(errdPhi,1e-08, ok) +
        ", rho="    + checkLess(errdens,3e-04, ok) +
        ", h="      + checkLess(errh,   1e-08, ok) +
        ", g="      + checkLess(errg,   1e-08, ok) +
        ", sphf="   + checkLess(errsphf,1e-06, ok) +
        ", fitf1="  + checkLess(errfitf1,0.02, ok) +
        ", fitf2="  + checkLess(errfitf2,0.02, ok) + ",";
    }
    std::cout <<
       " comf="     + checkLess(errcomf,2e-05, ok) +
       ", DF mass=" + checkLess(mass-1,  0.01, ok) + "\n";
    return ok;
}


int main()
{
    bool ok=true;
    potential::Plummer potp(1., 1.);
    potential::Dehnen  poth(1., 1., /*gamma*/1.0, 1., 1.);
    ok &= test< Plummer >(potp);
    ok &= test< Plummer >(potp, 0.0, 0.75);
    ok &= test< Plummer >(potp,-0.5, 0.50);
    ok &= test<Hernquist>(poth);
    ok &= test<Hernquist>(poth, 0.5);
    ok &= test<Hernquist>(poth, 0.2);
    ok &= test<Hernquist>(poth,-0.2);
    ok &= test<Hernquist>(poth,-0.5);
    if(ok)
        std::cout << "\033[1;32mALL TESTS PASSED\033[0m\n";
    else
        std::cout << "\033[1;31mSOME TESTS FAILED\033[0m\n";
    return 0;
}