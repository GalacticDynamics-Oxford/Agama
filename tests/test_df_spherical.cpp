/** \name   test_df_spherical.cpp
    \author Eugene Vasiliev
    \date   Sep 2016

    This program tests the accuracy of computation of phase volume 'h(E)',
    density of states 'g(E)', distribution function 'f(h)', and diffusion coefficients,
    for Plummer and Hernquist spherical isotropic models.
*/
#include "potential_analytic.h"
#include "potential_dehnen.h"
#include "potential_utils.h"
#include "math_core.h"
#include "math_specfunc.h"
#include "df_spherical.h"
#include "utils.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <stdexcept>

/// whether to produce output files
const bool output = utils::verbosityLevel >= utils::VL_VERBOSE;

/// analytic expressions for rmax(E) in the Plummer and Hernquist models
class RmaxPlummer: public math::IFunction {
public:
    RmaxPlummer(){}
    virtual void evalDeriv(double E, double* Rmax, double* dRmaxdE, double* =NULL) const {
        if(Rmax)
            *Rmax = -sqrt(1-E*E)/E;
        if(dRmaxdE)
            *dRmaxdE = 1/(E*E*sqrt(1-E*E));
    }
    virtual unsigned int numDerivs() const { return 1; }
};

class RmaxHernquist: public math::IFunction {
public:
    RmaxHernquist(){}
    virtual void evalDeriv(double E, double* Rmax, double* dRmaxdE, double* =NULL) const {
        if(Rmax)
            *Rmax = -1/E-1;
        if(dRmaxdE)
            *dRmaxdE = 1/pow_2(E);
    }
    virtual unsigned int numDerivs() const { return 1; }
};

/// analytic expressions for phase volume and density of states in the Plummer model
class PhasevolPlummer: public math::IFunction {
public:
    PhasevolPlummer(){}
    virtual void evalDeriv(double E, double* h, double* g, double* =NULL) const {
        if(E<-0.999) {  // asymptotic expressions for E -> -1
            double x = E+1;
            if(h)
                *h = pow_3(M_PI * x) *
                    (4./3 + x * (15./8 + x * (287./128 + x * 15561./1024)));
            if(g)
                *g = pow_2(M_PI * x) * M_PI *
                    (4.   + x * (15./2 + x * (1435./128 + x * 5187./2048)));
            return;
        }
        double x=sqrt((E+1)/2);
        double elE=math::ellintE(x, true);
        double elK=math::ellintK(x, true);
        double elP=math::ellintP(M_PI/2, x, -1-E, true);
        if(h)
            *h = 4*M_PI*M_PI/(9*E) *  ( elK * (3-34*E-8*E*E) + elE * (16*E*E-6) - elP * (36*E*E+3) );
        if(g)
            *g = 2*M_PI*M_PI/(3*E*E) * ( elK * (2*E-3-8*E*E) + elE * (16*E*E+6) - elP * (12*E*E-3) );
    }
    virtual unsigned int numDerivs() const { return 1; }
};

/// analytic expressions for phase volume and density of states in the Hernquist model
class PhasevolHernquist: public math::IFunction {
public:
    PhasevolHernquist(){}
    virtual void evalDeriv(double E, double* h, double* g, double* =NULL) const {
        if(E<-0.99) {  // asymptotic expressions for E -> -1
            double x = E+1;
            if(h)
                *h = M_PI*M_PI*M_SQRT2*512./945 * sqrt(x) * pow_2(x*x) *
                (1 + x * 24./11 * (1 + x * 20./13 * (1 + x * 4. /3 )));
            if(g)
                *g = M_PI*M_PI*M_SQRT2*256./105 * sqrt(x) * pow_3(x) *
                (1 + x * 8. /3  * (1 + x * 20./11 * (1 + x * 20./13)));
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

/// analytic distribution function of the Plummer model
class DFPlummer: public math::IFunctionNoDeriv {
    const df::PhaseVolume& pv;
public:
    DFPlummer(const df::PhaseVolume& _pv) : pv(_pv) {}
    virtual double value(double h) const {
        return 3*8*M_SQRT2/7/pow_3(M_PI) * pow(-pv.E(h), 3.5);
    }
};

/// same for the Hernquist model
class DFHernquist: public math::IFunctionNoDeriv {
    const df::PhaseVolume& pv;
public:
    DFHernquist(const df::PhaseVolume& _pv) : pv(_pv) {}
    virtual double value(double h) const {
        double E = pv.E(h);
        double q = sqrt(-E), sq1pE = sqrt(1+E);
        return 1/M_SQRT2/pow_3(2*M_PI) / ( pow_2(1+E) * sq1pE ) *
        (3*asin(q) + q * sq1pE * (1+2*E) * (8*E*E + 8*E - 3) );
    }
};

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

/// construct an interpolator for f(h) from the provided h(E) and f(E)
df::SphericalIsotropic createInterpolatedDF(const df::PhaseVolume& phasevol, const math::IFunction& trueDF)
{
    const unsigned int gridSize = 100;
    std::vector<double> gridh = math::createUniformGrid(gridSize, phasevol.logHmin(), phasevol.logHmax());
    std::vector<double> gridf(gridSize);
    for(unsigned int i=0; i<gridSize; i++) {
        gridh[i] = exp(gridh[i]);
        gridf[i] = trueDF(gridh[i]);
    }
    return df::SphericalIsotropic(gridh, gridf);
}

template<class RmaxFnc, class PhasevolFnc, class DistrFnc>
bool test(const potential::BasePotential& pot)
{
    bool ok=true;

    potential::Interpolator interp(pot);
    df::PhaseVolume phasevol((potential::PotentialWrapper(pot)));
    const RmaxFnc trueRmax;
    const PhasevolFnc truePhasevol;
    const DistrFnc trueDF(phasevol);
    df::SphericalIsotropic intDF = createInterpolatedDF(phasevol, trueDF);
    df::DiffusionCoefs dc(phasevol, intDF);

    const unsigned int npoints = 100000;
    std::vector<double> particle_h = sampleSphericalDF(dc, npoints);
    std::vector<double> particle_m(npoints, dc.cumulMass()/npoints);
    df::SphericalIsotropic fitDF = df::fitSphericalDF(particle_h, particle_m, 25);

    std::ofstream strm, strmd;
    if(output) {
        strm. open(("test_pot_"+std::string(pot.name())).c_str());
        strmd.open(("test_pot_"+std::string(pot.name())+"_dc").c_str());
    }
    strm << std::setprecision(15) << "E\t"
    "Rcirc(E),true Rcirc,root Rcirc,interp\t"
    "Lcirc(E),true Lcirc,root Lcirc,interp\t"
    "Rmax(E),true Rmax,root Rmax,interp\t"
    "dRmax(E)/dE,true dRmax/dE,interp\t"
    "Phi(Rcirc),true Phi,interp\t"
    "dPhi/dr,true dPhi/dr,interp\t"
    "d2Phi/dr2,true d2Phi/dr2,interp\t"
    "h(E),true h(E),interp g(E),true g(E),interp\t"
    "f(E),true f(E),interp f(E),fit\n";
    strmd << std::setprecision(15);

    double sumw=0, errRc=0, errRm=0, errPhi=0, errdPhi=0, errdens=0, errg=0, errh=0;
    for(double lr=-16; lr<=23; lr+=.25) {
        double r = pow(2., lr);
        double truePhi;
        coord::GradCyl grad;
        coord::HessCyl hess;
        pot.eval(coord::PosCyl(r,0,0), &truePhi, &grad, &hess);
        double E      = truePhi + 0.5*pow_2(v_circ(pot, r));
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
        double trueg, trueh;  // exact phase volume and density of states
        truePhasevol.evalDeriv(E, &trueh, &trueg);
        double intg, inth, intDE, intDEE;  // interpolated h and g
        phasevol.evalDeriv(E, &inth, &intg);
        dc.evalOrbitAvg(E, intDE, intDEE);
        double intPhi, intdPhi, intd2Phi;
        interp.evalDeriv(r, &intPhi, &intdPhi, &intd2Phi);
        double truedens = hess.dR2 + 2*grad.dR/r;
        double intdens  = intd2Phi + 2*intdPhi/r;
        double truef    = trueDF(trueh);
        double intf     = intDF(trueh);
        double fitf     = fitDF(trueh);

        // density-weighted error: integrate |x-x_true|^2 r^3 d log(r)
        double weight= pow_3(r) * pot.density(coord::PosCyl(r,0,0));
        sumw    += weight;
        errRc   += weight * pow_2((trueRc  - intRc)   / (trueRc  + intRc)   *2);
        errRm   += weight * pow_2((trueRm  - intRm)   / (trueRm  + intRm)   *2);
        errPhi  += weight * pow_2((truePhi - intPhi)  / (truePhi + intPhi)  *2);
        errdPhi += weight * pow_2((grad.dR - intdPhi) / (grad.dR + intdPhi) *2);
        errdens += weight * pow_2((truedens- intdens) / (truedens+ intdens) *2);
        errh    += weight * pow_2((trueh   - inth)    / (trueh   + inth)    *2);
        errg    += weight * pow_2((trueg   - intg)    / (trueg   + intg)    *2);

        strm << E << '\t' <<
            trueRc << ' ' << rootRc << ' ' << intRc << '\t' <<
            trueLc << ' ' << rootLc << ' ' << intLc << '\t' <<
            trueRm << ' ' << rootRm << ' ' << intRm << '\t' <<
            truedRmdE << ' ' << intdRmdE << '\t' <<
            truePhi<< ' ' << intPhi << '\t'<< 
            grad.dR<< ' ' << intdPhi<< '\t'<<
            truedens<<' ' << intdens<< '\t'<<
            trueh  << ' ' << inth   << ' ' << trueg << ' ' << intg << '\t' <<
            intDEE << ' ' << intDE  << '\t'<<
            truef  << ' ' << intf   << ' ' << fitf << '\n';

        for(double vrel=0; vrel<1.25; vrel+=0.03125) {
            double E = (1-pow_2(vrel)) * truePhi;
            double  intdvpar,  intdv2par,  intdv2per;
            double truedvpar, truedv2par, truedv2per;
            dc.evalLocal(truePhi, E, intdvpar, intdv2par, intdv2per);
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
    errPhi  = sqrt(errPhi/sumw);
    errdPhi = sqrt(errdPhi/sumw);
    errdens = sqrt(errdens/sumw);
    errh    = sqrt(errh/sumw);
    errg    = sqrt(errg/sumw);
    std::cout << pot.name() << ": weighted RMS error in Rcirc=" << errRc << ", Rmax=" << errRm <<
    ", Phi=" << errPhi << ", dPhi/dr=" << errdPhi << ", rho=" << errdens <<
    ", h=" << errh << ", g=" << errg << "\n";
    ok &= errRc  < 1e-10 && errRm   < 1e-10 &&
          errPhi < 1e-10 && errdPhi < 1e-08 && errdens < 1e-04 &&
          errh   < 1e-08 && errg    < 1e-08;
    return ok;
}

int main()
{
    bool ok=true;
    potential::Plummer potp(1., 1.);
    potential::Dehnen  poth(1., 1., 1., 1., 1.);

    ok &= test<RmaxPlummer,   PhasevolPlummer,   DFPlummer  >(potp);
    ok &= test<RmaxHernquist, PhasevolHernquist, DFHernquist>(poth);
    if(ok)
        std::cout << "\033[1;32mALL TESTS PASSED\033[0m\n";
    else
        std::cout << "\033[1;31mSOME TESTS FAILED\033[0m\n";
    return 0;
}