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
#include "galaxymodel_spherical.h"
#include "utils.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <algorithm>
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
    const potential::PhaseVolume& pv;
public:
    DFPlummer(const potential::PhaseVolume& _pv) : pv(_pv) {}
    virtual double value(double h) const {
        return 3*8*M_SQRT2/7/pow_3(M_PI) * pow(-pv.E(h), 3.5);
    }
};

/// same for the Hernquist model
class DFHernquist: public math::IFunctionNoDeriv {
    const potential::PhaseVolume& pv;
public:
    DFHernquist(const potential::PhaseVolume& _pv) : pv(_pv) {}
    virtual double value(double h) const {
        double E = pv.E(h);
        double q = sqrt(-E), sq1pE = sqrt(1+E);
        if(E>-1e-5)
            return 8./5*M_SQRT2/pow_3(M_PI) * math::powInt(q, 5) * (1 - 10./7*E + 40./21*E*E);
        if(E<-1+1e-5)
            return 3./32*M_SQRT2/pow_2(M_PI) * (math::powInt(sq1pE, -5) - 256./(15*M_PI));
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

/// construct an interpolator for f(h)
math::LogLogSpline createInterpolatedDF(const math::IFunction& trueDF)
{
    std::vector<double> gridh = math::createExpGrid(200, 1e-15, 1e15);
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
    return math::LogLogSpline(gridr, galaxymodel::densityFromCumulativeMass(gridr, gridm));
}

/// check the accuracy and print a warning if it's not within the required tolerance
std::string checkLess(double val, double max, bool &ok)
{
    if(!(val<max))
        ok = false;
    return utils::pp(val, 7) + (val<max ? "" : "\033[1;31m ** \033[0m");
}


template<class RmaxFnc, class PhasevolFnc, class DistrFnc>
bool test(const potential::BasePotential& pot)
{
    bool ok=true;

    potential::Interpolator interp(pot);
    potential::PhaseVolume phasevol((potential::PotentialWrapper(pot)));
    const RmaxFnc trueRmax;
    const PhasevolFnc truePhasevol;
    const DistrFnc trueDF(phasevol);
    galaxymodel::DiffusionCoefs dc(phasevol, trueDF);
    math::LogLogSpline intDF = createInterpolatedDF(trueDF);
    math::LogLogSpline intRho= createInterpolatedDensity(pot);
    math::LogLogSpline eddDF = galaxymodel::makeEddingtonDF(
        potential::DensityWrapper(pot), potential::PotentialWrapper(pot));
    const unsigned int npoints = 100000;
    particles::ParticleArraySph particles = galaxymodel::generatePosVelSamples(interp, trueDF, npoints);
    std::vector<double> particle_h(npoints), particle_m(npoints);
    for(unsigned int i=0; i<npoints; i++) {
        particle_h[i] = phasevol(totalEnergy(pot, particles[i].first));
        particle_m[i] = particles[i].second;
    }
    math::LogLogSpline fitDF = galaxymodel::fitSphericalDF(particle_h, particle_m, 25);

    std::ofstream strm, strmd;
    if(output) {
        std::string filename = std::string("test_pot_" )+pot.name();
        // gnuplot script for plotting the results
        strm.open((filename+".plt").c_str());
        strm << "set term pdf enhanced size 15cm,10cm\nset output '"+filename+".pdf'\n"
        "set logscale\nset xrange [1e-7:1e9]\nset yrange [1e-16:1e-4]\n"
        "set format x '10^{%T}'\nset format y '10^{%T}'\nset multiplot layout 2,2\n"
        "plot '"+filename+".dat' u 2:(abs(1-$3/$2)) w l title 'Rcirc(E),root', \\\n"
        "  '' u 2:(abs(1-$4 /$2))  w l title 'Rcirc(E),interp', \\\n"
        "  '' u 2:(abs(1-$6 /$5))  w l title 'Lcirc(E),root', \\\n"
        "  '' u 2:(abs(1-$7 /$5))  w l title 'Lcirc(E),interp'\n"
        "p '' u 2:(abs(1-$22/$21)) w l title 'h(E),interp', \\\n"
        "  '' u 2:(abs(1-$24/$23)) w l title 'g(E),interp'\n"
        "p '' u 2:(abs(1-$9 /$8))  w l title 'Rmax(E),root', \\\n"
        "  '' u 2:(abs(1-$10/$8))  w l title 'Rmax(E),interp', \\\n"
        "  '' u 2:(abs(1-$12/$11)) w l title 'dRmax/dE,interp'\n"
        "p '' u 2:(abs(1-$14/$13)) w l title 'Phi(r),interp', \\\n"
        "  '' u 2:(abs(1-$16/$15)) w l title 'dPhi/dr,interp', \\\n"
        "  '' u 2:(abs(1-$18/$17)) w l title 'rho(r),interp', \\\n"
        "  '' u 2:(abs(1-$19/$17)) w l title 'rho(r) from M(r)', \\\n"
        "  '' u 2:(abs(1-$20/$17)) w l title 'rho(r) from DF'\n";
        strm.close();
        strm. open((filename+".dat").c_str());
        strmd.open((filename+"_dc.dat").c_str());
    }
    strm << std::setprecision(16) << "E\t"
    "Rcirc(E),true Rcirc,root Rcirc,interp\t"
    "Lcirc(E),true Lcirc,root Lcirc,interp\t"
    "Rmax(E),true Rmax,root Rmax,interp\t"
    "dRmax(E)/dE,true dRmax/dE,interp\t"
    "Phi(Rcirc),true Phi,interp\t"
    "dPhi/dr,true dPhi/dr,interp\t"
    "rho,true rho,interp rho,fromCumulMass rho,fromDF\t"
    "h(E),true h(E),interp g(E),true g(E),interp\t"
    "f(E),true f(E),interp f(E),fit f(E),Eddington f(E),sphmodel\n";
    strmd << std::setprecision(15);

    double sumw=0, errRc=0, errRm=0, errPhi=0, errdPhi=0, errdens=0, errg=0, errh=0;
    std::vector<double> gridr = math::createExpGrid(321, 1e-7, 1e9);
    std::vector<double> gridPhi(gridr.size());
    for(unsigned int i=0; i<gridr.size(); i++)
        gridPhi[i] = pot.value(coord::PosCyl(gridr[i], 0, 0));
    std::vector<double> gridRhoDF = galaxymodel::computeDensity(trueDF, phasevol, gridPhi);
    for(unsigned int i=0; i<gridr.size(); i++) {
        double r = gridr[i];
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
        double truedens = (hess.dR2 + 2*grad.dR/r) / (4*M_PI);
        double intdens  = (intd2Phi + 2*intdPhi/r) / (4*M_PI);
        double truef    = trueDF(trueh);
        double intf     = intDF(trueh);
        double fitf     = fitDF(trueh);
        double eddf     = eddDF(trueh);
        double sphf     = dc.model.value(trueh);
        double cmdens   = intRho(gridr[i]);
        double dfdens   = gridRhoDF[i];

        // density-weighted error: integrate |x-x_true|^2 r^3 d log(r)
        double weight = pow_3(r) * truedens;
        if(weight>0) {
            sumw    += weight;
            errRc   += weight * pow_2((trueRc  - intRc)   / (trueRc  + intRc)   *2);
            errRm   += weight * pow_2((trueRm  - intRm)   / (trueRm  + intRm)   *2);
            errPhi  += weight * pow_2((truePhi - intPhi)  / (truePhi + intPhi)  *2);
            errdPhi += weight * pow_2((grad.dR - intdPhi) / (grad.dR + intdPhi) *2);
            errdens += weight * pow_2((truedens- intdens) / (truedens+ intdens) *2);
            errh    += weight * pow_2((trueh   - inth)    / (trueh   + inth)    *2);
            errg    += weight * pow_2((trueg   - intg)    / (trueg   + intg)    *2);
        }
        strm << E << '\t' <<
            trueRc << ' ' << rootRc << ' ' << intRc << '\t' <<
            trueLc << ' ' << rootLc << ' ' << intLc << '\t' <<
            trueRm << ' ' << rootRm << ' ' << intRm << '\t' <<
            truedRmdE << ' ' << intdRmdE << '\t' <<
            truePhi<< ' ' << intPhi << '\t'<< 
            grad.dR<< ' ' << intdPhi<< '\t'<<
            truedens<<' ' << intdens<< ' ' << cmdens << ' ' << dfdens << '\t' <<
            trueh  << ' ' << inth   << ' ' << trueg  << ' ' << intg   << '\t' <<
            intDEE << ' ' << intDE  << '\t'<<
            truef  << ' ' << intf   << ' ' << fitf << ' ' << eddf << ' ' << sphf << '\n';

        for(double vrel=0; vrel<1.25; vrel+=0.03125) {
            double E = (1-pow_2(vrel)) * truePhi;
            double intdvpar, intdv2par, intdv2per;
            double truedvpar=0, truedv2par=0, truedv2per=0;
            dc.evalLocal(truePhi, E, intdvpar, intdv2par, intdv2per);
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
    errPhi  = sqrt(errPhi/sumw);
    errdPhi = sqrt(errdPhi/sumw);
    errdens = sqrt(errdens/sumw);
    errh    = sqrt(errh/sumw);
    errg    = sqrt(errg/sumw);
    std::cout << "\033[1;33m " << pot.name() << " \033[0m: weighted RMS error in"
    "  Rcirc=" + checkLess(errRc,  1e-09, ok) +
    ", Rmax="  + checkLess(errRm,  1e-10, ok) +
    ", Phi="   + checkLess(errPhi, 1e-10, ok) +
    ", dPhi/dr="+checkLess(errdPhi,1e-08, ok) +
    ", rho="   + checkLess(errdens,1e-03, ok) +
    ", h="     + checkLess(errh,   1e-08, ok) +
    ", g="     + checkLess(errg,   1e-08, ok) + "\n";
    return ok;
}

void exportTable(const char* filename, const potential::BasePotential& pot)
{
    try{
        std::vector<double> h, f;
        potential::PhaseVolume phasevol((potential::PotentialWrapper(pot)));
        potential::Interpolator interp(pot);
        galaxymodel::makeEddingtonDF(potential::DensityWrapper(pot), potential::PotentialWrapper(pot), h, f);
        std::ofstream strm(filename);
        strm << "r\tM\tPhi\trho\tf\tg\th\n";
        for(unsigned int i=0; i<h.size(); i++) {
            double g, E= phasevol.E(h[i], &g);
            double r   = interp.R_max(E);
            double M   = pow_2(v_circ(pot, r)) * r;
            double rho = pot.density(coord::PosCyl(r, 0, 0));
            strm << utils::pp(r,14) + '\t' + utils::pp(M,14) + '\t' + 
                utils::pp(E,14) + '\t' + utils::pp(rho,14) + '\t' +
                utils::pp(f[i],14) + '\t' + utils::pp(g,14) + '\t' + utils::pp(h[i],14) + '\n';
        }
    }
    catch(std::exception& e){
        std::cout << filename << " => " << e.what() << '\n';
    }
}

int main()
{
    bool ok=true;
    potential::Plummer potp(1., 1.);
    potential::Dehnen  poth(1., 1., 1., 1., 1.);
    if(utils::verbosityLevel >= utils::VL_VERBOSE) {
        potential::Dehnen  pot0(1., 1., 0., 1., 1.);
        potential::Dehnen  pot2(1., 1., 2., 1., 1.);
        exportTable("test_Plummer.tab", potp);
        exportTable("test_Dehnen0.tab", pot0);
        exportTable("test_Dehnen1.tab", poth);
        exportTable("test_Dehnen2.tab", pot2);
    }
    ok &= test<RmaxPlummer,   PhasevolPlummer,   DFPlummer  >(potp);
    ok &= test<RmaxHernquist, PhasevolHernquist, DFHernquist>(poth);
    if(ok)
        std::cout << "\033[1;32mALL TESTS PASSED\033[0m\n";
    else
        std::cout << "\033[1;31mSOME TESTS FAILED\033[0m\n";
    return 0;
}