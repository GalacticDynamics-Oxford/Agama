#include "math_ode.h"
#include <cmath>
#include <stdexcept>

namespace math{

inline bool isFinite(double x) {
    return x==x && 1/x!=0;  // false for +-INFINITY or NAN
}

/* ----------------- ODE integrators ------------- */

/* --- DOP853 high-accuracy Runge-Kutta integrator --- */

OdeSolverDOP853::OdeSolverDOP853(const IOdeSystem& _odeSystem, double _accRel, double _accAbs):
    BaseOdeSolver(_odeSystem), accRel(_accRel), accAbs(_accAbs), timeStep(0)
{
    const unsigned int n = odeSystem.size();
    statePrev.assign(n, 0);
    stateCurr.assign(n, 0);
    /* allocate storage for intermediate calculations */
    ytemp.resize(n);
    k1.resize(n);
    k2.resize(n);
    k3.resize(n);
    k4.resize(n);
    k5.resize(n);
    k6.resize(n);
    k7.resize(n);
    k8.resize(n);
    k9.resize(n);
    k10.resize(n);
    rcont1.resize(n);
    rcont2.resize(n);
    rcont3.resize(n);
    rcont4.resize(n);
    rcont5.resize(n);
    rcont6.resize(n);
    rcont7.resize(n);
    rcont8.resize(n);
}

double OdeSolverDOP853::initTimeStep()
{
    // on entering this routine, stateCurr contains the current values of variables,
    // and k1 - their derivatives
    double dnf = 0, dny = 0;
    for (unsigned int i = 0; i < stateCurr.size(); i++) {
        double sk = accAbs + accRel * fabs(stateCurr[i]);
        if(sk==0) continue;
        double sqre = k1[i] / sk;
        dnf += sqre*sqre;
        sqre = stateCurr[i] / sk;
        dny += sqre*sqre;
    }
    double h = sqrt (dny/dnf) * 0.01;
    if(!isFinite(dnf+dny) || (dnf <= 1e-15) || (dny <= 1e-15))  // safety measures
        h = 1e-6;  // some arbitrary but small value

    /* perform an explicit Euler step */
    for (unsigned int i = 0; i < stateCurr.size(); i++)
        ytemp[i] = stateCurr[i] + h * k1[i];
    odeSystem.eval(timeCurr+h, ytemp, k2);
    
    /* estimate the second derivative of the solution */
    double der2 = 0.0;
    for (unsigned int i = 0; i < stateCurr.size(); i++) {
        double sk = accAbs + accRel * fabs(stateCurr[i]);
        double sqre = (k2[i] - k1[i]) / sk;
        if(sk!=0) der2 += sqre*sqre;
    }
    der2 = sqrt (der2) / h;
    
    /* step size is computed such that h^8 * max(norm(der),norm(der2)) = 0.01 */
    double der12 = fmax(fabs(der2), sqrt(dnf));
    double h1 = der12 > 1e-15 ? pow (0.01/der12, 1./8) : fmax(1e-6, h*1e-3);
    h = fmin(100.0 * h, h1);
    return h;
    
}
    
void OdeSolverDOP853::init(const OdeStateType& state)
{
    if(stateCurr.size() != state.size())
        throw std::runtime_error("ODE system size should not change");
    stateCurr = state;
    odeSystem.eval(timeCurr, stateCurr, k1);
    if(timeStep == 0)
        timeStep = initTimeStep();
}
    
double OdeSolverDOP853::doStep()
{
    // initialize some internal constants
    static const double
    c2  = 0.526001519587677318785587544488E-01,
    c3  = 0.789002279381515978178381316732E-01,
    c4  = 0.118350341907227396726757197510E+00,
    c5  = 0.281649658092772603273242802490E+00,
    c6  = 0.333333333333333333333333333333E+00,
    c7  = 0.25E+00,
    c8  = 0.307692307692307692307692307692E+00,
    c9  = 0.651282051282051282051282051282E+00,
    c10 = 0.6E+00,
    c11 = 0.857142857142857142857142857142E+00,
    b1  =  5.42937341165687622380535766363E-2,
    b6  =  4.45031289275240888144113950566E0,
    b7  =  1.89151789931450038304281599044E0,
    b8  = -5.8012039600105847814672114227E0,
    b9  =  3.1116436695781989440891606237E-1,
    b10 = -1.52160949662516078556178806805E-1,
    b11 =  2.01365400804030348374776537501E-1,
    b12 =  4.47106157277725905176885569043E-2,
    bhh1=  0.244094488188976377952755905512E+00,
    bhh2=  0.733846688281611857341361741547E+00,
    bhh3=  0.220588235294117647058823529412E-01,
    er1 =   0.1312004499419488073250102996E-01,
    er6 =  -0.1225156446376204440720569753E+01,
    er7 =  -0.4957589496572501915214079952E+00,
    er8 =   0.1664377182454986536961530415E+01,
    er9 =  -0.3503288487499736816886487290E+00,
    er10=   0.3341791187130174790297318841E+00,
    er11=   0.8192320648511571246570742613E-01,
    er12=  -0.2235530786388629525884427845E-01,
    a21 =    5.26001519587677318785587544488E-2,
    a31 =    1.97250569845378994544595329183E-2,
    a32 =    5.91751709536136983633785987549E-2,
    a41 =    2.95875854768068491816892993775E-2,
    a43 =    8.87627564304205475450678981324E-2,
    a51 =    2.41365134159266685502369798665E-1,
    a53 =   -8.84549479328286085344864962717E-1,
    a54 =    9.24834003261792003115737966543E-1,
    a61 =    3.7037037037037037037037037037E-2,
    a64 =    1.70828608729473871279604482173E-1,
    a65 =    1.25467687566822425016691814123E-1,
    a71 =    3.7109375E-2,
    a74 =    1.70252211019544039314978060272E-1,
    a75 =    6.02165389804559606850219397283E-2,
    a76 =   -1.7578125E-2,
    a81 =    3.70920001185047927108779319836E-2,
    a84 =    1.70383925712239993810214054705E-1,
    a85 =    1.07262030446373284651809199168E-1,
    a86 =   -1.53194377486244017527936158236E-2,
    a87 =    8.27378916381402288758473766002E-3,
    a91 =    6.24110958716075717114429577812E-1,
    a94 =   -3.36089262944694129406857109825E0,
    a95 =   -8.68219346841726006818189891453E-1,
    a96 =    2.75920996994467083049415600797E1,
    a97 =    2.01540675504778934086186788979E1,
    a98 =   -4.34898841810699588477366255144E1,
    a101 =   4.77662536438264365890433908527E-1,
    a104 =  -2.48811461997166764192642586468E0,
    a105 =  -5.90290826836842996371446475743E-1,
    a106 =   2.12300514481811942347288949897E1,
    a107 =   1.52792336328824235832596922938E1,
    a108 =  -3.32882109689848629194453265587E1,
    a109 =  -2.03312017085086261358222928593E-2,
    a111 =  -9.3714243008598732571704021658E-1,
    a114 =   5.18637242884406370830023853209E0,
    a115 =   1.09143734899672957818500254654E0,
    a116 =  -8.14978701074692612513997267357E0,
    a117 =  -1.85200656599969598641566180701E1,
    a118 =   2.27394870993505042818970056734E1,
    a119 =   2.49360555267965238987089396762E0,
    a1110=  -3.0467644718982195003823669022E0,
    a121 =   2.27331014751653820792359768449E0,
    a124 =  -1.05344954667372501984066689879E1,
    a125 =  -2.00087205822486249909675718444E0,
    a126 =  -1.79589318631187989172765950534E1,
    a127 =   2.79488845294199600508499808837E1,
    a128 =  -2.85899827713502369474065508674E0,
    a129 =  -8.87285693353062954433549289258E0,
    a1210=   1.23605671757943030647266201528E1,
    a1211=   6.43392746015763530355970484046E-1,
    // coefficients for 6th order interpolation instead of the original 8th order
    d41  = -5.40685903845352664250302,
    d46  = +367.268892700041893590281,
    d47  = +154.609958204083905482676,
    d48  = -505.920283865412564024766,
    d49  = +15.5975154819608130688200,
    d410 = -26.1936204184402805956691,
    d411 = -.740035123641222308447206,
    d412 = +1.11776539319431476294221,
    d413 = -.333333333333333333333333,
    d51  = +6.51987095363079615048119,
    d56  = -1066.34956011730205278592,
    d57  = -351.864047514639508625601,
    d58  = +1363.51955696662884408368,
    d59  = -112.727669432657582669864,
    d510 = +159.796191868560289612921,
    d511 = -2.13865100308788816220259,
    d512 = -3.75569172113289760348584,
    d513 = +7.00000000000000000000000,
    d61  = +10.4698004763293477204238,
    d66  = -1380.01473607038123167155,
    d67  = -531.219827862514074379012,
    d68  = +1866.98964341870892451324,
    d69  = -53.3302605020547902574560,
    d610 = +82.4147560258671369782481,
    d611 = +7.38443654502992069572676,
    d612 = +.417299080125877511498426,
    d613 = -3.11111111111111111111111,
    d71  = -16.6338582677165354330709,
    d76  = +4516.16568914956011730205,
    d77  = +1393.85185384057776465219,
    d78  = -5687.52042419481539670071,
    d79  = +473.965563750151263163661,
    d710 = -661.810776942355889724311,
    d711 = -18.0180473354013232598119,
    d712 = 0,
    d713 = 0,
    safe = 0.9,   // safety factor
    fac1 = 0.333, // parameters for step size selection
    fac2 = 6.0;
    const unsigned int n = stateCurr.size();
    do {
        if(timeStep <= fabs(timeCurr)*1e-15 || !isFinite(timeStep))
            return 0;   // error, integration must be terminated

        /* the twelve Runge-Kutta stages */
        for (unsigned int i = 0; i < n; i++)
            ytemp[i] = stateCurr[i] + timeStep * 
                a21 * k1[i];
        odeSystem.eval(timeCurr+c2*timeStep, ytemp, k2);
        for (unsigned int i = 0; i < n; i++)
            ytemp[i] = stateCurr[i] + timeStep * 
                (a31*k1[i] + a32*k2[i]);
        odeSystem.eval(timeCurr+c3*timeStep, ytemp, k3);
        for (unsigned int i = 0; i < n; i++)
            ytemp[i] = stateCurr[i] + timeStep * 
                (a41*k1[i] + a43*k3[i]);
        odeSystem.eval(timeCurr+c4*timeStep, ytemp, k4);
        for (unsigned int i = 0; i <n; i++)
            ytemp[i] = stateCurr[i] + timeStep * 
                (a51*k1[i] + a53*k3[i] + a54*k4[i]);
        odeSystem.eval(timeCurr+c5*timeStep, ytemp, k5);
        for (unsigned int i = 0; i < n; i++)
            ytemp[i] = stateCurr[i] + timeStep * 
                (a61*k1[i] + a64*k4[i] + a65*k5[i]);
        odeSystem.eval(timeCurr+c6*timeStep, ytemp, k6);
        for (unsigned int i = 0; i < n; i++)
            ytemp[i] = stateCurr[i] + timeStep * 
                (a71*k1[i] + a74*k4[i] + a75*k5[i] + a76*k6[i]);
        odeSystem.eval(timeCurr+c7*timeStep, ytemp, k7);
        for (unsigned int i = 0; i < n; i++)
            ytemp[i] = stateCurr[i] + timeStep * 
                (a81*k1[i] + a84*k4[i] + a85*k5[i] + a86*k6[i] + a87*k7[i]);
        odeSystem.eval(timeCurr+c8*timeStep, ytemp, k8);
        for (unsigned int i = 0; i <n; i++)
            ytemp[i] = stateCurr[i] + timeStep * 
                (a91*k1[i] + a94*k4[i] + a95*k5[i] + a96*k6[i] + a97*k7[i] + a98*k8[i]);
        odeSystem.eval(timeCurr+c9*timeStep, ytemp, k9);
        for (unsigned int i = 0; i < n; i++)
            ytemp[i] = stateCurr[i] + timeStep * 
                (a101*k1[i] + a104*k4[i] + a105*k5[i] + a106*k6[i] + a107*k7[i] + 
                 a108*k8[i] + a109*k9[i]);
        odeSystem.eval(timeCurr+c10*timeStep, ytemp, k10);
        for (unsigned int i = 0; i < n; i++)
            ytemp[i] = stateCurr[i] + timeStep * 
                (a111*k1[i] + a114*k4[i] + a115*k5[i] + a116*k6[i] + a117*k7[i] + 
                 a118*k8[i] + a119*k9[i] + a1110*k10[i]);
        odeSystem.eval(timeCurr+c11*timeStep, ytemp, k2);
        for (unsigned int i = 0; i < n; i++)
            ytemp[i] = stateCurr[i] + timeStep * 
                (a121*k1[i] + a124*k4[i] + a125*k5[i] + a126*k6[i] + a127*k7[i] + 
                 a128*k8[i] + a129*k9[i] + a1210*k10[i] + a1211*k2[i]);
        odeSystem.eval(timeCurr+timeStep, ytemp, k3);
        for (unsigned int i = 0; i < n; i++) {
            k4[i] = b1*k1[i] + b6*k6[i] + b7*k7[i] + b8*k8[i] + b9*k9[i] +
                    b10*k10[i] + b11*k2[i] + b12*k3[i];
            k5[i] = stateCurr[i] + timeStep * k4[i];
        }
     
        /* error estimation */
        double err = 0.0, err2 = 0.0;
        for (unsigned int i = 0; i < n; i++) {
            double sk = accAbs + accRel * fmax(fabs(stateCurr[i]), fabs(k5[i]));
            if(sk==0) continue;
            double erri = k4[i] - bhh1*k1[i] - bhh2*k9[i] - bhh3*k3[i];
            double sqre = erri / sk;
            err2 += sqre*sqre;
            erri = er1*k1[i] + er6*k6[i] + er7*k7[i] + er8*k8[i] + er9*k9[i] +
                   er10 * k10[i] + er11*k2[i] + er12*k3[i];
            sqre = erri / sk;
            err += sqre*sqre;
        }
        double deno = err + 0.01 * err2;
        if (deno <= 0.0)
            deno = 1.0;
        err *= timeStep / sqrt(deno*n);

        /* computation of hnew */
        double fac = pow(err, 1./8);
        /* we require fac1 <= hnew/h <= fac2 */
        fac = fmax(1.0/fac2, fmin(1.0/fac1, fac/safe));

        if (err <= 1.0) { // step accepted
            // make the full step, finally
            odeSystem.eval(timeCurr+timeStep, k5, k4);
            // preparation for dense output
            for(unsigned int i = 0; i < n; i++) {
                rcont1[i] = stateCurr[i];
                double ydiff = k5[i] - stateCurr[i];
                rcont2[i] = ydiff;
                double bspl = timeStep * k1[i] - ydiff;
                rcont3[i] = bspl;
                rcont4[i] = ydiff - timeStep*k4[i] - bspl;
                rcont5[i] = timeStep * (d41*k1[i] + d46*k6[i] + d47*k7[i] + d48*k8[i] +
                          d49*k9[i] + d410*k10[i] +d411*k2[i] +d412*k3[i] +d413*k4[i]);
                rcont6[i] = timeStep * (d51*k1[i] + d56*k6[i] + d57*k7[i] + d58*k8[i] +
                          d59*k9[i] + d510*k10[i] +d511*k2[i] +d512*k3[i] +d513*k4[i]);
                rcont7[i] = timeStep * (d61*k1[i] + d66*k6[i] + d67*k7[i] + d68*k8[i] +
                          d69*k9[i] + d610*k10[i] +d611*k2[i] +d612*k3[i] +d613*k4[i]);
                rcont8[i] = timeStep * (d71*k1[i] + d76*k6[i] + d77*k7[i] + d78*k8[i] +
                          d79*k9[i] + d710*k10[i] +d711*k2[i] +d712*k3[i] +d713*k4[i]);
            }
            k1 = k4;
            statePrev = stateCurr;
            stateCurr = k5;
            timePrev  = timeCurr;
            timeCurr += timeStep;
            timeStep /= fac;
            return timeCurr-timePrev;
        }

        // otherwise step rejected, make it smaller
        timeStep /= fmin(1.0/fac1, fac/safe);
    } while(1);
}

// dense output function
void OdeSolverDOP853::getSol(double t, double x[]) const
{
    const unsigned int n = stateCurr.size();
    if(t<timePrev || t>timeCurr) {
        for(unsigned int i=0; i<n; i++)
            x[i] = NAN;
        return;
    }
    if(t==timeCurr) {
        for(unsigned int i=0; i<n; i++)
            x[i] = stateCurr[i];
        return;
    }
    if(t==timePrev) {
        for(unsigned int i=0; i<n; i++)
            x[i] = rcont1[i];
        return;
    }
    double s = (t - timePrev) / (timeCurr-timePrev);
    double s1= 1.0 - s;
    for(unsigned int i=0; i<n; i++)
        x[i]   =  rcont1[i] + s * (rcont2[i] + s1 * (rcont3[i] + s * (rcont4[i] +
            s1 * (rcont5[i] + s * (rcont6[i] + s1 * (rcont7[i] + s *  rcont8[i]))))));
}

#ifdef HAVE_ODEINT
// Fancy C++ ODE integrator from boost

/// \cond INTERNAL_DOCS
// some machinery to be able to use member functions in templated class
template< class Obj>
class ode_wrapper
{
    Obj* m_obj;
public:
    ode_wrapper( Obj* obj ) : m_obj( obj ) { }

    template< class State , class Deriv , class Time >
    void operator()( const State &x , Deriv &dxdt , Time t )
    {
        m_obj->eval( t, x , &dxdt );
    }
};

template< class Obj >
ode_wrapper< Obj > make_ode_wrapper( Obj* obj )
{
    return ode_wrapper< Obj >( obj );
}

template< class Obj, class TempStorage >
class ode_wrapper_sympl
{
    Obj* obj;
    TempStorage* tmpx, *tmpdx;
public:
    ode_wrapper_sympl( Obj* _obj, TempStorage* _tmpx, TempStorage* _tmpdx) : 
      obj(_obj), tmpx(_tmpx), tmpdx(_tmpdx) { }

    template< class State , class Deriv >
    void operator()( const State &q , Deriv &dpdt )
    {
        size_t numVars2=tmpx->size()/2;
        for(size_t i=0; i<numVars2; i++)
            tmpx->at(i)=q[i];
        obj->odeFnc( 0, *tmpx , tmpdx );
        for(size_t i=0; i<numVars2; i++)
            dpdt[i]=tmpdx->at(i+numVars2);
    }
};

template< class Obj, class TempStorage >
ode_wrapper_sympl< Obj, TempStorage > make_ode_wrapper_sympl( Obj* obj, 
    TempStorage* tempStorageX, TempStorage* tempStorageDXDT)
{
    return ode_wrapper_sympl< Obj, TempStorage >(obj, tempStorageX, tempStorageDXDT);
}
/// \endcond

// Adaptive, dense-output 5th order Runge-Kutta (Dormand-Prince)
template<>
COdeIntegratorOdeint<StepperDP5>::COdeIntegratorOdeint(
    COdeSystem* _odeSystem, double _accAbs, double _accRel) :
  CBasicOdeIntegrator(_odeSystem),
  stepper(boost::numeric::odeint::controlled_runge_kutta< 
          boost::numeric::odeint::runge_kutta_dopri5< OdeStateType >, my_error_checker >
          (my_error_checker(_odeSystem, _accAbs, _accRel),
          boost::numeric::odeint::runge_kutta_dopri5< OdeStateType >() ) ),
  isStdHamiltonian(odeSystem->isStdHamiltonian())
{}

template<> 
const char* COdeIntegratorOdeint<StepperDP5>::myName() { return "DormandPrince5"; };

// Adaptive 5th order Runge-Kutta (Cash-Karp)
template<>
COdeIntegratorOdeint<StepperCK5>::COdeIntegratorOdeint(
    COdeSystem* _odeSystem, double _accAbs, double _accRel) :
  CBasicOdeIntegrator(_odeSystem),
  stepper(my_error_checker(_odeSystem, _accAbs, _accRel),
          boost::numeric::odeint::runge_kutta_cash_karp54< OdeStateType >() ),
  isStdHamiltonian(odeSystem->isStdHamiltonian())
{}

template<> 
const char* COdeIntegratorOdeint<StepperCK5>::myName() { return "CashKarp5"; };

// Fixed-step 4th order Runge-Kutta
template<>
COdeIntegratorOdeint<StepperRK4>::COdeIntegratorOdeint(
    COdeSystem* _odeSystem, double _timeStep, double /*_accRel*/) :
  CBasicOdeIntegrator(_odeSystem), 
  stepper(), 
  timeStep(_timeStep),
  isStdHamiltonian(odeSystem->isStdHamiltonian())
{}

template<> 
const char* COdeIntegratorOdeint<StepperRK4>::myName() { return "RungeKutta4"; };

// Bulirsch-Stoer (adaptive with dense output)
template<>
COdeIntegratorOdeint<StepperBS>::COdeIntegratorOdeint(
    COdeSystem* _odeSystem, double _accAbs, double _accRel) :
  CBasicOdeIntegrator(_odeSystem),
  stepper(_accAbs, _accRel), 
  isStdHamiltonian(odeSystem->isStdHamiltonian())
{}

template<> 
const char* COdeIntegratorOdeint<StepperBS>::myName() { return "BulirschStoer"; };

// Symplectic 4-th order Runge-Kutta (fixed-step)
template<>
COdeIntegratorOdeint<StepperSympl4>::COdeIntegratorOdeint(
    COdeSystem* _odeSystem, double _accAbs, double /*_accRel*/) :
  CBasicOdeIntegrator(_odeSystem), 
  stepper(), 
  timeStep(_accAbs),
  isStdHamiltonian(odeSystem->isStdHamiltonian())
{}

template<> 
const char* COdeIntegratorOdeint<StepperSympl4>::myName() { return "SymplecticRK4"; };

// generic integration function
template< class Stepper >
void COdeIntegratorOdeint<Stepper>::integrateToTime(double timeEnd)
{
    odeSystem->initState(&stateCurr);
    timeIntermediate=timeCurr;
    stateIntermediate=stateCurr;
    // do actual work differently for dense- and non-dense-output steppers
    integrateToTimeImpl(timeEnd, stepper_category());
}

// integration for fixed-timestep symplectic steppers
template<>
void COdeIntegratorOdeint<StepperSympl4>::integrateToTime(double timeEnd)
{
    if(!odeSystem->isStdHamiltonian()) return;   // cannot deal with systems which are not separable Hamiltonian in standard form
    odeSystem->initState(&stateCurr);
    size_t numVars=stateCurr.size();
    OdeStateType state_q(numVars/2), state_p(numVars/2);
    OdeStateType tmp1(numVars), tmp2(numVars);  // reserve space for intermediate computations
    size_t nstep = 0;
    bool needToTerminate=false;
    double tstep=timeStep;
    COdeSystem::STEPRESULT result = COdeSystem::SR_REINIT;
    while(!needToTerminate)
    {
        timePrev=timeCurr;
        statePrev=stateCurr;
        if(timeCurr<timeEnd && timeCurr+tstep > timeEnd) 
        {   // ensure that last step ends exactly at t_end
            tstep=timeEnd-timeCurr;
        } else
            tstep=timeStep;

        if(result == COdeSystem::SR_REINIT) {   // split combined phase space into coordinates and momenta
            state_q.assign(stateCurr.begin(), stateCurr.begin()+numVars/2);
            state_p.assign(stateCurr.begin()+numVars/2, stateCurr.end());
        }
        stepper.do_step( 
                make_ode_wrapper_sympl(odeSystem, &tmp1, &tmp2), 
                state_q, state_p, timeCurr , tstep );
        nstep++;
        timeCurr+=tstep;
        // map symplectic vars back onto the combined phase space
        stateCurr.assign(state_q.begin(), state_q.end());
        stateCurr.insert(stateCurr.end(), state_p.begin(), state_p.end());
        result = odeSystem->processStep(timePrev, timeCurr, &stateCurr);
        if(result == COdeSystem::SR_TERMINATE) 
            needToTerminate=true;
        if(!gsl_finite(timeCurr) || fabs(timeCurr-timePrev) <= fabs(timeCurr) * GSL_DBL_EPSILON*10) needToTerminate=true;   // timestep too small
        if(nstep>NUMSTEP_MAX) needToTerminate=true;  // too many steps or stepsize adjustments
    }
}

// integration for fixed-timestep steppers
template< class Stepper >
void COdeIntegratorOdeint<Stepper>::integrateToTimeImpl(double timeEnd, 
    boost::numeric::odeint::stepper_tag)
{
    size_t nstep = 0;
    bool needToTerminate=false;
    double tstep=timeStep;
    while(!needToTerminate)
    {
        timePrev=timeCurr;
        statePrev=stateCurr;
        if(timeCurr<timeEnd && timeCurr+tstep > timeEnd) 
        {   // ensure that last step ends exactly at t_end
            tstep=timeEnd-timeCurr;
        } else
            tstep=timeStep;

        stepper.do_step( 
                make_ode_wrapper(odeSystem ), 
                stateCurr, timeCurr , tstep );
        nstep++;
        timeCurr+=tstep;
        COdeSystem::STEPRESULT result = odeSystem->processStep(timePrev, timeCurr, &stateCurr);
        if(result == COdeSystem::SR_TERMINATE) 
            needToTerminate=true;
        if(!gsl_finite(timeCurr) || fabs(timeCurr-timePrev) <= fabs(timeCurr) * GSL_DBL_EPSILON*10) needToTerminate=true;   // timestep too small
        if(nstep>NUMSTEP_MAX) needToTerminate=true;  // too many steps or stepsize adjustments
    }
}

// integration for adaptive-timestep steppers
template< class Stepper >
void COdeIntegratorOdeint<Stepper>::integrateToTimeImpl(double timeEnd, 
    boost::numeric::odeint::controlled_stepper_tag)
{
    double tstep=std::max<double>(timeEnd*1e-4, 1e-4);
    size_t nstep = 0;
    bool needToTerminate=false;
    while(!needToTerminate)
    {
        timePrev=timeCurr;
        statePrev=stateCurr;
        if(timeCurr+tstep > timeEnd) 
        {   // ensure that last step ends exactly at t_end
            tstep=timeEnd-timeCurr;
        }

        const size_t max_attempts=100;
        size_t trials = 0;
        boost::numeric::odeint::controlled_step_result res;
        do {
            res = stepper.try_step( 
                make_ode_wrapper(odeSystem ), 
                stateCurr, timeCurr , tstep );
            trials++;
        }
        while( ( res == boost::numeric::odeint::fail ) && ( trials < max_attempts ) );
        nstep++;
        COdeSystem::STEPRESULT result = odeSystem->processStep(timePrev, timeCurr, &stateCurr);
        if(result == COdeSystem::SR_TERMINATE) 
            needToTerminate=true;
        if(!gsl_finite(timeCurr) || fabs(timeCurr-timePrev) <= fabs(timeCurr) * GSL_DBL_EPSILON*10) needToTerminate=true;   // timestep too small
        if(nstep>NUMSTEP_MAX || trials>=max_attempts) needToTerminate=true;  // too many steps or stepsize adjustments
    }
}

// integration for dense-output adaptive timestep
template< class Stepper >
void COdeIntegratorOdeint<Stepper>::integrateToTimeImpl(double timeEnd, 
    boost::numeric::odeint::dense_output_stepper_tag)
{
    double tstep=std::max<double>(timeEnd*1e-4, 1e-4);
    size_t nstep = 0;
    stepper.initialize(stateCurr, 0, tstep);
    bool needToTerminate=false;
    //double t_add=0;
    while(!needToTerminate)
    {
        timePrev=timeCurr;
        if(timeCurr+stepper.current_time_step() > timeEnd) 
        {   // ensure that last step ends exactly at t_end
            stepper.initialize( stepper.current_state(), timeCurr, timeEnd-timeCurr );
        }
        paird timeInterval = stepper.do_step( make_ode_wrapper(odeSystem ) );
        nstep++;
        timeCurr=stepper.current_time();
        stateCurr=stepper.current_state();
        COdeSystem::STEPRESULT result = odeSystem->processStep(timePrev, timeCurr, &stateCurr);
        if(result == COdeSystem::SR_TERMINATE) 
            needToTerminate=true;
        if(result == COdeSystem::SR_REINIT)
            stepper.initialize( stateCurr, timeCurr, stepper.current_time_step() );
        if(!gsl_finite(timeCurr) || fabs(timeCurr-timePrev) <= fabs(timeCurr) * GSL_DBL_EPSILON*10) needToTerminate=true;   // timestep too small
        if(nstep>NUMSTEP_MAX) needToTerminate=true;  // too many steps
    }
}

// generic interpolation routine (calls the appropriate implementation based on whether dense output is available or not)
template< class Stepper >
double COdeIntegratorOdeint<Stepper>::getInterpolatedSolution(unsigned int c, double t) const
{
    if(t<timePrev || t>timeCurr || c>=stateCurr.size()) return gsl_nan();
    if(t==timeCurr) return stateCurr[c];
    return getInterpolatedSolutionImpl(c, t, stepper_category());
}

// linear interpolation
template< class Stepper >
double COdeIntegratorOdeint<Stepper>::getInterpolatedSolutionGeneric(unsigned int c, double t) const 
{
    return (statePrev[c] * (timeCurr-t) + stateCurr[c] * (t-timePrev)) / (timeCurr-timePrev);
}

// generic 3(2) order interpolation in coordinate(velocity) for Hamiltonian systems in the standard form, 
// using only the phase-space coordinates at the ends of interval
template< class Stepper >
double COdeIntegratorOdeint<Stepper>::getInterpolatedSolutionStdHamiltonian(unsigned int c, double t) const 
{
    unsigned int dim=stateCurr.size()/2;  // dimension of either coordinate or momentum vector
    double x_old=statePrev[c%dim];
    double x_new=stateCurr[c%dim];
    double v_old=statePrev[c%dim+dim];
    double v_new=stateCurr[c%dim+dim];
    double timestep=timeCurr-timePrev;
    double tau=(t-timePrev)/timestep;
    // compute acceleration and its derivative for the given coordinate
    double accel2 = 3*(x_new-x_old)-timestep*(2*v_old+v_new);
    double acder6 = 2*(x_old-x_new)+timestep*(v_old+v_new);
    // do 3rd order interpolation for coordinate or 2nd order for velocity
    if(c<dim)
        return x_old + tau*(v_old*timestep + tau*(accel2 + tau*acder6));
    else
        return v_old + tau/timestep*(2*accel2 + 3*tau*acder6);
}

// high-order interpolation for dense output methods
template< class Stepper >
double COdeIntegratorOdeint<Stepper>::getInterpolatedSolutionImpl(unsigned int c, double t, 
    boost::numeric::odeint::dense_output_stepper_tag) const
{
    if(timeIntermediate!=t) { // perform interpolation of the entire system to a given time
        stepper.calc_state(t, stateIntermediate);
        timeIntermediate=t;   // store (cache) state for subsequent calls
    }
    return stateIntermediate[c];
}
#endif

/* ----- IAS15 ----- */
#if 0
void COdeIntegratorIAS15::integrateToTime(double timeEnd)
{
    if(!odeSystem->isStdHamiltonian()) {
        my_error(FUNCNAME, "IAS15 can only deal with standard Hamiltonian variables (coordinate+velocity)");
        return;
    }
    odeSystem->initState(&stateCurr);
    statePrev=stateCurr;
    deriv.assign(stateCurr.size(),0);
    int N3 = stateCurr.size()/2;
    for (int l=0;l<7;++l) {
        g[l].assign(N3, 0);
        b[l].assign(N3, 0);
        e[l].assign(N3, 0);
        br[l].assign(N3,0);
        er[l].assign(N3,0);
    }
    at.resize(N3);
    a0.resize(N3);
    compsum.assign(stateCurr.size(), 0);
    nstep=naccpt=nrejct=nfcn=0;
    dt_last_success=dt=std::max<double>(timeEnd*1e-4, 1e-4); // maybe assign initial timestep in a more fancy way?
    while(timeCurr<timeEnd) {
        nstep++;
        if(integrator_ias15_step()) {  // step successful
            naccpt++;
            COdeSystem::STEPRESULT result =   // ensure that the timestep function will be called only if rounded-off time interval is positive
                timeCurr>timePrev ? odeSystem->processStep(timePrev, timeCurr, &stateCurr) : COdeSystem::SR_CONTINUE;
            if(result == COdeSystem::SR_TERMINATE)
                return;
            if(timeCurr==timePrev) { // timestep is too small, integration does not advance
#ifdef DEBUGPRINT
                my_message(FUNCNAME, "Timestep is too small - integration stopped at t="+convertToString(timeCurr));
#endif
                return;
            }
            if(timeCurr<timeEnd && timeCurr+dt>timeEnd)
                dt=timeEnd-timeCurr; // end exactly at desired time
        } else {
            nrejct++;  // will repeat with a smaller timestep
        }
    }
}

int COdeIntegratorIAS15::integrator_ias15_step() 
{
    // Gauss Radau spacings
    static const double h[8]  = { 0.0, 0.0562625605369221464656521910, 0.1802406917368923649875799428, 
        0.3526247171131696373739077702, 0.5471536263305553830014485577, 0.7342101772154105410531523211, 
        0.8853209468390957680903597629, 0.9775206135612875018911745004}; 
    // Other constants
    static const double r[28] = {0.0562625605369221464656522, 0.1802406917368923649875799, 
        0.1239781311999702185219278, 0.3526247171131696373739078, 0.2963621565762474909082556, 
        0.1723840253762772723863278, 0.5471536263305553830014486, 0.4908910657936332365357964, 
        0.3669129345936630180138686, 0.1945289092173857456275408, 0.7342101772154105410531523, 
        0.6779476166784883945875001, 0.5539694854785181760655724, 0.3815854601022409036792446, 
        0.1870565508848551580517038, 0.8853209468390957680903598, 0.8290583863021736216247076, 
        0.7050802551022034031027798, 0.5326962297259261307164520, 0.3381673205085403850889112, 
        0.1511107696236852270372074, 0.9775206135612875018911745, 0.9212580530243653554255223, 
        0.7972799218243951369035946, 0.6248958964481178645172667, 0.4303669872307321188897259, 
        0.2433104363458769608380222, 0.0921996667221917338008147};
    static const double c[21] = {-0.0562625605369221464656522, 0.0101408028300636299864818, 
        -0.2365032522738145114532321, -0.0035758977292516175949345, 0.0935376952594620658957485, 
        -0.5891279693869841488271399, 0.0019565654099472210769006, -0.0547553868890686864408084, 
        0.4158812000823068616886219, -1.1362815957175395318285885, -0.0014365302363708915610919, 
        0.0421585277212687082291130, -0.3600995965020568162530901, 1.2501507118406910366792415, 
        -1.8704917729329500728817408, 0.0012717903090268677658020, -0.0387603579159067708505249, 
        0.3609622434528459872559689, -1.4668842084004269779203515, 2.9061362593084293206895457, 
        -2.7558127197720458409721005};
    static const double d[21] = {0.0562625605369221464656522, 0.0031654757181708292499905, 
        0.2365032522738145114532321, 0.0001780977692217433881125, 0.0457929855060279188954539, 
        0.5891279693869841488271399, 0.0000100202365223291272096, 0.0084318571535257015445000, 
        0.2535340690545692665214616, 1.1362815957175395318285885, 0.0000005637641639318207610, 
        0.0015297840025004658189490, 0.0978342365324440053653648, 0.8752546646840910912297246, 
        1.8704917729329500728817408, 0.0000000317188154017613665, 0.0002762930909826476593130, 
        0.0360285539837364596003871, 0.5767330002770787313544596, 2.2485887607691598182153473, 
        2.7558127197720458409721005};
    const double safety_factor = 0.5;    // Maximum increase/deacrease of consecutve timesteps. (rebound orig.=0.25)
    const double max_predictor_corrector_error = 1e-8;  // convergence criterion (rebound orig.=1e-16)
    const bool integrator_epsilon_global=true;  // left in place from rebound
    double s[9];                // Summation coefficients 

    timePrev=timeCurr;
    statePrev=stateCurr;
    int N3=stateCurr.size()/2;

    odeSystem->odeFnc(timeCurr, stateCurr, &deriv);
    nfcn++;
    for(int k=0; k<N3; k++) {
        a0[k] = deriv[k+N3];
    }

    for(int k=0;k<N3;k++) {
        g[0][k] = b[6][k]*d[15] + b[5][k]*d[10] + b[4][k]*d[6] + b[3][k]*d[3]  + b[2][k]*d[1]  + b[1][k]*d[0]  + b[0][k];
        g[1][k] = b[6][k]*d[16] + b[5][k]*d[11] + b[4][k]*d[7] + b[3][k]*d[4]  + b[2][k]*d[2]  + b[1][k];
        g[2][k] = b[6][k]*d[17] + b[5][k]*d[12] + b[4][k]*d[8] + b[3][k]*d[5]  + b[2][k];
        g[3][k] = b[6][k]*d[18] + b[5][k]*d[13] + b[4][k]*d[9] + b[3][k];
        g[4][k] = b[6][k]*d[19] + b[5][k]*d[14] + b[4][k];
        g[5][k] = b[6][k]*d[20] + b[5][k];
        g[6][k] = b[6][k];
    }

    double predictor_corrector_error = 1e300;
    double predictor_corrector_error_last = 2;
    int iterations = 0;
    // Predictor corrector loop
    // Stops if one of the following conditions is satisfied: 
    //   1) predictor_corrector_error better than the threshold 
    //   2) predictor_corrector_error starts to oscillate
    //   3) more than 12 iterations
    while(1) {
        if(predictor_corrector_error<max_predictor_corrector_error) {
            break;
        }
        if(iterations > 2 && predictor_corrector_error_last <= predictor_corrector_error) {
            break;
        }
        if(iterations>=12){
            /*integrator_iterations_max_exceeded++;
            const int integrator_iterations_warning = 10;
            if (integrator_iterations_max_exceeded==integrator_iterations_warning ){
                my_message(FUNCNAME, "At least "+convertToString(integrator_iterations_warning)+
                    " predictor corrector loops in integrator IAS15 did not converge. "
                    "This is typically an indication of the timestep being too large.");
            }*/
            break;   // Quit predictor corrector loop, will result in abandoning this step and repeating with a smaller timestep
        }
        predictor_corrector_error_last = predictor_corrector_error;
        predictor_corrector_error = 0;
        iterations++;

        for(int n=1;n<8;n++) {  // Loop over interval using Gauss-Radau spacings

            s[0] = dt * h[n];
            s[1] = s[0] * s[0] / 2.;
            s[2] = s[1] * h[n] / 3.;
            s[3] = s[2] * h[n] / 2.;
            s[4] = 3. * s[3] * h[n] / 5.;
            s[5] = 2. * s[4] * h[n] / 3.;
            s[6] = 5. * s[5] * h[n] / 7.;
            s[7] = 3. * s[6] * h[n] / 4.;
            s[8] = 7. * s[7] * h[n] / 9.;

            // Prepare particles arrays for force calculation
            for(int k=0;k<N3;k++) {  // Predict positions at interval n using b values
                double xk = compsum[k] + (s[8]*b[6][k] + s[7]*b[5][k] + s[6]*b[4][k] + s[5]*b[3][k] + 
                    s[4]*b[2][k] + s[3]*b[1][k] + s[2]*b[0][k] + s[1]*a0[k] + s[0]*statePrev[N3+k] );
                stateCurr[k] = xk + statePrev[k];
            }
 
            if(true /*problem_additional_forces && integrator_force_is_velocitydependent*/){
                s[0] = dt * h[n];
                s[1] =      s[0] * h[n] / 2.;
                s[2] = 2. * s[1] * h[n] / 3.;
                s[3] = 3. * s[2] * h[n] / 4.;
                s[4] = 4. * s[3] * h[n] / 5.;
                s[5] = 5. * s[4] * h[n] / 6.;
                s[6] = 6. * s[5] * h[n] / 7.;
                s[7] = 7. * s[6] * h[n] / 8.;

                for(int k=0;k<N3;k++) {  // Predict velocities at interval n using b values
                    double vk =  compsum[k+N3] + s[7]*b[6][k] + s[6]*b[5][k] + s[5]*b[4][k] + s[4]*b[3][k] + 
                        s[3]*b[2][k] + s[2]*b[1][k] + s[1]*b[0][k] + s[0]*a0[k];
                    stateCurr[k+N3] = vk + statePrev[N3+k];
                }
            }

            odeSystem->odeFnc(timeCurr+dt*h[n], stateCurr, &deriv);
            nfcn++;
            for(int k=0; k<N3; k++) {
                at[k] = deriv[k+N3];
            }

            switch(n) {                            // Improve b and g values
                case 1: 
                    for(int k=0;k<N3;++k) {
                        double tmp = g[0][k];
                        g[0][k]  = (at[k] - a0[k]) / r[0];
                        b[0][k] += g[0][k] - tmp;
                    } break;
                case 2: 
                    for(int k=0;k<N3;++k) {
                        double tmp = g[1][k];
                        const double gk = at[k] - a0[k];
                        g[1][k] = (gk/r[1] - g[0][k])/r[2];
                        tmp = g[1][k] - tmp;
                        b[0][k] += tmp * c[0];
                        b[1][k] += tmp;
                    } break;
                case 3: 
                    for(int k=0;k<N3;++k) {
                        double tmp = g[2][k];
                        const double gk = at[k] - a0[k];
                        g[2][k] = ((gk/r[3] - g[0][k])/r[4] - g[1][k])/r[5];
                        tmp = g[2][k] - tmp;
                        b[0][k] += tmp * c[1];
                        b[1][k] += tmp * c[2];
                        b[2][k] += tmp;
                    } break;
                case 4:
                    for(int k=0;k<N3;++k) {
                        double tmp = g[3][k];
                        const double gk = at[k] - a0[k];
                        g[3][k] = (((gk/r[6] - g[0][k])/r[7] - g[1][k])/r[8] - g[2][k])/r[9];
                        tmp = g[3][k] - tmp;
                        b[0][k] += tmp * c[3];
                        b[1][k] += tmp * c[4];
                        b[2][k] += tmp * c[5];
                        b[3][k] += tmp;
                    } break;
                case 5:
                    for(int k=0;k<N3;++k) {
                        double tmp = g[4][k];
                        const double gk = at[k] - a0[k];
                        g[4][k] = ((((gk/r[10] - g[0][k])/r[11] - g[1][k])/r[12] - g[2][k])/r[13] - g[3][k])/r[14];
                        tmp = g[4][k] - tmp;
                        b[0][k] += tmp * c[6];
                        b[1][k] += tmp * c[7];
                        b[2][k] += tmp * c[8];
                        b[3][k] += tmp * c[9];
                        b[4][k] += tmp;
                    } break;
                case 6:
                    for(int k=0;k<N3;++k) {
                        double tmp = g[5][k];
                        const double gk = at[k] - a0[k];
                        g[5][k] = (((((gk/r[15] - g[0][k])/r[16] - g[1][k])/r[17] - g[2][k])/r[18] - g[3][k])/r[19] - g[4][k])/r[20];
                        tmp = g[5][k] - tmp;
                        b[0][k] += tmp * c[10];
                        b[1][k] += tmp * c[11];
                        b[2][k] += tmp * c[12];
                        b[3][k] += tmp * c[13];
                        b[4][k] += tmp * c[14];
                        b[5][k] += tmp;
                    } break;
                case 7:
                {
                    double maxak = 0.0;
                    double maxb6ktmp = 0.0;
                    for(int k=0;k<N3;++k) {
                        double tmp = g[6][k];
                        const double gk = at[k] - a0[k];
                        g[6][k] = ((((((gk/r[21] - g[0][k])/r[22] - g[1][k])/r[23] - g[2][k])/r[24] - g[3][k])/r[25] - g[4][k])/r[26] - g[5][k])/r[27];
                        tmp = g[6][k] - tmp; 
                        b[0][k] += tmp * c[15];
                        b[1][k] += tmp * c[16];
                        b[2][k] += tmp * c[17];
                        b[3][k] += tmp * c[18];
                        b[4][k] += tmp * c[19];
                        b[5][k] += tmp * c[20];
                        b[6][k] += tmp;
 
                        // Monitor change in b[6][k] relative to at[k]. The predictor corrector scheme is converged if it is close to 0.
                        if(integrator_epsilon_global) {
                            const double ak  = fabs(at[k]);
                            if(gsl_finite(ak) && ak>maxak) {
                                maxak = ak;
                            }
                            const double b6ktmp = fabs(tmp);  // change of b6ktmp coefficient
                            if(gsl_finite(b6ktmp) && b6ktmp>maxb6ktmp) {
                                maxb6ktmp = b6ktmp;
                            }
                        } else {
                            const double ak  = at[k];
                            const double b6ktmp = tmp; 
                            const double errork = fabs(b6ktmp/ak);
                            if(gsl_finite(errork) && errork>predictor_corrector_error) {
                                predictor_corrector_error = errork;
                            }
                        }
                    } 
                    if(integrator_epsilon_global) {
                        predictor_corrector_error = maxb6ktmp/maxak;
                    }
 
                    break;
                }
            }
        }
    }
    // Find new timestep
    const double dt_done = dt;

    if(integrator_epsilon>0) {
        // Estimate error (given by last term in series expansion) 
        // There are two options:
        // integrator_epsilon_global==1  (default)
        //   First, we determine the maximum acceleration and the maximum of the last term in the series. 
        //   Then, the two are divided.
        // integrator_epsilon_global==0
        //   Here, the fractional error is calculated for each particle individually and we use the maximum of the fractional error.
        //   This might fail in cases where a particle does not experience any (physical) acceleration besides roundoff errors. 
        double integrator_error = 0.0;
        if(integrator_epsilon_global) {
            double maxak = 0.0;
            double maxb6k = 0.0;
            for(int k=0;k<N3;k++) { 
                const double ak  = fabs(at[k]);
                if (gsl_finite(ak) && ak>maxak){
                    maxak = ak;
                }
                const double b6k = fabs(b[6][k]); 
                if (gsl_finite(b6k) && b6k>maxb6k){
                    maxb6k = b6k;
                }
            }
            integrator_error = maxb6k/maxak;
        } else {
            for(int k=0;k<N3;k++) {
                const double ak  = at[k];
                const double b6k = b[6][k]; 
                const double errork = fabs(b6k/ak);
                if (gsl_finite(errork) && errork>integrator_error){
                    integrator_error = errork;
                }
            }
        }

        double dt_new;
        if(gsl_finite(integrator_error)) {
            // if error estimate is available increase by more educated guess
             dt_new = pow(integrator_epsilon/integrator_error,1./7.)*dt_done;
        } else {   // In the rare case that the error estimate doesn't give a finite number 
             // (e.g. when all forces accidentally cancel up to machine precission).
             dt_new = dt_done/safety_factor; // by default, increase timestep a little
        }

        if(fabs(dt_new/dt_done) < safety_factor) {    // New timestep is significantly smaller.
            stateCurr=statePrev;  // reset state to the one at the beginning of timestep
            dt = dt_new;
            double ratio = dt/dt_last_success;
            predict_next_step(ratio, N3, er, br);
            return 0; // Step rejected. Do again. 
        } 
        if(fabs(dt_new/dt_done) > 1.0) {    // New timestep is larger.
            if(dt_new/dt_done > 1./safety_factor) 
                dt_new = dt_done /safety_factor;    // Don't increase the timestep by too much compared to the last one.
        }
        dt = dt_new;
    }

    // Find new position and velocity values at end of the sequence
    const double dt_done2 = dt_done * dt_done;
    for(int k=0;k<N3;++k) {
        double a = statePrev[k];
        compsum[k] += (b[6][k]/72. + b[5][k]/56. + b[4][k]/42. + b[3][k]/30. + b[2][k]/20. + b[1][k]/12. + 
            b[0][k]/6. + a0[k]/2.) * dt_done2 + statePrev[N3+k] * dt_done;
        stateCurr[k] = a + compsum[k];
        compsum[k] += a - stateCurr[k]; 

        a = statePrev[k+N3]; 
        compsum[k+N3] += (b[6][k]/8. + b[5][k]/7. + b[4][k]/6. + b[3][k]/5. + b[2][k]/4. + b[1][k]/3. +
            b[0][k]/2. + a0[k]) * dt_done;
        stateCurr[N3+k] = a + compsum[k+N3];
        compsum[k+N3] += a - stateCurr[N3+k];
    }

    timeCurr += dt_done;
    dt_last_success = dt_done;
    for(int j=0; j<7; j++) {
        er[j]=e[j];  // copy entire array
        br[j]=b[j];
    }
    double ratio = dt/dt_done;
    predict_next_step(ratio, N3, e, b);
    return 1; // Success.
}

void COdeIntegratorIAS15::predict_next_step(double ratio, int N3, vectord _e[7], vectord _b[7])
{
    // Predict new B values to use at the start of the next sequence. The predicted
    // values from the last call are saved as E. The correction, BD, between the
    // actual and predicted values of B is applied in advance as a correction.
    const double q1 = ratio;
    const double q2 = q1 * q1;
    const double q3 = q1 * q2;
    const double q4 = q2 * q2;
    const double q5 = q2 * q3;
    const double q6 = q3 * q3;
    const double q7 = q3 * q4;

    for(int k=0;k<N3;++k) {
        double be0 = _b[0][k] - _e[0][k];
        double be1 = _b[1][k] - _e[1][k];
        double be2 = _b[2][k] - _e[2][k];
        double be3 = _b[3][k] - _e[3][k];
        double be4 = _b[4][k] - _e[4][k];
        double be5 = _b[5][k] - _e[5][k];
        double be6 = _b[6][k] - _e[6][k];

        e[0][k] = q1*(_b[6][k]* 7.0 + _b[5][k]* 6.0 + _b[4][k]* 5.0 + _b[3][k]* 4.0 + _b[2][k]* 3.0 + _b[1][k]*2.0 + _b[0][k]);
        e[1][k] = q2*(_b[6][k]*21.0 + _b[5][k]*15.0 + _b[4][k]*10.0 + _b[3][k]* 6.0 + _b[2][k]* 3.0 + _b[1][k]);
        e[2][k] = q3*(_b[6][k]*35.0 + _b[5][k]*20.0 + _b[4][k]*10.0 + _b[3][k]* 4.0 + _b[2][k]);
        e[3][k] = q4*(_b[6][k]*35.0 + _b[5][k]*15.0 + _b[4][k]* 5.0 + _b[3][k]);
        e[4][k] = q5*(_b[6][k]*21.0 + _b[5][k]* 6.0 + _b[4][k]);
        e[5][k] = q6*(_b[6][k]* 7.0 + _b[5][k]);
        e[6][k] = q7* _b[6][k];

        b[0][k] = e[0][k] + be0;
        b[1][k] = e[1][k] + be1;
        b[2][k] = e[2][k] + be2;
        b[3][k] = e[3][k] + be3;
        b[4][k] = e[4][k] + be4;
        b[5][k] = e[5][k] + be5;
        b[6][k] = e[6][k] + be6;
    }
}

double COdeIntegratorIAS15::getInterpolatedSolution(unsigned int c, double t) const 
{
    unsigned int dim=stateCurr.size()/2;  // dimension of either coordinate or momentum vector
    unsigned int k=c%dim;
    double timestep=timeCurr-timePrev;
    double tau=(t-timePrev)/timestep;
    double s[9]; // temp.array
    if(c<dim) {  // interpolate position
        s[0] = timestep * tau;
        s[1] = s[0] * s[0] / 2.;
        s[2] = s[1] * tau / 3.;
        s[3] = s[2] * tau / 2.;
        s[4] = 3. * s[3] * tau / 5.;
        s[5] = 2. * s[4] * tau / 3.;
        s[6] = 5. * s[5] * tau / 7.;
        s[7] = 3. * s[6] * tau / 4.;
        s[8] = 7. * s[7] * tau / 9.;
        return compsum[c] + (s[8]*br[6][k] + s[7]*br[5][k] + s[6]*br[4][k] + s[5]*br[3][k] + 
            s[4]*br[2][k] + s[3]*br[1][k] + s[2]*br[0][k] + s[1]*a0[k] + s[0]*statePrev[k+dim] ) + statePrev[k];
    } else {  // interpolate velocity
        s[0] = timestep * tau;
        s[1] =      s[0] * tau / 2.;
        s[2] = 2. * s[1] * tau / 3.;
        s[3] = 3. * s[2] * tau / 4.;
        s[4] = 4. * s[3] * tau / 5.;
        s[5] = 5. * s[4] * tau / 6.;
        s[6] = 6. * s[5] * tau / 7.;
        s[7] = 7. * s[6] * tau / 8.;
        return compsum[c] + s[7]*br[6][k] + s[6]*br[5][k] + s[5]*br[4][k] + s[4]*br[3][k] + 
            s[3]*br[2][k] + s[2]*br[1][k] + s[1]*br[0][k] + s[0]*a0[k] + statePrev[k+dim];
    }
}

/* ------ Hermite integrator ------ */

void COdeIntegratorHermite::integrateToTime(double timeEnd)
{
    if(!odeSystem->isStdHamiltonian()) {
        my_error(FUNCNAME, "Hermite can only deal with standard Hamiltonian variables (coordinate+velocity)");
        return;
    }
    odeSystem->initState(&stateCurr);
    statePrev=stateCurr;  // allocate arrays
    derivPrev=stateCurr;
    derivCurr=stateCurr;
    snapcrac =stateCurr;
    while(timeCurr<timeEnd) {
        if(timeCurr<timeEnd && timeCurr+dt>timeEnd)
            dt=timeEnd-timeCurr; // end exactly at desired time
        hermite_step();
        COdeSystem::STEPRESULT result =   // ensure that the timestep function will be called only if rounded-off time interval is positive
            timeCurr>timePrev ? odeSystem->processStep(timePrev, timeCurr, &stateCurr) : COdeSystem::SR_CONTINUE;
        if(result == COdeSystem::SR_TERMINATE)
            return;
        if(timeCurr==timePrev)  // timestep is too small, integration does not advance
            return;
    }
}

void COdeIntegratorHermite::hermite_step()
{
    size_t numVar = stateCurr.size()/2;
    timePrev=timeCurr;
    statePrev=stateCurr;
    // compute acceleration and jerk at the beginning of step
    odeSystem->odeFnc(timePrev, statePrev, &derivPrev);
    if(!gsl_finite(dt) || dt==0) {  // initial timestep assignment
        double sumasq=0, sumadotsq=0, sumxsq=0;
        for(size_t i=0; i<numVar; i++) {
            sumxsq += pow_2(statePrev[i]);
            sumasq += pow_2(derivPrev[i+numVar]);
            sumadotsq += pow_2(derivPrev[i]);
        }
        dt = accur * std::min<double>(sqrt(sumasq/sumadotsq), pow(sumxsq/sumasq, 0.25));
        if(dt==0 || !gsl_finite(dt)) dt=1e-10;  // arbitrary small number!
    }
    double dt_new=dt;
    int nrej=0;
    do {
        dt=dt_new;
        // predict the coordinates/velocities at the end of timestep
        for(size_t i=0; i<numVar; i++) {
            stateCurr[i] = statePrev[i] + dt*(statePrev[i+numVar] + dt/2*(derivPrev[i+numVar] + dt/3*derivPrev[i]));
            stateCurr[i+numVar] = statePrev[i+numVar] + dt*(derivPrev[i+numVar] + dt/2*derivPrev[i]);
        }
        // compute acceleration and jerk at the end of step, using predicted pos/vel
        odeSystem->odeFnc(timePrev+dt, stateCurr, &derivCurr);
        // in case of rotating potential, need another corrector step
        if(twoCorrectorSteps) {
            for(size_t i=0; i<numVar; i++) {
                stateCurr[i+numVar] = statePrev[i+numVar] + 
                    dt/2*(derivPrev[i+numVar]+derivCurr[i+numVar] + dt/6*(derivPrev[i]-derivCurr[i]));
                stateCurr[i] = statePrev[i] + dt/2*(statePrev[i+numVar]+stateCurr[i+numVar] + 
                    dt/6*(derivPrev[i+numVar]-derivCurr[i+numVar]));
            }
            odeSystem->odeFnc(timePrev+dt, stateCurr, &derivCurr);
        }
        // compute new timestep using Aarseth's magic criterion
        double sumasq=0, sumadotsq=0, sumaddotsq=0, sumadddotsq=0;
        for(size_t i=0; i<numVar; i++) {
            sumasq += pow_2(derivCurr[i+numVar]);
            sumadotsq += pow_2(derivCurr[i]);
            sumaddotsq += pow_2(2*(3*(derivCurr[i+numVar]-derivPrev[i+numVar])/dt-derivCurr[i]-2*derivPrev[i])/dt);
            sumadddotsq+= pow_2(6*(2*(derivPrev[i+numVar]-derivCurr[i+numVar])/dt+derivCurr[i]+derivPrev[i])/pow_2(dt));
        }
        dt_new = sqrt( (sqrt(sumasq*sumaddotsq)+sumadotsq)/(sqrt(sumadotsq*sumadddotsq)+sumaddotsq) * accur);
    } while(dt_new<dt*0.5 && ++nrej<3);  // repeat if new estimate of timestep was substantially less than the actually used one
    // compute corrected positions/velocities at the end of timestep
    for(size_t i=0; i<numVar; i++) {
        // first compute corrected velocity, then position - otherwise won't get 4th order accuracy
        // (see Hut&Makino, The Art of computational science, vol.2, ch.11.4)
        stateCurr[i+numVar] = statePrev[i+numVar] + dt/2*(derivPrev[i+numVar]+derivCurr[i+numVar] + 
            dt/6*(derivPrev[i]-derivCurr[i]));
        double delta_acc=derivCurr[i+numVar]-derivPrev[i+numVar];
        stateCurr[i] = statePrev[i] + dt/2*(statePrev[i+numVar]+stateCurr[i+numVar] - 
            dt/6*delta_acc);
        // also store the estimated values of snap and crackle at the beginning of timestep, to use in interpolation
        snapcrac[i] = -2*(2*derivPrev[i]+derivCurr[i])/dt + 6*delta_acc/(dt*dt);
        snapcrac[i+numVar] = 6*(derivPrev[i]+derivCurr[i] - 2*delta_acc/dt)/(dt*dt);
    }
    timeCurr += dt;
    dt=dt_new;
}

double COdeIntegratorHermite::getInterpolatedSolution(unsigned int c, double t) const
{
    size_t numVar=stateCurr.size()/2;  // dimension of either coordinate or momentum vector
    size_t i=c%numVar;
    double deltat=t-timePrev;
    if(c<numVar) {  // interpolate position
        return statePrev[i] + deltat*(statePrev[i+numVar] + 
            deltat/2*(derivPrev[i+numVar] + deltat/3*(derivPrev[i] + deltat/4*snapcrac[i])));
    } else {  // interpolate velocity
        return statePrev[i+numVar] + deltat*(derivPrev[i+numVar] + 
            deltat/2*(derivPrev[i] + deltat/3*(snapcrac[i] + deltat/4*snapcrac[i+numVar])));
    }
}

/* ------ utility functions ------- */
// return the name for a given integrator type
const char* getIntegratorNameByType(CBasicOdeIntegrator::STEPPERKIND integratorType)
{
    switch(integratorType) {
    case CBasicOdeIntegrator::SK_LEAPFROG_NB:   return COdeIntegratorLeapfrog::myName();
    case CBasicOdeIntegrator::SK_DOP853:        return COdeIntegratorDOP853::myName();
    case CBasicOdeIntegrator::SK_IAS15:         return COdeIntegratorIAS15::myName();
    case CBasicOdeIntegrator::SK_HERMITE:       return COdeIntegratorHermite::myName();
#ifdef HAVE_ODEINT
    case CBasicOdeIntegrator::SK_ODEINT_CK5:    return COdeIntegratorOdeint<StepperCK5>::myName();
    case CBasicOdeIntegrator::SK_ODEINT_DP5:    return COdeIntegratorOdeint<StepperDP5>::myName();
    case CBasicOdeIntegrator::SK_ODEINT_BS3:    return COdeIntegratorOdeint<StepperBS3>::myName();
    case CBasicOdeIntegrator::SK_ODEINT_RK4:    return COdeIntegratorOdeint<StepperRK4>::myName();
    case CBasicOdeIntegrator::SK_ODEINT_BS:     return COdeIntegratorOdeint<StepperBS>::myName();
    case CBasicOdeIntegrator::SK_ODEINT_SYMPL4: return COdeIntegratorOdeint<StepperSympl4>::myName();
#endif
    default: return "Default";
    }
};

// return integrator type associated with the text name
CBasicOdeIntegrator::STEPPERKIND getIntegratorTypeByName(const std::string& integratorName)
{
    if(strings_equal(integratorName, COdeIntegratorLeapfrog::myName())) return CBasicOdeIntegrator::SK_LEAPFROG_NB;
    if(strings_equal(integratorName, COdeIntegratorDOP853  ::myName())) return CBasicOdeIntegrator::SK_DOP853;
    if(strings_equal(integratorName, COdeIntegratorIAS15   ::myName())) return CBasicOdeIntegrator::SK_IAS15;
    if(strings_equal(integratorName, COdeIntegratorHermite ::myName())) return CBasicOdeIntegrator::SK_HERMITE;
#ifdef HAVE_ODEINT
    if(strings_equal(integratorName, COdeIntegratorOdeint<StepperCK5>   ::myName())) return CBasicOdeIntegrator::SK_ODEINT_CK5;
    if(strings_equal(integratorName, COdeIntegratorOdeint<StepperDP5>   ::myName())) return CBasicOdeIntegrator::SK_ODEINT_DP5;
    if(strings_equal(integratorName, COdeIntegratorOdeint<StepperBS3>   ::myName())) return CBasicOdeIntegrator::SK_ODEINT_BS3;
    if(strings_equal(integratorName, COdeIntegratorOdeint<StepperRK4>   ::myName())) return CBasicOdeIntegrator::SK_ODEINT_RK4;
    if(strings_equal(integratorName, COdeIntegratorOdeint<StepperBS>    ::myName())) return CBasicOdeIntegrator::SK_ODEINT_BS;
    if(strings_equal(integratorName, COdeIntegratorOdeint<StepperSympl4>::myName())) return CBasicOdeIntegrator::SK_ODEINT_SYMPL4;
#endif
    return CBasicOdeIntegrator::SK_DEFAULT;
};
#endif
};  // namespace
