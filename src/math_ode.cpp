#include "math_ode.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#ifndef _MSC_VER
#include <alloca.h>
#else
#include <malloc.h>
#endif

namespace math{

void IOdeSystem2ndOrder::eval(const double t, const double w[], double dwdt[], double* af) const
{
    const unsigned int N = size() / 2;  // dimension of position or velocity
    for(unsigned int i=0; i<N; i++)
        dwdt[i] = w[i + N];
    eval2(t, w, dwdt + N, NULL, af);
}

void IOdeSystem2ndOrderLinear::eval(const double t, const double w[], double dwdt[], double*) const
{
    // temp storage for matrices a, b
    unsigned int N = size() / 2;  // length of x and dx/dt vectors
    double *A = static_cast<double*>(alloca(N*N * sizeof(double) * 2)), *B = A + N*N;
    evalMat(t, A, B);
    for(unsigned int i=0; i<N; i++) {
        dwdt[i] = w[i+N];
        dwdt[i+N] = 0;
        for(unsigned int k=0; k<N; k++)
            dwdt[i+N] += A[i*N+k] * w[k] + B[i*N+k] * w[k+N];
    }
}

double initTimeStep(const IOdeSystem& odeSystem, const double x[], double accAbs, double accRel)
{
    const int NDIM = odeSystem.size();

    // min/max limits on the initial timestep
    static const double HMIN = 1e-12, HMAX = 1e12;

    // temporary storage allocated on the stack
    double *xt = static_cast<double*>(alloca(NDIM*4 * sizeof(double))),
    *k1 = xt + NDIM,  // dx/dt(t) at the initial point
    *k2 = k1 + NDIM,  // local temporary storage
    *k3 = k2 + NDIM;

    // compute the derivatives at the initial point
    odeSystem.eval(/*time offset*/ 0, x, k1);

    // compute the L2-norm of x and dx/dt (use the sum of all components for the crude estimate)
    double normx0 = 0, normd0 = 0;
    for(int i=0; i<NDIM; i++) {
        normx0 += pow_2(x [i]);
        normd0 += pow_2(k1[i]);
    }

    // estimate a reasonable timestep as |x| / |dx/dt|, with appropriate safety cutoffs
    double h1 = fmax(HMIN, fmin(HMAX, sqrt(normx0 / normd0) * 0.01));

    // perform an explicit Euler step with a length h1 estimated from the 1st derivative
    for(int i=0; i<NDIM; i++)
        xt[i] = x[i] + h1 * k1[i];
    odeSystem.eval(/*time offset*/ h1, xt, k2);

    // Heun's method (corrector step using xt estimated during the predictor step)
    for(int i=0; i<NDIM; i++)
        xt[i] = x[i] + 0.5 * h1 * (k1[i] + k2[i]);
    odeSystem.eval(/*time offset*/ h1, xt, k3);

    // estimate the second and third derivatives of the solution (separately for each component),
    // and use them to choose the timestep (take the shortest one among all components);
    // this is a modification of the original algorithm aimed at improving robustness:
    // it uses a combination of higher derivatives in both the numerator and the denominator,
    // which is less likely to be identically zero, even if some of them are;
    // moreover, the estimate is performed for each component separately, better adapted to
    // their possibly very different scales
    double h2 = INFINITY;
    for(int i=0; i<NDIM; i++) {
        double d0 = fmax(accAbs, fmax(fabs(x[i]), fabs(xt[i]))); // |x|
        double d1 = fmax(fabs(k1[i]), fabs(k3[i]));              // |dx/dt|
        double d2 = fabs((3 * k2[i] - k1[i] - 2 * k3[i]) / h1 ); // |d2x/dt2|
        double d3 = fabs(6 * (k3[i] - k2[i]) / (h1*h1) );        // |d3x/dt3|
        double hh = (d0 * d2 + d1 * d1) / (d1 * d3 + d2 * d2);   // Aarseth-like criterion
        h2 = fmin(h2, hh);
    }

    double h = fmax(HMIN, fmin(HMAX, pow(accRel, 1./8) * sqrt(h2)));
    return h;
}

/** --- DOP853 high-accuracy Runge-Kutta integrator --- **/

void OdeStepperDOP853::init(const double stateNew[])
{
    // copy the vector x
    for(int d=0; d<NDIM; d++)
        state[d] = stateNew[d];
    // obtain the derivatives dx/dt and the accuracy factor
    double accFac = 1;
    odeSystem.eval(0, stateNew, /*where to store the derivs*/ &state[NDIM],
        nextTimeStep==0 ? &accFac : NULL);
    if(nextTimeStep == 0)  // first call to init() at the beginning of integration, step is not known
        nextTimeStep = initTimeStep(odeSystem, stateNew, accAbs, accRel) * accFac;
}

double OdeStepperDOP853::doStep(double maxTimeStep)
{
    if(maxTimeStep == 0)
        return 0;  // something must be wrong, no progress made
    static const double
    // fractions of timestep at each RK stage
    c2   =  0.05260015195876773187856,
    c3   =  0.07890022793815159781784,
    c4   =  0.11835034190722739672676,
    c5   =  0.28164965809277260327324,
    c6   =  0.33333333333333333333333,
    c7   =  0.25000000000000000000000,
    c8   =  0.30769230769230769230769,
    c9   =  0.65128205128205128205128,
    c10  =  0.60000000000000000000000,
    c11  =  0.85714285714285714285714,
    c12  =  1.00000000000000000000000,
    // coefficients for Runge-Kutta stages
    a21  =  0.05260015195876773187856,
    a31  =  0.01972505698453789945446,
    a32  =  0.05917517095361369836338,
    a41  =  0.02958758547680684918169,
    a43  =  0.08876275643042054754507,
    a51  =  0.24136513415926668550237,
    a53  = -0.88454947932828608534486,
    a54  =  0.92483400326179200311574,
    a61  =  0.03703703703703703703704,
    a64  =  0.17082860872947387127960,
    a65  =  0.12546768756682242501669,
    a71  =  0.03710937500000000000000,
    a74  =  0.17025221101954403931498,
    a75  =  0.06021653898045596068502,
    a76  = -0.01757812500000000000000,
    a81  =  0.03709200011850479271088,
    a84  =  0.17038392571223999381021,
    a85  =  0.10726203044637328465181,
    a86  = -0.01531943774862440175279,
    a87  =  0.00827378916381402288758,
    a91  =  0.62411095871607571711443,
    a94  = -3.36089262944694129406857,
    a95  = -0.86821934684172600681819,
    a96  =  27.5920996994467083049416,
    a97  =  20.1540675504778934086187,
    a98  = -43.4898841810699588477366,
    a101 =  0.47766253643826436589043,
    a104 = -2.48811461997166764192642,
    a105 = -0.59029082683684299637145,
    a106 =  21.2300514481811942347289,
    a107 =  15.2792336328824235832597,
    a108 = -33.2882109689848629194453,
    a109 = -0.02033120170850862613582,
    a111 = -0.93714243008598732571704,
    a114 =  5.18637242884406370830024,
    a115 =  1.09143734899672957818500,
    a116 = -8.14978701074692612513997,
    a117 = -18.5200656599969598641566,
    a118 =  22.7394870993505042818970,
    a119 =  2.49360555267965238987089,
    a1110= -3.04676447189821950038237,
    a121 =  2.27331014751653820792360,
    a124 = -10.5344954667372501984067,
    a125 = -2.00087205822486249909676,
    a126 = -17.9589318631187989172766,
    a127 =  27.9488845294199600508500,
    a128 = -2.85899827713502369474066,
    a129 = -8.87285693353062954433549,
    a1210=  12.3605671757943030647266,
    a1211=  0.64339274601576353035597,
    // Runge-Kutta coefficients for the final stage
    b1   =  0.05429373411656876223805,
    b6   =  4.45031289275240888144114,
    b7   =  1.89151789931450038304282,
    b8   = -5.80120396001058478146721,
    b9   =  0.31116436695781989440892,
    b10  = -0.15216094966251607855618,
    b11  =  0.20136540080403034837478,
    b12  =  0.04471061572777259051769,
    // coefficients for error estimates (3rd and 5th order)
    bhh1 =  0.24409448818897637795276,
    bhh2 =  0.73384668828161185734136,
    bhh3 =  0.02205882352941176470588,
    er1  =  0.01312004499419488073250,
    er6  = -1.22515644637620444072057,
    er7  = -0.49575894965725019152141,
    er8  =  1.66437718245498653696153,
    er9  = -0.35032884874997368168865,
    er10 =  0.33417911871301747902973,
    er11 =  0.08192320648511571246571,
    er12 = -0.02235530786388629525884,
    // coefficients for 7th order interpolation instead of the original 8th order
    d41  = -5.40685903845352664250302,
    d46  =  367.268892700041893590281,
    d47  =  154.609958204083905482676,
    d48  = -505.920283865412564024766,
    d49  =  15.5975154819608130688200,
    d410 = -26.1936204184402805956691,
    d411 = -0.74003512364122230844721,
    d412 =  1.11776539319431476294221,
    d413 = -0.33333333333333333333333,
    d51  =  6.51987095363079615048119,
    d56  = -1066.34956011730205278592,
    d57  = -351.864047514639508625601,
    d58  =  1363.51955696662884408368,
    d59  = -112.727669432657582669864,
    d510 =  159.796191868560289612921,
    d511 = -2.13865100308788816220259,
    d512 = -3.75569172113289760348584,
    d513 =  7.00000000000000000000000,
    d61  =  10.4698004763293477204238,
    d66  = -1380.01473607038123167155,
    d67  = -531.219827862514074379012,
    d68  =  1866.98964341870892451324,
    d69  = -53.3302605020547902574560,
    d610 =  82.4147560258671369782481,
    d611 =  7.38443654502992069572676,
    d612 =  0.41729908012587751149843,
    d613 = -3.11111111111111111111111,
    d71  = -16.6338582677165354330709,
    d76  =  4516.16568914956011730205,
    d77  =  1393.85185384057776465219,
    d78  = -5687.52042419481539670071,
    d79  =  473.965563750151263163661,
    d710 = -661.810776942355889724311,
    d711 = -18.0180473354013232598119,
    d712 = 0,
    d713 = 0,
    // parameters for step size selection
    fdec = 0.333, // maximum instantaneous decrease factor
    finc = 6.0,   // maximum increase factor
    safe = 0.9;   // safety factor in timestep

    // temporary storage for intermediate Runge-Kutta steps
    const int tempSize = NDIM * 10;
    double *xt = static_cast<double*>(alloca(tempSize * sizeof(double)));
    double
    // persistent data (conserved between timesteps)
    *x  = &state[0],   // x     at the beginning of the timestep
    *k1 = x  + NDIM,   // dx/dt at the beginning of the timestep
    // temporary data (used only inside this routine)
    *k2 = xt + NDIM,
    *k3 = k2 + NDIM,
    *k4 = k3 + NDIM,
    *k5 = k4 + NDIM,
    *k6 = k5 + NDIM,
    *k7 = k6 + NDIM,
    *k8 = k7 + NDIM,
    *k9 = k8 + NDIM,
    *k10= k9 + NDIM,
    *k11= k2,  // last stages reuse the memory from earlier stages
    *k12= k3,
    *k13= k4,
    *k1b= k5;
    // use the previously estimated timestep or the requested max step, whichever is shorter
    double sign = maxTimeStep >= 0 ? +1 : -1;
    double timeStep = fmin(maxTimeStep * sign, nextTimeStep) * sign;
    // track the number of rejected attempts and the progress in error reduction
    int nbad = 0;
    double preverr = INFINITY;

    // repeat until the step is accepted
    while(1) {
        if(timeStep==0 || !isFinite(timeStep))
            return 0;   // error, integration must be terminated

        // the twelve Runge-Kutta stages
        for(int i=0; i<NDIM; i++)
            xt[i] = x[i] + timeStep *
                a21 * k1[i];
        odeSystem.eval(c2*timeStep, xt, k2);
        for(int i=0; i<NDIM; i++)
            xt[i] = x[i] + timeStep *
                (a31*k1[i] + a32*k2[i]);
        odeSystem.eval(c3*timeStep, xt, k3);
        for(int i=0; i<NDIM; i++)
            xt[i] = x[i] + timeStep *
                (a41*k1[i] + a43*k3[i]);
        odeSystem.eval(c4*timeStep, xt, k4);
        for(int i=0; i<NDIM; i++)
            xt[i] = x[i] + timeStep *
                (a51*k1[i] + a53*k3[i] + a54*k4[i]);
        odeSystem.eval(c5*timeStep, xt, k5);
        for(int i=0; i<NDIM; i++)
            xt[i] = x[i] + timeStep *
                (a61*k1[i] + a64*k4[i] + a65*k5[i]);
        odeSystem.eval(c6*timeStep, xt, k6);
        for(int i=0; i<NDIM; i++)
            xt[i] = x[i] + timeStep *
                (a71*k1[i] + a74*k4[i] + a75*k5[i] + a76*k6[i]);
        odeSystem.eval(c7*timeStep, xt, k7);
        for(int i=0; i<NDIM; i++)
            xt[i] = x[i] + timeStep *
                (a81*k1[i] + a84*k4[i] + a85*k5[i] + a86*k6[i] + a87*k7[i]);
        odeSystem.eval(c8*timeStep, xt, k8);
        for(int i=0; i<NDIM; i++)
            xt[i] = x[i] + timeStep *
                (a91*k1[i] + a94*k4[i] + a95*k5[i] + a96*k6[i] + a97*k7[i] + a98*k8[i]);
        odeSystem.eval(c9*timeStep, xt, k9);
        for(int i=0; i<NDIM; i++)
            xt[i] = x[i] + timeStep *
                (a101*k1[i] + a104*k4[i] + a105*k5[i] + a106*k6[i] + a107*k7[i] +
                 a108*k8[i] + a109*k9[i]);
        odeSystem.eval(c10*timeStep, xt, k10);
        for(int i=0; i<NDIM; i++)
            xt[i] = x[i] + timeStep *
                (a111*k1[i] + a114*k4[i] + a115 *k5 [i] + a116*k6[i] + a117*k7[i] +
                 a118*k8[i] + a119*k9[i] + a1110*k10[i]);
        odeSystem.eval(c11*timeStep, xt, k11);
        for(int i=0; i<NDIM; i++)
            xt[i] = x[i] + timeStep *
                (a121*k1[i] + a124*k4[i] + a125 *k5 [i] + a126 *k6 [i] + a127*k7[i] +
                 a128*k8[i] + a129*k9[i] + a1210*k10[i] + a1211*k11[i]);
        odeSystem.eval(c12*timeStep, xt, k12);
        for(int i=0; i<NDIM; i++) {
            k1b[i] = b1*k1 [i] + b6 *k6 [i] + b7 *k7 [i] + b8*k8[i] + b9*k9[i] +
                    b10*k10[i] + b11*k11[i] + b12*k12[i];
            xt[i] = x[i] + timeStep * k1b[i];
        }

        // compute the solution at the end of the timestep and the accuracy factor
        double accFac = 1.0;
        odeSystem.eval(timeStep, xt, k13, &accFac);

        // error estimation
        double err5 = 0.0, err3 = 0.0;
        for(int i=0; i<NDIM; i++) {
            double sk = (accAbs + accRel * fmax(fabs(x[i]), fabs(xt[i]))) * accFac;
            if(sk==0) continue;
            err3 += pow_2( (   k1b[i] - bhh1*k1 [i] - bhh2*k9 [i] - bhh3*k12[i]) / sk);
            err5 += pow_2( (er1*k1[i] + er6 *k6 [i] + er7 *k7 [i] + er8 *k8 [i]  +
                            er9*k9[i] + er10*k10[i] + er11*k11[i] + er12*k12[i]) / sk);
        }
        double den = sqrt(NDIM * (err5 + 0.01 * err3));
        double err = den==0 ? 0 : err5 * fabs(timeStep) / den;

        if(!isFinite(err))
            return 0;   // error, integration must be terminated

        // computation of nextTimeStep
        double fac = sqrt(sqrt(sqrt(err)));  // = pow(err, 1./8);

        // step accepted if the internal error estimate is small enough
        if(err <= 1) {
            // adjust the prediction for the next timestep
            nextTimeStep = fabs(timeStep) * fmin(finc, safe/fac);
            break;
        }
        else {
            if(err > 0.5*preverr) {
                // the error is supposed to improve as the timestep is reduced;
                // if that's not happening, something might be wrong
                nbad++;
                if(nbad >= 2)
                    fac = 1/fdec;  // apply maximum reduction in timestep
                if(nbad >= 4) {
                    // no improvement likely means that something is wrong with the error estimate
                    // (e.g. one of the variables is nearly zero so that the relative error is huge);
                    // just accept the current step and try to proceed further
                    nextTimeStep = fabs(timeStep);
                    break;
                }
            } else
                nbad = 0;  // reset the counter of badly failed steps which didn't reduce the error
            preverr = err;
            // if the step is rejected, make it smaller
            timeStep *= fmax(fdec, safe/fac);
        }
    }

    // preparation of interpolation coefficients for dense output
    double  // the interpolation coefficients are stored in 'state' at different offsets
    *rcont1 = k1     + NDIM,
    *rcont2 = rcont1 + NDIM,
    *rcont3 = rcont2 + NDIM,
    *rcont4 = rcont3 + NDIM,
    *rcont5 = rcont4 + NDIM,
    *rcont6 = rcont5 + NDIM,
    *rcont7 = rcont6 + NDIM,
    *rcont8 = rcont7 + NDIM;
    for(int i=0; i<NDIM; i++) {
        rcont1[i] = x[i];
        double xd = xt[i] - x[i];   // x(t+dt) - x(t)
        rcont2[i] = xd;
        double xc = timeStep * k1[i] - xd;
        rcont3[i] = xc;
        rcont4[i] = xd - timeStep*k13[i] - xc;
        rcont5[i] = timeStep * (d41 *k1 [i] + d46 *k6 [i] + d47 *k7 [i] + d48 *k8 [i] +
                    d49*k9[i] + d410*k10[i] + d411*k11[i] + d412*k12[i] + d413*k13[i]);
        rcont6[i] = timeStep * (d51 *k1 [i] + d56 *k6 [i] + d57 *k7 [i] + d58 *k8 [i] +
                    d59*k9[i] + d510*k10[i] + d511*k11[i] + d512*k12[i] + d513*k13[i]);
        rcont7[i] = timeStep * (d61 *k1 [i] + d66 *k6 [i] + d67 *k7 [i] + d68 *k8 [i] +
                    d69*k9[i] + d610*k10[i] + d611*k11[i] + d612*k12[i] + d613*k13[i]);
        rcont8[i] = timeStep * (d71 *k1 [i] + d76 *k6 [i] + d77 *k7 [i] + d78 *k8 [i] +
                    d79*k9[i] + d710*k10[i] + d711*k11[i] + d712*k12[i] + d713*k13[i]);
        // store the new values and derivatives of x at the end of the current timestep
        x [i] = xt [i];
        k1[i] = k13[i];
    }
    prevTimeStep = timeStep;
    return timeStep;
}

// dense output function
double OdeStepperDOP853::getSol(double timeOffset, unsigned int i) const
{
    if(i >= (unsigned int)NDIM)
        throw std::out_of_range("OdeStepperDOP853: element index out of range");
    if(timeOffset == prevTimeStep)
        return state[i];  // state vector at the end of the last timestep
    if(timeOffset == 0)
        return state[i+2*NDIM];  // = rcont1[i], state vector at the beginning of the last timestep
    double p = timeOffset / prevTimeStep, q = 1.0 - p;  // both p & q should be between 0 and 1
    if(!(q>=0 && p>=0))
        throw std::out_of_range("OdeStepperDOP853: requested time is outside the last completed timestep");
    // interpolation coefs rcont1..rcont8 are stored in the 'state' array at various offsets
    return   state[i+2*NDIM] + p * (state[i+3*NDIM] + q * (state[i+4*NDIM] + p * (state[i+5*NDIM] +
        q * (state[i+6*NDIM] + p * (state[i+7*NDIM] + q * (state[i+8*NDIM] + p *  state[i+9*NDIM]))))));
}


/** --- 8th order Runge-Kutta-Nystrom integrator for second-order ODEs --- **/

OdeStepperDPRKN8::OdeStepperDPRKN8(const IOdeSystem2ndOrder& _odeSystem, double _accRel) :
    odeSystem(_odeSystem),
    NDIM(odeSystem.size()),
    accRel(10 * pow(_accRel, 0.9)),  // empirical approximate match to dop853's accuracy parameter
    nextTimeStep(0),
    state(NDIM * 3)
{}

void OdeStepperDPRKN8::init(const double stateNew[])
{
    for(int d=0; d<NDIM; d++)  // copy the vector x
        state[d] = stateNew[d];
    // obtain the derivative d2x/dt2 and the accuracy factor
    double accFac = 1;
    odeSystem.eval2(0, stateNew, /*output*/ &state[NDIM], NULL,
        nextTimeStep==0 ? &accFac : NULL);
    if(nextTimeStep == 0)  // initial timestep assignment
        nextTimeStep = initTimeStep(odeSystem, stateNew, 0, accRel) * accFac;
    qold = 0.0001;
}

double OdeStepperDPRKN8::doStep(double maxTimeStep)
{
    if(maxTimeStep == 0)
        return 0;
    int numVar = NDIM/2;
    static const double
    c1  = 1.0 / 20,
    c2  = 1.0 / 10,
    c3  = 3.0 / 10,
    c4  = 1.0 / 2,
    c5  = 7.0 / 10,
    c6  = 9.0 / 10,
    c7  = 1.0,
    c8  = 1.0,
    a21 = 1.0 / 800,
    a31 = 1.0 / 600,
    a32 = 1.0 / 300,
    a41 = 9.0 / 200,
    a42 =-9.0 / 100,
    a43 = 9.0 / 100,
    a51 =-66701.0 / 197352,
    a52 = 28325.0 / 32892,
    a53 =-2665.0  / 5482,
    a54 = 2170.0  / 24669,
    a61 = 227015747.0 / 304251000,
    a62 =-54897451.0 / 30425100,
    a63 = 12942349.0 / 10141700,
    a64 =-9499.0 / 304251,
    a65 = 539.0 / 9250,
    a71 =-1131891597.0 / 901789000,
    a72 = 41964921.0 / 12882700,
    a73 =-6663147.0 / 3220675,
    a74 = 270954.0 / 644135,
    a75 =-108.0 / 5875,
    a76 = 114.0 / 1645,
    a81 = 13836959.0 / 3667458,
    a82 =-17731450.0 / 1833729,
    a83 = 1063919505.0 / 156478208,
    a84 =-33213845.0 / 39119552,
    a85 = 13335.0 / 28544,
    a86 =-705.0  / 14272,
    a87 = 1645.0 / 57088,
    a91 = 223.0  / 7938,
    a93 = 1175.0 / 8064,
    a94 = 925.0  / 6048,
    a95 = 41.0   / 448,
    a96 = 925.0  / 14112,
    a97 = 1175.0 / 72576,
    b1  = 223.0  / 7938,
    b3  = 1175.0 / 8064,
    b4  = 925.0  / 6048,
    b5  = 41.0   / 448,
    b6  = 925.0  / 14112,
    b7  = 1175.0 / 72576,
    bp1 = 223.0  / 7938,
    bp3 = 5875.0 / 36288,
    bp4 = 4625.0 / 21168,
    bp5 = 41.0   / 224,
    bp6 = 4625.0 / 21168,
    bp7 = 5875.0 / 36288,
    bp8 = 223.0  / 7938,
    btilde1  = 223.0  / 7938  - 7987313.0  / 109941300,
    btilde3  = 1175.0 / 8064  - 1610737.0  / 44674560,
    btilde4  = 925.0  / 6048  - 10023263.0 / 33505920,
    btilde5  = 41.0   / 448   + 497221.0   / 12409600,
    btilde6  = 925.0  / 14112 - 10023263.0 / 78180480,
    btilde7  = 1175.0 / 72576 - 1610737.0  / 402071040,
    bptilde1 = 223.0  / 7938  - 7987313.0  / 109941300,
    bptilde3 = 5875.0 / 36288 - 1610737.0  / 40207104,
    bptilde4 = 4625.0 / 21168 - 10023263.0 / 23454144,
    bptilde5 = 41.0   / 224   + 497221.0   / 6204800,
    bptilde6 = 4625.0 / 21168 - 10023263.0 / 23454144,
    bptilde7 = 5875.0 / 36288 - 1610737.0  / 40207104,
    bptilde8 = 223.0  / 7938  + 4251941.0  / 54970650,
    bptilde9 = -3.0   /  20;

    // temporary storage for intermediate Runge-Kutta steps
    const int tempSize = numVar * 13;
    double *xt = static_cast<double*>(alloca(tempSize * sizeof(double)));
    double
    // persistent data (conserved between timesteps)
    *x  = &state[0],   // x       at the beginning of the timestep
    *v  = x  + numVar, // dx/dt   at the beginning of the timestep
    *k1 = v  + numVar, // d2x/dt2 at the beginning of the step
    *j1 = k1 + numVar, // d3x/dt3 at the beginning of the timestep
    // temporary data (used only inside this routine)
    *k2 = xt + numVar,
    *k3 = k2 + numVar,
    *k4 = k3 + numVar,
    *k5 = k4 + numVar,
    *k6 = k5 + numVar,
    *k7 = k6 + numVar,
    *k8 = k7 + numVar,
    *k9 = k8 + numVar,
    // final x and v at the end of the timestep
    *xn = k9 + numVar,
    *vn = xn + numVar,
    *kn = vn + numVar,
    *jn = kn + numVar;
    // use the previously estimated timestep or the requested max step, whichever is shorter
    double sign = maxTimeStep >= 0 ? +1 : -1;
    double timeStep = fmin(maxTimeStep * sign, nextTimeStep) * sign;

    // track the number of rejected attempts and the progress in error reduction
    int nbad = 0, niter = 0;
    double preverr = INFINITY;

    // repeat until the step is accepted
    while(true) {
        if(timeStep==0 || !isFinite(timeStep))
            return 0;   // error, integration must be terminated

        // nine Runge-Kutta stages
        for(int i=0; i<numVar; i++)
            xt[i] = x[i] + timeStep * (c1 * v[i] + timeStep *
                (a21 * k1[i]));
        odeSystem.eval2(c1 * timeStep, xt, k2);
        for(int i=0; i<numVar; i++)
            xt[i] = x[i] + timeStep * (c2 * v[i] + timeStep *
                (a31 * k1[i] + a32 * k2[i]));
        odeSystem.eval2(c2 * timeStep, xt, k3);
        for(int i=0; i<numVar; i++)
            xt[i] = x[i] + timeStep * (c3 * v[i] + timeStep *
                (a41 * k1[i] + a42 * k2[i] + a43 * k3[i]));
        odeSystem.eval2(c3 * timeStep, xt, k4);
        for(int i=0; i<numVar; i++)
            xt[i] = x[i] + timeStep * (c4 * v[i] + timeStep *
                (a51 * k1[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i]));
        odeSystem.eval2(c4 * timeStep, xt, k5);
        for(int i=0; i<numVar; i++)
            xt[i] = x[i] + timeStep * (c5 * v[i] + timeStep *
                (a61 * k1[i] + a62 * k2[i] + a63 * k3[i] + a64 * k4[i] + a65 * k5[i]));
        odeSystem.eval2(c5 * timeStep, xt, k6);
        for(int i=0; i<numVar; i++)
            xt[i] = x[i] + timeStep * (c6 * v[i] + timeStep *
                (a71 * k1[i] + a72 * k2[i] + a73 * k3[i] + a74 * k4[i] + a75 * k5[i] + a76 * k6[i]));
        odeSystem.eval2(c6 * timeStep, xt, k7);
        for(int i=0; i<numVar; i++)
            xt[i] = x[i] + timeStep * (c7 * v[i] + timeStep *
                (a81 * k1[i] + a82 * k2[i] + a83 * k3[i] + a84 * k4[i] + a85 * k5[i] + a86 * k6[i] +
                 a87 * k7[i]));
        odeSystem.eval2(c7 * timeStep, xt, k8);
        for(int i=0; i<numVar; i++)
            xt[i] = x[i] + timeStep * (c8 * v[i] + timeStep *  // no a92 and a98
                (a91 * k1[i] + a93 * k3[i] + a94 * k4[i] + a95 * k5[i] + a96 * k6[i] + a97 * k7[i]));
        odeSystem.eval2(c8 * timeStep, xt, k9);

        // compute variables at the end of the timestep
        for(int i=0; i<numVar; i++) {
            xn[i] = x[i] + timeStep * (v[i] + timeStep *  // no b2 and b8
                (b1 * k1[i] + b3 * k3[i] + b4 * k4[i] + b5 * k5[i] + b6 * k6[i] + b7 * k7[i]));
            vn[i] = v[i] + timeStep *  // no bp2
                (bp1* k1[i] + bp3* k3[i] + bp4* k4[i] + bp5* k5[i] + bp6* k6[i] + bp7* k7[i] + bp8* k8[i]);
        }

        // final evaluation of ODE at the end of the timestep and the computation of accuracy factor
        double accFac = 1.0;
        odeSystem.eval2(timeStep, xn, /*output*/kn, /*no 3rd deriv*/NULL, &accFac);

        // error estimation
        double err = 0.0;
        for(int i=0; i<numVar; i++) {
            double denom = (/*accAbs +*/ accRel * fmax(fabs(x[i]), fabs(xn[i]))) * accFac;
            if(denom!=0) {
                double xtilde = pow_2(timeStep) * (  // no btilde2,8,9
                    btilde1 * k1[i] + btilde3 * k3[i] + btilde4 * k4[i] + btilde5 * k5[i] +
                    btilde6 * k6[i] + btilde7 * k7[i]);
                err += pow_2(xtilde / denom);
            }
            denom = (/*accAbs +*/ accRel * fmax(fabs(v[i]), fabs(vn[i]))) * accFac;
            if(denom!=0) {
                double vtilde = timeStep * (  // no bptilde2
                    bptilde1* k1[i] + bptilde3* k3[i] + bptilde4* k4[i] + bptilde5* k5[i] +
                    bptilde6* k6[i] + bptilde7* k7[i] + bptilde8* k8[i] + bptilde9* k9[i]);
                err += pow_2(vtilde / denom);
            }
        }
        err = sqrt(err / NDIM);

        // step estimation either by a proportional-integral (PI) controller
        const double gamma = 0.9, qmax = 10.0, qmin = 0.2, beta1 = 7./80, beta2 = 4./80;
        double q1 = pow(err, beta1);
        if(err <= 1 || ++niter >= 12) {  // step accepted
            double q = accRel * accFac==INFINITY /*ignore accuracy control*/ ? 0 :
                fmin(1 / qmin, fmax(1 / qmax, q1 / pow(qold, beta2) / gamma));
            // adjust the prediction for the next timestep
            nextTimeStep = fabs(timeStep) / q;
            qold = err;
            break;
        }

        // same precautions as in dop853; more aggressive step reduction in case of persistent troubles
        if(err > 0.5*preverr) {
            // the error is supposed to improve as the timestep is reduced;
            // if that's not happening, something might be wrong
            if(++nbad >= 2)
                q1 = 1/qmin;  // apply maximum reduction in timestep
        } else
            nbad = 0;  // reset the counter of badly failed steps which didn't reduce the error

        // step rejected
        preverr = err;
        timeStep /= fmin(1 / qmin, q1 / gamma);
    }

    // preparation of interpolation coefficients for dense output
    for(int i=0; i<numVar; i++) {
        // compute coefficients of Taylor expansion d3x/dt3 .. d5x/dt5
        jn[i]             = ((
            -60 * (x[i] -       xn[i]) / timeStep +
            -24 *  v[i] -  36 * vn[i]) / timeStep +
            -3  * k1[i] +   9 * kn[i]) / timeStep;
        state[4*numVar+i] = ((
            -360 * (x[i] -       xn[i]) / timeStep +
            -168 *  v[i] - 192 * vn[i]) / timeStep +
            -24  * k1[i] +  36 * kn[i]) / pow_2(timeStep);
        state[5*numVar+i] = ((
            -720 * (x[i] -       xn[i]) / timeStep +
            -360 *  v[i] - 360 * vn[i]) / timeStep +
            -60  * k1[i] +  60 * kn[i]) / pow_3(timeStep);
        x [i] = xn[i];
        v [i] = vn[i];
        k1[i] = kn[i];
        j1[i] = jn[i];
    }
    prevTimeStep = timeStep;
    return timeStep;
}

double OdeStepperDPRKN8::getSol(double timeOffset, unsigned int ind) const
{
    if(ind >= (unsigned int)NDIM)
        throw std::out_of_range("OdeStepperDPRKN8: element index out of range");
    double deltat = timeOffset - prevTimeStep;  // expected to be between -prevTimeStep and 0
    if(deltat == 0)
        return state[ind];
    unsigned int numVar = NDIM/2;  // dimension of either coordinate or velocity vector
    int i = ind % numVar;
    if(ind < numVar) {  // interpolate position
        return                 state[i+0*numVar] +
            deltat *          (state[i+1*numVar] +
            deltat * (1./2) * (state[i+2*numVar] +
            deltat * (1./3) * (state[i+3*numVar] +
            deltat * (1./4) * (state[i+4*numVar] +
            deltat * (1./5) *  state[i+5*numVar] ))));
    } else {  // interpolate velocity
        return                 state[i+1*numVar] +
            deltat *          (state[i+2*numVar] +
            deltat * (1./2) * (state[i+3*numVar] +
            deltat * (1./3) * (state[i+4*numVar] +
            deltat * (1./4) *  state[i+5*numVar] )));
    }
}


/** --- 4th order Hermite scheme --- **/

OdeStepperHermite::OdeStepperHermite(const IOdeSystem2ndOrder& _odeSystem, double _accRel) :
    odeSystem(_odeSystem),
    NDIM(odeSystem.size()),
    accRel(1.5 * pow(_accRel, 0.2)),  // empirical approximate match to dop853's accuracy parameter
    prevTimeStep(0),
    nextTimeStep(0),
    state(NDIM * 5)
{}

void OdeStepperHermite::init(const double stateNew[])
{
    for(int d=0; d<NDIM; d++)  // copy the vector x
        state[d+NDIM*2] = stateNew[d];
    if(nextTimeStep == 0)  // initial timestep assignment
        nextTimeStep = initTimeStep(odeSystem, stateNew, 0, pow(accRel, 8));
}

double OdeStepperHermite::doStep(double maxTimeStep)
{
    if(maxTimeStep == 0)
        return 0;
    int numVar = NDIM/2;
    double
        *oldpos = &state[0],
        *oldvel = oldpos + numVar,
        *oldacc = oldvel + numVar,
        *oldjrk = oldacc + numVar,
        *newpos = oldjrk + numVar,
        *newvel = newpos + numVar,
        *newacc = newvel + numVar,
        *newjrk = newacc + numVar,
        *snap   = newjrk + numVar,
        *crackle= snap   + numVar;
    for(int i=0; i<NDIM; i++)
        state[i] = state[i+NDIM*2];
    // compute acceleration and jerk at the beginning of step
    odeSystem.eval2(0, oldpos, /*output*/ oldacc, oldjrk);
    // use the previously estimated timestep or the requested max step, whichever is shorter
    double sign = maxTimeStep >= 0 ? +1 : -1, dt;
    nextTimeStep = fmin(maxTimeStep * sign, nextTimeStep);
    int nreject = 0;
    do {
        dt = nextTimeStep * sign;
        // predict the coordinates/velocities at the end of timestep
        for(int i=0; i<numVar; i++) {
            newpos[i] = oldpos[i] + dt * (oldvel[i] + 0.5 * dt * (oldacc[i] + 1./3 * dt * oldjrk[i]));
            newvel[i] = oldvel[i] + dt * (oldacc[i] + 0.5 * dt *  oldjrk[i]);
        }
        // compute acceleration and jerk at the end of step, using predicted pos/vel
        odeSystem.eval2(dt, newpos, /*output*/ newacc, newjrk);
        double sumasq=0, sumjsq=0, sumssq=0, sumcsq=0;
        for(int i=0; i<numVar; i++) {
            // store the estimated values of snap and crackle at the beginning of timestep
            snap   [i] =-(4 * oldjrk[i] + 2 * newjrk[i]) / dt + 6 * (newacc[i] - oldacc[i]) / pow_2(dt);
            crackle[i] = 6 * (oldjrk[i] +     newjrk[i] - 2 * (newacc[i] - oldacc[i]) / dt) / pow_2(dt);
            sumasq += pow_2(newacc[i]);
            sumjsq += pow_2(newjrk[i]);
            sumssq += pow_2(snap  [i]);
            sumcsq += pow_2(crackle[i]);
        }
        // compute new timestep using Aarseth's magic criterion
        nextTimeStep = accRel * sqrt((sqrt(sumasq * sumssq) + sumjsq) / (sqrt(sumjsq * sumcsq) + sumssq));
        // repeat if new estimate of timestep was substantially less than the actually used one
    } while(nextTimeStep < 0.5 * dt * sign && ++nreject < 3);

    // compute corrected positions/velocities at the end of timestep
    for(int i=0; i<numVar; i++) {
        // first compute corrected velocity, then position - otherwise won't get 4th order accuracy
        // (see Hut&Makino, The Art of computational science, vol.2, ch.11.4)
        newvel[i] = oldvel[i] + 0.5 * dt * (oldacc[i] + newacc[i] + 1./6 * dt * (oldjrk[i] - newjrk[i]));
        newpos[i] = oldpos[i] + 0.5 * dt * (oldvel[i] + newvel[i] + 1./6 * dt * (oldacc[i] - newacc[i]));
    }

    prevTimeStep = dt;
    return dt;
}

double OdeStepperHermite::getSol(double timeOffset, unsigned int ind) const
{
    unsigned int numVar = NDIM/2;  // dimension of either coordinate or momentum vector
    int i = ind % numVar;
    if(i >= NDIM)
        throw std::out_of_range("OdeStepperHermite: element index out of range");
    double h = timeOffset / prevTimeStep;
    if(h<0 || h>1 || (prevTimeStep==0 && h!=0))
        throw std::out_of_range("OdeStepperHermite: requested time is outside the last completed timestep");
    if(ind < numVar) {  // interpolate position
        return                     state[i         ] +
            timeOffset *          (state[i+  numVar] +
            timeOffset * (1./2) * (state[i+2*numVar] +
            timeOffset * (1./3) * (state[i+3*numVar] +
            timeOffset * (1./4) * (state[i+8*numVar] +
            timeOffset * (1./5) *  state[i+9*numVar]))));
    } else {  // interpolate velocity
        return                     state[i+  numVar] +
            timeOffset *          (state[i+2*numVar] +
            timeOffset * (1./2) * (state[i+3*numVar] +
            timeOffset * (1./3) * (state[i+8*numVar] +
            timeOffset * (1./4) *  state[i+9*numVar])));
    }
}


/** --- Integrators for second-order linear ODE systems --- */

template<int NDIM>
Ode2StepperGL3<NDIM>::Ode2StepperGL3(
    const IOdeSystem2ndOrderLinear& _odeSystem, unsigned int _numVectors) :
    odeSystem(_odeSystem),
    numVectors(_numVectors),
    prevTimeStep(0),
    newstep(true),
    state(5 * NDIM * numVectors)
{
    if(numVectors <= 0)
        throw std::invalid_argument("Ode2StepperGL3: invalid number of vectors");
    if(odeSystem.size() != NDIM * 2)
        throw std::invalid_argument("Ode2StepperGL3: invalid size of the ODE system");
}

template<int NDIM>
void Ode2StepperGL3<NDIM>::init(const double stateNew[])
{
    state.assign(5*NDIM * numVectors, 0);
    for(unsigned int v=0; v<numVectors; v++)
        std::copy(stateNew + v * 2*NDIM, stateNew + (v+1) * 2*NDIM, state.begin() + v * 5*NDIM);
    newstep = true;
}

template<int NDIM>
double Ode2StepperGL3<NDIM>::getSol(double timeOffset, unsigned int i) const
{
    double T = timeOffset - prevTimeStep;
    // T is expected to be between -prevTimeStep and 0, i.e. interpolating on the previous timestep
    if(! ( (prevTimeStep>=0 && timeOffset>=0 && timeOffset<=prevTimeStep)
        || (prevTimeStep<=0 && timeOffset<=0 && timeOffset>=prevTimeStep) ) )
        throw std::out_of_range("Ode2StepperGL3: requested time is outside the last completed timestep");
    unsigned int vec = i / (2*NDIM);  // index of the requested vector
    if(vec >= numVectors)
        throw std::out_of_range("Ode2StepperGL3: element index out of range");
    i -= vec * 2*NDIM;          // index of the requested element in the given vector
    if(T==0)  // fast track: no interpolation needed, just copy the element of the state vector
        return state[vec * 5*NDIM + i];
    unsigned int d = i % NDIM;  // index of the requested dimension in either position or velocity
    const double *x = &state[vec * 5*NDIM + d], *xdot = x+NDIM,  // state vector (pos, vel)
        *p = xdot+NDIM, *q = p+NDIM, *r = q+NDIM;  // higher-order interpolation coefficients
    if(i == d)  // d-th component of x(t)
        return (*x) + T * ((*xdot) + T * ((*p) + T * ((*q) + T * (*r))));
    else        // d-th component of dx(t)/dt
        return (*xdot) + T * (2 * (*p) + T * (3 * (*q) + T * 4 * (*r)));
}

template<int NDIM>
double Ode2StepperGL3<NDIM>::doStep(double dt)
{
    double dt2 = dt*dt, idt = 1./dt, idt2 = idt*idt;

    // collocation points are the nodes of Gauss-Legendre quadrature of degree 3:
    // this gives the highest possible accuracy of the whole scheme (it has order 6)
    static const double h0 = 0.1127016653792583, h1 = 0.5, h2 = 1-h0,
    // various combinations of collocation points
    su0 = 0.5 *  h0 * h0,
    sv0 = su0 *  h0 * -2 / 3,
    sw0 = sv0 * (0.25 * h0 - h1),
    su1 = 0.5 *  h1 * h1,
    sv1 = su1 * (h1 / 3 - h0),
    sw1 = su1 *  h1 / 6 * (4 * h0 - h1),
    su2 = 0.5 *  h2 * h2,
    sv2 = su2 * (h2 / 3 - h0),
    sw2 = su2 * (h2 / 3 * (0.5 * h2 - h0 - h1) + h0 * h1),
    du0 = h0,
    dv0 = h0  * -h0 / 2,
    dw0 = h0  * h0 * (h1 / 2 - h0 / 6),
    du1 = h1,
    dv1 = h1  * (h1 / 2 - h0),
    dw1 = h1  * h1 * (h0 / 2 - h1 / 6),
    du2 = h2,
    dv2 = h2  * (h2 / 2 - h0),
    dw2 = h2  * (h2 * (h2 / 3 - h1 / 2 - h0 / 2) + h0 * h1),
    mvu = 1 / (h1 - h0),
    mwu = 1 / (h2 - h0) / (h2 - h1),
    mwv = 1 / (h2 - h1),
    mpu = 1./2,
    mpv = 1./2 * (1 - h0),
    mpw = 1./2 * (1 - h0) * (1 - h1),
    mqv = 1./6,
    mqw = 1./6 * (2 - h0 - h1),
    mrw = 1./12;

    // collect the values of the matrices A and B in the RHS of the ODE at the collocation points h_k,
    // corresponding to times  t_0 + h_k * dt, where 0<=h<=1 is the time normalized to timestep;
    // values of A(h_k) and B(h_k) are stored as flattened arrays in row-major order
    double a0[NDIM * NDIM], a1[NDIM * NDIM], a2[NDIM * NDIM];
    double b0[NDIM * NDIM], b1[NDIM * NDIM], b2[NDIM * NDIM];
    odeSystem.evalMat(dt * h0, a0, b0);
    odeSystem.evalMat(dt * h1, a1, b1);
    odeSystem.evalMat(dt * h2, a2, b2);

    // The second derivative of d-th component x_d is approximated by a quadratic polynomial in h,
    // with coefficients u_d, v_d, w_d to be determined:
    // x_d''(h) = u_d + (h-h0) * (v_d + (h-h1) * w_d)                                       [*]
    // accordingly, the first derivative and the value of x_d itself are
    // x_d'(h)  = x_d'(0) + dt * (pu(h) * u_d + pv(h) * v_d + pw(h) * w_d),
    // x_d(h)   = x_d(0)  + dt * h * x_d'(0) + dt^2 * (su(h) * u_d + sv(h) * v_d + sw(h) * w_d),
    // where pu, pv, pw, su, sv, sw are known polynomials of h
    // (pre-computed for all collocation points h_k and for the final point h=1).
    // The RHS of our ODE prescribes that
    // x_d''(h_k) = \sum_{j=1}^{NDIM} A_{dj}(h_k) x_j(h_k) + B_{dj}(h_k) x_j'(h_j).
    // We iteratively recompute the coefficients u,v,w:
    // first evaluate x_d(h_k) with the current values of these coefs at the given point h_k,
    // then compute the rhs x_d''(h_k),
    // then use the relation [*] to update u, v, and w - it is written in such a way that
    // for each point h_k we update only one of these coefs (h0 -> u, h1 -> v, h2 -> w);
    // the updated coefs are then used for the next point h_{k+1}, and then the whole iteration
    // cycle is repeated a few times (three seems to be enough).

    // storage for temporary arrays defined below
    const int tempSize = NDIM * 8;
    double *temp = static_cast<double*>(alloca(tempSize * sizeof(double)));
    // approximate values of x and xdot at the current iteration and the current collocation point h_k
    double *xh = temp, *dh = temp + NDIM;
    // pre-computed values of x_prev[d] + x_prev'[d] * dt * h_{0,1,2}
    double *x0 = temp + 2*NDIM, *x1 = temp + 3*NDIM, *x2 = temp + 4*NDIM;
    // polynomial coefficients to be calculated
    double *u = temp + 5*NDIM, *v = temp + 6*NDIM, *w = temp + 7*NDIM;

    // process each vector independently
    for(unsigned int vec=0; vec<numVectors; vec++) {
        // values of x and xdot at the beginning of the current timestep
        double *x = &state[vec * 5*NDIM], *xdot = x + NDIM;
        // values of higher-order interpolation coefficients at the beginning of the timestep
        double *p = xdot + NDIM, *q = p + NDIM, *r = q + NDIM;
        for(int d=0; d<NDIM; d++) {
            x0[d] = x[d] + xdot[d] * dt * h0;
            x1[d] = x[d] + xdot[d] * dt * h1;
            x2[d] = x[d] + xdot[d] * dt * h2;
            // predict the polynomial coefs in x'' from the previous timestep
            double Q = dt*q[d], R = dt2*r[d];
            u[d] = 2  * p[d] + 6*h0 * Q + 12*h0*h0 * R;
            v[d] = 6  * Q + 12*(h0+h1) * R;
            w[d] = 12 * R;
        }

        // iteratively find the coefficients u_d, v_d, w_d for each component of vector x
        const int NUMITER = newstep ? 6 : 3;
        for(int i=0; i<NUMITER; i++) {
            // consider each collocation point h_k in turn, and use it to update u, v, w (one by one)

            // h_0 is used for calculating u
            // first predict the values of x and xdot at time h_0
            for(int d=0; d<NDIM; d++) {
                xh[d] = x0  [d] + dt2 * (su0 * u[d] + sv0 * v[d] + sw0 * w[d]);
                dh[d] = xdot[d] + dt  * (du0 * u[d] + dv0 * v[d] + dw0 * w[d]);
            }
            // then compute the RHS at time h_0 and calculate u
            for(int d=0; d<NDIM; d++) {
                // second derivative of d'th component (RHS of the ODE) at the current point
                double rhs = 0;
                for(int j=0; j<NDIM; j++)
                    rhs += a0[d * NDIM + j] * xh[j] + b0[d * NDIM + j] * dh[j];
                u[d] = rhs;
            }

            // h_1 is used for calculating v (third derivative of d'th component)
            for(int d=0; d<NDIM; d++) {
                xh[d] = x1  [d] + dt2 * (su1 * u[d] + sv1 * v[d] + sw1 * w[d]);
                dh[d] = xdot[d] + dt  * (du1 * u[d] + dv1 * v[d] + dw1 * w[d]);
            }
            for(int d=0; d<NDIM; d++) {
                double rhs = 0;
                for(int j=0; j<NDIM; j++)
                    rhs += a1[d * NDIM + j] * xh[j] + b1[d * NDIM + j] * dh[j];
                v[d] = (rhs - u[d]) * mvu;
            }

            // finally, h_2 is used for calculating w (fourth derivative)
            for(int d=0; d<NDIM; d++) {
                xh[d] = x2  [d] + dt2 * (su2 * u[d] + sv2 * v[d] + sw2 * w[d]);
                dh[d] = xdot[d] + dt  * (du2 * u[d] + dv2 * v[d] + dw2 * w[d]);
            }
            for(int d=0; d<NDIM; d++) {
                double rhs = 0;
                for(int j=0; j<NDIM; j++)
                    rhs += a2[d * NDIM + j] * xh[j] + b2[d * NDIM + j] * dh[j];
                w[d] = (rhs - u[d]) * mwu - v[d] * mwv;
            }
        }

        // now that the coefficients u, v, w are known,
        // compute the new values of x_d, xdot_d at the end of timestep (h=1)
        // and the coefficients p, q, r for interpolating the solution at any moment of time
        // inside the completed timestep
        for(int d=0; d<NDIM; d++) {
            double
            P = mpw * w[d] + mpv * v[d] + mpu * u[d],
            Q = mqw * w[d] + mqv * v[d],
            R = mrw * w[d];
            p[d] = P;
            q[d] = Q * idt;
            r[d] = R * idt2;
            x[d]    += (    P - 2 * Q + 3 * R) * dt2 + xdot[d] * dt;
            xdot[d] += (2 * P - 3 * Q + 4 * R) * dt;
        }
    }

    prevTimeStep = dt;
    newstep = false;
    return dt;
}

template<int NDIM>
Ode2StepperGL4<NDIM>::Ode2StepperGL4(
    const IOdeSystem2ndOrderLinear& _odeSystem, unsigned int _numVectors) :
    odeSystem(_odeSystem),
    numVectors(_numVectors),
    prevTimeStep(0),
    newstep(true),
    state(6*NDIM * numVectors)
{
    if(numVectors <= 0)
        throw std::invalid_argument("Ode2StepperGL4: invalid number of vectors");
    if(odeSystem.size() != NDIM * 2)
        throw std::invalid_argument("Ode2StepperGL4: invalid size of the ODE system");
}

template<int NDIM>
void Ode2StepperGL4<NDIM>::init(const double stateNew[])
{
    state.assign(6*NDIM * numVectors, 0);
    for(unsigned int v=0; v<numVectors; v++)
        std::copy(stateNew + v * 2*NDIM, stateNew + (v+1) * 2*NDIM, state.begin() + v * 6*NDIM);
    newstep = true;
}

template<int NDIM>
double Ode2StepperGL4<NDIM>::getSol(double timeOffset, unsigned int i) const
{
    double T = timeOffset - prevTimeStep;
    // T is expected to be between -prevTimeStep and 0, i.e. interpolating on the previous timestep
    if(! ( (prevTimeStep>=0 && timeOffset>=0 && timeOffset<=prevTimeStep)
        || (prevTimeStep<=0 && timeOffset<=0 && timeOffset>=prevTimeStep) ) )
        throw std::out_of_range("Ode2StepperGL4: requested time is outside the last completed timestep");
    unsigned int vec = i / (2*NDIM);  // index of the requested vector
    if(vec >= numVectors)
        throw std::out_of_range("Ode2StepperGL4: element index out of range");
    i -= vec * 2*NDIM;          // index of the requested element in the given vector
    if(T==0)  // fast track: no interpolation needed, just copy the element of the state vector
        return state[vec * 6*NDIM + i];
    unsigned int d = i % NDIM;  // index of the requested dimension in either position or velocity
    const double *x = &state[vec * 6*NDIM + d], *xdot = x+NDIM,  // state vector (pos, vel)
        *p = xdot+NDIM, *q = p+NDIM, *r = q+NDIM, *s = r+NDIM;   // higher-order interpolation coefs
    if(i == d)  // d-th component of x(t)
        return (*x) + T * ((*xdot) + T * ((*p) + T * ((*q) + T * ((*r) + T * (*s)))));
    else        // d-th component of dx(t)/dt
        return (*xdot) + T * (2 * (*p) + T * (3 * (*q) + T * 4 * ((*r) + T * 5 * (*s))));
}

template<int NDIM>
double Ode2StepperGL4<NDIM>::doStep(double dt)
{
    double dt2 = dt*dt, dt3 = dt2*dt, idt = 1./dt, idt2 = idt*idt, idt3 = idt2*idt;

    // collocation points are the nodes of Gauss-Legendre quadrature of degree 4:
    // this gives the highest possible accuracy of the whole scheme (it has order 8)
    static const double h0 = 0.069431844202973713, h1 = 0.33000947820757187, h2 = 1-h1, h3 = 1-h0,
    // various combinations of collocation points
    su0 = 0.5 * h0 * h0,
    sv0 = su0 * h0 * -2/3,
    sw0 = su0 * h0 * (h1*2/3 - h0/6),
    sz0 = su0 * h0 * (h0 * (h1+h2)/6 - h0*h0/15 - h1*h2*2/3),
    su1 = 0.5 * h1 * h1,
    sv1 = su1 *(h1/3 - h0),
    sw1 = su1 * h1 * (h0*2/3 - h1/6),
    sz1 = su1 * h1 * (h1 * (h0+h2)/6 - h1*h1/15 - h0*h2*2/3),
    su2 = 0.5 * h2 * h2,
    sv2 = su2 *(h2/3 - h0),
    sw2 = su2 *(h2/3 * (h2/2 - h0 - h1) + h0*h1),
    sz2 = su2 * h2 * (h2 * (h0+h1)/6 - h2*h2/15 - h0*h1*2/3),
    su3 = 0.5 * h3 * h3,
    sv3 = su3 *(h3/3 - h0),
    sw3 = su3 *(h3/3 * (h3/2 - h0 - h1) + h0*h1),
    sz3 = su3 *(h3 * (h3 * (h3*0.6-h0-h1-h2)/6 + (h0*h1+h1*h2+h0*h2)/3) - h0*h1*h2),
    du0 = h0,
    dv0 = h0  * -h0 / 2,
    dw0 = h0  * h0 * (h1 / 2 - h0 / 6),
    dz0 = h0  * h0 * (h0 * (-h0 / 12 + h1 / 6 + h2 / 6) - h1 * h2 / 2),
    du1 = h1,
    dv1 = h1  * (h1 / 2 - h0),
    dw1 = h1  * h1 * (h0 / 2 - h1 / 6),
    dz1 = h1  * h1 * (h1 * (-h1 / 12 + h0 / 6 + h2 / 6) - h0 * h2 / 2),
    du2 = h2,
    dv2 = h2  * (h2 / 2 - h0),
    dw2 = h2  * (h2 * (h2 / 3 - h1 / 2 - h0 / 2) + h0 * h1),
    dz2 = h2  *  h2 * (h2 * (-h2 / 12 + h0 / 6 + h1 / 6) - h0 * h1 / 2),
    du3 = h3,
    dv3 = h3  * (h3 / 2 - h0),
    dw3 = h3  * (h3 * (h3 / 3 - h1 / 2 - h0 / 2) + h0 * h1),
    dz3 = h3  * (h3 * (h0*h1/2 + h0*h2/2 + h1*h2/2 - h3 * (h0/3 + h1/3 + h2/3 - h3/4)) - h0*h1*h2),
    mvu = 1 / (h1 - h0),
    mwu = 1 / (h2 - h0) / (h2 - h1),
    mwv = 1 / (h2 - h1),
    mzu = 1 / (h3 - h0) / (h3 - h1) / (h3 - h2),
    mzv = 1 / (h3 - h1) / (h3 - h2),
    mzw = 1 / (h3 - h2),
    mpu = 1./2,
    mpv = 1./2 * (1 - h0),
    mpw = 1./2 * (1 - h0) * (1 - h1),
    mpz = 1./2 * (1 - h0) * (1 - h1) * (1 - h2),
    mqv = 1./6,
    mqw = 1./6 * (2 - h0 - h1),
    mqz = 1./6 * ((1-h0) * (1-h1) + (1-h1) * (1-h2) + (1-h2) * (1-h0)),
    mrw = 1./12,
    mrz = 1./12 * (3 - h0 - h1 - h2),
    msz = 1./20;

    // collect the values of the matrices A and B in the RHS of the ODE at the collocation points h_k,
    // corresponding to times  t_0 + h_k * dt, where 0<=h<=1 is the time normalized to timestep;
    // values of A(h_k) and B(h_k) are stored as flattened arrays in row-major order
    double a0[NDIM * NDIM], a1[NDIM * NDIM], a2[NDIM * NDIM], a3[NDIM * NDIM];
    double b0[NDIM * NDIM], b1[NDIM * NDIM], b2[NDIM * NDIM], b3[NDIM * NDIM];
    odeSystem.evalMat(dt * h0, a0, b0);
    odeSystem.evalMat(dt * h1, a1, b1);
    odeSystem.evalMat(dt * h2, a2, b2);
    odeSystem.evalMat(dt * h3, a3, b3);

    // The second derivative of d-th component x_d is approximated by a cubic polynomial in h,
    // with coefficients u_d, v_d, w_d, z_d to be determined:
    // x_d''(h) = u_d + (h-h0) * (v_d + (h-h1) * (w_d + (h-h2) * z_d))
    // the procedure is analogous to the 6-th order method.

    // storage for temporary arrays defined below
    const int tempSize = NDIM * 10;
    double *temp = static_cast<double*>(alloca(tempSize * sizeof(double)));
    // approximate values of x and xdot at the current iteration and the current collocation point h_k
    double *xh = temp, *dh = temp + NDIM;
    // pre-computed values of x_prev[d] + x_prev'[d] * dt * h_{0,1,2,3}
    double *x0 = temp + 2*NDIM, *x1 = temp + 3*NDIM, *x2 = temp + 4*NDIM, *x3 = temp + 5*NDIM;
    // polynomial coefficients to be calculated
    double *u = temp + 6*NDIM, *v = temp + 7*NDIM, *w = temp + 8*NDIM, *z = temp + 9*NDIM;

    // process each vector independently
    for(unsigned int vec=0; vec<numVectors; vec++) {
        // values of x and xdot at the beginning of the current timestep
        double *x = &state[vec * 6*NDIM], *xdot = x + NDIM;
        // values of higher-order interpolation coefficients at the beginning of the timestep
        double *p = xdot + NDIM, *q = p + NDIM, *r = q + NDIM, *s = r + NDIM;
        for(int d=0; d<NDIM; d++) {
            x0[d] = x[d] + xdot[d] * dt * h0;
            x1[d] = x[d] + xdot[d] * dt * h1;
            x2[d] = x[d] + xdot[d] * dt * h2;
            x3[d] = x[d] + xdot[d] * dt * h3;
            // predict the polynomial coefs in x'' from the previous timestep
            double Q = dt*q[d], R = dt2*r[d], S = dt3*s[d];
            u[d] = 2  * p[d] + 6*h0 * Q + 12*h0*h0 * R + 20*h0*h0*h0 * S;
            v[d] = 6  * Q + 12*(h0+h1) * R + 20*(h0*h0+h0*h1+h1*h1) * S;
            w[d] = 12 * R + 20*(h0+h1+h2) * S,
            z[d] = 20 * S;
        }

        // iteratively find the coefficients u_d, v_d, w_d, z_d for each component of vector x
        const int NUMITER = newstep ? 8 : 4;
        for(int iter=0; iter<NUMITER; iter++) {
            // consider each collocation point h_k in turn, and use it to update u, v, w, z (one by one)

            // h_0 is used for calculating u
            // first predict the values of x and xdot at time h_0
            for(int d=0; d<NDIM; d++) {
                xh[d] = x0  [d] + dt2 * (su0 * u[d] + sv0 * v[d] + sw0 * w[d] + sz0 * z[d]);
                dh[d] = xdot[d] + dt  * (du0 * u[d] + dv0 * v[d] + dw0 * w[d] + dz0 * z[d]);
            }
            // then compute the RHS at time h_0 and calculate u
            for(int d=0; d<NDIM; d++) {
                // second derivative of d-th component (RHS of the ODE) at the current point
                double rhs = 0;
                for(int j=0; j<NDIM; j++)
                    rhs += a0[d * NDIM + j] * xh[j] + b0[d * NDIM + j] * dh[j];
                u[d] = rhs;
            }

            // h_1 is used for calculating v
            for(int d=0; d<NDIM; d++) {
                xh[d] = x1  [d] + dt2 * (su1 * u[d] + sv1 * v[d] + sw1 * w[d] + sz1 * z[d]);
                dh[d] = xdot[d] + dt  * (du1 * u[d] + dv1 * v[d] + dw1 * w[d] + dz1 * z[d]);
            }
            for(int d=0; d<NDIM; d++) {
                double rhs = 0;
                for(int j=0; j<NDIM; j++)
                    rhs += a1[d * NDIM + j] * xh[j] + b1[d * NDIM + j] * dh[j];
                v[d] = (rhs - u[d]) * mvu;
            }

            // h_2 is used for calculating w
            for(int d=0; d<NDIM; d++) {
                xh[d] = x2  [d] + dt2 * (su2 * u[d] + sv2 * v[d] + sw2 * w[d] + sz2 * z[d]);
                dh[d] = xdot[d] + dt  * (du2 * u[d] + dv2 * v[d] + dw2 * w[d] + dz2 * z[d]);
            }
            for(int d=0; d<NDIM; d++) {
                double rhs = 0;
                for(int j=0; j<NDIM; j++)
                    rhs += a2[d * NDIM + j] * xh[j] + b2[d * NDIM + j] * dh[j];
                w[d] = (rhs - u[d]) * mwu - v[d] * mwv;
            }

            // finally, h_3 is used for calculating z
            for(int d=0; d<NDIM; d++) {
                xh[d] = x3  [d] + dt2 * (su3 * u[d] + sv3 * v[d] + sw3 * w[d] + sz3 * z[d]);
                dh[d] = xdot[d] + dt  * (du3 * u[d] + dv3 * v[d] + dw3 * w[d] + dz3 * z[d]);
            }
            for(int d=0; d<NDIM; d++) {
                double rhs = 0;
                for(int j=0; j<NDIM; j++)
                    rhs += a3[d * NDIM + j] * xh[j] + b3[d * NDIM + j] * dh[j];
                z[d] = (rhs - u[d]) * mzu - v[d] * mzv - w[d] * mzw;
            }
        }

        // now that the coefficients u, v, w, z are known,
        // compute the new values of x_d, x_d' at the end of timestep (h=1)
        // and the coefficients p, q, r, s for interpolating the solution at any moment of time
        // inside the completed timestep
        for(int d=0; d<NDIM; d++) {
            double
            P = mpz * z[d] + mpw * w[d] + mpv * v[d] + mpu * u[d],
            Q = mqz * z[d] + mqw * w[d] + mqv * v[d],
            R = mrz * z[d] + mrw * w[d],
            S = msz * z[d];
            p[d] = P;
            q[d] = Q * idt;
            r[d] = R * idt2;
            s[d] = S * idt3;
            x[d]    += (    P - 2 * Q + 3 * R - 4 * S) * dt2 + xdot[d] * dt;
            xdot[d] += (2 * P - 3 * Q + 4 * R - 5 * S) * dt;
        }
    }

    prevTimeStep = dt;
    newstep = false;
    return dt;
}

// compile the template instantiation for 1d,2d and 3d systems only
template class Ode2StepperGL3<1>;
template class Ode2StepperGL3<2>;
template class Ode2StepperGL3<3>;
template class Ode2StepperGL4<1>;
template class Ode2StepperGL4<2>;
template class Ode2StepperGL4<3>;

}  // namespace
