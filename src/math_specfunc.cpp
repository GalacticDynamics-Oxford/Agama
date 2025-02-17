#include "math_base.h"
#include "math_core.h"
#include "math_specfunc.h"
#include <string>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <gsl/gsl_sf_hyperg.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_psi.h>
#include <gsl/gsl_sf_ellint.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_lambert.h>

/* Most of the functions here are implemented by calling corresponding routines from GSL,
   but having library-independent wrappers makes it possible to switch the back-end if necessary */

namespace math {

// if any GSL function triggers an error, it will be stored in this variable (defined in math_core.cpp)
extern bool exceptionFlag;

// call a GSL function and return NAN in case of an error;
// the error message is recorded in math::exceptionText and may be later examined by the caller
#define CALL_FUNCTION_OR_NAN(x) { \
    exceptionFlag = false; \
    double _result = x; \
    return !exceptionFlag ? _result : NAN; \
}

double erf(double xx)
{
    // note: erf from <cmath> is not accurate enough, at least in some versions of standard library,
    // whereas the one from GSL is too slow. This one is fast and should be accurate to machine precision.
    double x = xx>=0 ? xx : -xx;
    if(x >= 6)
        return xx>0 ? 1 : -1;
    double coef = x<=1 ?
        (1 + x * (1.1819872144928538 + x * (0.71431605350552105 + x *
        (0.24307301039438656 + x * (0.046377180766266783 + x * 0.0039203315598425230))))) /
        (1 + x * (2.3103663815883269 + x * (2.3212853468498850 + x * (1.3042494330881013 + x *
        (0.43475925241505070 + x * (0.082155448430511544 + x * 0.0069509962985487110))))))
    : /* x>1 */
        (0.99999988128286680 + x * (1.3880993103643025 + x * (0.94660090636459344 + x *
        (0.37193392527165884 + x * (0.083848202518174942 + x * 0.0088622939090319692))))) /
        (1 + x * (2.5164772417854029 + x * (2.7861475430536739 + x * (1.7515208175152142 + x *
        (0.66715178052942912 + x * (0.14861294386339330 + x * 0.015708138463035560))))));
    return (xx>=0 ? 1 : -1) * (1 - exp(-x*x) * coef);
}

double erfinv(const double x)
{
    if(x < -1 || x > 1) return NAN;
    if(x == -1) return -INFINITY;
    if(x == +1) return +INFINITY;
    if(x ==  0) return 0;
    double z;   // first approximation
    if(fabs(x)<=0.7) {
        double x2 = x*x;
        z  = x * (((-0.140543331 * x2 + 0.914624893) * x2 - 1.645349621) * x2 + 0.886226899) /
            (1 + (((0.012229801 * x2 - 0.329097515) * x2 + 1.442710462) * x2 - 2.118377725) * x2);
    }
    else {
        double y = sqrt(-log((1-fabs(x))/2));
        z = (((1.641345311 * y + 3.429567803) * y - 1.62490649) * y - 1.970840454) /
            (1 + (1.6370678 * y + 3.5438892) * y);
        if(x<0) z = -z;
    }
    // improve by Halley iteration
    double f = math::erf(z) - x, fp = 2/M_SQRTPI * exp(-z*z), fpp = -2*z*fp;
    z -= f*fp / (fp*fp - 0.5*f*fpp);
    return z;
}

double hypergeom2F1(const double a, const double b, const double c, const double x)
{
    exceptionFlag = false;
    double _result = NAN;
    if(-1.<=x && x<1)
        _result = gsl_sf_hyperg_2F1(a, b, c, x);
    // extension for 2F1 into the range x<-1 which is not provided by GSL; code from Heiko Bauke
    else if(x<-1.) {
        // choose one of the two Pfaff transformations, whichever is expected to be more accurate
        if(a*(c-b) < (c-a)*b)
            _result = std::pow(1.-x, -a) * gsl_sf_hyperg_2F1(a, c-b, c, x/(x-1.));
        else
            _result = std::pow(1.-x, -b) * gsl_sf_hyperg_2F1(c-a, b, c, x/(x-1.));
    }  // otherwise (x>=1) not a real-valued function
    return !exceptionFlag ? _result : NAN;
}

// ------ approximations for specific instances of hypergeometric function ------ //
namespace {
// data for approximations for several individual values of m, from 0 up to MMAX.
// max relative error in function value is ~1e-8 for m=0 and up to few x 1e-6 for m=12;
// error in derivative is ~10x higher, and average error is ~10x lower than max error.
const int MMAX_HYPERGEOM = 12;

/* continued fraction expansions of the form
                        A1
 f = A0 + ---------------------------------
                             A3
          x + A2 + ------------------------
                                  A5
                   x + A4 + ---------------
                                       A7
                            x + A6 + ------
                                     x + A8
*/
const double HYPERGEOM_0[MMAX_HYPERGEOM+1][9] = {
    {0.4529874071816636,-2.692539697590362,-6.480598406507461,-3.860216328610213,-2.609146033646615,
    -0.1909432553353492,-1.460299624021841,-0.01541563568408259,-1.115369096436195},
    {0.08938502676040028,-2.525406068615592,-3.802041706399501,-2.691157667804647,-2.741391856196645,
    -0.1823978151361306,-1.467764617695739,-0.01514303374682610,-1.116215391130251},
    {-0.01069787527041158,-1.537005620596391,-1.697345768635404,-0.3378529865322015,-1.983063657443297,
    -0.09488668216801367,-1.362931464322307,-0.00940689438170889,-1.093693723592968},
    {-0.01711490718455377,-0.6009550456991466,3.394854752517220,36.72266736665577,-9.201204857656629,
    0.01516766397285159,-1.228118125014300,-0.005976086542472318,-1.080099304233194},
    {-0.02749247751103787,-0.3072229139612030,5.687071537359105,55.96560640229840,-9.348637781981967,
    0.0007050923061239599,-1.068784792886700,0.0006587322854650128,0.1314557788692205},
    {0.006592755012603758,0.1271086283761586,-20.85294538326066,368.2891340792655,17.55960611497124,
    0.01287836941499270,-2.119217276127404,0.005841336094937430,-1.041259452130866},
    {0.02415357420175733,0.2760551779728967,-8.304528092063684,49.00778641555581,5.706942819375404,
    0.0000135380497674,-1.077581157358874,0.01898377989044321,-0.8802786435315881},
    {-0.4755438570635217,-15.23176265534132,-26.14525965096805,22.72480434937186,-3.726962896593618,
    12.26068097538548,2.374616974037387,-0.00001072561873862000,-0.9939982148579712},
    {-0.01778950682637795,-0.1621708071527029,-6.722812444954199,-3.605415274607454,42.56877984182697,
    1763.410384466364,-40.89722407762218,4.04152404e-8,-0.9643299106460461},
    {-0.01640040314046018,-0.1567441655078544,-6.696210685394554,5.878476915868881,-15.59083749781898,
    297.2840667256496,18.02878411264815,-1.6468589e-9,-0.9155934138921334},
    {-0.04537030719843308,-0.3438471580930914,-5.110748703261416,4.979640874599663,-4.697175985956599,
    33.93946493242602,5.914298150113606,9.7752486955e-9,0.01489956391058121},
    {0.2560046720818122,3.738379858847606,-10.25803052150991,11.73970408392119,-1.965832605102681,
    0.3053195680701161,-35.13200601587556,1238.732680559852,35.14765318693473},
    {0.01532856961552287,0.1296876459958492,-5.869778335825212,4.144607847312670,-1.999697044584623,
    -29.36865667344516,5.786781200597918,63.58059663540046,-3.806468298143671}
};
/* same approach for a different interval */
const double HYPERGEOM_I[MMAX_HYPERGEOM+1][9] = {
    {0.6265483823527411,-0.8488996006421011,-2.430767712870059,-0.2216645603123820,-1.377915333154397,
    -0.01067779281977426,-1.106258117483859,-0.0008017018734870802,-1.025715949003203},
    {0.2556745290319481,-1.302202831061045,-1.856484357772605,-0.1506639551665747,-1.365766354550209,
    -0.008801806475184472,-1.097860353907581,-0.0006463098611572502,-1.023023211337991},
    {0.06793076058208223,-1.259938656832443,-1.383753616440758,-0.04811851897121551,-1.252578275690214,
    -0.004438147879886131,-1.071030099270125,-0.0003371916228163400,-1.016826439397124},
    {-0.1438872110578193,-1.318526106712988,-1.165709435421967,-0.01078909626071359,-1.140430116301888,
    -0.001811274363462059,-1.048091679713478,-0.0001577543940147300,-1.011553061701041},
    {-0.5623160680982770,-1.576021970674381,-1.076471896362168,-0.001780342522184680,-1.062427091799673,
    0.004018205261775791,-0.6514030690056149,0.005561736279330680,-1.027642872383092},
    {-9.162872732083632,42.14305648246658,5.624924575599729,2.272916747830027,-1.389775678893547,
    -0.00073079680223201,-1.034210205811073,-0.00008747485028171,-1.0085136703608},
    {0.2017739963700901,0.5462246106199197,-5.109468125375064,14.41419452992633,2.502624381778485,
    -0.00006809974429250000,-1.030883958557467,-0.00008191929998314000,-1.008456613288245},
    {0.3185170028368671,0.8886260784862031,-3.580049820836553,5.651162454123797,1.186628617227599,
    -0.00003820523068187999,-1.023368519956452,-0.00004924296499382999,-1.006649542714069},
    {0.5045523625514224,1.485758862452970,-2.752242779713227,2.595936432732558,0.4785798705675604,
    -0.00002433498096823000,-1.019227289991061,-0.00003585972755994999,-1.005789911366520},
    {1.398160625476730,2.954532545810714,-2.136522020336696,1.061026744232020,-0.06909753819383704,
    -0.00001821885545569000,-1.015887980712423,-0.00002582418432982,-1.004995632391522},
    {3.620215546579624,5.686835654840336,-1.799598166133843,0.5134745736144935,-0.3602442057709179,
    -0.00001410577594344001,-1.013913058391083,-0.00002176004393978001,-1.004725750313438},
    {8.776955057607215,10.69299243000038,-1.596330222811836,0.2810073271616607,-0.5309379037693800,
    -0.00001164458682589000,-1.013280572890286,-0.00002307480563412999,-1.005031422945444},
    {20.02358305143908,19.69122956965990,-1.465053947210626,0.1693355984048097,-0.6378402328091336,
    -0.00001088292906364000,-1.014557343074211,-0.00003173874282459000,-1.005763792645454}
};
/* asymptotic expansion of F(x) at x -> 1 of the following form (with y = 1-x and z = ln(y)):
   f = (A0 + A1*z) + (A2 + A3*z) * y + (A4 + A5*z) * y^2 + (A6 + A7*z) * y^3 + (A8 + A9*z) * y^4;
   the coefficients are given by eq. 15.3.10 of Abramowicz&Stegun.
*/
const double HYPERGEOM_1[MMAX_HYPERGEOM+1][10] = {
  {  0.9360775742346216,-0.2250790790392765, 0.0348401207694437,-0.0422023273198643, 0.0104808433089421,
    -0.0230793977530508, 0.0049617494377955,-0.0158670859552224, 0.0028807359141383,-0.0120862568799546},
    {0.1430450323100617,-0.9003163161571056,0.02156517827104491,-0.8440465463972865,0.008039995073102123,
    -0.8308583191098289,0.004137595481807615,-0.8250884696715662,0.002509932082129108,-0.8218654678369117},
    {-2.819671260176213,-2.400843509752283,-2.866871055726074,-5.251845177583119,-2.875549627231282,
    -8.123948009073887,-2.878479952408139,-11.00117959562089,-2.879798366033066,-13.88039456791229},
    {-11.3768305631473,-5.76202442340548,-22.46842570169623,-22.68797116715908,-33.54140985134508,
    -50.69343557662106,-44.61066123524255,-89.76962550026647,-55.67872510000488,-139.914377244556},
    {-33.53009359531525,-13.17034153921252,-110.3361852693207,-81.49148827387747,-230.4962906372769,
    -248.2943783344704,-394.0183070924650,-556.9380847363468,-600.9038975602715,-1050.785527061154},
    {-87.51906383078700,-29.26742564269449,-434.6509534806616,-261.5776166815820,-1212.744309671997,
    -1042.223316465678,-2593.120194540037,-2887.827106040317,-4747.096184506519,-6486.330413957741},
    {-214.1711215952598,-63.85620140224254,-1500.709045078266,-778.2474545898310,-5397.230112323440,
    -3927.717622383053,-14120.94007905760,-13174.21952507649,-30568.64501900970,-34736.71163838528},
    {-503.6105492173122,-137.5364337894455,-4742.610771427904,-2191.986913519288,-21381.51289794725,
    -13665.66841397181,-67219.94462809997,-54567.77318079023,-169894.9790469529,-166900.6500021825},
    {-1152.612120663817,-293.4110587508170,-14062.58521759393,-5923.235748532118,-77715.55133278912,
    -44701.91978970333,-289167.5872258646,-209540.2490142344,-843996.0107326956,-735846.4213429555},
    {-2587.023565073958,-621.3410655899655,-39737.92516363194,-15494.69282314976,-264294.3690284689,
    -139210.1308329862,-1149012.999729593,-756955.0864043624,-3836087.008386949,-3024863.489811182},
    {-5721.752022023828,-1308.086453873611,-108138.6205048347,-39487.85982630963,-852386.5359668258,
    -416473.5216056094,-4282240.812990009,-2600067.332801687,-16217451.44075212,-11730772.53666386},
    {-12510.48091662534,-2740.752570020901,-285492.8478112217,-98495.79548512613,-2632527.346250018,
    -1205034.497888340,-15137019.29939817,-8560765.912081749,-64550174.40362200,-43305436.93806978},
    {-27103.58303477956,-5719.831450478403,-735179.4395018670,-241305.3893170576,-7842282.258628712,
    -3389586.640563044,-51179318.48413728,-27187309.51284942,-244144745.5117011,-153247217.2931317}
};

/* pre-factors in legendreQ are  2^(-1/2-m)*sqrt(Pi)*Gamma(m+1/2)/GAMMA(m+1) */
const double Q_PREFACTOR[MMAX_HYPERGEOM+1] = {
    2.221441469079183,
    0.5553603672697958,
    0.2082601377261734,
    0.08677505738590559,
    0.03796408760633369,
    0.01708383942285016,
    0.007830093068806324,
    0.003635400353374365,
    0.001704093915644234,
    0.0008047110157208882,
    0.0003822377324674218,
    0.0001824316450412695,
    0.0000874151632489416
};

/* choose the asymptotic form at x=1 if x is larger than this value */
const double X_THRESHOLD1[MMAX_HYPERGEOM+1] = {
    0.94, 0.956,0.968,0.98, 0.98, 0.983,0.987,0.989,0.99, 0.992,0.992,0.9928,0.993
};
/* boundary between two continued fraction approximations */
const double X_THRESHOLD0[MMAX_HYPERGEOM+1] = {
    0.72, 0.72, 0.80, 0.80, 0.83, 0.86, 0.85, 0.88, 0.88, 0.88, 0.885,0.90, 0.91
};

double hypergeom_m(int m, double x, double* deriv)
{
    if(x < X_THRESHOLD1[m]) {  // use continued fraction approximation for x not too close to unity
        const double* A = x<X_THRESHOLD0[m] ? HYPERGEOM_0[m] : HYPERGEOM_I[m];
        double xA8 = x + A[8];
        double xA6 = x + A[6] + A[7] / xA8;
        double xA4 = x + A[4] + A[5] / xA6;
        double xA2 = x + A[2] + A[3] / xA4;
        if(deriv!=NULL)
            *deriv =-A[1] / pow_2(xA2) * (1 - A[3] / pow_2(xA4) *
                (1 - A[5] / pow_2(xA6) * (1 - A[7] / pow_2(xA8) ) ) );
        return A[0] + A[1] / xA2;
    }
    else {  // use asymptotic series expansion close to the log-singular point (x=1)
        const double* A = HYPERGEOM_1[m], y = 1-x, y2 = y*y, z = log(y);
        if(deriv!=NULL)
            *deriv = -A[1]/y - (A[2] + A[3] + A[3]*z) - (2*A[4] + A[5] + 2*A[5]*z) * y -
            (3*A[6] + A[7] + 3*A[7]*z + (4*A[8] + A[9] + 4*A[9]*z) * y) * y2;
        return A[0] + A[1]*z + (A[2] + A[3]*z) * y + 
        ( A[4] + A[5]*z + (A[6] + A[7]*z) * y + (A[8] + A[9]*z) * y2 ) * y2;
    }
}
}  // internal namespace

double legendreQ(const double n, const double x, double* deriv)
{
    int m = static_cast<int>(n+0.5);
    double prefactor, F;
    if(m == n+0.5 && m >= 0 && m<= MMAX_HYPERGEOM) {
        prefactor = Q_PREFACTOR[m] / sqrt(x) / pow(x, m);
        F = prefactor * hypergeom_m(m, 1/(x*x), deriv);
    } else {
        prefactor = std::pow(2*x,-1-n) * M_SQRTPI * gamma(n+1) / gamma(n+1.5);
        F = prefactor * hypergeom2F1(1+n/2, 0.5+n/2, 1.5+n, 1/(x*x));
        if(deriv)
            *deriv = (1+n/2) * (0.5+n/2) / (1.5+n) * hypergeom2F1(2+n/2, 1.5+n/2, 2.5+n, 1/(x*x));
    }
    if(deriv)
        *deriv = ( (-1-n) * F + prefactor * (*deriv) * (-2/(x*x)) ) / x;
    return F;
}

double factorial(const unsigned int n) {
     CALL_FUNCTION_OR_NAN( gsl_sf_fact(n) )
}

double lnfactorial(const unsigned int n) {
    CALL_FUNCTION_OR_NAN( gsl_sf_lnfact(n) )
}

double dfactorial(const unsigned int n) {
    CALL_FUNCTION_OR_NAN( gsl_sf_doublefact(n) )
}

double gamma(const double x) {
    CALL_FUNCTION_OR_NAN( gsl_sf_gamma(x) )
}

double lngamma(const double x) {
    CALL_FUNCTION_OR_NAN( gsl_sf_lngamma(x) )
}

double gammainc(const double x, const double y) {
    CALL_FUNCTION_OR_NAN( gsl_sf_gamma_inc(x, y) )
}

double digamma(const double x) {
    CALL_FUNCTION_OR_NAN( gsl_sf_psi(x) )
}

double digamma(const int x) {
    CALL_FUNCTION_OR_NAN( gsl_sf_psi_int(x) )
}

double ellintK(const double k) {
    CALL_FUNCTION_OR_NAN( gsl_sf_ellint_Kcomp(k, GSL_PREC_DOUBLE) )
}

double ellintE(const double k) {
    CALL_FUNCTION_OR_NAN( gsl_sf_ellint_Ecomp(k, GSL_PREC_DOUBLE) )
}

double ellintF(const double phi, const double k) {
    CALL_FUNCTION_OR_NAN( gsl_sf_ellint_F(phi, k, GSL_PREC_DOUBLE) )
}

double ellintE(const double phi, const double k) {
    CALL_FUNCTION_OR_NAN( gsl_sf_ellint_E(phi, k, GSL_PREC_DOUBLE) )
}

double ellintP(const double phi, const double k, const double n) {
    CALL_FUNCTION_OR_NAN( gsl_sf_ellint_P(phi, k, n, GSL_PREC_DOUBLE) )
}

double besselJ(const int n, const double x) {
    CALL_FUNCTION_OR_NAN( gsl_sf_bessel_Jn(n, x) )
}

double besselY(const int n, const double x) {
    CALL_FUNCTION_OR_NAN( gsl_sf_bessel_Yn(n, x) )
}

double besselI(const int n, const double x) {
    CALL_FUNCTION_OR_NAN( gsl_sf_bessel_In(n, x) )
}

double besselK(const int n, const double x) {
    CALL_FUNCTION_OR_NAN( gsl_sf_bessel_Kn(n, x) )
}

double lambertW(const double x, bool Wminus1branch) {
    CALL_FUNCTION_OR_NAN( Wminus1branch ? gsl_sf_lambert_Wm1(x) : gsl_sf_lambert_W0(x) )
}

double qexp(const double x, double q) {
    if(fabs(q) > 1e-7)
        return math::pow(1 + q*x, 1/q);
    else  // asymptotic expansion for small q, including the limiting case q=0
        return exp(x) * (q!=0 ? 1 + q*x*x * (-0.5 + q*x * (1./3 + 1./8*x)) : 1);
}

double solveKepler(double ecc, double phase)
{
    phase -= 2*M_PI * floor(1/(2*M_PI) * phase);
    if(phase==0 || phase==M_PI)
        return phase;
    double eta, sineta, coseta, deltaeta = 0;
    // initial approximation
    if(ecc>0.95 && (phase<0.3 || phase>6.0)) {  // very eccentric orbits near periapsis
        double phase1 = phase<=M_PI ? phase : phase - 2*M_PI;
        eta = phase + pow_2(ecc) * (cbrt(6*phase1) - phase1);
    } else {  // all other cases
        sincos(phase, sineta, coseta);
        eta = phase + ecc * sineta / sqrt(1 - ecc * (2*coseta - ecc));
    }
    do {  // Halley's method
        sincos(eta, sineta, coseta);
        double f  = eta - ecc * sineta - phase;
        double df = 1.  - ecc * coseta;
        double d2f= ecc * sineta;
        deltaeta  = -f * df / (df * df - 0.5 * f * d2f);
        eta      += deltaeta;
    } while(fabs(deltaeta) > 1e-5);
    // since the Halley method converges cubically, a correction < 1e-5 at the current iteration
    // implies that it would be <~1e-15 at the next iteration, which is beyond the precision limit
    return eta;
}

}  // namespace