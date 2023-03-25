#include "potential_ferrers.h"
#include "math_core.h"
#include "math_specfunc.h"
#include <cmath>
#include <stdexcept>

namespace potential {

/// relative accuracy of root-finder for lambda
const double ACCURACY_ROOT = 1e-6;

/// max. radius (in units of scale radii) beyond which to use an asymptotic quadrupole expansion
const double MAX_RADIUS = 10.0;

// Ferrers n=2 potential

Ferrers::Ferrers(double _mass, double _R, double _p, double _q):
    BasePotentialCar(), a(_R), b(_R*_p), c(_R*_q), mass(_mass), rho0( mass*105./(32*M_PI*a*b*c) )
{
    if(!(_R > 0))
        throw std::invalid_argument("Ferrers potential: scale radius should be positive");
    if(!(1 > _p && _p > _q && _q > 0))
        throw std::invalid_argument("Ferrers potential: axis ratios must satisfy 0 < q < p < 1");
    computeW(0, W0);
}

double Ferrers::densityCar(const coord::PosCar& pos, double /*time*/) const
{
    double m2 = pow_2(pos.x/a) + pow_2(pos.y/b) + pow_2(pos.z/c);
    return m2>1 ? 0 : rho0*pow_2(1-m2);
}

/// helper class to find lambda as the root of equation
/// x^2/(lambda+a^2) + y^2/(lambda+b^2) + z^2/(lambda+c^2) = 1
class FerrersLambdaRootFinder: public math::IFunction {
public:
    FerrersLambdaRootFinder(double x, double y, double z, double a, double b, double c) :
        x2(x*x), y2(y*y), z2(z*z), a2(a*a), b2(b*b), c2(c*c) {};
    virtual void evalDeriv(double lambda, double* val, double* der, double*) const {
        if(val)
            *val = x2/(lambda+a2) + y2/(lambda+b2) + z2/(lambda+c2) - 1;
        if(der)
            *der = -x2/pow_2(lambda+a2) - y2/pow_2(lambda+b2) - z2/pow_2(lambda+c2);
    }
    virtual unsigned int numDerivs() const { return 1; }
private:
    double x2, y2, z2, a2, b2, c2;
};

void Ferrers::evalCar(const coord::PosCar &pos,
    double* potential, coord::GradCar* grad, coord::HessCar* hess, double /*time*/) const
{
    double X2 = pow_2(pos.x), Y2 = pow_2(pos.y), Z2 = pow_2(pos.z);
    double m2 = X2/(a*a) + Y2/(b*b) + Z2/(c*c);
    double r2 = X2+Y2+Z2;
    if(r2 == INFINITY) {
        if(potential) *potential = 0;
        if(grad) grad->dx = grad->dy = grad->dz = 0;
        if(hess) hess->dx2 = hess->dy2 = hess->dz2 = hess->dxdy = hess->dydz = hess->dxdz = 0;
        return;
    }
    if(r2 > pow_2(MAX_RADIUS * a)) {
        // use spherical-harmonic expansion up to l=2 at large radii to speed up computation
        double Moverr = (32*M_PI/105) * rho0 * a*b*c / sqrt(r2);
        if(potential)
            *potential = -Moverr *
                (1 + ( 1./36 * (2*c*c-a*a-b*b) * (2*Z2-X2-Y2) + 1./12 * (b*b-a*a) * (Y2-X2) ) / (r2*r2) );
        double Moverr3 = Moverr/r2;
        double C20 = 1./36 * (2*c*c-a*a-b*b), C22 = 1./12 * (b*b-a*a);
        double add = 5 * (C20 * (2*Z2-X2-Y2) + C22 * (Y2-X2) ) / (r2*r2);
        if(grad) {
            grad->dx = pos.x * Moverr3 * (1+add+2*(C20+C22)/r2);
            grad->dy = pos.y * Moverr3 * (1+add+2*(C20-C22)/r2);
            grad->dz = pos.z * Moverr3 * (1+add-4*C20/r2);
        }
        if(hess) {
            hess->dx2  = Moverr3 * ( 1 - 3*X2/r2 + add*(1-7*X2/r2) + 2*(C20+C22)/r2*(1-10*X2/r2) );
            hess->dy2  = Moverr3 * ( 1 - 3*Y2/r2 + add*(1-7*Y2/r2) + 2*(C20-C22)/r2*(1-10*Y2/r2) );
            hess->dz2  = Moverr3 * ( 1 - 3*Z2/r2 + add*(1-7*Z2/r2) - 4*C20/r2*(1-10*Z2/r2) );
            hess->dxdy = Moverr3 * pos.x*pos.y/r2 * ( - 3 - add*7 - 20*C20/r2 );
            hess->dydz = Moverr3 * pos.y*pos.z/r2 * ( - 3 - add*7 + 10*(C20+C22)/r2 );
            hess->dxdz = Moverr3 * pos.z*pos.x/r2 * ( - 3 - add*7 + 10*(C20-C22)/r2 );
        }
        return;
    }
    double Wcurr[20];  // temp.coefs for lambda>0 if needed
    const double *W;   // coefs used in computation (either pre-computed or temp.)
    if(m2>1) {
        FerrersLambdaRootFinder fnc(pos.x, pos.y, pos.z, a, b, c);
        double lambda = math::findRoot(fnc, math::ScalingSemiInf(), ACCURACY_ROOT);
        computeW(lambda, Wcurr);
        W = Wcurr;
    } else 
        W=W0;  // use pre-computed coefs inside the model
    if(potential) {
        *potential = -M_PI/3 * rho0 * a*b*c * ( W[0] - 6*W[10]*X2*Y2*Z2
            + X2 * ( X2*(3*W[7]-X2*W[17]) + 3*Y2*(2*W[4]-Y2*W[11]-X2*W[14]) - 3*W[1] )
            + Y2 * ( Y2*(3*W[8]-Y2*W[18]) + 3*Z2*(2*W[5]-Z2*W[12]-Y2*W[15]) - 3*W[2] )
            + Z2 * ( Z2*(3*W[9]-Z2*W[19]) + 3*X2*(2*W[6]-X2*W[13]-Z2*W[16]) - 3*W[3] ) );
    }
    double mult = 2*M_PI*rho0*a*b*c;
    if(grad) {
        grad->dx = pos.x*mult*(W[1] +
            X2*(X2*W[17]+2*Y2*W[14]-2*W[7]) +
            Y2*(Y2*W[11]+2*Z2*W[10]-2*W[4]) +
            Z2*(Z2*W[16]+2*X2*W[13]-2*W[6]) );
        grad->dy = pos.y*mult*(W[2] +
            X2*(X2*W[14]+2*Y2*W[11]-2*W[4]) +
            Y2*(Y2*W[18]+2*Z2*W[15]-2*W[8]) +
            Z2*(Z2*W[12]+2*X2*W[10]-2*W[5]) );
        grad->dz = pos.z*mult*(W[3] +
            X2*(X2*W[13]+2*Y2*W[10]-2*W[6]) +
            Y2*(Y2*W[15]+2*Z2*W[12]-2*W[5]) +
            Z2*(Z2*W[19]+2*X2*W[16]-2*W[9]) );
    }
    if(hess) {
        hess->dx2 = mult*(W[1] + 
            X2*(5*X2*W[17]+6*Y2*W[14]-6*W[7]) +
            Y2*(  Y2*W[11]+2*Z2*W[10]-2*W[4]) +
            Z2*(  Z2*W[16]+6*X2*W[13]-2*W[6]) );
        hess->dy2 = mult*(W[2] +
            X2*(  X2*W[14]+6*Y2*W[11]-2*W[4]) +
            Y2*(5*Y2*W[18]+6*Z2*W[15]-6*W[8]) +
            Z2*(  Z2*W[12]+2*X2*W[10]-2*W[5]) );
        hess->dz2 = mult*(W[3] +
            X2*(  X2*W[13]+2*Y2*W[10]-2*W[6]) +
            Y2*(  Y2*W[15]+6*Z2*W[12]-2*W[5]) +
            Z2*(5*Z2*W[19]+6*X2*W[16]-6*W[9]) );
        hess->dxdy = mult*4*pos.x*pos.y*(X2*W[14]+Y2*W[11]+Z2*W[10]-W[4]);
        hess->dydz = mult*4*pos.y*pos.z*(X2*W[10]+Y2*W[15]+Z2*W[12]-W[5]);
        hess->dxdz = mult*4*pos.z*pos.x*(X2*W[13]+Y2*W[10]+Z2*W[16]-W[6]);
    }
}

void Ferrers::computeW(double lambda, double W[20]) const
{
    double ab2 = (a*a-b*b), ab = sqrt(ab2);
    double ac2 = (a*a-c*c), ac = sqrt(ac2);
    double bc2 = (b*b-c*c);
    double denom = sqrt((lambda+a*a)*(lambda+b*b)*(lambda+c*c));
    double argphi = asin(sqrt(ac2/(a*a+lambda)));
    double E = math::ellintE(argphi, ab/ac);
    double F = math::ellintF(argphi, ab/ac);
    W[0] = 2*F/ac;   // W_000
    W[1] = 2*(F-E)/(ab2*ac);  // W_100
    W[3] = 2/bc2*(b*b+lambda)/denom - 2*E/bc2/ac;  // W_001
    W[2] = 2/denom - W[1] - W[3]; // W_010
    W[4] = (W[2] - W[1])/ab2;     // W_110
    W[5] = (W[3] - W[2])/bc2;     // W_011
    W[6] =-(W[1] - W[3])/ac2;     // W_101
    W[7] = (2/denom/(a*a+lambda) - W[4] - W[6])/3;  // W_200
    W[8] = (2/denom/(b*b+lambda) - W[5] - W[4])/3;  // W_020
    W[9] = (2/denom/(c*c+lambda) - W[6] - W[5])/3;  // W_002
    W[10]=-(W[4] - W[5])/ac2;     // W_111
    W[11]= (W[8] - W[4])/ab2;     // W_120
    W[12]= (W[9] - W[5])/bc2;     // W_012
    W[13]=-(W[7] - W[6])/ac2;     // W_201
    W[14]= (W[4] - W[7])/ab2;     // W_210  !! inverse sign w.r.t. Pfenniger'1984 paper,
    W[15]= (W[5] - W[8])/bc2;     // W_021  !! but matches his fortran code
    W[16]=-(W[6] - W[9])/ac2;     // W_102  !! --"--
    W[17]= (2/denom/pow_2(a*a+lambda) - W[14] - W[13])/5;  // W_300
    W[18]= (2/denom/pow_2(b*b+lambda) - W[15] - W[11])/5;  // W_030
    W[19]= (2/denom/pow_2(c*c+lambda) - W[16] - W[12])/5;  // W_003
}

}  // namespace potential
