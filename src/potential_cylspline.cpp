#include "potential_cylspline.h"
#include <cmath>
#include <cassert>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_sf_legendre.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_hyperg.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_trig.h>
#include <gsl/gsl_sf_ellint.h>
#include <gsl/gsl_fit.h>
#include <gsl/gsl_multifit.h>
#include "utils.h"
#ifdef HAVE_CUBATURE
#include <cubature.h>
#endif

namespace potential {
#if 0
//-------- CylSpline section --------//
// faster evaluation of hypergeometric function by spline approximation
#define HYPERGEOM_APPROX
#ifdef HYPERGEOM_APPROX
gsl_spline* init_spline_hypergeom(double m)
{
    const double ymin=log(1e-6), ymax=log(100.0), ystepinit=1;
    const size_t numpt=static_cast<size_t>((ymax-ymin)/ystepinit);
    vectord yval(numpt), fval(numpt);
    for(size_t i=0; i<numpt; i++) {
        double y=ymin+ystepinit*i;
        double x=pow_2(exp(y)+1);
        double val=x*(gsl_sf_hyperg_2F1(1+m/2, 0.5+m/2, 1.5+m, 1/x)-1);
        yval[i]=y;
        fval[i]=val;
    }
    size_t ind=1;
    while(ind<yval.size()-1) {
        // check second derivative
        double der2 = 
            ((fval[ind+1]-fval[ind])/(yval[ind+1]-yval[ind]) - 
            ( fval[ind]-fval[ind-1])/(yval[ind]-yval[ind-1])) / 
            ((yval[ind+1]-yval[ind-1]) * fval[ind]);
        if( yval[ind]-yval[ind-1]>0.1 && 
            der2*(yval[ind+1]-yval[ind])*pow_2(yval[ind]-yval[ind-1])*(yval[ind+1]-yval[ind-1])>1e-3) {  // add a point in between
            double yadd=(yval[ind]+yval[ind-1])/2;
            double x=pow_2(exp(yadd)+1);
            double val=x*(gsl_sf_hyperg_2F1(1+m/2, 0.5+m/2, 1.5+m, 1/x)-1);
            yval.insert(yval.begin()+ind, yadd);
            fval.insert(fval.begin()+ind, val); 
        }
        else ind++;
    }
    gsl_spline* spl=gsl_spline_alloc(gsl_interp_cspline, yval.size());
    gsl_spline_init(spl, &(yval.front()), &(fval.front()), yval.size());
    return spl;
}
double my_hypergeom(double x, const gsl_spline* spl)
{
    /*if(x>=1) {  // shouldn't occur
        my_message("Invalid argument for hypergeometric function!");
        return 0;
    }*/
    return 1+x*gsl_spline_evalx(spl, log(1/sqrt(x)-1), NULL);
}
#endif

DirectPotential::DirectPotential(const BaseDensity* _density, size_t _mmax, ACCURACYMODE _accuracymode) : 
  density(_density), mysymmetry(_density->symmetry()), accuracymode(_accuracymode), points(NULL)
{
#ifdef HYPERGEOM_APPROX
    // initialize approximating spline for faster hypergeometric function evaluation
    spl_hyperg.assign(_mmax+1, NULL);
    for(size_t m=0; m<=_mmax; m++)
        spl_hyperg[m]=init_spline_hypergeom(m-0.5);
#endif
    if((mysymmetry & ST_AXISYMMETRIC)==ST_AXISYMMETRIC)
        return;  // no further action necessary
    // otherwise prepare interpolating splines in (R,z) for Fourier expansion of density in angle phi
    int mmax=_mmax;
    splines.assign(mmax*2+1, static_cast<interp2d_spline*>(NULL));
    double Rmin=0.001, Rmax=1000;  // arbirary values, but if they are inappropriate, it only will slowdown the computation 
    // but not deteriorate its accuracy, because the interpolation is not used outside the grid
    double delta=0.05;  // relative difference between grid nodes = log(x[n+1]/x[n]) 
    size_t numNodes = static_cast<size_t>(log(Rmax/Rmin)/delta);
    vectord grid;
    createNonuniformGrid(numNodes, Rmin, Rmax, true, &grid);
    vectord gridz(2*grid.size()-1);
    for(size_t i=0; i<grid.size(); i++) {
        gridz[grid.size()-1-i] =-grid[i];
        gridz[grid.size()-1+i] = grid[i];
    }
    vectord values(grid.size()*gridz.size());
    bool zsymmetry = (density->symmetry()&ST_PLANESYM)==ST_PLANESYM;      // whether densities at z and -z are different
    int mmin = (density->symmetry() & ST_PLANESYM)==ST_PLANESYM ? 0 :-1;  // if triaxial symmetry, do not use sine terms which correspond to m<0
    int mstep= (density->symmetry() & ST_PLANESYM)==ST_PLANESYM ? 2 : 1;  // if triaxial symmetry, use only even m
    for(int m=mmax*mmin; m<=mmax; m+=mstep) {
        splines[mmax+m] = interp2d_spline_alloc(interp2d_bicubic, grid.size(), gridz.size());
        for(size_t iR=0; iR<grid.size(); iR++)
            for(size_t iz=0; iz<grid.size(); iz++) {
                double val=computeRho_m(grid[iR], grid[iz], m);
                if(gsl_isnan(val) || gsl_isinf(val)) {
                    if(iR==0 && iz==0)  // may have a singularity at origin, substitute the infinite density with something reasonable
                        val=std::max<double>(computeRho_m(grid[1], grid[0], m), computeRho_m(grid[0], grid[1], m));
                    else val=0;
                }
                values[INDEX_2D(iR, grid.size()-1+iz, grid.size(), gridz.size())] = val;
                if(!zsymmetry && iz>0) {
                    val=computeRho_m(grid[iR], -grid[iz], m);
                    if(gsl_isnan(val) || gsl_isinf(val)) val=0;  // don't let rubbish in
                }
                values[INDEX_2D(iR, grid.size()-1-iz, grid.size(), gridz.size())] = val;
            }
        interp2d_spline_init(splines[mmax+m], &(grid.front()), &(gridz.front()), &(values.front()), grid.size(), gridz.size());
    }
}

DirectPotential::DirectPotential(const CPointMassSet<double> *_points, size_t _mmax, SYMMETRYTYPE sym) :
  density(NULL), mysymmetry(sym), accuracymode(AM_FAST), points(_points) 
{
#ifdef HYPERGEOM_APPROX
    spl_hyperg.assign(_mmax+1, NULL);
    for(size_t m=0; m<=_mmax; m++)
        spl_hyperg[m]=init_spline_hypergeom(m-0.5);
#endif
};

DirectPotential::~DirectPotential()
{
    for(size_t m=0; m<splines.size(); m++)
        if(splines[m]!=NULL)
            interp2d_spline_free(splines[m]);
    for(size_t m=0; m<spl_hyperg.size(); m++)
        if(spl_hyperg[m]!=NULL)
            gsl_spline_free(spl_hyperg[m]);
}

double DirectPotential::totalMass() const
{
    if(density!=NULL) return density->totalMass();
    if(points!=NULL) {
        double mass=0;
        for(CPointMassSet<double>::Type::const_iterator pt=points->data.begin(); pt!=points->data.end(); pt++) 
            mass+=pt->second;
        return mass;
    }
    return 0;  // shouldn't reach here
}

double DirectPotential::Mass(const double r) const
{
    if(density!=NULL) return density->Mass(r);
    if(points!=NULL) {
        double mass=0;
        for(CPointMassSet<double>::Type::const_iterator pt=points->data.begin(); pt!=points->data.end(); pt++) 
            if(pow_2(pt->first.Pos[0])+pow_2(pt->first.Pos[1])+pow_2(pt->first.Pos[2]) <= pow_2(r))
                mass+=pt->second;
        return mass;
    }
    return 0;  // shouldn't reach here
}

double DirectPotential::Rho(double X, double Y, double Z, double) const
{
    if(density==NULL) return 0;  // not applicable in discrete point set mode
    if(splines.size()==0)  // no interpolation
        return density->Rho(X,Y,Z); 
    else {
        double R=sqrt(X*X+Y*Y);
        double phi=atan2(Y, X);
        double val=0;
        int mmax=splines.size()/2;
        for(int m=-mmax; m<=mmax; m++)
            val+=Rho_m(R, Z, m) * (m>=0 ? cos(m*phi) : sin(-m*phi));
        return std::max<double>(val, 0);
    }
}

double DirectPotential::Rho_m(double R, double z, int m) const
{
    if(splines.size()==0) {
        if(m==0 && density!=NULL) return density->Rho(R,0,z);
        else return 0;
    }
    size_t mmax=splines.size()/2;
    if(splines[mmax+m]==NULL) return 0;
    if( R>splines[mmax+m]->interp_object.xmax || (fabs(z)+R)<splines[mmax+m]->xarr[1]*2 || 
        z<splines[mmax+m]->interp_object.ymin || z>splines[mmax+m]->interp_object.ymax)
        return computeRho_m(R, z, m);  // outside interpolating grid -- compute directly by integration
    else 
        return interp2d_spline_eval(splines[mmax+m], R, z, NULL, NULL);
}

/// \cond INTERNAL_DOCS
struct DirectPotentialParam {
    const BaseDensity* density;
    const DirectPotential* potential;
    double R, z;   // point at which the potential is evaluated
    int m;         // azimuthal harmonic (m>=0 correspond to cos(m*phi), m<0 -- to sin(|m|*phi)
    double R1, z1; // point at which the density is taken
    bool zfirst;   // determines the order of integration (R,z or z,R)
    gsl_integration_workspace* ws; // workspace for adaptive computation of the inner integral
    gsl_spline* spl_hypergeom;     // spline approximation for faster evaluation of hypergeometric function
    int num_eval;  // count the number of function evaluations
};
/// \endcond
double integrandPotentialDirectDensity(double phi, void* param)
{
    double mult=((DirectPotentialParam*)param)->m==0 ? 1 : ((DirectPotentialParam*)param)->m>0 ? 
        2*cos(((DirectPotentialParam*)param)->m*phi) : 2*sin(-((DirectPotentialParam*)param)->m*phi);
    return mult*((DirectPotentialParam*)param)->density->Rho( 
        ((DirectPotentialParam*)param)->R*cos(phi), ((DirectPotentialParam*)param)->R*sin(phi), ((DirectPotentialParam*)param)->z);
}

double DirectPotential::computeRho_m(double R, double z, int m) const
{
    gsl_function F;
    DirectPotentialParam params;
    F.function=&integrandPotentialDirectDensity;
    F.params=&params;
    params.density=density;
    params.R=R;
    params.z=z;
    params.m=m;
    double result, error;
    size_t neval;
    double phimax=(density->symmetry() & ST_PLANESYM)==ST_PLANESYM ? M_PI_2 : 2*M_PI;
    gsl_integration_qng(&F, 0, phimax, 0, EPSREL_DENSITY_INT, &result, &error, &neval);
    return result/phimax;
}

#define Q_HYPERG 1
/// Legendre function Q(m,x) expressed through Gauss hypergeometric function 
double LegendreQ(double m, double x, const gsl_spline* spl)
{
#if Q_HYPERG
#ifdef HYPERGEOM_APPROX
    double val=my_hypergeom(1/(x*x), spl);
#else
    double val= gsl_sf_hyperg_2F1(1+m/2, 0.5+m/2, 1.5+m, 1/(x*x));// : (1+(0.345+0.25*m)/pow_2(x));
#endif
    if(gsl_isnan(val)) val=1;  // not quite correct, but allows to continue computation
    val*=pow(2*x,-m-1);
    return val;
#else
    double mu=sqrt(2.0/(1+x));
    double m1,m0=mu*gsl_sf_ellint_Kcomp(mu, GSL_PREC_SINGLE);
    if(m==-0.5) return m0;
    if(m>=0.5) m1=x*m0-(1+x)*mu*gsl_sf_ellint_Ecomp(mu, GSL_PREC_SINGLE);
    if(m==0.5) return m1;
    // otherwise use recurrence relations from Cohl&Tohline1999
    double mcurr=2.0;
    do{
        double m2=4*(mcurr-1)/(2*mcurr-1)*x*m1 - (2*mcurr-3)/(2*mcurr-1)*m0;
        if(mcurr==m+0.5)
            return m2;
        m0=m1; m1=m2; mcurr+=1;
    } while(mcurr<100);  // avoid infinite loop
    return 0;  // shouldn't happen?
#endif
}
/// $\int_0^\infinity J_m(a*x)*J_m(b*x)*exp(-|c|*x) dx / (\sqrt(\pi)*\Gamma(m+1/2)/\Gamma(m+1))$
double IntBessel(double m, double a, double b, double c, const gsl_spline* spl, double eps2=0)
{
    if(fabs(a)<1e-10 || fabs(b)<1e-10)
        return m==0 ? 1/(M_PI*sqrt(a*a+b*b+c*c+eps2)) : 0;
    else
        return LegendreQ(m-0.5, (a*a+b*b+c*c+eps2)/(2*a*b), spl)/(M_PI*sqrt(a*b));
}

inline double integrandPotentialDirect(double R1, double z1, void* params)
{
    double val = ((DirectPotentialParam*)params)->potential->Rho_m(R1, z1, ((DirectPotentialParam*)params)->m) * 
        -2*M_PI*R1 * IntBessel( abs(((DirectPotentialParam*)params)->m), 
        ((DirectPotentialParam*)params)->R, R1, ((DirectPotentialParam*)params)->z-z1, 
        ((DirectPotentialParam*)params)->spl_hypergeom );
    z1*=-1;
    val += ((DirectPotentialParam*)params)->potential->Rho_m(R1, z1, ((DirectPotentialParam*)params)->m) * 
        -2*M_PI*R1 * IntBessel( abs(((DirectPotentialParam*)params)->m), 
        ((DirectPotentialParam*)params)->R, R1, ((DirectPotentialParam*)params)->z-z1, 
        ((DirectPotentialParam*)params)->spl_hypergeom );
    //((DirectPotentialParam*)params)->num_eval++;
    return val;
}

#ifdef HAVE_CUBATURE
int integrandPotentialDirectCubature(unsigned int /*ndim*/, const double Rz[],
    void* params, unsigned int /*fdim*/, double* output)
{
    *output= Rz[0]>=1. || Rz[1]>=1. || (  // scaled coords point at infinity
        ((DirectPotentialParam*)params)->R==Rz[0] &&
        ((DirectPotentialParam*)params)->z==Rz[1] ) ? 0. :
        integrandPotentialDirect(Rz[0]/(1-Rz[0]), Rz[1]/(1-Rz[1]), params) / 
        pow_2((1-Rz[0])*(1-Rz[1]));  // jacobian of scaled coord transformation
    return 0; // no error
}
#endif

double integrandPotentialDirectVar2(double var, void* params)
{
    double R1,z1;
    if(((DirectPotentialParam*)params)->zfirst) {
        R1=var/(1-var);
        z1=((DirectPotentialParam*)params)->z1;
    } else {
        z1=var/(1-var);
        R1=((DirectPotentialParam*)params)->R1;
    } 
    return integrandPotentialDirect(R1, z1, params)/pow_2(1-var);
}

double integrandPotentialDirectVar1(double var, void* params)
{
    gsl_function F;
    F.function=&integrandPotentialDirectVar2;
    F.params=params;
    if(((DirectPotentialParam*)params)->zfirst) 
        ((DirectPotentialParam*)params)->z1=var/(1-var);
    else
        ((DirectPotentialParam*)params)->R1=var/(1-var);
    double result, errr;
    size_t neval;
    if(((DirectPotentialParam*)params)->ws==NULL)
        gsl_integration_qng(&F, 0, 1, 0, EPSREL_POTENTIAL_INT, &result, &errr, &neval);
    else 
        gsl_integration_qags(&F, 0, 1, 0, EPSREL_POTENTIAL_INT, 1000, ((DirectPotentialParam*)params)->ws, &result, &errr);
    return result/pow_2(1-var);
}

double DirectPotential::Phi_m(double R, double Z, int m) const
{
    DirectPotentialParam params;
#ifdef HYPERGEOM_APPROX
    params.spl_hypergeom=spl_hyperg[abs(m)];
#endif
    if(density==NULL) {  // invoked in the discrete point set mode
        if(points==NULL) return 0;  // shouldn't normally happen
        double val=0;
        for(CPointMassSet<double>::Type::const_iterator pt=points->data.begin(); pt!=points->data.end(); pt++) {
            double R1=sqrt(pow_2(pt->first.Pos[0])+pow_2(pt->first.Pos[1]));
            double phi1=atan2(pt->first.Pos[1], pt->first.Pos[0]);
            double Z1=pt->first.Pos[2];
            double eps2=0;
            double val1=IntBessel(abs(m), R, R1, Z-Z1, params.spl_hypergeom,eps2);
            if((mysymmetry & ST_PLANESYM)==ST_PLANESYM)   // add symmetric contribution from -Z
                val1 = (val1 + IntBessel(abs(m), R, R1, Z+Z1, params.spl_hypergeom))/2;
            if(!gsl_isnan(val1) && gsl_finite(val1))
                val += pt->second * val1 /* R1*/ * (m==0 ? 1 : m>0 ? 2*cos(m*phi1) : 2*sin(-m*phi1) );
        }
#if Q_HYPERG
        val*=M_SQRTPI*gsl_sf_gamma(abs(m)+0.5)/gsl_sf_gamma(abs(m)+1);
#endif
        return -val;
    }
    // otherwise invoked in the smooth density profile mode
    int mmax=splines.size()/2;
    if(splines.size()>0 && splines[mmax+m]==NULL) return 0;  // using splines for m-components of density but it is identically zero at this m
    gsl_function F;
    F.function=&integrandPotentialDirectVar1;
    F.params=&params;
    params.potential=this;
    params.R=R;
    params.z=Z;
    params.zfirst=true;
    params.m=m;
    params.num_eval=0;
    double result, error;
#ifdef HAVE_CUBATURE
    double Rzmin[2]={0.,0.}, Rzmax[2]={1.,1.}; // integration box in scaled coords
    hcubature(1, &integrandPotentialDirectCubature, &params, 2, Rzmin, Rzmax, 65536/*max_eval*/, 
        EPSABS_POTENTIAL_INT, EPSREL_POTENTIAL_INT, ERROR_L1/*ignored*/, &result, &error);
#else
    size_t neval;
    if(accuracymode!=AM_FAST && abs(m)<=4)  // more precise computation only for low-m terms
        params.ws=gsl_integration_workspace_alloc(1000);
    else
        params.ws=NULL;
    if(accuracymode!=AM_SLOW)
        gsl_integration_qng(&F, 0., 1., 0, EPSREL_POTENTIAL_INT, &result, &error, &neval);
    else {
        gsl_integration_workspace* ws=gsl_integration_workspace_alloc(1000);
        gsl_integration_qags(&F, 0., 1., 0, EPSREL_POTENTIAL_INT, 1000, ws, &result, &error);
        gsl_integration_workspace_free(ws);
    }
    if(params.ws!=NULL)
        gsl_integration_workspace_free(params.ws);
#endif
#if Q_HYPERG
    result*=M_SQRTPI*gsl_sf_gamma(abs(params.m)+0.5)/gsl_sf_gamma(abs(params.m)+1);
#endif
    return result;
};

double DirectPotential::Phi(double X, double Y, double Z, double /*t*/) const
{
    double R=sqrt(X*X+Y*Y);
    double phi=atan2(Y, X);
    double val=0;
    int mmax=splines.size()/2;
    for(int m=-mmax; m<=mmax; m++)
        val+=Phi_m(R, Z, m) * (m>=0 ? cos(m*phi) : sin(-m*phi));
    return val;
}

//----------------------------------------------------------------------------//
// Cylindrical spline potential 

CylSplineExp::CylSplineExp(size_t Ncoefs_R, size_t Ncoefs_z, size_t Ncoefs_phi, 
    const BaseDensity* density, double radius_min, double radius_max, double z_min, double z_max)
{
    const BasePotential* pot_tmp=new DirectPotential(density, Ncoefs_phi, DirectPotential::AM_MEDIUM);
    initPot(Ncoefs_R, Ncoefs_z, Ncoefs_phi, pot_tmp, radius_min, radius_max, z_min, z_max);
    delete pot_tmp;
}

CylSplineExp::CylSplineExp(size_t Ncoefs_R, size_t Ncoefs_z, size_t Ncoefs_phi, 
    const BasePotential* potential, double radius_min, double radius_max, double z_min, double z_max)
{
    initPot(Ncoefs_R, Ncoefs_z, Ncoefs_phi, potential, radius_min, radius_max, z_min, z_max);
}

CylSplineExp::CylSplineExp(size_t Ncoefs_R, size_t Ncoefs_z, size_t Ncoefs_phi, 
    const CPointMassSet<double> &points, SYMMETRYTYPE _sym, double radius_min, double radius_max, double z_min, double z_max)
{
    mysymmetry=_sym;
    if(Ncoefs_phi==0)
        mysymmetry=(SYMMETRYTYPE)(mysymmetry | ST_ZROTSYM);
    const BasePotential* pot_tmp=new DirectPotential(&points, Ncoefs_phi, mysymmetry);
    initPot(Ncoefs_R, Ncoefs_z, Ncoefs_phi, pot_tmp, radius_min, radius_max, z_min, z_max);
    delete pot_tmp;
}

CylSplineExp::CylSplineExp(const vectord &gridR, const vectord& gridz, const std::vector< vectord > &coefs)
{
    if(coefs.size()==0 || coefs.size()%2!=1 || coefs[coefs.size()/2].size()!=gridR.size()*gridz.size()) {
        my_error(FUNCNAME, "Invalid parameters in the constructor");
        initDefault();
    } else {
        grid_R=gridR;
        grid_z=gridz;
        initSplines(coefs);
        // check symmetry
        int mmax=static_cast<int>(splines.size()/2);
        mysymmetry=mmax==0 ? ST_AXISYMMETRIC : ST_TRIAXIAL;
        for(int m=-mmax; m<=mmax; m++)
            if(splines[m+mmax]!=NULL) {
                if(m<0 || (m>0 && m%2==1))
                    mysymmetry = ST_NONE;//(SYMMETRYTYPE)(mysymmetry & ~ST_PLANESYM & ~ST_ZROTSYM);
            }
    }
}

CylSplineExp::~CylSplineExp()
{
    for(size_t k=0; k<splines.size(); k++)
        if(splines[k]!=NULL)
            interp2d_spline_free(splines[k]);
}

double integrandPotAvgPhi(double phi, void* param)
{
    int m=((BasePotentialParam*)param)->m;
    return ((BasePotentialParam*)param)->P->Phi( 
        ((BasePotentialParam*)param)->R*cos(phi), ((BasePotentialParam*)param)->R*sin(phi), ((BasePotentialParam*)param)->Z) *
    (m==0 ? 1 : m>0 ? 2*cos(m*phi) : 2*sin(-m*phi));
}

double CylSplineExp::computePhi_m(double R, double z, int m, const BasePotential* potential) const
{
    if(potential->PotentialType()==PT_DIRECT) {
        return ((const DirectPotential*)potential)->Phi_m(R, z, m);
    }
    else {  // compute azimuthal Fourier harmonic coefficient for the given m
        if(R==0 && m!=0) return 0;
        gsl_function F;
        BasePotentialParam PP;
        PP.P=potential;
        PP.R=R;
        PP.Z=z;
        PP.m=m;
        F.function=integrandPotAvgPhi;
        F.params=&PP;
        double result, error;
        size_t neval;
        double phimax=(potential->symmetry() & ST_PLANESYM)==ST_PLANESYM ? M_PI_2 : 2*M_PI;
        gsl_integration_qng(&F, 0, phimax, 0, EPSREL_POTENTIAL_INT, &result, &error, &neval);
        return result/phimax;
    }
}

void CylSplineExp::initPot(size_t _Ncoefs_R, size_t _Ncoefs_z, size_t _Ncoefs_phi, 
    const BasePotential* potential, double radius_min, double radius_max, double z_min, double z_max)
{
    mysymmetry=potential->symmetry();
    bool zsymmetry= (mysymmetry & ST_PLANESYM)==ST_PLANESYM;  // whether we need to compute potential at z<0 independently from z>0
    int mmax = (mysymmetry & ST_AXISYMMETRIC) == ST_AXISYMMETRIC ? 0 : _Ncoefs_phi;
    int mmin = (mysymmetry & ST_PLANESYM)==ST_PLANESYM ? 0 :-1;  // if triaxial symmetry, do not use sine terms which correspond to m<0
    int mstep= (mysymmetry & ST_PLANESYM)==ST_PLANESYM ? 2 : 1;  // if triaxial symmetry, use only even m
    if(radius_max==0 || radius_min==0) {
        double totalmass=potential->totalMass();
        if(radius_max==0)
            radius_max=potential->getRadiusByMass(totalmass*(1-1.0/(_Ncoefs_R*_Ncoefs_z)));
        if(radius_max<=0) radius_max=100; // arbitrary!
        if(radius_min==0) 
            radius_min=std::min<double>(radius_max/_Ncoefs_R, potential->getRadiusByMass(totalmass/(_Ncoefs_R*_Ncoefs_z)));
        if(radius_min<=0) radius_min=radius_max/_Ncoefs_R;
    }
    vectord splineRad;
    createNonuniformGrid(_Ncoefs_R, radius_min, radius_max, true, &splineRad);
    grid_R=splineRad;
    if(z_max==0) z_max=radius_max;
    if(z_min==0) z_min=radius_min;
    z_min=std::min<double>(z_min, z_max/_Ncoefs_z);
    createNonuniformGrid(_Ncoefs_z, z_min, z_max, true, &splineRad);
    grid_z.assign(2*_Ncoefs_z-1,0);
    for(size_t i=0; i<_Ncoefs_z; i++) {
        grid_z[_Ncoefs_z-1-i] =-splineRad[i];
        grid_z[_Ncoefs_z-1+i] = splineRad[i];
    }
    size_t Ncoefs_R=grid_R.size();
    size_t Ncoefs_z=grid_z.size();
    std::vector<vectord> coefs(2*mmax+1);
    for(int m=mmax*mmin; m<=mmax; m+=mstep) {
        coefs[mmax+m].assign(Ncoefs_R*Ncoefs_z,0);
    }
    bool correct=true;
    // for some unknown reason, switching on the OpenMP parallelization makes the results 
    // of computation of m=4 coef irreproducible (adds a negligible random error). 
    // Assumed to be unimportant, thus OpenMP is enabled...
#pragma omp parallel for
    for(int iR=0; iR<static_cast<int>(Ncoefs_R); iR++) {
        for(size_t iz=0; iz<=Ncoefs_z/2; iz++) {
            for(int m=mmax*mmin; m<=mmax; m+=mstep) {
                double val=computePhi_m(grid_R[iR], grid_z[Ncoefs_z/2+iz], m, potential);
                if(!gsl_finite(val)) {
#ifdef DEBUGPRINT
                    my_message(FUNCNAME, "Error in computing potential at R="+convertToString(grid_R[iR])+
                        ", z="+convertToString(grid_z[iz+Ncoefs_z/2])+", m="+convertToString(m));
#endif
                    val=0;
                    correct=false;
                }
                coefs[mmax+m][(Ncoefs_z/2+iz)*Ncoefs_R+iR] = val;
                if(!zsymmetry && iz>0)   // no symmetry about x-y plane
                    val=computePhi_m(grid_R[iR], grid_z[Ncoefs_z/2-iz], m, potential);  // compute potential at -z independently
                coefs[mmax+m][(Ncoefs_z/2-iz)*Ncoefs_R+iR] = val;
            }
        }
#ifdef DEBUGPRINT
        my_message(FUNCNAME, "R="+convertToString(grid_R[iR]));
#endif
    }
    if(!correct) 
        my_message(FUNCNAME, "Errors in computing the potential are replaced by zeros");
    initSplines(coefs);
}

void CylSplineExp::initSplines(const std::vector< vectord > &coefs)
{
    size_t Ncoefs_R=grid_R.size();
    size_t Ncoefs_z=grid_z.size();
    int mmax=coefs.size()/2;
    assert(coefs[mmax].size()==Ncoefs_R*Ncoefs_z);  // check that at least m=0 coefficients are present
    // compute multipole coefficients for extrapolating the potential and forces beyond the grid,
    // by fitting them to the potential at the grid boundary
    C00=C20=C22=C40=0;
    bool fitm2=mmax>=2 && coefs[mmax+2].size()==Ncoefs_R*Ncoefs_z;  // whether to fit m=2
    size_t npointsboundary=2*(Ncoefs_R-1)+Ncoefs_z;
    gsl_matrix* X0=gsl_matrix_alloc(npointsboundary, 3);  // matrix of coefficients  for m=0
    gsl_vector* Y0=gsl_vector_alloc(npointsboundary);     // vector of r.h.s. values for m=0
    gsl_vector* W0=gsl_vector_alloc(npointsboundary);     // vector of weights
    vectord X2(npointsboundary);     // vector of coefficients  for m=2
    vectord Y2(npointsboundary);     // vector of r.h.s. values for m=2
    for(size_t i=0; i<npointsboundary; i++) {
        size_t iR=i<2*Ncoefs_R ? i/2 : Ncoefs_R-1;
        size_t iz=i<2*Ncoefs_R ? (i%2)*(Ncoefs_z-1) : i-2*Ncoefs_R+1;
        double R=grid_R[iR];
        double z=grid_z[iz];
        double oneoverr=1/sqrt(R*R+z*z);
        gsl_vector_set(Y0, i, coefs[mmax][iz*Ncoefs_R+iR]);
        gsl_matrix_set(X0, i, 0, oneoverr);
        gsl_matrix_set(X0, i, 1, pow(oneoverr,5.0)*(2*z*z-R*R));
        gsl_matrix_set(X0, i, 2, pow(oneoverr,9.0)*(8*pow(z,4.0)-24*z*z*R*R+3*pow(R,4.0)) );
        // weight proportionally to the value of potential itself (so that we minimize sum of squares of relative differences)
        gsl_vector_set(W0, i, 1.0/pow_2(coefs[mmax][iz*Ncoefs_R+iR]));
        if(fitm2) {
            X2[i]=R*R*pow(oneoverr,5.0);
            Y2[i]=coefs[mmax+2][iz*Ncoefs_R+iR];
        }
    }
    // fit m=0 by three parameters
    gsl_vector* fit=gsl_vector_alloc(3);
    gsl_matrix* cov=gsl_matrix_alloc(3,3);
    double chisq;
    gsl_multifit_linear_workspace* ws=gsl_multifit_linear_alloc(npointsboundary, 3);
    if(gsl_multifit_wlinear(X0, W0, Y0, fit, cov, &chisq, ws) == GSL_SUCCESS) {
        C00=gsl_vector_get(fit, 0);  // C00 ~= -Mtotal
        C20=gsl_vector_get(fit, 1);
        C40=gsl_vector_get(fit, 2);
    }
    gsl_multifit_linear_free(ws);
    gsl_vector_free(fit);
    gsl_matrix_free(cov);
    // fit m=2 if necessary
    if(fitm2) {
        double dummy1, dummy2;
        if(gsl_fit_mul(&(X2.front()), 1, &(Y2.front()), 1, npointsboundary, &C22, &dummy1, &dummy2) != GSL_SUCCESS)
            C22=0;
    }
    // assign Rscale so that it approximately equals -Mtotal/Phi(r=0)
    Rscale=C00/coefs[mmax][(Ncoefs_z/2)*Ncoefs_R];
    if(Rscale<=0) Rscale=std::min<double>(grid_R.back(), grid_z.back())*0.5; // shouldn't occur?
#ifdef DEBUGPRINT
    my_message(FUNCNAME,  "Rscale="+convertToString(Rscale)+", C00="+convertToString(C00)+", C20="+convertToString(C20)+", C22="+convertToString(C22)+", C40="+convertToString(C40));
#endif

    const size_t N_ADD=4;
    vectord grid_Rscaled(Ncoefs_R+N_ADD);
    vectord grid_zscaled(Ncoefs_z);
    for(size_t i=0; i<Ncoefs_R; i++) {
        grid_Rscaled[i+N_ADD] = log(1+grid_R[i]/Rscale);
        if(i>=1 && i<=N_ADD)   // add three points symmetric about 0, to ensure that spline derivative is (close to) zero at R=0
            grid_Rscaled[N_ADD-i] = -grid_Rscaled[N_ADD+i];
    }
    for(size_t i=0; i<Ncoefs_z; i++) {
        grid_zscaled[i] = log(1+fabs(grid_z[i])/Rscale)*(grid_z[i]>=0?1:-1);
    }
    splines.assign(coefs.size(), NULL);
    vectord values((Ncoefs_R+N_ADD)*Ncoefs_z);
    for(size_t m=0; m<coefs.size(); m++) {
        if(coefs[m].size()!=Ncoefs_R*Ncoefs_z) 
            continue;
        bool allzero=true;
        for(size_t iR=0; iR<Ncoefs_R; iR++) {
            for(size_t iz=0; iz<Ncoefs_z; iz++) {
                double scaling=sqrt(pow_2(Rscale)+pow_2(grid_R[iR])+pow_2(grid_z[iz]));
                double val=coefs[m][iz*Ncoefs_R+iR] * scaling;
                values[INDEX_2D(iR+N_ADD, iz, Ncoefs_R+N_ADD, Ncoefs_z)] = val;
                if(iR>=1&&iR<=N_ADD)
                    values[INDEX_2D(N_ADD-iR, iz, Ncoefs_R+N_ADD, Ncoefs_z)] = val;
                if(val!=0) allzero=false;
            }
        }
        if(allzero)
            continue;
        splines[m]=interp2d_spline_alloc(interp2d_bicubic, Ncoefs_R+N_ADD, Ncoefs_z);
        if(splines[m]==NULL) { 
            my_error(FUNCNAME, "Error allocating memory for splines");
            initDefault();
            return;
        }
        if(interp2d_spline_init(splines[m], &(grid_Rscaled.front()), &(grid_zscaled.front()), 
            &(values.front()), Ncoefs_R+N_ADD, Ncoefs_z) != GSL_SUCCESS)  {
            my_error(FUNCNAME, "Can't initialize splines");
            initDefault();
            return;
        }
    }
}

void CylSplineExp::initDefault()
{
    Rscale=1.0;
    C00=C20=C22=0;
    mysymmetry=ST_SPHERICAL;
    grid_R.resize(2); grid_R[0]=0; grid_R[1]=1;
    grid_z.resize(3); grid_z[0]=-1; grid_z[1]=0; grid_z[2]=1;
    splines.assign(1, static_cast<interp2d_spline*>(NULL));
}

BasePotential* CylSplineExp::clone() const 
{   // need to duplicate spline arrays explicitly; everything else is copied automatically
    CylSplineExp* newpot = new CylSplineExp(*this); 
    for(size_t i=0; i<splines.size(); i++)
        if(splines[i]!=NULL)
        {
            interp2d_spline *spline = interp2d_spline_alloc(interp2d_bicubic, 
                splines[i]->interp_object.xsize, splines[i]->interp_object.ysize);
            interp2d_spline_init(spline, splines[i]->xarr, splines[i]->yarr, splines[i]->zarr, 
                splines[i]->interp_object.xsize, splines[i]->interp_object.ysize);
            newpot->splines[i]=spline;
        }
        else newpot->splines[i]=NULL;
    return newpot;
}

void CylSplineExp::getCoefs(vectord *gridR, vectord* gridz, std::vector< vectord > *coefs) const
{
    *gridR = grid_R;
    *gridz = grid_z;
    coefs->resize(splines.size());
    for(size_t m=0; m<splines.size(); m++)
        coefs->at(m).assign(splines[m]!=NULL ? gridz->size()*gridR->size() : 0, 0);
    for(size_t iz=0; iz<gridz->size(); iz++)
        for(size_t iR=0; iR<gridR->size(); iR++) {
            double Rscaled=log(1+gridR->at(iR)/Rscale);
            double zscaled=log(1+fabs(gridz->at(iz))/Rscale)*(gridz->at(iz)>=0?1:-1);
            for(size_t m=0; m<splines.size(); m++)
                if(splines[m]!=NULL) {
                    double scaling=sqrt(pow_2(Rscale)+pow_2(grid_R[iR])+pow_2(grid_z[iz]));
                    coefs->at(m)[iz*gridR->size()+iR] = interp2d_spline_eval(splines[m], Rscaled, zscaled, NULL, NULL)/scaling;
                }
        }
}

double CylSplineExp::Mass(const double r) const
{
    if(r<=0) return 0;
    gsl_function F;
    BaseDensityParam params;
    F.function=&getDensityRcyl;
    F.params=&params;
    params.P=this;
    params.r=r;  // upper limit of the integration
    params.mumble=grid_z.back();  // upper limit on integration in Z
    double R=std::min<double>(r, grid_R.back());
    double result, error;
    size_t neval;
    gsl_integration_qng(&F, 0, R/(1+R), 0, EPSREL_DENSITY_INT, &result, &error, &neval);
    return result;
}

double CylSplineExp::Phi(double X, double Y, double Z, double /*t*/) const
{
    double R=sqrt(X*X+Y*Y);
    if(R>=grid_R.back() || Z<=grid_z.front() || Z>=grid_z.back()) 
    { // fallback mechanism for extrapolation beyond grid definition region
        double r2=X*X+Y*Y+Z*Z;
        return (C00 + ((2*Z*Z-R*R)*C20 + (X*X-Y*Y)*C22 + (35*pow_2(Z*Z/r2)-30*Z*Z/r2+3)*C40)/pow_2(r2))/sqrt(r2);
    }
    double S=1/sqrt(pow_2(Rscale)+pow_2(R)+pow_2(Z));  // scaling
    double phi=atan2(Y,X);
    double Rscaled=log(1+R/Rscale);
    double zscaled=log(1+fabs(Z)/Rscale)*(Z>=0?1:-1);
    double val=0;
    int mmax=splines.size()/2;
    for(int m=-mmax; m<=mmax; m++)
        if(splines[m+mmax]!=NULL) {
            val += interp2d_spline_eval(splines[m+mmax], Rscaled, zscaled, NULL, NULL) * (m>=0 ? cos(m*phi) : sin(-m*phi));
        }
    return val*S;
}

double CylSplineExp::Rho(double X, double Y, double Z, double /*t*/) const
{
    double R=sqrt(X*X+Y*Y);
    if(R>=grid_R.back() || fabs(Z)>=grid_z.back())
        return 0;  // no extrapolation beyong grid 
    double phi=atan2(Y,X);
    double Rscaled=log(1+R/Rscale);
    double zscaled=log(1+fabs(Z)/Rscale)*(Z>=0?1:-1);
    double dRscaleddR=1/(Rscale+R), d2RscaleddR2=-pow_2(dRscaleddR);
    double dzscaleddz=1/(Rscale+fabs(Z)), d2zscaleddz2=-pow_2(dzscaleddz)*(Z>=0?1:-1);
    double Phi_tot=0, dPhidRscaled=0, dPhidzscaled=0, d2PhidRscaled2=0, d2Phidzscaled2=0, d2Phidphi2=0;
    int mmax=splines.size()/2;
    for(int m=-mmax; m<=mmax; m++)
        if(splines[m+mmax]!=NULL) { 
            double cosmphi=m>=0 ? cos(m*phi) : sin(-m*phi);
            double Phi_m, dPhi_m_dRscaled, dPhi_m_dzscaled, d2Phi_m_dRscaled2, d2Phi_m_dzscaled2;
            interp2d_spline_eval_all(splines[m+mmax], Rscaled, zscaled, 
                &Phi_m, &dPhi_m_dRscaled, &dPhi_m_dzscaled, &d2Phi_m_dRscaled2, NULL, &d2Phi_m_dzscaled2);
            dPhidRscaled+=dPhi_m_dRscaled * cosmphi;
            dPhidzscaled+=dPhi_m_dzscaled * cosmphi;
            d2PhidRscaled2+=d2Phi_m_dRscaled2 * cosmphi;
            d2Phidzscaled2+=d2Phi_m_dzscaled2 * cosmphi;
            Phi_tot+=Phi_m*cosmphi;
            d2Phidphi2+=Phi_m * -m*m*cosmphi;
        }
    double S=1/sqrt(pow_2(Rscale)+pow_2(R)+pow_2(Z));  // scaling
    double dSdr_over_r=-S*S*S;
    double d2Sdr2=(pow_2(Rscale)-2*(R*R+Z*Z))*dSdr_over_r*S*S;
    double ddd=R*dPhidRscaled*dRscaleddR + Z*dPhidzscaled*dzscaleddz + Phi_tot;
    double LaplacePhi = (R>0 ? dPhidRscaled*(dRscaleddR/R+d2RscaleddR2) + d2Phidphi2/R/R : 0)
        + d2PhidRscaled2*pow_2(dRscaleddR) + dPhidzscaled*d2zscaleddz2 + d2Phidzscaled2*pow_2(dzscaleddz);
    double result = S*LaplacePhi + 2*dSdr_over_r*ddd + d2Sdr2*Phi_tot;
    return std::max<double>(0, result)/(4*M_PI);
}

void CylSplineExp::Force(const double v[N_DIM], const double /*t*/, double* force, double* forceDeriv) const
{
    double X=v[0], Y=v[1], Z=v[2];
    double R2=X*X+Y*Y, R=sqrt(R2);
    if(R>=grid_R.back() || fabs(Z)>=grid_z.back()) { // fallback mechanism for extrapolation beyond grid definition region
        double X2=X*X, Y2=Y*Y, Z2=Z*Z, R2=X2+Y2, r2=R2+Z2;
        double oneoverr3=1/(r2*sqrt(r2)), oneoverr4=1/(r2*r2), oneoverr8=pow_2(oneoverr4);
        double add =5*(C20 * (2*Z2-R2) + C22 * (X2-Y2) ) * oneoverr4;
        double add4=C40*15*(21*Z2*Z2 - 14*Z2*r2 + r2*r2) * oneoverr8;
        double add4z=C40*5*(63*Z2*Z2 - 70*Z2*r2 + 15*r2*r2) * oneoverr8;
        force[0] = X*oneoverr3*(C00+add+2*(C20-C22)/r2 + add4);
        force[1] = Y*oneoverr3*(C00+add+2*(C20+C22)/r2 + add4);
        force[2] = Z*oneoverr3*(C00+add-4*C20/r2 + add4z);
        if(forceDeriv!=NULL) {
            double X2r2=X2/r2, Y2r2=Y2/r2, Z2r2=Z2/r2;
            double add7=C40*60*(1-7*Z2r2)*oneoverr4 - add4*11;
            forceDeriv[0] = oneoverr3*( C00*(1 - 3*X2r2) + add*(1-7*X2r2) + 2*(C20-C22)/r2*(1-10*X2r2) 
                                      + add4 + add7*X2r2 );
            forceDeriv[1] = oneoverr3*( C00*(1 - 3*Y2r2) + add*(1-7*Y2r2) + 2*(C20+C22)/r2*(1-10*Y2r2)
                                      + add4 + add7*Y2r2 );
            forceDeriv[2] = oneoverr3*( C00*(1 - 3*Z2r2) + add*(1-7*Z2r2) - 4*C20/r2*(1-10*Z2r2) 
                                      + add4z*(1-11*Z2r2) + C40*80*Z2r2*(7*Z2r2-5)*oneoverr4);
            double addmix = - 3*C00 - add*7;
            forceDeriv[3] = oneoverr3*X*Y/r2*(addmix - 20*C20/r2 + add7);
            addmix += -11*add4z + C40*100*(3-7*Z2r2)*oneoverr4;
            forceDeriv[4] = oneoverr3*Y*Z/r2*(addmix + 10*(C20-C22)/r2);
            forceDeriv[5] = oneoverr3*Z*X/r2*(addmix + 10*(C20+C22)/r2);
        }
        return;
    }
    double phi=atan2(Y,X);
    double Rscaled=log(1+R/Rscale);
    double zscaled=log(1+fabs(Z)/Rscale)*(Z>=0?1:-1);
    double Phi_tot=0, dPhidRscaled=0, dPhidzscaled=0, dPhidphi=0;
    double d2PhidRscaled2=0, d2PhidRscaleddzscaled=0, d2Phidzscaled2=0, d2Phidphi2=0, d2PhidRscaleddphi=0, d2Phidzscaleddphi=0;
    int mmax=splines.size()/2;
    for(int m=-mmax; m<=mmax; m++)
        if(splines[m+mmax]!=NULL) {
            double cosmphi=m>=0?cos(m*phi):sin(-m*phi);
            double sinmphi=m>=0?sin(m*phi):cos(-m*phi);
#ifdef NO_OPTIMIZATION
            double Phi_m=interp2d_spline_eval(splines[m+mmax], Rscaled, zscaled, NULL, NULL);
            double dPhi_m_dRscaled=interp2d_spline_eval_deriv_x(splines[m+mmax], Rscaled, zscaled, NULL, NULL);
            double dPhi_m_dzscaled=interp2d_spline_eval_deriv_y(splines[m+mmax], Rscaled, zscaled, NULL, NULL);
#else
            double Phi_m, dPhi_m_dRscaled, dPhi_m_dzscaled, d2Phi_m_dRscaled2, d2Phi_m_dRscaleddzscaled, d2Phi_m_dzscaled2;
            interp2d_spline_eval_all(splines[m+mmax], Rscaled, zscaled, &Phi_m, 
                &dPhi_m_dRscaled, &dPhi_m_dzscaled, &d2Phi_m_dRscaled2, &d2Phi_m_dRscaleddzscaled, &d2Phi_m_dzscaled2);
#endif
            dPhidRscaled+=dPhi_m_dRscaled*cosmphi;
            dPhidzscaled+=dPhi_m_dzscaled*cosmphi;
            Phi_tot+=Phi_m*cosmphi;
            dPhidphi+=Phi_m * -m*sinmphi;
            if(forceDeriv!=NULL)
            {
#ifdef NO_OPTIMIZATION
                double d2Phi_m_dRscaled2=interp2d_spline_eval_deriv_xx(splines[m+mmax], Rscaled, zscaled, NULL, NULL);
                double d2Phi_m_dRscaleddzscaled=interp2d_spline_eval_deriv_xy(splines[m+mmax], Rscaled, zscaled, NULL, NULL);
                double d2Phi_m_dzscaled2=interp2d_spline_eval_deriv_yy(splines[m+mmax], Rscaled, zscaled, NULL, NULL);
#endif
                d2PhidRscaled2+=d2Phi_m_dRscaled2 * cosmphi;
                d2Phidzscaled2+=d2Phi_m_dzscaled2 * cosmphi;
                d2Phidphi2+=Phi_m * -m*m*cosmphi;
                d2PhidRscaleddzscaled+=d2Phi_m_dRscaleddzscaled * cosmphi;
                d2PhidRscaleddphi+=dPhi_m_dRscaled* -m*sinmphi;
                d2Phidzscaleddphi+=dPhi_m_dzscaled* -m*sinmphi;
            }
        }
    double r2=R*R+Z*Z;
    double S=1/sqrt(pow_2(Rscale)+r2);  // scaling
    double dSdr_over_r=-S*S*S;
    double dRscaleddR=1/(Rscale+R);
    double dzscaleddz=1/(Rscale+fabs(Z));
    dPhidphi*=S;
    double dPhidR=dPhidRscaled*dRscaleddR*S + Phi_tot*R*dSdr_over_r;
    if(R>0) {
        force[0] = -(dPhidR*X-dPhidphi*Y/R)/R;
        force[1] = -(dPhidR*Y+dPhidphi*X/R)/R;
    } else { 
        force[0]=force[1]=0;
    }
    if(Z==0 && (mysymmetry & ST_PLANESYM)==ST_PLANESYM) { // symmetric about z -> -z
        dPhidzscaled=0;
        d2Phidzscaleddphi=0;
        d2PhidRscaleddzscaled=0;
    }
    double dPhidz=dPhidzscaled*dzscaleddz*S + Phi_tot*Z*dSdr_over_r;
    force[2] = -dPhidz;
    if(forceDeriv!=NULL && R>0)
    {
        d2Phidphi2*=S;
        double d2RscaleddR2=-pow_2(dRscaleddR);
        double d2zscaleddz2=-pow_2(dzscaleddz)*(Z>=0?1:-1);
        double d2Sdr2=(pow_2(Rscale)-2*r2)*dSdr_over_r*S*S;
        double d2PhidR2 = (R*R*d2Sdr2+Z*Z*dSdr_over_r)/r2*Phi_tot + 2*dSdr_over_r*R*dRscaleddR*dPhidRscaled +
            S*(d2PhidRscaled2*pow_2(dRscaleddR)+dPhidRscaled*d2RscaleddR2);
        double d2Phidz2 = (Z*Z*d2Sdr2+R*R*dSdr_over_r)/r2*Phi_tot + 2*dSdr_over_r*Z*dzscaleddz*dPhidzscaled +
            S*(d2Phidzscaled2*pow_2(dzscaleddz)+dPhidzscaled*d2zscaleddz2);
        double d2PhidRdz = (d2Sdr2-dSdr_over_r)*R*Z/r2*Phi_tot + d2PhidRscaleddzscaled*S*dRscaleddR*dzscaleddz +
            dSdr_over_r*(Z*dRscaleddR*dPhidRscaled+R*dzscaleddz*dPhidzscaled);
        double d2PhidRdphi = dSdr_over_r*R*dPhidphi/S + S*dRscaleddR*d2PhidRscaleddphi;
        double d2Phidzdphi = dSdr_over_r*Z*dPhidphi/S + S*dzscaleddz*d2Phidzscaleddphi;
        double Rinv4 = 1/pow_2(R2);
        double X2=X*X, Y2=Y*Y, XY=X*Y, X2mY2=X2-Y2;
        forceDeriv[0] = -(dPhidR*Y2*R + dPhidphi* 2*XY + d2PhidR2*X2*R2 + d2PhidRdphi*-2*XY*R + d2Phidphi2*Y2) *Rinv4;
        forceDeriv[1] = -(dPhidR*X2*R + dPhidphi*-2*XY + d2PhidR2*Y2*R2 + d2PhidRdphi* 2*XY*R + d2Phidphi2*X2) *Rinv4;
        forceDeriv[3] = -(dPhidR*-XY*R + dPhidphi*-X2mY2 + d2PhidR2*XY*R2 + d2PhidRdphi*X2mY2*R + d2Phidphi2*-XY) *Rinv4;
        forceDeriv[2] = -d2Phidz2;
        forceDeriv[4] = -(d2PhidRdz*Y*R + d2Phidzdphi*X)/R2;
        forceDeriv[5] = -(d2PhidRdz*X*R - d2Phidzdphi*Y)/R2;
    }
}
    
#else
CylSplineExp::CylSplineExp(unsigned int _Ncoefs_R, unsigned int _Ncoefs_z, unsigned int _Ncoefs_phi, 
    const particles::PointMassSet<coord::Car> &points, SymmetryType _sym, 
    double radius_min, double radius_max, double z_min, double z_max) {};
CylSplineExp::~CylSplineExp() {};
void CylSplineExp::eval_cyl(const coord::PosCyl &pos,
    double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const {};
#endif
}; // namespace
