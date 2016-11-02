#include "actions_newtorus.h"
#include "actions_spherical.h"
#include "actions_staeckel.h"
#include "actions_genfnc.h"
#include "potential_utils.h"
#include "math_core.h"
#include "math_fit.h"
#include <stdexcept>
#include <cassert>
#include <cmath>
#include <utility>
#include <map>

#include <iostream>
#include <fstream>
#include "utils.h"

namespace actions{

namespace {  // internal routines

// ----- Point transformations, i.e. change of coordinates ----- //

/** Point transform as used during the fitting process, augmented with the routine for
    computing derivatives of Hamiltonian by coordinate/momenta in the toy phase space
*/
class PointTransformFit: public BasePointTransform<coord::SphMod> {
public:
    PointTransformFit(const potential::OblatePerfectEllipsoid &poten, const Actions &/*acts*/) :
        cs(0*sqrt(poten.coordsys().delta))
    {
        //computeIntegralsStaeckel(poten, acts, rhomap, taumap);
    }
    coord::PosVelProlMod transform(const coord::PosVelSphMod &ps,
        double &drho_dtoyr, double &dtau_dtoytau) const
    {
        double rho, tau;
        rho=ps.r; drho_dtoyr=1;  //!!!
        tau=ps.tau; dtau_dtoytau=1;
        //rhomap->evalDeriv(ps.r, &rho, &drho_dtoyr);
        //taumap->evalDeriv(ps.tau, &tau, &dtau_dtoytau);
        double prho = ps.pr / drho_dtoyr;
        double ptau = ps.ptau / dtau_dtoytau;
        return coord::PosVelProlMod(
            coord::PosProlMod(rho, tau, ps.phi, cs), coord::VelProlMod(prho, ptau, ps.pphi));
    }
    virtual coord::PosVelCyl map(const coord::PosVelSphMod &ps) const
    {
        double drho_dtoyr, dtau_dtoytau;
        return toPosVelCyl(transform(ps, drho_dtoyr, dtau_dtoytau));
    }
    double Hamiltonian(const coord::PosVelSphMod &ps,
        const potential::BasePotential& potential, coord::PosVelSphMod *dHby=NULL) const
    {
        double drho_dtoyr, dtau_dtoytau;
        coord::PosVelProlMod pp(transform(ps, drho_dtoyr, dtau_dtoytau));
        coord::PosVelCyl pcy(toPosVelCyl(pp));
        if(dHby) {
            coord::PosDerivT<coord::ProlMod, coord::Cyl> der;
            coord::GradCyl grad;
            double H;
            potential.eval(toPosDeriv(pp, &der), &H, &grad);
            coord::GradProlMod dHp;
            coord::VelProlMod dHv;
            H += coord::Ekin(pp, &dHp, &dHv);
            dHby->r   =(dHp.drho + grad.dR * der.dRdrho + grad.dz * der.dzdrho) * drho_dtoyr;
            dHby->tau =(dHp.dtau + grad.dR * der.dRdtau + grad.dz * der.dzdtau) * dtau_dtoytau;
            dHby->phi = dHp.dphi + grad.dphi;
            dHby->pr  = dHv.prho / drho_dtoyr;
            dHby->ptau= dHv.ptau / dtau_dtoytau;
            dHby->pphi= dHv.pphi;
            return H;
        } else {
            return potential.value(coord::toPosDeriv<coord::ProlMod, coord::Cyl>(pp, NULL))
            + coord::Ekin(pp);
        }
    }
private:
    const coord::ProlMod cs;  ///< coordinate system of point transformation
    math::PtrFunction rhomap, taumap;
};

// ----- A few auxiliary routines ----- //

/// create an array of angles uniformly covering the range [0:pi]  (NB: why not 2pi?)
static std::vector<Angles> makeGridAngles(unsigned int nr, unsigned int nz, unsigned int nphi=1)
{
    std::vector<Angles> vec(nr*nz*nphi);
    for(unsigned int ir=0; ir<nr; ir++) {
        double thetar = ir * M_PI / nr;
        for(unsigned int iz=0; iz<nz; iz++) {
            double thetaz = iz * M_PI / nz;
            for(unsigned int iphi=0; iphi<nphi; iphi++)
                vec[ (ir*nz + iz) * nphi + iphi ] =
                    Angles(thetar, thetaz, iphi * M_PI / fmax(nphi-1, 1));
        }
    }
    return vec;
}

/// create grid in angles with size determined by the maximal Fourier harmonic in the indices array
static std::vector<Angles> makeGridAngles(const GenFncIndices& indices)
{
    int maxmr=4, maxmz=4, maxmphi=0;
    for(unsigned int i=0; i<indices.size(); i++) {
        maxmr   = std::max<int>(maxmr,   math::abs(indices[i].mr));
        maxmz   = std::max<int>(maxmz,   math::abs(indices[i].mz));
        maxmphi = std::max<int>(maxmphi, math::abs(indices[i].mphi));
    }
    return makeGridAngles(6*(maxmr/4+1), 6*(maxmz/4+1), maxmphi>0 ? 6*(maxmphi/4+1) : 1);
}

/// create the array of indices of the generating function with all terms up to the given maximum order
static GenFncIndices makeGridIndices(int irmax, int izmax)
{   /// NOTE: here we specialize for the case of axisymmetric systems!
    GenFncIndices indices;
    for(int ir=0; ir<=irmax; ir++)
        for(int iz=-izmax; iz<=(ir==0?-2:izmax); iz+=2)
            indices.push_back(GenFncIndex(ir, iz, 0));
    return indices;
}

/// return the absolute value of an element in a map, or zero if it doesn't exist
static inline double absvalue(const std::map< std::pair<int,int>, double >& indPairs, int ir, int iz)
{
    if(indPairs.find(std::make_pair(ir, iz)) != indPairs.end())
        return fabs(indPairs.find(std::make_pair(ir, iz))->second);
    else
        return 0;
}

/// debugging: print a 2d table (mr,mz) of gen.fnc.coefs (log magnitude)
static void printoutGenFncCoefs(const GenFncIndices& indices, const double values[])
{   /// NOTE: axisymmetric case only!
    std::map< std::pair<int,int>, double > indPairs;
    // 1. determine the extent of existing grid in (mr,mz)
    int maxmr=0, minmz=0, maxmz=0;
    for(unsigned int i=0; i<indices.size(); i++) {
        indPairs[std::make_pair(indices[i].mr, indices[i].mz)] = values[i];
        maxmr = std::max<int>(maxmr, indices[i].mr);
        maxmz = std::max<int>(maxmz, indices[i].mz);
        minmz = std::min<int>(minmz, indices[i].mz);
    }
    for(int iz=minmz; iz<=maxmz; iz+=2) {
        std::cout << utils::pp(iz,3);
        for(int ir=0; ir<=maxmr; ir++) {
            double val = absvalue(indPairs, ir, iz);
            std::cout << ' ' << (val>0 ? utils::pp(int(-10.*log10(val)),3) : " - ");
        }
        std::cout << '\n';
    }
}

// ----- The class that performs torus fitting ----- //

class Mapping;
typedef shared_ptr<const Mapping> PtrMapping;

/** Complete description of mapping {real actions, toy angles} => {real pos/vel},
    as used in the process of torus fitting (using a grid of toy angles).
    The task of this class is to perform a complete cycle of finding the best-fit
    values for all parameters used in the entire mapping procedure
    (gen.fnc. => toy map => point transform => real potential).
    It operates on the combined array of parameters that needs to be passed as an argument
    to most member functions.
    The rationale is that the instance of the class is an immutable object, as usual,
    but the parameters that change in the course of fitting are stored externally.
    The layout of this array and scaling of parameters is handled internally by this class;
    the caller does not need to care about the content of this array, only should pass it
    to the methods of this class.
    The class also provides an IFunctionNdimDeriv interface that is used by
    the Levenberg-Marquardt minimization routine (called internally from `fitTorus` method).
*/
class Mapping : public math::IFunctionNdimDeriv{
public:
    /** Create the mapping for the provided values of real actions, real potential, toy map,
        indices of terms of gen.fnc. to be adjusted during the fit,
        and the regularization parameter lambda.
        A suitable grid in the space of toy angles is constructed internally.
    */
    Mapping(const Actions &_acts,
        const potential::BasePotential &_potential, 
        const PointTransformFit &_pointTrans,
        const BaseToyMap<coord::SphMod> &_toyMapFit,
        const GenFncIndices &_indices,
        const double _lambda)
    :
        acts(_acts),
        potential(_potential),
        pointTrans(_pointTrans),
        toyMapFit(_toyMapFit),
        indices(_indices),
        genFncFit(indices, acts, makeGridAngles(indices)),
        lambda(_lambda)
    {}

    /** Perform a complete cycle of Levenberg-Marquardt fitting.
        \param[in,out]  params is the array of parameters of the entire mapping.
        On input, this array may be empty, or contain values left from the previous cycle,
        possibly performed for a smaller set of parameters
        (in any case, all new parameters are initialized with zero values).
        After several iterations of L-M procedure, the array of parameters will contain
        current best-fit values.
        \return the number of iterations taken; a negative value means that the maximum
        number of iterations was reached, and zero means that the solver encountered
        a severe error, so that the array of parameters likely contains nonsense values.
    */
    int fitTorus(std::vector<double> &params) const;

    /** Compute the dispersion of Hamiltonian averaged over the grid of toy angles,
        \param[in]  params  are their respective parameters;
        \returns  the dispersion of Hamiltonian.
    */
    double computeHamiltonianDisp(const std::vector<double> &params) const;

    /** Construct the components of torus mapping to be used in the Torus class itself.
        During the fitting process, variants of these mapping operators are created and
        used internally; when the fitting is done, the caller needs to create
        the corresponding components with the best-fit parameters.
        Moreover, during the fit we do not need to deal with angle mapping
        (real to toy angles), but it is required for the completed torus.
        The procedure of angle mapping, which has non-negligible cost and computes
        both the frequencies and the derivatives of gen.fnc. by actions,
        is also performed by this method.
        \param[in]  params  is the array of best-fit parameters;
        \param[out] toyMap  will contain an instance of Toy Map
        (replacing the existing one if the smart pointer was not empty);
        \param[out] genFnc  will contain an instance of generating function (same remark);
        \param[out] freqs   will contain the estimate of frequencies on this torus;
        \return  the dispersion of real Hamiltonian over the grid of toy angles
        (same value as returned by `computeHamiltonianDisp`, obtained as a by-product).
    */
    double construct(const std::vector<double> &params,
        PtrCanonicalMap &genFnc, Frequencies &freqs) const;

    /** Create a new instance of Mapping class with an expanded list of gen.fnc.terms,
        adding new terms adjacent to the existing ones.
        \param[in]  values  are the current best-fit parameters, 
        used to decide where to add new terms (next to the largest of existing ones).
        \return  a new Mapping class.
    */
    PtrMapping expandMapping(const std::vector<double> &params) const;

    /** Debugging: export the torus with the given transformation parameters to a text file */
    void printoutTorus(const char* filename, const double params[]) const;

private:
    const Actions acts;               ///< values of real actions on this torus
    const potential::BasePotential &potential;  ///< real potential
    const PointTransformFit &pointTrans;        ///< point transformation of coordinate/momentum
    const BaseToyMap<coord::SphMod> &toyMapFit; ///< map between toy action/angles and position/velocity
    const GenFncIndices indices;      ///< indices of allowed terms in gen.fnc.
    const GenFncFit genFncFit;        ///< map between real and toy actions at a grid of angles
    const double lambda;              ///< regularization coefficient
    
    /** methods providing the interface of IFunctionNdimDeriv: number of parameters to fit */
    virtual unsigned int numVars() const {
        return genFncFit.numParams(); }

    /** number of equations in the least-square fit: all elements of the toy angle grid,
        plus optionally additional equations responsible for the regularization */
    virtual unsigned int numValues() const {
        return genFncFit.numPoints() + (lambda!=0 ? genFncFit.numParams() : 0); }
    
    /** Compute the deviations of Hamiltonian from its average value for an array of toy angles.
        For each point it returns the relative difference between `H_k` and `<H>`, and optionally
        computes the derivatives of `H_k` w.r.t. the parameters of toy map and generating function.
        This routine is called from the Levenberg-Marquardt solver.
        \param[in]  mapping  is the sequence of mapping operators;
        \param[in]  params  are their respective parameters;
        \param[out] deltaHvalues  is the array of deviations of Hamiltonian at each point;
        \param[out] dHdParams  is the Jacobian matrix of derivatives of Hamiltonian w.r.t.
        all input parameters;
    */
    virtual void evalDeriv(const double params[],
        double* deltaHvalues=NULL, double* dHdParams=NULL) const;

    /** Perform the torus mapping for the given point on the grid of toy angles.
        \param[in]  params  is the array of parameters for mapping operators;
        \param[in]  indPoint  is the index of point in the array of toy angles (contained in gen.fnc.);
        \param[out] dHdJ  if not NULL, will contain the derivatives of Hamiltonian by toy actions;
        \param[out] derivGenFnc  if not NULL, will contain the derivs of H by params of gen.fnc.;
        \return  the value of real Hamiltonian at the mapped pos/vel point.
    */
    double computeHamiltonianAtPoint(const double params[], const unsigned int indPoint,
        Actions *dHdJ=NULL, double *derivGenFnc=NULL) const;

    /** Compute the frequencies and the derivatives of generating function by real actions,
        used in angle mapping.
        The three arrays of derivatives dS_i/dJ_{r,z,phi}, i=0..numParamsGenFnc-1,
        together with three frequencies Omega_{r,z,phi}, are the solutions of
        an overdetermined system of linear equations:
        \f$  M_{k,i} X_{i} = RHS_{k}, k=0..numPoints-1  \f$,
        where numPoints is the number of individual triplets of toy angles,
        \f$  X_i  \f$ is the solution vector {Omega, dS_i/dJ} for each direction (r,z,theta),
        \f$  RHS_k = dH/dJ(\theta_k)  \f$, again for three directions independently, and
        \f$  M_{k,i}  \f$ is the matrix of coefficients shared between all three equation systems:
        \f$  M_{k,0} = 1, M_{k,i+1} = -dH/dS_i(\theta_k)  \f$.
        The matrix M and three RHS vectors are filled using the same approach as during
        the Levenberg-Marquardt minimization, from the provided parameters of toy map and
        generating function; then the three linear systems are solved using
        the singular-value decomposition of the shared coefficient matrix,
        and the output frequencies and gen.fnc.derivatives are returned in corresponding arguments.
        The return value of this function is the same as `computeHamiltonianDisp()`.
    */
    double fitAngleMap(const double params[],
        Frequencies& freqs, GenFncDerivs& derivs) const;
};

// ----- public member functions of Mapping ----- //

int Mapping::fitTorus(std::vector<double>& params) const
{
    // number of iterations in Levenberg-Marquardt algorithm
    const unsigned int maxNumIter = 10;
    // stopping criterion for LM fit (relative change in parameters during step)
    const double relToler = 0.1;
    // allocate or extend (if it was not empty) the array of initial parameter values
    params.resize(genFncFit.numParams());
    // perform the Levenberg-Marquardt minimization and store best-fit parameters in fitParams
    try{
        int numIter = math::nonlinearMultiFit(*this, &params[0], relToler, maxNumIter, &params[0]);
        std::cout << numIter << " iterations; " << indices.size() << " GF terms; ";
        return numIter;
    }
    catch(std::exception& e) {
        std::cout << "\033[1;31mException in Torus\033[0m: " << e.what() <<'\n';
        return 0;  // signal of error, will restart the fit from default initial params
    }
}

double Mapping::construct(const std::vector<double> &params,
    PtrCanonicalMap &genFnc, Frequencies &freqs) const
{
    assert(params.size() == genFncFit.numParams());
    GenFncDerivs derivs;
    double dispH = fitAngleMap(&params[0], freqs, derivs);
    genFnc.reset(new GenFnc(indices, &params[0], &derivs[0]));
    return dispH;
}

double Mapping::computeHamiltonianDisp(const std::vector<double> &params) const
{
    assert(params.size() == genFncFit.numParams());
    math::Averager Havg;
    for(unsigned int indPoint=0; indPoint < genFncFit.numPoints(); indPoint++)
        Havg.add(computeHamiltonianAtPoint(&params[0], indPoint));
    return Havg.disp();    
}

PtrMapping Mapping::expandMapping(const std::vector<double> &params) const
{   /// NOTE: here we specialize for the case of axisymmetric systems!
    assert(params.size() == genFncFit.numParams());
    if(params.size() == 0)  // existing mapping has empty gen.fnc.
        return PtrMapping(new Mapping(acts, potential, pointTrans, toyMapFit, makeGridIndices(4, 4), lambda));

    std::map< std::pair<int,int>, double > indPairs;

    // 1. determine the extent of existing grid in (mr,mz)
    int maxmr=0, maxmz=0;
    for(unsigned int i=0; i<indices.size(); i++) {
        indPairs[std::make_pair(indices[i].mr, indices[i].mz)] = params[i];
        maxmr = std::max<int>(maxmr, math::abs(indices[i].mr));
        maxmz = std::max<int>(maxmz, math::abs(indices[i].mz));
    }
    GenFncIndices newIndices = indices;
    /*if(maxmz==0) {  // dealing with the case Jz==0 -- add only two elements in m_r
        newIndices.push_back(GenFncIndex(maxmr+1, 0, 0));
        newIndices.push_back(GenFncIndex(maxmr+2, 0, 0));
        return newIndices;
    }*/

    // 2. determine the largest amplitude of coefs that are at the boundary of existing values
    double maxval = 0;
    for(int ir=0; ir<=maxmr+2; ir++)
        for(int iz=-maxmz-2; iz<=maxmz+2; iz+=2) {
            if(indPairs.find(std::make_pair(ir, iz)) != indPairs.end() &&
               ((iz<=0 && indPairs.find(std::make_pair(ir, iz-2)) == indPairs.end()) ||
                (iz>=0 && indPairs.find(std::make_pair(ir, iz+2)) == indPairs.end()) ||
                indPairs.find(std::make_pair(ir+1, iz)) == indPairs.end()) )
                maxval = fmax(fabs(indPairs[std::make_pair(ir, iz)]), maxval);
        }

    // 3. add more terms adjacent to the existing ones at the boundary, if they are large enough
    double thresh = maxval * 0.1;
    int numadd = 0;
    for(int ir=0; ir<=maxmr+2; ir++)
        for(int iz=-maxmz-2; iz<=maxmz+2; iz+=2) {
            if(indPairs.find(std::make_pair(ir, iz)) != indPairs.end() || (ir==0 && iz>=0))
                continue;  // already exists or not required
            if (absvalue(indPairs, ir-2, iz)   >= thresh ||
                absvalue(indPairs, ir-1, iz)   >= thresh ||
                absvalue(indPairs, ir  , iz-2) >= thresh ||
                absvalue(indPairs, ir  , iz+2) >= thresh ||
                absvalue(indPairs, ir+1, iz-2) >= thresh ||
                absvalue(indPairs, ir+1, iz+2) >= thresh ||
                absvalue(indPairs, ir+1, iz)   >= thresh)
            {   // add a term if any of its neighbours are large enough
                newIndices.push_back(GenFncIndex(ir, iz, 0));
                numadd++;
            }
        }

    // 4. create a new mapping
    assert(numadd>0);
    return PtrMapping(new Mapping(acts, potential, pointTrans, toyMapFit, newIndices, lambda));
}

void Mapping::printoutTorus(const char* filename, const double params[]) const
{
    printoutGenFncCoefs(indices, params);
    std::ofstream strm(filename);
    GenFncDerivs derivs;
    Frequencies freqs;
    fitAngleMap(&params[0], freqs, derivs);
    GenFnc genFnc(indices, &params[0], &derivs[0]);
    const int NR=32, NZ=64;
    for(int iR=0; iR<=NR; iR++) {
        double thetar = M_PI * iR / NR;
        for(int iz=0; iz<=NZ; iz++) {
            double thetaz = 2*M_PI * iz / NZ;
            ActionAngles toyAA = genFnc.map(ActionAngles(acts, Angles(thetar, thetaz, 0)));
            if(toyAA.thetar>M_PI*1.5)
                toyAA.thetar-=2*M_PI;
            coord::PosVelSphMod ps = toyMapFit.map(toyAA);
            coord::PosVelCyl pc = pointTrans.map(ps);
            double H = totalEnergy(potential, pc);
            //double H1= pointTrans.Hamiltonian(ps, potential, NULL);
            double Phi = potential.value(pc);
            double Ekin= H-Phi;
            strm << thetar << ' ' << thetaz << '\t' <<
            toyAA.thetar << ' ' << toyAA.thetaz << '\t' <<
            toyAA.Jr << ' ' << toyAA.Jz << '\t' <<
            ps.r << ' ' << ps.tau << '\t' <<
            pc.R << ' ' << pc.z << '\t' << H << ' ' <<Phi<<' '<<Ekin<< '\t' << 
            ps.r*(1-ps.tau*ps.tau)/(1+ps.tau*ps.tau) << ' ' << ps.r*2*ps.tau/(1+ps.tau*ps.tau) << '\t' <<
            0.5*(pow_2(ps.pr)+pow_2((1+ps.tau*ps.tau)/ps.r)*(0.25*pow_2(ps.ptau)+pow_2(ps.pphi/(1-ps.tau*ps.tau)))) << '\n';
        }
        strm << '\n';
    }
}
    
// ----- private member functions of Mapping ----- //

/** Compute the derivative of Hamiltonian by toy actions:
    dH/dJ = dH/d{x,v} d{x,v}/dJ, where the lhs is a covector of length 3,
    the first term on rhs is a covector of length 6 (the gradient dPhi/dx and the velocity),
    and the second term is a 6x3 matrix of partial derivs provided by the toy map.
*/
static inline Actions dHbydJ(const coord::PosVelSphMod& dHby,
    const DerivAct<coord::SphMod>& derivAct)
{
    return Actions(
        derivAct.dbyJr.r * dHby.r + derivAct.dbyJr.tau * dHby.tau + derivAct.dbyJr.phi * dHby.phi +
        derivAct.dbyJr.pr* dHby.pr+ derivAct.dbyJr.ptau* dHby.ptau+ derivAct.dbyJr.pphi* dHby.pphi,
        derivAct.dbyJz.r * dHby.r + derivAct.dbyJz.tau * dHby.tau + derivAct.dbyJz.phi * dHby.phi +
        derivAct.dbyJz.pr* dHby.pr+ derivAct.dbyJz.ptau* dHby.ptau+ derivAct.dbyJz.pphi* dHby.pphi,
        derivAct.dbyJphi.r * dHby.r + derivAct.dbyJphi.tau * dHby.tau + derivAct.dbyJphi.phi * dHby.phi +
        derivAct.dbyJphi.pr* dHby.pr+ derivAct.dbyJphi.ptau* dHby.ptau+ derivAct.dbyJphi.pphi* dHby.pphi);
}

double Mapping::computeHamiltonianAtPoint(const double params[], const unsigned int indPoint,
    Actions *dHdJ, double *derivGenFnc) const
{
    // Generating function computes the toy actions from the real actions
    // at the given point in the grid of toy angles grid
    ActionAngles toyAA = genFncFit.toyActionAngles(indPoint, params);

    // do not allow to stray into forbidden region of negative actions
    if(toyAA.Jr<0 || toyAA.Jz<0)
        return NAN;
    // Toy map computes the position and velocity from the toy actions and angles,
    // and optionally their derivatives w.r.t. toy actions and toy map parameters,
    DerivAct<coord::SphMod> derivAct;
    coord::PosVelSphMod point = toyMapFit.map(toyAA, NULL, derivGenFnc!=NULL ? &derivAct : NULL);

    // obtain the value real potential at the given point and its derivatives w.r.t. coordinates/momenta
    coord::PosVelSphMod dHbyCM;
    double H = pointTrans.Hamiltonian(point, potential, derivGenFnc ? &dHbyCM : NULL);

    // derivatives of Hamiltonian w.r.t. parameters of gen.fnc.
    if(derivGenFnc) {
        // derivative of Hamiltonian by toy actions
        Actions dHby = dHbydJ(dHbyCM, derivAct);
        if(dHdJ)
            *dHdJ = dHby;
        for(unsigned int p = 0; p<genFncFit.numParams(); p++) {
            // derivs of toy actions by gen.fnc.params
            Actions dbyS = genFncFit.deriv(indPoint, p);
            // derivs of Hamiltonian by gen.fnc.params
            double  dHdS = dHby.Jr * dbyS.Jr + dHby.Jz * dbyS.Jz + dHby.Jphi * dbyS.Jphi;
            derivGenFnc[p] = dHdS;
        }
    }
    return H;
}

void Mapping::evalDeriv(const double params[],
    double* deltaHvalues, double* dHdParams) const
{
    const unsigned int numPoints = genFncFit.numPoints();
    const unsigned int numParams = genFncFit.numParams();

    // we need to store the values of Hamiltonian at grid points even if this is not requested,
    // because they are used to correct the entries of the Jacobian matrix
    // to account for the fact that the mean <H> also depends on the parameters
    std::vector<double> Hvalues(numPoints);
    double Havg = 0;  // accumulator for the average Hamiltonian

    // loop over grid of toy angles
    for(unsigned int indPoint=0; indPoint < numPoints; indPoint++) {
        double H = computeHamiltonianAtPoint(params, indPoint, NULL,
            dHdParams ? dHdParams + indPoint * numParams : NULL); 
        
        // accumulate the average value and store the output
        Havg += H;
        Hvalues[indPoint] = H;
    }

    // convert from  H_k  to  deltaH_k = H_k / <H> - 1
    Havg /= numPoints;
    if(deltaHvalues) {
        double disp = 0;
        for(unsigned int indPoint=0; indPoint < numPoints; indPoint++) {
            deltaHvalues[indPoint] = Hvalues[indPoint] / Havg - 1;
            disp += pow_2(deltaHvalues[indPoint]);
        }
        if(lambda != 0)  // regularization: add penalty proportional to the magnitude of parameters
            for(unsigned int indReg=0; indReg < numParams; indReg++)
                deltaHvalues[indReg + numPoints] = 
                lambda * params[indReg];
        std::cout << "Havg="<<Havg<<", dH/H="<<sqrt(disp/numPoints)<<"\n";
        /*for(unsigned int i=0; i<numParamsGenFnc; i++)
            std::cout<<indices[i].mr<<','<<indices[i].mz<<'='<<params[i+offsetGenFncParam]<<' ';
        std::cout<<'\n';*/
    }
    if(params[0]==42)
        printoutTorus("bla", params);
    // convert derivatives:  d(deltaH_k) / dP_p = (1/<H>) dH_k / dP_p - (H_k / <H>^2) d<H> / dP_p
    if(dHdParams) {
        std::vector<double> dHavgdP(numPoints);
        for(unsigned int pp=0; pp < numPoints * numParams; pp++)
            dHavgdP[pp % numParams] += dHdParams[pp] / numPoints;
        for(unsigned int pp=0; pp < numPoints * numParams; pp++) {
            unsigned int indPoint = pp / numParams;
            unsigned int indParam = pp % numParams;
            dHdParams[pp] = (dHdParams[pp] - dHavgdP[indParam] * Hvalues[indPoint] / Havg) / Havg;
        }
        if(lambda != 0)  // regularization
            for(unsigned int indReg=0; indReg < numParams; indReg++)
                for(unsigned int indParam=0; indParam<numParams; indParam++)
                    dHdParams[(indReg + numPoints) * numParams + indParam] = 
                        indParam == indReg ? lambda : 0;
    }
}

double Mapping::fitAngleMap(const double params[],
    Frequencies& freqs, GenFncDerivs& derivs) const
{
    const unsigned int numPoints = genFncFit.numPoints();
    const unsigned int numParams = genFncFit.numParams();
    // the matrix of coefficients shared between three linear systems
    math::Matrix<double> coefsdHdS(numPoints, numParams+1);
    // tmp storage for dH/dS
    std::vector<double> derivGenFnc(numParams+1);
    // derivs of Hamiltonian by toy actions (RHS vectors)
    std::vector<double> dHdJr(numPoints), dHdJz(numPoints), dHdJphi(numPoints);
    // accumulator for computing dispersion in H
    math::Averager Havg;

    // loop over grid of toy angles
    for(unsigned int indPoint=0; indPoint < numPoints; indPoint++) {
        Actions dHby(0,0,0);  // derivative of Hamiltonian by toy actions
        Havg.add(computeHamiltonianAtPoint(params, indPoint,
            &dHby, &derivGenFnc.front()) );
        // fill the elements of each of three rhs vectors
        dHdJr  [indPoint] = dHby.Jr;
        dHdJz  [indPoint] = dHby.Jz;
        dHdJphi[indPoint] = dHby.Jphi;
        // fill the matrix row
        coefsdHdS(indPoint, 0) = 1;  // matrix coef for omega
        for(unsigned int p=0; p<numParams; p++)
            coefsdHdS(indPoint, p+1) = -derivGenFnc[p];  // matrix coef for dS_p/dJ
    }

    // solve the overdetermined linear system in the least-square sense:
    // step 1: prepare the SVD of coefs matrix
    math::SVDecomp SVD(coefsdHdS);

    // step 2: solve three linear systems with the same matrix but different rhs
    std::vector<double>
        dSdJr(SVD.solve(dHdJr)),
        dSdJz(SVD.solve(dHdJz)),
        dSdJphi(SVD.solve(dHdJphi));

    // store output
    freqs.Omegar   = dSdJr[0];
    freqs.Omegaz   = dSdJz[0];
    freqs.Omegaphi = dSdJphi[0];
    derivs.resize(numParams);
    for(unsigned int p=0; p<numParams; p++) {
        derivs[p].Jr   = dSdJr[p+1];
        derivs[p].Jz   = dSdJz[p+1];
        derivs[p].Jphi = dSdJphi[p+1];
    }
    return Havg.disp();
}

}  // internal namespace

ActionMapperNewTorus::ActionMapperNewTorus(const potential::BasePotential& poten,
    const Actions& _acts, double toler) :
    acts(_acts), toyMap(new ToyMapSpherical(poten))
{
    if(!isAxisymmetric(poten))
        throw std::invalid_argument("ActionMapperNewTorus only works for axisymmetric potentials");
    if(acts.Jr<0 || acts.Jz<0)
        throw std::invalid_argument("ActionMapperNewTorus: negative actions");

    // number of complete cycles of Levenberg-Marquardt fitting procedure
    const unsigned int maxNumCycles = 6;
    
    // point transformation
    const PointTransformFit* pointTransFit = new PointTransformFit(
        dynamic_cast<const potential::OblatePerfectEllipsoid&>(poten), acts);
    pointTrans.reset(pointTransFit);

    // parameters of generating function obtained during the fit
    std::vector<double> params;

    // create a first approximation without any generating function, using point transform only
    PtrMapping mapping(new Mapping(acts, poten, *pointTransFit, *toyMap, GenFncIndices(), 0));

    // after the first cycle, we construct all components of torus, because we need to estimate
    // the frequencies required to compute the maximum allowed dispersion of Hamiltonian;
    // since frequencies are obtained in the course of angle mapping procedure, we have now
    // the complete description of torus, and if we're lucky to reach the required accuracy
    // at the first attempt, the entire fitting procedure is done.
    double dispH = mapping->construct(params, genFnc, freqs);

    // translate the required tolerance into the maximum allowed dispersion of H
    double dispHmax = pow_2(toler) * 
        (pow_2(freqs.Omegar) + pow_2(freqs.Omegaz)) *
        (acts.Jr==0 || acts.Jz==0 ? pow_2(acts.Jr + acts.Jz) : acts.Jr * acts.Jz);
    std::cout << "; dispJ/J=" << toler*(sqrt(dispH/dispHmax));
    
    if(!isFinite(dispH + dispHmax))
        throw std::runtime_error("Error in Torus: first fit attempt failed");

    converged = dispH <= dispHmax;
    if(converged) {
        std::cout << " \033[1;32mCONVERGED INSTANTLY\033[0m\n";
        return;
    }
    // if not converged on the first cycle, repeat the fit while increasing the number of parameters.

    // keep track of fit quality from the previous iteration
    double dispHprev = INFINITY;

    // keep track of best-fit set of parameters and their values, and the measure of fit quality
    double dispHbest = dispH;
    std::vector<double> bestParams = params;
    PtrMapping bestMapping = mapping;
    
    // flag denoting whether the best-fit parameters have changed
    // and the torus components need to be constructed anew at the end of the loop
    bool needToConstruct = false;

    // perform one or more complete fit cycles, expanding the set of indices after each cycle
    // if the residuals in Hamiltonian are not sufficiently small
    unsigned int numCycles = 0, numIter = 1;
    while(++numCycles < maxNumCycles && !converged)
    {
        // add more terms to gen.fnc., taking into account the magnitudes of existing terms,
        // and reinitialize the mapping itself, to account for the updated gen.fnc.
        mapping = mapping->expandMapping(params);

        // if not converged yet, take appropriate measures        
        if(!isFinite(dispH) || numIter == 0 || dispH > dispHprev*0.9)
        {   // the process does not seem to converge: start afresh
            params.assign(params.size(), 0);
            dispHprev = INFINITY;
            std::cout << " \033[1;33mRESTARTING\033[0m\n";
        } else {
            dispHprev = dispH;
            std::cout << "\n";
        }
        
        // perform Levenberg-Marquardt least-square fit for the given set of parameters
        numIter = mapping->fitTorus(params);
        
        // measure the fit quality - dispersion of Hamiltonian at grid points
        dispH = mapping->computeHamiltonianDisp(params);
        converged = dispH <= dispHmax;
        std::cout << "; dispJ/J=" << toler*(sqrt(dispH/dispHmax));

        // on every cycle we update the best parameters
        if(dispH < dispHbest) {
            bestMapping = mapping;
            bestParams  = params;
            dispHbest   = dispH;
            needToConstruct = true;
            std::cout << " *";
        }
    }
    if(numCycles==42 || !converged)
        bestMapping->printoutTorus("torus.dat", &bestParams[0]);

    // redo the angle fitting if the process took more than one cycle
    if(needToConstruct)
        bestMapping->construct(bestParams, genFnc, freqs);
    std::cout << (converged ? " \033[1;32mCONVERGED\033[0m\n" : " \033[1;31mNOT CONVERGED\033[0m\n");
}

coord::PosVelCyl ActionMapperNewTorus::map(const ActionAngles& actAng, Frequencies* freq) const
{
    // make sure that the input actions are the same as in the Torus object
    if( math::fcmp(actAng.Jr,   acts.Jr) != 0 ||
        math::fcmp(actAng.Jz,   acts.Jz) != 0 ||
        math::fcmp(actAng.Jphi, acts.Jphi) != 0 )
        throw std::invalid_argument("ActionMapperNewTorus: "
            "values of actions are different from those provided to the constructor");
    if(freq)
        *freq = freqs;
    return pointTrans->map(toyMap->map(genFnc->map(actAng)));
}

}  // namespace actions
