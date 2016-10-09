/** Wrapper for AGAMA potentials to be used in NEMO
    (in particular, as an external potential in the N-body code gyrfalcON).
*/
#define POT_DEF
#include <defacc.h> // from NEMOINC
#include <cmath>
#include "potential_factory.h"

/** defines a wrapper class that is used in constructing NEMO potential
    and acceleration objects from an AGAMA potential that is specified
    by parameters in an ini file, or by a potential coefficients file.
    The usage is as follows: in gyrfalcON, for instance, one adds
    `accname=agama accfile=params.ini`
    to the command line arguments, and provides all necessary parameters 
    in the params.ini text file (see readme.pdf for explanation of the options).
    Alternatively, a previously created file with coefficients of a potential 
    expansion can be given as `accfile=mypotential`.
    The choice is determined by the file extesion (.ini vs any other).
**/
struct agama {
    potential::PtrPotential pot;  ///< pointer to the actual instance of potential
    double Omega;                 ///< rotation pattern speed
    mutable double cost, sint;    ///< cos/sin of time-dependent rotation angle
    ///< ("mutable" qualifier is a hack to circumvent the API restriction)
    static const char* name() { return "agama"; }
    bool NeedMass() const { return false; }
    bool NeedVels() const { return false; }

    template<int NDIM, typename scalar>
    void set_time(double t, int, const scalar*, const scalar*, const scalar*) const  // why is it defined as const??
    {
        cost=cos(Omega*t);
        sint=sin(Omega*t);
    }

    // constructor
    agama(const double* pars, int npar, const char *file)
    {
        if(file==NULL || file[0]==0) {
            nemo_dprintf(0, "Should provide the name of INI or potential coefficients file in accfile=...\n");
            return;
        }
        const std::string filename(file);
        try{
            if(filename.size()>4 && filename.substr(filename.size()-4)==".ini")
                pot = potential::createPotential(filename);
            else
                pot = potential::readPotential(filename);
        }
        catch(std::exception& e){
            nemo_dprintf(0, "Error in creating potential specified in %s: %s\n", file, e.what());
            return;
        }
        Omega=npar>=1 ? pars[0] : 0;
        cost=1; sint=0;
        nemo_dprintf(0, "Created an instance of %s potential with pattern speed=%g\n", pot->name(), Omega);
    }

    // function that computes potential and forces
    template<int NDIM, typename scalar>
    inline void acc(const scalar*, const scalar*pos, const scalar*, scalar &poten, scalar *accel) const
    {
        if(!pot) {
            poten=0;
            for(int i=0; i<NDIM; i++) accel[i]=0;
            return;
        }
        coord::PosCar point(pos[0]*cost+pos[1]*sint, pos[1]*cost-pos[0]*sint, NDIM>=3?pos[2]:0);
        coord::GradCar grad;
        double Phi;
        pot->eval(point, &Phi, &grad);
        poten = static_cast<scalar>(Phi);
        accel[0] = -static_cast<scalar>(grad.dx*cost-grad.dy*sint);
        accel[1] = -static_cast<scalar>(grad.dy*cost+grad.dx*sint);
        if(NDIM==3) accel[2] = -static_cast<scalar>(grad.dz);
    }
};
__DEF__ACC(agama)
__DEF__POT(agama)
