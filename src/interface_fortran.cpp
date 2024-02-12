/** \name   interface_fortran.cpp
    \brief  Fortran interface to potentials of AGAMA library
    \author Eugene Vasiliev
    \date   2015-2016

This file provides the interface to the potentials of AGAMA library accessible from FORTRAN.

One may create a potential in several ways:
(1)  loading the parameters from an INI file (one or several components);
(2)  passing the parameters for one component as a single string;
(3)  providing a FORTRAN routine that returns a density at a given point,
     and using it to create a potential approximation with the parameters
     provided in a text string;
(4)  providing a FORTRAN routine that returns potential and force at a given point,
     and creating a potential approximation for it in the same way as above.

There are functions for computing density, potential, force and force derivatives.

Due to the absense of a native pointer type in FORTRAN, the pointer to the C++ object
should be stored in a placeholder variable of type CHAR*8, which is passed
as the first argument to all functions in this module.
See the FORTRAN example for more details.
*/
#include <cstring>
#include "potential_factory.h"
#include "utils_config.h"
#include "utils.h"

namespace{

// routine defined in FORTRAN that computes the density at point X:
// FUNCTION DENSITY(X)
//   DOUBLE PRECISION X(3), DENSITY
//   DENSITY = density at point X
// END
typedef double(*densityfnc)(double* X);

// routine defined in FORTRAN that computes the potential and force at point X:
// FUNCTION POTENTIAL(X, FORCE)
//   DOUBLE PRECISION X(3), FORCE(3), POTENTIAL
//   POTENTIAL = potential at point X (in cartesian coordinates)
//   FORCE(1)  = -dPhi/dx, etc.
// END
typedef double(*potentialfnc)(double* X, double* FORCE);

// C++-compatible wrapper for the potential defined in FORTRAN
class DensityWrapper: public potential::BaseDensity{
public:
    DensityWrapper(densityfnc _dens, coord::SymmetryType _sym) :
        dens(_dens), sym(_sym)
    {
        FILTERMSG(utils::VL_DEBUG, "FortranWrapper",
            "Created a C++ wrapper for a Fortran density routine at "+
            utils::toString((void*)dens));
    }
private:
    densityfnc dens;
    coord::SymmetryType sym;
    virtual std::string name() const { return "DensityWrapper"; };
    virtual coord::SymmetryType symmetry() const { return sym; }
    virtual double densityCyl(const coord::PosCyl &pos, double /*time*/) const {
        return densityCar(toPosCar(pos), 0); }
    virtual double densitySph(const coord::PosSph &pos, double /*time*/) const {
        return densityCar(toPosCar(pos), 0); }
    virtual double densityCar(const coord::PosCar &pos, double /*time*/) const {
        double x[3] = {pos.x, pos.y, pos.z};
        return dens(x);  // call the FORTRAN routine
    }
};

// C++-compatible wrapper for the potential defined in FORTRAN
class PotentialWrapper: public potential::BasePotentialCar {
public:
    PotentialWrapper(potentialfnc _pot, coord::SymmetryType _sym) :
        pot(_pot), sym(_sym)
    {
        FILTERMSG(utils::VL_DEBUG, "FortranWrapper",
            "Created a C++ wrapper for a Fortran potential routine at "+
            utils::toString((void*)pot));
    }
private:
    potentialfnc pot;
    coord::SymmetryType sym;
    virtual std::string name() const { return "PotentialWrapper"; };
    virtual coord::SymmetryType symmetry() const { return sym; }
    virtual void evalCar(const coord::PosCar &pos,
        double* potential, coord::GradCar* deriv, coord::HessCar*, double /*time*/) const
    {
        double x[3] = {pos.x, pos.y, pos.z};
        double f[3];
        double p=pot(x, f);  // call the FORTRAN routine
        if(potential)
            *potential = p;
        if(deriv) {
            deriv->dx = -f[0];
            deriv->dy = -f[1];
            deriv->dz = -f[2];
        }
    }
};

/// convert FORTRAN string to C++ string
static std::string stdstr(char* fortranString, long len)
{
    std::string str(len, '\0');
    memcpy(&(str[0]), fortranString, len);
    return str;
}

/// *smart* pointers that should exist until the end of the program
std::vector<potential::PtrPotential> potentials;

} // internal namespace


/// Routine that should be called from FORTRAN to construct a (possibly composite)
/// potential specified by parameters provided in an INI file.
/// INPUT:  inifilename - a string with the name of INI file with all parameters.
/// OUTPUT: c_obj - the placeholder for storing the pointer to the C++ potential object.
/// EXAMPLE IN FORTRAN:
///     char*8 C_OBJ
///     call agama_initfromfile(C_OBJ, "file.ini")  ! now C_OBJ contains a valid instance of potential
extern "C" void agama_initfromfile_(void* c_obj, char* inifilename, long /*len(c_obj)=8*/, long len)
{
    potentials.push_back(potential::readPotential(stdstr(inifilename, len)));
    const potential::BasePotential* ptr = potentials.back().get();
    memcpy(c_obj, &ptr, sizeof(void*));
}

/// Routine that should be called from FORTRAN to construct a potential specified by
/// the parameters in a single string.
/// INPUT:  params - a string with parameters of the potential expansion
/// (e.g., "type=Dehnen p=0.8 q=0.5 mass=10 scaleRadius=5").
/// OUTPUT: c_obj - the placeholder for storing the pointer to the C++ potential object.
extern "C" void agama_initfromparam_(void* c_obj, char* params, long, long len)
{
    potentials.push_back(potential::createPotential(utils::KeyValueMap(stdstr(params, len))));
    const potential::BasePotential* ptr = potentials.back().get();
    memcpy(c_obj, &ptr, sizeof(void*));
}

/// Routine that should be called from FORTRAN to construct a potential approximation
/// to the user-defined potential.
/// INPUT:  paramstr - a string with parameters of the potential expansion
/// (e.g., "type=Multipole rmin=0.01 rmax=100 lmax=8 mmax=6 symmetry=Triaxial").
/// INPUT:  pot  - the FORTRAN routine that computes the potential and its derivative;
/// OUTPUT: c_obj - the placeholder for storing the pointer to the C++ potential object.
extern "C" void agama_initfrompot_(void* c_obj, char* params, potentialfnc pot, long, long len)
{
    utils::KeyValueMap param(stdstr(params, len));
    potentials.push_back(potential::createPotential(param, potential::PtrPotential(
        new PotentialWrapper(pot, potential::getSymmetryTypeByName(param.getString("Symmetry"))))));
    const potential::BasePotential* ptr = potentials.back().get();
    memcpy(c_obj, &ptr, sizeof(void*));
}

/// same as above, but for a user-defined density function
extern "C" void agama_initfromdens_(void* c_obj, char* paramstr, densityfnc pot, long, long len)
{
    utils::KeyValueMap param(stdstr(paramstr, len));
    potentials.push_back(potential::createPotential(param, potential::PtrDensity(
        new DensityWrapper(pot, potential::getSymmetryTypeByName(param.getString("Symmetry"))))));
    const potential::BasePotential* ptr = potentials.back().get();
    memcpy(c_obj, &ptr, sizeof(void*));
}

/// Routine that should be called from FORTRAN to compute the potential at the given point.
/// INPUT:  c_obj  is the placeholder for the pointer to a previously created potential.
/// INPUT:  X[3]   is the array of coordinates (x,y,z).
/// RETURN: the potential at the point Phi(x,y,z).
extern "C" double agama_potential_(void* c_obj, double* X, long)
{
    const potential::BasePotential* pot;
    memcpy(&pot, c_obj, sizeof(void*));
    return pot->value(coord::PosCar(X[0], X[1], X[2]));
}

/// Routine that should be called from FORTRAN to compute the potential and force at the given point.
/// INPUT:  c_obj  is the placeholder for the pointer to a previously created potential.
/// INPUT:  X[3]   is the array of coordinates (x,y,z).
/// OUTPUT: FORCE[3] will contain the force -dPhi/dx, etc.
/// RETURN: the potential at the point Phi(x,y,z).
extern "C" double agama_potforce_(void* c_obj, double* X, double* FORCE, long)
{
    const potential::BasePotential* pot;
    memcpy(&pot, c_obj, sizeof(void*));
    coord::GradCar grad;
    double value;
    pot->eval(coord::PosCar(X[0], X[1], X[2]), &value, &grad);
    FORCE[0] = -grad.dx;
    FORCE[1] = -grad.dy;
    FORCE[2] = -grad.dz;
    return value;
}

/// Routine that should be called from FORTRAN to compute the potential, force and its derivative.
/// INPUT:  c_obj  is the placeholder for the pointer to a previously created potential.
/// INPUT:  X[3]   is the array of coordinates (x,y,z).
/// OUTPUT: FORCE[3] will contain the force -dPhi/dx, etc.
/// OUTPUT: DERIV[6] will contain the force derivatives in the following order:
/// dFx/dx, dFy/dy, dFz/dz, dFx/dy, dFx/dz, dFy/dz.
/// RETURN: the potential at the point Phi(x,y,z).
extern "C" double agama_potforcederiv_(void* c_obj, double* X, double* FORCE, double* DERIV, long)
{
    const potential::BasePotential* pot;
    memcpy(&pot, c_obj, sizeof(void*));
    coord::GradCar grad;
    coord::HessCar hess;
    double value;
    pot->eval(coord::PosCar(X[0], X[1], X[2]), &value, &grad, &hess);
    FORCE[0] = -grad.dx;
    FORCE[1] = -grad.dy;
    FORCE[2] = -grad.dz;
    DERIV[0] = -hess.dx2;
    DERIV[1] = -hess.dy2;
    DERIV[2] = -hess.dz2;
    DERIV[3] = -hess.dxdy;
    DERIV[4] = -hess.dxdz;
    DERIV[5] = -hess.dydz;
    return value;
}

// function that should be called from FORTRAN to compute the density at the given point.
// INPUT:  c_obj  is the placeholder for the pointer to a previously created potential.
// INPUT:  X[3]   is the array of coordinates (x,y,z).
// RETURN: the value of density.
extern "C" double agama_density_(void* c_obj, double* X, long)
{
    const potential::BasePotential* pot;
    memcpy(&pot, c_obj, sizeof(void*));
    return pot->density(coord::PosCar(X[0], X[1], X[2]));
}
