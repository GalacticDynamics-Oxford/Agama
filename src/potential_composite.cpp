#include "potential_composite.h"
#include "math_core.h"
#include <stdexcept>
#ifndef _MSC_VER
#include <alloca.h>
#else
#include <malloc.h>
#endif

// utility snippet for allocating temporary storage either on stack (if small) or on heap otherwise
#define ALLOC(NPOINTS, TYPE, NAME) \
    std::vector< TYPE > tmparray; \
    TYPE* NAME; \
    if(NPOINTS * sizeof(TYPE) > 65536) { \
        tmparray.resize(NPOINTS); \
        NAME = &tmparray[0]; \
    } else \
    NAME = static_cast<TYPE*>(alloca(NPOINTS * sizeof(TYPE)));

namespace potential{

namespace {
/// identifiers of preferred coordinate system for potential components
template<typename CoordType> struct CoordT;
template<> struct CoordT<coord::Car> { static const char id = 1; };
template<> struct CoordT<coord::Cyl> { static const char id = 2; };
template<> struct CoordT<coord::Sph> { static const char id = 3; };

/** All-mighty routine for evaluating the components of a composite potential in their preferred
    coordinate systems.
    The idea is that each potential is more easily (natively) evaluated in its own CS,
    and we respect these preferences when evaluating the composite potential and its derivatives.
    Namely, if the native CS of a given component equals internalCS (this is determined
    in the costructor and assigned an indentifier in the componentTypes array),
    we transform the input point to the internal CS, evaluate the component,
    accumulate the results of all such components in a single instance of gradient/hessian,
    and finally transform the gradient/hessian back to the output CS
    (which is identical to the CS of the input point).
    There are three standard coordinate systems, and one of them is the outputCS, while two
    others are designated as internalCS for non-overlapping subsets of components.
    The components whose preferred CS coincides with the outputCS are called directly,
    while the components which prefer either of the two internal CS are accumulated separately.
    The key point is that we only perform the coordinate and gradient/hessian conversion once,
    not each time a new component is evaluated.

    \tparam internalCS1: the preferred coordinate system for one subset of the components;
    \tparam internalCS2: the preferred coordinate system for another subset;
    \tparam outputCS:    the coordinate system of the input point and output gradient/hessian,
    which is also used to evaluate the remaining components not belonging to the other two subsets.
*/
template<typename internalCS1, typename internalCS2, typename outputCS>
inline void evalComponents(
    const std::vector<PtrPotential>& components,
    const std::vector<char>& componentTypes,
    const coord::PosT<outputCS>& pos,
    double* potential,
    coord::GradT<outputCS>* deriv,
    coord::HessT<outputCS>* deriv2,
    double time)
{
    bool needDeriv = deriv!=NULL || deriv2!=NULL;  // whether one needs first or
    bool needDeriv2= deriv2!=NULL;                 // second derivs of coordinate transformations

    if(potential) *potential=0;
    if(deriv)  coord::clear(*deriv);
    if(deriv2) coord::clear(*deriv2);

    // position, derivatives of coordinate transformations and gradient/hessian in the internalCS1:
    // they may remain unused if no component of the potential prefers this internalCS1.
    coord::PosT<internalCS1> internal1Pos;
    coord::PosDerivT <outputCS, internalCS1> coord1Deriv;
    coord::PosDeriv2T<outputCS, internalCS1> coord1Deriv2;
    coord::GradT<internalCS1> internal1Grad;
    coord::HessT<internalCS1> internal1Hess;
    bool haveInternal1Pos = false;   // will be changed to true if any of the component needs it

    // same for internalCS2
    coord::PosT<internalCS2> internal2Pos;
    coord::PosDerivT <outputCS, internalCS2> coord2Deriv;
    coord::PosDeriv2T<outputCS, internalCS2> coord2Deriv2;
    coord::GradT<internalCS2> internal2Grad;
    coord::HessT<internalCS2> internal2Hess;
    bool haveInternal2Pos = false;   // will be changed to true if any of the component needs it

    // loop over potential components and evaluate each one in its preferred coordinate system
    for(unsigned int i=0; i<components.size(); i++)
        switch(componentTypes[i]) {
        case CoordT<internalCS1>::id : {
            // the preferred coordinate system for this component is internalCS1:
            // check if we have already transformed the input coords into this system
            if(!haveInternal1Pos) {
                internal1Pos = needDeriv ?
                    coord::toPosDeriv<outputCS, internalCS1>(
                        pos, &coord1Deriv, needDeriv2 ? &coord1Deriv2 : NULL) :
                    coord::toPos<outputCS, internalCS1>(pos);
                coord::clear(internal1Grad);
                coord::clear(internal1Hess);
                haveInternal1Pos = true;
            }
            // evaluate this component of the potential in the internalCS1
            double pot;
            coord::GradT<internalCS1> grad;
            coord::HessT<internalCS1> hess;
            components[i]->eval(internal1Pos,
                potential? &pot : NULL, needDeriv?  &grad : NULL, needDeriv2? &hess : NULL, time);
            // accumulate results in the internalCS1
            if(potential) *potential += pot;
            if(needDeriv)  coord::combine(internal1Grad, grad);
            if(needDeriv2) coord::combine(internal1Hess, hess);
            break;
        }
        case CoordT<internalCS2>::id : {
            // same as above, but for internalCS2
            if(!haveInternal2Pos) {
                internal2Pos = needDeriv ?
                    coord::toPosDeriv<outputCS, internalCS2>(
                        pos, &coord2Deriv, needDeriv2 ? &coord2Deriv2 : NULL) :
                    coord::toPos<outputCS, internalCS2>(pos);
                coord::clear(internal2Grad);
                coord::clear(internal2Hess);
                haveInternal2Pos = true;
            }
            // evaluate this component of the potential in the internalCS2
            double pot;
            coord::GradT<internalCS2> grad;
            coord::HessT<internalCS2> hess;
            components[i]->eval(internal2Pos,
                potential? &pot : NULL, needDeriv?  &grad : NULL, needDeriv2? &hess : NULL, time);
            // accumulate results in the internalCS2
            if(potential) *potential += pot;
            if(needDeriv)  coord::combine(internal2Grad, grad);
            if(needDeriv2) coord::combine(internal2Hess, hess);
            break;
        }
        default : {
            // the preferred coordinate system for this component is outputCS
            // (or maybe something else, but we will anyway evaluate it in outputCS)
            double pot;
            coord::GradT<outputCS> grad;
            coord::HessT<outputCS> hess;
            components[i]->eval(pos,
                potential? &pot : NULL, deriv?  &grad : NULL, deriv2? &hess : NULL, time);
            // accumulate results in the outputCS
            if(potential) *potential += pot;
            if(deriv)  coord::combine(*deriv,  grad);
            if(deriv2) coord::combine(*deriv2, hess);
        }
    }

    // if the gradient/hessian of any of the components were evaluated in the internalCS1,
    // now need to transform them back to the outputCS and add to the output grad/hess
    if(haveInternal1Pos && deriv)
        coord::combine(*deriv, coord::toGrad<internalCS1, outputCS> (internal1Grad, coord1Deriv));
    if(haveInternal1Pos && deriv2)
        coord::combine(*deriv2, coord::toHess<internalCS1, outputCS> (
            internal1Grad, internal1Hess, coord1Deriv, coord1Deriv2));
    // same for internalCS2
    if(haveInternal2Pos && deriv)
        coord::combine(*deriv, coord::toGrad<internalCS2, outputCS> (internal2Grad, coord2Deriv));
    if(haveInternal2Pos && deriv2)
        coord::combine(*deriv2, coord::toHess<internalCS2, outputCS> (
            internal2Grad, internal2Hess, coord2Deriv, coord2Deriv2));
}

/** search a sorted array for a linear interpolator and determine the interpolation weights */
inline void searchInterp(
    /*input: value to search for*/ double val,
    /*input: array to search in (sorted)*/ const std::vector<double>& arr,
    /*input: whether to interpolate within segments or take the nearest node*/ bool interpLinear,
    /*output: index of the leftmost node*/ ptrdiff_t& index,
    /*output: weight of this node (between 0 and 1)*/ double& weightLeft)
{
    ptrdiff_t size = arr.size();
    index = math::binSearch(val, &arr.front(), size);
    if(index<0) {
        index = 0;
        weightLeft = 1;
    } else if(index>=size-1) {
        index = size-1;
        weightLeft = 1;
    } else {
        if(interpLinear)
            weightLeft = (arr[index+1] - val) / (arr[index+1] - arr[index]);
        else { // take the nearest of the two timestamps (index or index+1), leaving weight=1
            if( arr[index+1] - val < val - arr[index] )
                index++;
            weightLeft = 1;
        }
    }
}

}  // namespace


CompositeDensity::CompositeDensity(const std::vector<PtrDensity>& _components) : 
    BaseDensity(), components(_components)
{
    if(_components.empty())
        throw std::invalid_argument("CompositeDensity: List of density components cannot be empty");
}

double CompositeDensity::densityCar(const coord::PosCar &pos, double time) const {
    double sum=0;
    for(unsigned int i=0; i<components.size(); i++) 
        sum += components[i]->density(pos, time);
    return sum;
}

double CompositeDensity::densityCyl(const coord::PosCyl &pos, double time) const {
    double sum=0;
    for(unsigned int i=0; i<components.size(); i++) 
        sum += components[i]->density(pos, time);
    return sum;
}

double CompositeDensity::densitySph(const coord::PosSph &pos, double time) const {
    double sum=0;
    for(unsigned int i=0; i<components.size(); i++) 
        sum += components[i]->density(pos, time);
    return sum;
}

void CompositeDensity::evalmanyDensityCar(const size_t npoints, const coord::PosCar pos[],
    /*output*/ double values[], /*input*/ double time) const
{
    components[0]->evalmanyDensityCar(npoints, pos, values, time);
    ALLOC(npoints, double, tmpvalues)
    for(unsigned int i=1; i<components.size(); i++) {
        components[i]->evalmanyDensityCar(npoints, pos, tmpvalues, time);
        for(size_t p=0; p<npoints; p++)
            values[p] += tmpvalues[p];
    }
}

void CompositeDensity::evalmanyDensityCyl(const size_t npoints, const coord::PosCyl pos[],
    /*output*/ double values[], /*input*/ double time) const
{
    components[0]->evalmanyDensityCyl(npoints, pos, values, time);
    ALLOC(npoints, double, tmpvalues)
    for(unsigned int i=1; i<components.size(); i++) {
        components[i]->evalmanyDensityCyl(npoints, pos, tmpvalues, time);
        for(size_t p=0; p<npoints; p++)
            values[p] += tmpvalues[p];
    }
}

void CompositeDensity::evalmanyDensitySph(const size_t npoints, const coord::PosSph pos[],
    /*output*/ double values[], /*input*/ double time) const
{
    components[0]->evalmanyDensitySph(npoints, pos, values, time);
    ALLOC(npoints, double, tmpvalues)
    for(unsigned int i=1; i<components.size(); i++) {
        components[i]->evalmanyDensitySph(npoints, pos, tmpvalues, time);
        for(size_t p=0; p<npoints; p++)
            values[p] += tmpvalues[p];
    }
}

double CompositeDensity::totalMass() const {
    double sum = 0;
    for(unsigned int i=0; i<components.size(); i++)
        sum += components[i]->totalMass();
    return sum;
}

coord::SymmetryType CompositeDensity::symmetry() const {
    int sym = static_cast<int>(coord::ST_SPHERICAL);
    for(unsigned int index=0; index<components.size(); index++) {
        coord::SymmetryType csym = components[index]->symmetry();
        if(isUnknown(csym))
            return csym;  // contagious value, overrides all other variants
        sym &= csym;
    }
    return static_cast<coord::SymmetryType>(sym);
}

std::string CompositeDensity::name() const
{
    std::string name = "CompositeDensity{ ";
    for(unsigned int i=0; i<components.size(); i++) {
        if(i>0) name += ", ";
        name += components[i]->name();
    }
    return name + " }";
}

Composite::Composite(const std::vector<PtrPotential>& _components) :
    BasePotential(), components(_components), componentTypes(components.size())
{
    if(_components.empty())
        throw std::invalid_argument("Composite: List of potential components cannot be empty");
    // determine the preferred coordinate systems for each component
    for(size_t i=0; i<components.size(); i++) {
        const BasePotential* ptr = components[i].get();
        // if the dynamic cast succeeds (does not return NULL),
        // the potential is derived from the base class being tested
        if     (dynamic_cast<const BasePotentialCar*>(ptr) != NULL)
            componentTypes[i] = CoordT<coord::Car>::id;   // prefer cartesian
        else if(dynamic_cast<const BasePotentialCyl*>(ptr) != NULL)
            componentTypes[i] = CoordT<coord::Cyl>::id;   // prefer cylindrical
        else if(dynamic_cast<const BasePotentialSph*>(ptr) != NULL)
            componentTypes[i] = CoordT<coord::Sph>::id;   // prefer spherical
        // otherwise (e.g. for spherically-symmetric potentials or for Composite)
        // evaluate them in the same coordinate system as the input point
    }
}

void Composite::evalCar(const coord::PosCar &pos,
    double* potential, coord::GradCar* deriv, coord::HessCar* deriv2, double time) const
{
    evalComponents<coord::Cyl, coord::Sph>(
        components, componentTypes, pos, potential, deriv, deriv2, time);
}

void Composite::evalCyl(const coord::PosCyl &pos,
    double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2, double time) const
{
    evalComponents<coord::Car, coord::Sph>(
        components, componentTypes, pos, potential, deriv, deriv2, time);
}

void Composite::evalSph(const coord::PosSph &pos,
    double* potential, coord::GradSph* deriv, coord::HessSph* deriv2, double time) const
{
    evalComponents<coord::Car, coord::Cyl>(
        components, componentTypes, pos, potential, deriv, deriv2, time);
}

double Composite::densityCar(const coord::PosCar &pos, double time) const
{
    double sum=0;
    for(unsigned int i=0; i<components.size(); i++)
        sum += components[i]->density(pos, time);
    return sum;
}
double Composite::densityCyl(const coord::PosCyl &pos, double time) const
{
    double sum=0;
    for(unsigned int i=0; i<components.size(); i++)
        sum += components[i]->density(pos, time);
    return sum;
}
double Composite::densitySph(const coord::PosSph &pos, double time) const
{
    double sum=0;
    for(unsigned int i=0; i<components.size(); i++)
        sum += components[i]->density(pos, time);
    return sum;
}

void Composite::evalmanyDensityCar(const size_t npoints, const coord::PosCar pos[],
    /*output*/ double values[], /*input*/ double time) const
{
    components[0]->evalmanyDensityCar(npoints, pos, values, time);
    ALLOC(npoints, double, tmpvalues)
    for(unsigned int i=1; i<components.size(); i++) {
        components[i]->evalmanyDensityCar(npoints, pos, tmpvalues, time);
        for(size_t p=0; p<npoints; p++)
            values[p] += tmpvalues[p];
    }
}

void Composite::evalmanyDensityCyl(const size_t npoints, const coord::PosCyl pos[],
    /*output*/ double values[], /*input*/ double time) const
{
    components[0]->evalmanyDensityCyl(npoints, pos, values, time);
    ALLOC(npoints, double, tmpvalues)
    for(unsigned int i=1; i<components.size(); i++) {
        components[i]->evalmanyDensityCyl(npoints, pos, tmpvalues, time);
        for(size_t p=0; p<npoints; p++)
            values[p] += tmpvalues[p];
    }
}

void Composite::evalmanyDensitySph(const size_t npoints, const coord::PosSph pos[],
    /*output*/ double values[], /*input*/ double time) const
{
    components[0]->evalmanyDensitySph(npoints, pos, values, time);
    ALLOC(npoints, double, tmpvalues)
    for(unsigned int i=1; i<components.size(); i++) {
        components[i]->evalmanyDensitySph(npoints, pos, tmpvalues, time);
        for(size_t p=0; p<npoints; p++)
            values[p] += tmpvalues[p];
    }
}

double Composite::totalMass() const
{
    double sum = 0;
    for(unsigned int i=0; i<components.size(); i++)
        sum += components[i]->totalMass();
    return sum;
}

coord::SymmetryType Composite::symmetry() const
{
    int sym = static_cast<int>(coord::ST_SPHERICAL);
    for(unsigned int index=0; index<components.size(); index++) {
        coord::SymmetryType csym = components[index]->symmetry();
        if(isUnknown(csym))
            return csym;  // contagious value, overrides all other variants
        sym &= csym;
    }
    return static_cast<coord::SymmetryType>(sym);
}

std::string Composite::name() const
{
    std::string name = "CompositePotential{ ";
    for(unsigned int i=0; i<components.size(); i++) {
        if(i>0) name += ", ";
        name += components[i]->name();
    }
    return name + " }";
}


Evolving::Evolving(const std::vector<double> _times,
    const std::vector<PtrPotential> _instances,
    bool _interpLinear)
:
    times(_times), instances(_instances), interpLinear(_interpLinear)
{
    if(times.size() != instances.size())
        throw std::length_error("Evolving: input arrays are not equal in length");
    if(times.size() == 0)
        throw std::invalid_argument("Evolving: empty list of potentials");
    sym = instances[0]->symmetry();
    for(size_t i=1; i<times.size(); i++) {
        if(!(times[i] > times[i-1]))
            throw std::invalid_argument("Evolving: times must be sorted in increasing order");
        coord::SymmetryType isym = instances[i]->symmetry();
        if(isUnknown(isym) || isUnknown(sym))
            sym = coord::ST_UNKNOWN;  // contagious value
        else
            sym = static_cast<coord::SymmetryType>(sym & isym);
    }
}

void Evolving::evalCar(const coord::PosCar &pos,
    double* potential, coord::GradCar* deriv, coord::HessCar* deriv2, double time) const
{
    ptrdiff_t index;
    double weight;
    searchInterp(time, times, interpLinear, /*output*/ index, weight);
    instances[index]->eval(pos, potential, deriv, deriv2, time);
    if(weight!=1) {
        // evaluate the potential at the other time stamp and interpolate between them
        double pot;
        coord::GradCar grad;
        coord::HessCar hess;
        instances[index+1]->eval(pos,
            potential? &pot : NULL, deriv? &grad : NULL, deriv2? &hess : NULL, time);
        if(potential)
            *potential = weight * (*potential) + (1-weight) * pot;
        if(deriv)
            coord::combine(*deriv,  grad, weight, 1-weight);
        if(deriv2)
            coord::combine(*deriv2, hess, weight, 1-weight);
    }
}

double Evolving::densityCar(const coord::PosCar &pos, double time) const
{
    ptrdiff_t index;
    double weight;
    searchInterp(time, times, interpLinear, /*output*/ index, weight);
    double result = instances[index]->density(pos, time);
    if(weight!=1)
        result = weight * result + (1-weight) * instances[index+1]->density(pos, time);
    return result;
}

//--------- Modifier classes --------//

// common function for evaluating density in the given coordinate system,
// shared between Shifted<BaseDensity> and Shifted<BasePotential>
template<typename CoordT>
inline void evalmanyShifted(const BaseDensity& dens,
    const double centerx, const double centery, const double centerz,
    const size_t npoints, const coord::PosT<CoordT> pos[],
    /*output*/ double values[], /*input*/ double time)
{
    ALLOC(npoints, coord::PosCar, poscar)
    for(size_t i=0; i<npoints; i++) {
        poscar[i] = toPosCar(pos[i]);
        poscar[i].x -= centerx;
        poscar[i].y -= centery;
        poscar[i].z -= centerz;
    }
    dens.evalmanyDensityCar(npoints, poscar, values, time);
}

void Shifted<BaseDensity>::evalmanyDensityCar(const size_t npoints, const coord::PosCar pos[],
    /*output*/ double values[], /*input*/ double time) const
{ evalmanyShifted(*dens, centerx(time), centery(time), centerz(time), npoints, pos, values, time); }

void Shifted<BaseDensity>::evalmanyDensityCyl(const size_t npoints, const coord::PosCyl pos[],
    /*output*/ double values[], /*input*/ double time) const
{ evalmanyShifted(*dens, centerx(time), centery(time), centerz(time), npoints, pos, values, time); }

void Shifted<BaseDensity>::evalmanyDensitySph(const size_t npoints, const coord::PosSph pos[],
    /*output*/ double values[], /*input*/ double time) const
{ evalmanyShifted(*dens, centerx(time), centery(time), centerz(time), npoints, pos, values, time); }

void Shifted<BasePotential>::evalmanyDensityCar(const size_t npoints, const coord::PosCar pos[],
    /*output*/ double values[], /*input*/ double time) const
{ evalmanyShifted(*pot, centerx(time), centery(time), centerz(time), npoints, pos, values, time); }

void Shifted<BasePotential>::evalmanyDensityCyl(const size_t npoints, const coord::PosCyl pos[],
    /*output*/ double values[], /*input*/ double time) const
{ evalmanyShifted(*pot, centerx(time), centery(time), centerz(time), npoints, pos, values, time); }

void Shifted<BasePotential>::evalmanyDensitySph(const size_t npoints, const coord::PosSph pos[],
    /*output*/ double values[], /*input*/ double time) const
{ evalmanyShifted(*pot, centerx(time), centery(time), centerz(time), npoints, pos, values, time); }


// common function for evaluating density in the given coordinate system,
// shared between Tilted<BaseDensity> and Tilted<BasePotential>
template<typename CoordT>
inline void evalmanyTilted(const BaseDensity& dens, const coord::Orientation& orientation,
    const size_t npoints, const coord::PosT<CoordT> pos[],
    /*output*/ double values[], /*input*/ double time)
{
    ALLOC(npoints, coord::PosCar, poscar)
    for(size_t i=0; i<npoints; i++)
        poscar[i] = orientation.toRotated(toPosCar(pos[i]));
    dens.evalmanyDensityCar(npoints, poscar, values, time);
}

void Tilted<BaseDensity>::evalmanyDensityCar(const size_t npoints, const coord::PosCar pos[],
    /*output*/ double values[], /*input*/ double time) const
{ evalmanyTilted(*dens, orientation, npoints, pos, values, time); }

void Tilted<BaseDensity>::evalmanyDensityCyl(const size_t npoints, const coord::PosCyl pos[],
    /*output*/ double values[], /*input*/ double time) const
{ evalmanyTilted(*dens, orientation, npoints, pos, values, time); }

void Tilted<BaseDensity>::evalmanyDensitySph(const size_t npoints, const coord::PosSph pos[],
    /*output*/ double values[], /*input*/ double time) const
{ evalmanyTilted(*dens, orientation, npoints, pos, values, time); }

void Tilted<BasePotential>::evalmanyDensityCar(const size_t npoints, const coord::PosCar pos[],
    /*output*/ double values[], /*input*/ double time) const
{ evalmanyTilted(*pot, orientation, npoints, pos, values, time); }

void Tilted<BasePotential>::evalmanyDensityCyl(const size_t npoints, const coord::PosCyl pos[],
    /*output*/ double values[], /*input*/ double time) const
{ evalmanyTilted(*pot, orientation, npoints, pos, values, time); }

void Tilted<BasePotential>::evalmanyDensitySph(const size_t npoints, const coord::PosSph pos[],
    /*output*/ double values[], /*input*/ double time) const
{ evalmanyTilted(*pot, orientation, npoints, pos, values, time); }

coord::SymmetryType Tilted<BaseDensity>::symmetry() const {
    coord::SymmetryType sym = dens->symmetry();
    return isUnknown(sym) || isSpherical(sym) ? sym :
        static_cast<coord::SymmetryType>(sym & coord::ST_REFLECTION);
}

coord::SymmetryType Tilted<BasePotential>::symmetry() const {
    coord::SymmetryType sym = pot->symmetry();
    return isUnknown(sym) || isSpherical(sym) ? sym :
        static_cast<coord::SymmetryType>(sym & coord::ST_REFLECTION);
}


double Rotating<BaseDensity>::densityCar(const coord::PosCar &pos, double time) const
{
    double sa, ca;
    math::sincos(angle(time), sa, ca);
    return dens->density(coord::PosCar(pos.x * ca + pos.y * sa, pos.y * ca - pos.x * sa, pos.z), time);
}

void Rotating<BaseDensity>::evalmanyDensityCar(const size_t npoints, const coord::PosCar pos[],
    /*output*/ double values[], /*input*/ double time) const
{
    ALLOC(npoints, coord::PosCar, poscar)
    double sa, ca;
    math::sincos(angle(time), sa, ca);
    for(size_t i=0; i<npoints; i++)
        poscar[i] = coord::PosCar(pos[i].x * ca + pos[i].y * sa, pos[i].y * ca - pos[i].x * sa, pos[i].z);
    dens->evalmanyDensityCar(npoints, poscar, values, time);
}

void Rotating<BaseDensity>::evalmanyDensityCyl(const size_t npoints, const coord::PosCyl pos[],
    /*output*/ double values[], /*input*/ double time) const
{
    ALLOC(npoints, coord::PosCyl, poscyl)
    double ang = angle(time);
    for(size_t i=0; i<npoints; i++)
        poscyl[i] = coord::PosCyl(pos[i].R, pos[i].z, pos[i].phi - ang);
    dens->evalmanyDensityCyl(npoints, poscyl, values, time);
}

void Rotating<BaseDensity>::evalmanyDensitySph(const size_t npoints, const coord::PosSph pos[],
    /*output*/ double values[], /*input*/ double time) const
{
    ALLOC(npoints, coord::PosSph, possph)
    double ang = angle(time);
    for(size_t i=0; i<npoints; i++)
        possph[i] = coord::PosSph(pos[i].r, pos[i].theta, pos[i].phi - ang);
    dens->evalmanyDensitySph(npoints, possph, values, time);
}

double Rotating<BasePotential>::densityCar(const coord::PosCar &pos, double time) const
{
    double sa, ca;
    math::sincos(angle(time), sa, ca);
    return pot->density(coord::PosCar(pos.x * ca + pos.y * sa, pos.y * ca - pos.x * sa, pos.z), time);
}

void Rotating<BasePotential>::evalmanyDensityCar(const size_t npoints, const coord::PosCar pos[],
    /*output*/ double values[], /*input*/ double time) const
{
    ALLOC(npoints, coord::PosCar, poscar)
    double sa, ca;
    math::sincos(angle(time), sa, ca);
    for(size_t i=0; i<npoints; i++)
        poscar[i] = coord::PosCar(pos[i].x * ca + pos[i].y * sa, pos[i].y * ca - pos[i].x * sa, pos[i].z);
    pot->evalmanyDensityCar(npoints, poscar, values, time);
}

void Rotating<BasePotential>::evalmanyDensityCyl(const size_t npoints, const coord::PosCyl pos[],
    /*output*/ double values[], /*input*/ double time) const
{
    ALLOC(npoints, coord::PosCyl, poscyl)
    double ang = angle(time);
    for(size_t i=0; i<npoints; i++)
        poscyl[i] = coord::PosCyl(pos[i].R, pos[i].z, pos[i].phi - ang);
    pot->evalmanyDensityCyl(npoints, poscyl, values, time);
}

void Rotating<BasePotential>::evalmanyDensitySph(const size_t npoints, const coord::PosSph pos[],
    /*output*/ double values[], /*input*/ double time) const
{
    ALLOC(npoints, coord::PosSph, possph)
    double ang = angle(time);
    for(size_t i=0; i<npoints; i++)
        possph[i] = coord::PosSph(pos[i].r, pos[i].theta, pos[i].phi - ang);
    pot->evalmanyDensitySph(npoints, possph, values, time);
}

void Rotating<BasePotential>::evalCar(const coord::PosCar &pos,
    double* potential, coord::GradCar* deriv, coord::HessCar* deriv2, double time) const
{
    double sa, ca;
    coord::GradCar derivrot;
    coord::HessCar deriv2rot;
    math::sincos(angle(time), sa, ca);
    pot->eval(coord::PosCar(pos.x * ca + pos.y * sa, pos.y * ca - pos.x * sa, pos.z),
        potential, deriv ? &derivrot : NULL, deriv2 ? &deriv2rot : NULL, time);
    if(deriv) {
        deriv->dx = derivrot.dx * ca - derivrot.dy * sa;
        deriv->dy = derivrot.dy * ca + derivrot.dx * sa;
        deriv->dz = derivrot.dz;
    }
    if(deriv2) {
        deriv2->dx2 = ca*ca * deriv2rot.dx2 + sa*sa * deriv2rot.dy2 - 2*ca*sa * deriv2rot.dxdy;
        deriv2->dy2 = sa*sa * deriv2rot.dx2 + ca*ca * deriv2rot.dy2 + 2*ca*sa * deriv2rot.dxdy;
        deriv2->dxdy= ca*sa * (deriv2rot.dx2 - deriv2rot.dy2) + (ca*ca-sa*sa) * deriv2rot.dxdy;
        deriv2->dxdz= ca * deriv2rot.dxdz - sa * deriv2rot.dydz;
        deriv2->dydz= ca * deriv2rot.dydz + sa * deriv2rot.dxdz;
        deriv2->dz2 = deriv2rot.dz2;
    }
}

void Rotating<BasePotential>::evalCyl(const coord::PosCyl &pos,
    double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2, double time) const
{
    pot->eval(coord::PosCyl(pos.R, pos.z, pos.phi - angle(time)), potential, deriv, deriv2, time);
}

void Rotating<BasePotential>::evalSph(const coord::PosSph &pos,
    double* potential, coord::GradSph* deriv, coord::HessSph* deriv2, double time) const
{
    pot->eval(coord::PosSph(pos.r, pos.theta, pos.phi - angle(time)), potential, deriv, deriv2, time);
}

coord::SymmetryType Rotating<BaseDensity>::symmetry() const {
    coord::SymmetryType sym = dens->symmetry();
    return isUnknown(sym) || isZRotSymmetric(sym) ? sym :
        static_cast<coord::SymmetryType>(sym & coord::ST_BISYMMETRIC);
}

coord::SymmetryType Rotating<BasePotential>::symmetry() const {
    coord::SymmetryType sym = pot->symmetry();
    return isUnknown(sym) || isZRotSymmetric(sym) ? sym :
        static_cast<coord::SymmetryType>(sym & coord::ST_BISYMMETRIC);
}


void Scaled<BaseDensity>::evalmanyDensityCar(const size_t npoints, const coord::PosCar pos[],
    /*output*/ double values[], /*input*/ double time) const
{
    double s = 1 / scale(time), as3 = pow_3(s) * ampl(time);
    ALLOC(npoints, coord::PosCar, poscar)
    for(size_t i=0; i<npoints; i++)
        poscar[i] = coord::PosCar(pos[i].x * s, pos[i].y * s, pos[i].z * s);
    dens->evalmanyDensityCar(npoints, poscar, values, time);
    for(size_t i=0; i<npoints; i++)
        values[i] *= as3;
}

void Scaled<BaseDensity>::evalmanyDensityCyl(const size_t npoints, const coord::PosCyl pos[],
    /*output*/ double values[], /*input*/ double time) const
{
    double s = 1 / scale(time), as3 = pow_3(s) * ampl(time);
    ALLOC(npoints, coord::PosCyl, poscyl)
    for(size_t i=0; i<npoints; i++)
        poscyl[i] = coord::PosCyl(pos[i].R * s, pos[i].z * s, pos[i].phi);
    dens->evalmanyDensityCyl(npoints, poscyl, values, time);
    for(size_t i=0; i<npoints; i++)
        values[i] *= as3;
}

void Scaled<BaseDensity>::evalmanyDensitySph(const size_t npoints, const coord::PosSph pos[],
    /*output*/ double values[], /*input*/ double time) const
{
    double s = 1 / scale(time), as3 = pow_3(s) * ampl(time);
    ALLOC(npoints, coord::PosSph, possph)
    for(size_t i=0; i<npoints; i++)
        possph[i] = coord::PosSph(pos[i].r * s, pos[i].theta, pos[i].phi);
    dens->evalmanyDensitySph(npoints, possph, values, time);
    for(size_t i=0; i<npoints; i++)
        values[i] *= as3;
}

void Scaled<BasePotential>::evalmanyDensityCar(const size_t npoints, const coord::PosCar pos[],
    /*output*/ double values[], /*input*/ double time) const
{
    double s = 1 / scale(time), as3 = pow_3(s) * ampl(time);
    ALLOC(npoints, coord::PosCar, poscar)
    for(size_t i=0; i<npoints; i++)
        poscar[i] = coord::PosCar(pos[i].x * s, pos[i].y * s, pos[i].z * s);
    pot->evalmanyDensityCar(npoints, poscar, values, time);
    for(size_t i=0; i<npoints; i++)
        values[i] *= as3;
}

void Scaled<BasePotential>::evalmanyDensityCyl(const size_t npoints, const coord::PosCyl pos[],
    /*output*/ double values[], /*input*/ double time) const
{
    double s = 1 / scale(time), as3 = pow_3(s) * ampl(time);
    ALLOC(npoints, coord::PosCyl, poscyl)
    for(size_t i=0; i<npoints; i++)
        poscyl[i] = coord::PosCyl(pos[i].R * s, pos[i].z * s, pos[i].phi);
    pot->evalmanyDensityCyl(npoints, poscyl, values, time);
    for(size_t i=0; i<npoints; i++)
        values[i] *= as3;
}

void Scaled<BasePotential>::evalmanyDensitySph(const size_t npoints, const coord::PosSph pos[],
    /*output*/ double values[], /*input*/ double time) const
{
    double s = 1 / scale(time), as3 = pow_3(s) * ampl(time);
    ALLOC(npoints, coord::PosSph, possph)
    for(size_t i=0; i<npoints; i++)
        possph[i] = coord::PosSph(pos[i].r * s, pos[i].theta, pos[i].phi);
    pot->evalmanyDensitySph(npoints, possph, values, time);
    for(size_t i=0; i<npoints; i++)
        values[i] *= as3;
}

void Scaled<BasePotential>::evalCar(const coord::PosCar &pos,
    double* potential, coord::GradCar* deriv, coord::HessCar* deriv2, double time) const
{
    double a = ampl(time);
    if(a == 0) {   // when amplitude is zero, skip evaluating the potential
        if(potential)  *potential = 0;
        if(deriv)      coord::clear(*deriv);
        if(deriv2)     coord::clear(*deriv2);
        return;
    }
    double s = 1 / scale(time), as1 = a * s, as2 = as1 * s, as3 = as2 * s;
    pot->eval(coord::PosCar(pos.x * s, pos.y * s, pos.z * s), potential, deriv, deriv2, time);
    if(potential)  *potential *= as1;
    if(deriv)  {
        deriv->dx *= as2;
        deriv->dy *= as2;
        deriv->dz *= as2;
    }
    if(deriv2) {
        deriv2->dx2  *= as3;
        deriv2->dy2  *= as3;
        deriv2->dz2  *= as3;
        deriv2->dxdy *= as3;
        deriv2->dxdz *= as3;
        deriv2->dydz *= as3;
    }
}

void Scaled<BasePotential>::evalCyl(const coord::PosCyl &pos,
    double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2, double time) const
{
    double a = ampl(time);
    if(a == 0) {   // when amplitude is zero, skip evaluating the potential
        if(potential)  *potential = 0;
        if(deriv)      coord::clear(*deriv);
        if(deriv2)     coord::clear(*deriv2);
        return;
    }
    double s = 1 / scale(time), as1 = a * s, as2 = as1 * s, as3 = as2 * s;
    pot->eval(coord::PosCyl(pos.R * s, pos.z * s, pos.phi), potential, deriv, deriv2, time);
    if(potential)  *potential *= as1;
    if(deriv)  {
        deriv->dR   *= as2;
        deriv->dz   *= as2;
        deriv->dphi *= as1;
    }
    if(deriv2) {
        deriv2->dR2    *= as3;
        deriv2->dz2    *= as3;
        deriv2->dphi2  *= as1;
        deriv2->dRdz   *= as3;
        deriv2->dRdphi *= as2;
        deriv2->dzdphi *= as2;
    }
}

void Scaled<BasePotential>::evalSph(const coord::PosSph &pos,
    double* potential, coord::GradSph* deriv, coord::HessSph* deriv2, double time) const
{
    double a = ampl(time);
    if(a == 0) {   // when amplitude is zero, skip evaluating the potential
        if(potential)  *potential = 0;
        if(deriv)      coord::clear(*deriv);
        if(deriv2)     coord::clear(*deriv2);
        return;
    }
    double s = 1 / scale(time), as1 = a * s, as2 = as1 * s, as3 = as2 * s;
    pot->eval(coord::PosSph(pos.r * s, pos.theta, pos.phi), potential, deriv, deriv2, time);
    if(potential)  *potential *= as1;
    if(deriv)  {
        deriv->dr *= as2;
        deriv->dtheta *= as1;
        deriv->dphi *= as1;
    }
    if(deriv2) {
        deriv2->dr2        *= as3;
        deriv2->dtheta2    *= as1;
        deriv2->dphi2      *= as1;
        deriv2->drdtheta   *= as2;
        deriv2->drdphi     *= as2;
        deriv2->dthetadphi *= as1;
    }
}

} // namespace potential
