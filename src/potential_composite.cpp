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
    for(unsigned int index=0; index<components.size(); index++)
        sym &= static_cast<int>(components[index]->symmetry());
    return static_cast<coord::SymmetryType>(sym);
}

void ShiftedDensity::evalmanyDensityCar(const size_t npoints, const coord::PosCar pos[],
    /*output*/ double values[], /*input*/ double time) const
{
    ALLOC(npoints, coord::PosCar, poscar)
    for(size_t i=0; i<npoints; i++) {
        poscar[i] = pos[i];
        poscar[i].x -= centerx(time);
        poscar[i].y -= centery(time);
        poscar[i].z -= centerz(time);
    }
    dens->evalmanyDensityCar(npoints, poscar, values, time);
}

void ShiftedDensity::evalmanyDensityCyl(const size_t npoints, const coord::PosCyl pos[],
    /*output*/ double values[], /*input*/ double time) const
{
    ALLOC(npoints, coord::PosCar, poscar)
    for(size_t i=0; i<npoints; i++) {
        poscar[i] = toPosCar(pos[i]);
        poscar[i].x -= centerx(time);
        poscar[i].y -= centery(time);
        poscar[i].z -= centerz(time);
    }
    dens->evalmanyDensityCar(npoints, poscar, values, time);
}

void ShiftedDensity::evalmanyDensitySph(const size_t npoints, const coord::PosSph pos[],
    /*output*/ double values[], /*input*/ double time) const
{
    ALLOC(npoints, coord::PosCar, poscar)
    for(size_t i=0; i<npoints; i++) {
        poscar[i] = toPosCar(pos[i]);
        poscar[i].x -= centerx(time);
        poscar[i].y -= centery(time);
        poscar[i].z -= centerz(time);
    }
    dens->evalmanyDensityCar(npoints, poscar, values, time);
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

double Composite::densityCar(const coord::PosCar &pos, double time) const {
    double sum=0;
    for(unsigned int i=0; i<components.size(); i++)
        sum += components[i]->density(pos, time);
    return sum;
}
double Composite::densityCyl(const coord::PosCyl &pos, double time) const {
    double sum=0;
    for(unsigned int i=0; i<components.size(); i++)
        sum += components[i]->density(pos, time);
    return sum;
}
double Composite::densitySph(const coord::PosSph &pos, double time) const {
    double sum=0;
    for(unsigned int i=0; i<components.size(); i++)
        sum += components[i]->density(pos, time);
    return sum;
}

double Composite::totalMass() const {
    double sum = 0;
    for(unsigned int i=0; i<components.size(); i++)
        sum += components[i]->totalMass();
    return sum;
}

coord::SymmetryType Composite::symmetry() const {
    int sym = static_cast<int>(coord::ST_SPHERICAL);
    for(unsigned int index=0; index<components.size(); index++)
        sym &= static_cast<int>(components[index]->symmetry());
    return static_cast<coord::SymmetryType>(sym);
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
    for(size_t i=1; i<times.size(); i++)
        if(!(times[i] > times[i-1]))
            throw std::invalid_argument("Evolving: Times must be sorted in increasing order");
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
        instances[index+1]->eval(pos, potential? &pot : NULL, deriv? &grad : NULL, deriv2? &hess : NULL);
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
    double result = instances[index]->density(pos);
    if(weight!=1)
        result = weight * result + (1-weight) * instances[index+1]->density(pos);
    return result;
}

} // namespace potential
