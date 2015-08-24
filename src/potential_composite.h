/** \file    potential_composite.h
    \brief   Composite density and potential classes
    \author  Eugene Vasiliev
    \date    2014-2015
 
\note: the semantics of creating a composite density and potential classes
is far from ideal because the user needs to create each component manually,
then create the composite, and _do_not_ delete the components - this will 
be done by the composite's destructor. A better design approach is needed.
*/
#pragma once
#include "potential_base.h"
#include <vector>

namespace potential{

/** A trivial collection of several density objects */
class CompositeDensity: public BaseDensity{
public:
    /** Construct from the provided array of components; 
        note that they are 'taken over' and will be deleted in the destructor */
    CompositeDensity(const std::vector<const BaseDensity*> _components) : 
        BaseDensity(), components(_components) {};

    virtual ~CompositeDensity() {
        for(unsigned int i=0; i<components.size(); i++) delete components[i]; }

    /** provides the 'least common denominator' for the symmetry degree */
    virtual SymmetryType symmetry() const;

    virtual const char* name() const { return myName(); };
    static const char* myName() { return "CompositeDensity"; };

private:
    std::vector<const BaseDensity*> components;
    virtual double densityCar(const coord::PosCar &pos) const;
    virtual double densityCyl(const coord::PosCyl &pos) const;
    virtual double densitySph(const coord::PosSph &pos) const;
};

/** A trivial collection of several potential objects, evaluated in cylindrical coordinates */
class CompositeCyl: public BasePotentialCyl{
public:
    /** Construct from the provided array of components; 
        note that they are 'taken over' and will be deleted in the destructor */
    CompositeCyl(const std::vector<const BasePotential*> _components) : 
        BasePotentialCyl(), components(_components) {};

    virtual ~CompositeCyl() {
        for(unsigned int i=0; i<components.size(); i++) delete components[i]; }

    /** provides the 'least common denominator' for the symmetry degree */
    virtual SymmetryType symmetry() const;

    virtual const char* name() const { return myName(); };
    static const char* myName() { return "CompositePotential"; };

private:
    std::vector<const BasePotential*> components;
    virtual void evalCyl(const coord::PosCyl &pos,
        double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const;
};

}  // namespace potential