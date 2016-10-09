/** \file    potential_composite.h
    \brief   Composite density and potential classes
    \author  Eugene Vasiliev
    \date    2014-2015
*/
#pragma once
#include "potential_base.h"
#include "smart.h"
#include <vector>

namespace potential{

/** A trivial collection of several density objects */
class CompositeDensity: public BaseDensity{
public:
    /** construct from the provided array of components */
    CompositeDensity(const std::vector<PtrDensity>& _components);

    /** provides the 'least common denominator' for the symmetry degree */
    virtual coord::SymmetryType symmetry() const;

    virtual const char* name() const { return myName(); }
    static const char* myName() { static const char* text = "CompositeDensity"; return text; }

    unsigned int size() const { return components.size(); }
    PtrDensity component(unsigned int index) const { return components.at(index); }
private:
    std::vector<PtrDensity> components;
    virtual double densityCar(const coord::PosCar &pos) const;
    virtual double densityCyl(const coord::PosCyl &pos) const;
    virtual double densitySph(const coord::PosSph &pos) const;
};

/** A trivial collection of several potential objects, evaluated in cylindrical coordinates */
class CompositeCyl: public BasePotentialCyl{
public:
    /** construct from the provided array of components */ 
    CompositeCyl(const std::vector<PtrPotential>& _components);
    
    /** provides the 'least common denominator' for the symmetry degree */
    virtual coord::SymmetryType symmetry() const;

    virtual const char* name() const { return myName(); };
    static const char* myName() { static const char* text = "CompositePotential"; return text; }

    unsigned int size() const { return components.size(); }
    PtrPotential component(unsigned int index) const { return components.at(index); }
private:
    std::vector<PtrPotential> components;
    virtual void evalCyl(const coord::PosCyl &pos,
        double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const;
};

}  // namespace potential