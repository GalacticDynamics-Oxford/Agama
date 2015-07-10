/**  \brief   Composite density and potential classes
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
    virtual SYMMETRYTYPE symmetry() const;

private:
    std::vector<const BaseDensity*> components;
    virtual double density_car(const coord::PosCar &pos) const;
    virtual double density_cyl(const coord::PosCyl &pos) const;
    virtual double density_sph(const coord::PosSph &pos) const;
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
    virtual SYMMETRYTYPE symmetry() const;

private:
    std::vector<const BasePotential*> components;
    virtual void eval_cyl(const coord::PosCyl &pos,
        double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const;
};

}  // namespace potential