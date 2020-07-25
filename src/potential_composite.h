/** \file    potential_composite.h
    \brief   Composite density and potential classes
    \author  Eugene Vasiliev
    \date    2014-2015
*/
#pragma once
#include "potential_base.h"
#include "smart.h"
#include "math_spline.h"

namespace potential{

/** A trivial collection of several density objects */
class CompositeDensity: public BaseDensity{
public:
    /** construct from the provided array of components */
    CompositeDensity(const std::vector<PtrDensity>& _components);

    /** provides the 'least common denominator' for the symmetry degree */
    virtual coord::SymmetryType symmetry() const;
    
    /** sum up masses of all components */
    virtual double totalMass() const;

    virtual const char* name() const { return myName(); }
    static const char* myName() { static const char* text = "CompositeDensity"; return text; }

    unsigned int size() const { return components.size(); }
    PtrDensity component(unsigned int index) const { return components.at(index); }
private:
    std::vector<PtrDensity> components;
    virtual double densityCar(const coord::PosCar &pos, double time) const;
    virtual double densityCyl(const coord::PosCyl &pos, double time) const;
    virtual double densitySph(const coord::PosSph &pos, double time) const;
};


/** A trivial collection of several potential objects */
class Composite: public BasePotential {
public:
    /** construct from the provided array of components */
    Composite(const std::vector<PtrPotential>& _components);

    /** provides the 'least common denominator' for the symmetry degree */
    virtual coord::SymmetryType symmetry() const;

    /** sum up masses of all components */
    virtual double totalMass() const;

    virtual const char* name() const { return myName(); };
    static const char* myName() { static const char* text = "CompositePotential"; return text; }

    unsigned int size() const { return components.size(); }
    PtrPotential component(unsigned int index) const { return components.at(index); }

private:
    std::vector<PtrPotential> components;
    std::vector<char> componentTypes;
    virtual void evalCar(const coord::PosCar &pos,
        double* potential, coord::GradCar* deriv, coord::HessCar* deriv2, double time) const;
    virtual void evalCyl(const coord::PosCyl &pos,
        double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2, double time) const;
    virtual void evalSph(const coord::PosSph &pos,
        double* potential, coord::GradSph* deriv, coord::HessSph* deriv2, double time) const;
    virtual double densityCar(const coord::PosCar &pos, double time) const;
    virtual double densityCyl(const coord::PosCyl &pos, double time) const;
    virtual double densitySph(const coord::PosSph &pos, double time) const;
};


class ShiftedDensity: public BaseDensity {
public:
    /// the instance of the actual density
    const PtrDensity dens;

    /// initialize from the given density and splines representing time-dependent offsets
    ShiftedDensity(const PtrDensity& _dens,
        const math::CubicSpline& _centerx,
        const math::CubicSpline& _centery,
        const math::CubicSpline& _centerz)
    :
        dens(_dens), centerx(_centerx), centery(_centery), centerz(_centerz) {}

    /// convenience constructor using fixed offset values
    ShiftedDensity(const PtrDensity& _dens, double x, double y, double z) :
        dens(_dens),
        centerx(math::constantInterp<math::CubicSpline>(x)),
        centery(math::constantInterp<math::CubicSpline>(y)),
        centerz(math::constantInterp<math::CubicSpline>(z)) {}
    
    virtual double totalMass() const { return dens->totalMass(); }
    virtual coord::SymmetryType symmetry() const { return coord::ST_NONE; }  // in general...
    virtual const char* name() const { return dens->name(); };  // pretend to be that guy

private:
    /// possibly time-dependent offsets of the potential center from origin
    const math::CubicSpline centerx, centery, centerz;

    virtual double densityCar(const coord::PosCar &pos, double time) const
    {
        return dens->density(
            coord::PosCar(pos.x-centerx(time), pos.y-centery(time), pos.z-centerz(time)));
    }

    virtual double densityCyl(const coord::PosCyl &pos, double time) const
    { return densityCar(toPosCar(pos), time); }

    virtual double densitySph(const coord::PosSph &pos, double time) const
    { return densityCar(toPosCar(pos), time); }
};


class Shifted: public BasePotentialCar {
public:
    /// the instance of the actual potential
    const PtrPotential pot;

    /// initialize from the given potential and splines representing time-dependent offsets
    Shifted(const PtrPotential& _pot,
        const math::CubicSpline& _centerx,
        const math::CubicSpline& _centery,
        const math::CubicSpline& _centerz)
    :
        pot(_pot), centerx(_centerx), centery(_centery), centerz(_centerz) {}

    /// convenience constructor using fixed offset values
    Shifted(const PtrPotential& _pot, double x, double y, double z) :
        pot(_pot),
        centerx(math::constantInterp<math::CubicSpline>(x)),
        centery(math::constantInterp<math::CubicSpline>(y)),
        centerz(math::constantInterp<math::CubicSpline>(z)) {}

    virtual double totalMass() const { return pot->totalMass(); }
    virtual coord::SymmetryType symmetry() const { return coord::ST_NONE; }  // in general...
    virtual const char* name() const { return pot->name(); };  // pretend to be that guy

private:
    /// possibly time-dependent offsets of the potential center from origin
    const math::CubicSpline centerx, centery, centerz;

    virtual void evalCar(const coord::PosCar &pos,
        double* potential, coord::GradCar* deriv, coord::HessCar* deriv2, double time) const
    {
        return pot->eval(
            coord::PosCar(pos.x-centerx(time), pos.y-centery(time), pos.z-centerz(time)),
            potential, deriv, deriv2, time);
    }

    virtual double densityCar(const coord::PosCar &pos, double time) const
    {
        return pot->density(
            coord::PosCar(pos.x-centerx(time), pos.y-centery(time), pos.z-centerz(time)));
    }

    virtual double densityCyl(const coord::PosCyl &pos, double time) const
    { return densityCar(toPosCar(pos), time); }

    virtual double densitySph(const coord::PosSph &pos, double time) const
    { return densityCar(toPosCar(pos), time); }
};


class Evolving: public BasePotentialCar {
public:
    Evolving(const std::vector<double> _times,
        const std::vector<PtrPotential> _instances,
        bool _interpLinear=false);
    virtual coord::SymmetryType symmetry() const { return coord::ST_NONE; }  // in general...
    virtual const char* name() const { return myName(); };
    static const char* myName() { static const char* text = "Evolving"; return text; }

private:
    /// array of time stamps for a time-dependent potential
    std::vector<double> times;
    /// array of potentials corresponding to each moment of time (possibly just one)
    std::vector<PtrPotential> instances;
    /// use linear or nearest-point interpolation of a time-dependent potential
    bool interpLinear;

    virtual void evalCar(const coord::PosCar &pos,
        double* potential, coord::GradCar* deriv, coord::HessCar* deriv2, double time) const;
    virtual double densityCar(const coord::PosCar &pos, double time) const;
};


class UniformAcceleration: public BasePotentialCar {
public:
    /// initialize from a triplet of splines representing time-dependent acceleration
    UniformAcceleration(const math::CubicSpline& _accx,
        const math::CubicSpline& _accy, const math::CubicSpline& _accz)
    :
        accx(_accx), accy(_accy), accz(_accz) {}

    virtual coord::SymmetryType symmetry() const { return coord::ST_NONE; }
    virtual const char* name() const { return myName(); };
    static const char* myName() { static const char* text = "UniformAcceleration"; return text; }

private:
    const math::CubicSpline accx, accy, accz;

    virtual void evalCar(const coord::PosCar &pos,
        double* potential, coord::GradCar* deriv, coord::HessCar* deriv2, double time) const
    {
        double dx = -accx(time), dy = -accy(time), dz = -accz(time);
        if(potential)
            *potential = pos.x * dx + pos.y * dy + pos.z * dz;
        if(deriv) {
            deriv->dx = dx;
            deriv->dy = dy;
            deriv->dz = dz;
        }
        if(deriv2)
            coord::clear(*deriv2);
    }
};

}  // namespace potential
