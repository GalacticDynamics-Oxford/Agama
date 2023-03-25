/** \file    potential_composite.h
    \brief   Composite density and potential classes and various modifiers
    \author  Eugene Vasiliev
    \date    2014-2022
*/
#pragma once
#include "potential_base.h"
#include "smart.h"
#include "math_spline.h"

namespace potential{

/** Interface for composite Density or Potential classes:
    these may contain multiple individual components, as in CompositeDensity or Composite potential,
    or wrap a single object and modify its behaviour, as in Shifted/Rotating/etc.
*/
template<class BaseDensityOrPotential> class BaseComposite {
public:
    virtual ~BaseComposite() {}
    virtual unsigned int size() const = 0;
    virtual shared_ptr<const BaseDensityOrPotential> component(unsigned int index) const = 0;
};

/** A collection of several density objects */
class CompositeDensity: public BaseDensity, public BaseComposite<BaseDensity> {
public:
    /** construct from the provided array of components */
    CompositeDensity(const std::vector<PtrDensity>& _components);

    /** provides the 'least common denominator' for the symmetry degree */
    virtual coord::SymmetryType symmetry() const;

    /** sum up masses of all components */
    virtual double totalMass() const;

    /** joins the names of all components */
    virtual std::string name() const;

    virtual unsigned int size() const { return components.size(); }
    virtual PtrDensity component(unsigned int index) const { return components.at(index); }

private:
    std::vector<PtrDensity> components;
    virtual double densityCar(const coord::PosCar &pos, double time) const;
    virtual double densityCyl(const coord::PosCyl &pos, double time) const;
    virtual double densitySph(const coord::PosSph &pos, double time) const;
    virtual void evalmanyDensityCar(const size_t npoints, const coord::PosCar pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
    virtual void evalmanyDensityCyl(const size_t npoints, const coord::PosCyl pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
    virtual void evalmanyDensitySph(const size_t npoints, const coord::PosSph pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
};


/** A collection of several potential objects */
class Composite: public BasePotential, public BaseComposite<BasePotential> {
public:
    /** construct from the provided array of components */
    Composite(const std::vector<PtrPotential>& _components);

    /** provides the 'least common denominator' for the symmetry degree */
    virtual coord::SymmetryType symmetry() const;

    /** sum up masses of all components */
    virtual double totalMass() const;

    /** joins the names of all components */
    virtual std::string name() const;

    virtual unsigned int size() const { return components.size(); }
    virtual PtrPotential component(unsigned int index) const { return components.at(index); }

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
    virtual void evalmanyDensityCar(const size_t npoints, const coord::PosCar pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
    virtual void evalmanyDensityCyl(const size_t npoints, const coord::PosCyl pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
    virtual void evalmanyDensitySph(const size_t npoints, const coord::PosSph pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
};


/** Time-dependent potential represented by a collection of other potentials,
    either piecewise-constant or linearly interpolated in time */
class Evolving: public BasePotentialCar {
public:
    Evolving(const std::vector<double> _times,
        const std::vector<PtrPotential> _instances,
        bool _interpLinear=false);
    virtual coord::SymmetryType symmetry() const { return sym; }
    virtual std::string name() const { return myName(); }
    static std::string myName() { return "Evolving"; }

private:
    /// array of time stamps for a time-dependent potential
    std::vector<double> times;
    /// array of potentials corresponding to each moment of time (possibly just one)
    std::vector<PtrPotential> instances;
    /// use linear or nearest-point interpolation of a time-dependent potential
    bool interpLinear;
    /// highest symmetry level shared by all instances of the potential
    coord::SymmetryType sym;

    virtual void evalCar(const coord::PosCar &pos,
        double* potential, coord::GradCar* deriv, coord::HessCar* deriv2, double time) const;
    virtual double densityCar(const coord::PosCar &pos, double time) const;
};


/** Time-dependent but spatially uniform acceleration arising from a non-inertial reference frame */
class UniformAcceleration: public BasePotentialCar {
public:
    /// initialize from a triplet of splines representing time-dependent acceleration
    UniformAcceleration(
        const math::CubicSpline& _accx,
        const math::CubicSpline& _accy,
        const math::CubicSpline& _accz)
    :
        accx(_accx), accy(_accy), accz(_accz) {}

    virtual coord::SymmetryType symmetry() const { return coord::ST_NONE; }
    virtual std::string name() const { return myName(); }
    static std::string myName() { return "UniformAcceleration"; }

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


// four kinds of modifiers, which can be applied to density or potential classes
// (unfortunately, some code duplication is inevitable here)
template<class BaseDensityOrPotential> class Shifted;
template<class BaseDensityOrPotential> class Tilted;
template<class BaseDensityOrPotential> class Rotating;
template<class BaseDensityOrPotential> class Scaled;


/** Modifier of any density profile adding an arbitrary, possibly time-dependent offset */
template<> class Shifted<BaseDensity>: public BaseDensity, public BaseComposite<BaseDensity> {
public:
    /// initialize from the given density and splines representing time-dependent offsets
    Shifted(const PtrDensity& _dens,
        const math::CubicSpline& _centerx,
        const math::CubicSpline& _centery,
        const math::CubicSpline& _centerz)
    :
        dens(_dens), centerx(_centerx), centery(_centery), centerz(_centerz) {}

    virtual double totalMass() const { return dens->totalMass(); }
    virtual coord::SymmetryType symmetry() const
    { return isUnknown(dens->symmetry()) ? coord::ST_UNKNOWN : coord::ST_NONE; }  // in general...
    virtual std::string name() const { return myName() + " " + dens->name(); }
    static std::string myName() { return "Shifted"; }
    virtual unsigned int size() const { return 1; }
    virtual PtrDensity component(unsigned int) const { return dens; }  // should check if index==0?

private:
    /// the instance of the actual density
    const PtrDensity dens;

    /// time-dependent offsets of the potential center from origin
    const math::CubicSpline centerx, centery, centerz;

    virtual double densityCar(const coord::PosCar &pos, double time) const {
        return dens->density(
            coord::PosCar(pos.x-centerx(time), pos.y-centery(time), pos.z-centerz(time)), time);
    }

    virtual double densityCyl(const coord::PosCyl &pos, double time) const
    { return densityCar(toPosCar(pos), time); }

    virtual double densitySph(const coord::PosSph &pos, double time) const
    { return densityCar(toPosCar(pos), time); }

    virtual void evalmanyDensityCar(const size_t npoints, const coord::PosCar pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
    virtual void evalmanyDensityCyl(const size_t npoints, const coord::PosCyl pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
    virtual void evalmanyDensitySph(const size_t npoints, const coord::PosSph pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
};

/** Modifier of any potential profile adding an arbitrary, possibly time-dependent offset */
template<> class Shifted<BasePotential>: public BasePotentialCar, public BaseComposite<BasePotential> {
public:
    /// initialize from the given potential and splines representing time-dependent offsets
    Shifted(const PtrPotential& _pot,
        const math::CubicSpline& _centerx,
        const math::CubicSpline& _centery,
        const math::CubicSpline& _centerz)
    :
        pot(_pot), centerx(_centerx), centery(_centery), centerz(_centerz) {}

    virtual double totalMass() const { return pot->totalMass(); }
    virtual coord::SymmetryType symmetry() const
    { return isUnknown(pot->symmetry()) ? coord::ST_UNKNOWN : coord::ST_NONE; }  // in general...
    virtual std::string name() const { return Shifted<BaseDensity>::myName() + " " + pot->name(); }
    virtual unsigned int size() const { return 1; }
    virtual PtrPotential component(unsigned int) const { return pot; }

private:
    /// the instance of the actual potential
    const PtrPotential pot;

    /// time-dependent offsets of the potential center from origin
    const math::CubicSpline centerx, centery, centerz;

    virtual void evalCar(const coord::PosCar &pos,
        double* potential, coord::GradCar* deriv, coord::HessCar* deriv2, double time) const
    {
        pot->eval(coord::PosCar(pos.x-centerx(time), pos.y-centery(time), pos.z-centerz(time)),
            potential, deriv, deriv2, time);
    }

    virtual double densityCar(const coord::PosCar &pos, double time) const {
        return pot->density(
            coord::PosCar(pos.x-centerx(time), pos.y-centery(time), pos.z-centerz(time)), time);
    }

    virtual double densityCyl(const coord::PosCyl &pos, double time) const
    { return densityCar(toPosCar(pos), time); }

    virtual double densitySph(const coord::PosSph &pos, double time) const
    { return densityCar(toPosCar(pos), time); }

    virtual void evalmanyDensityCar(const size_t npoints, const coord::PosCar pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
    virtual void evalmanyDensityCyl(const size_t npoints, const coord::PosCyl pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
    virtual void evalmanyDensitySph(const size_t npoints, const coord::PosSph pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
};


/** Modifier of any density profile tilting the coordinate axes by a triplet of Euler angles */
template<> class Tilted<BaseDensity>: public BaseDensity, public BaseComposite<BaseDensity> {
public:
    /// initialize from the given density and a triplet of Euler angles
    Tilted(const PtrDensity& _dens, double alpha, double beta, double gamma) :
        dens(_dens), orientation(alpha, beta, gamma) {}

    virtual double totalMass() const { return dens->totalMass(); }
    virtual double enclosedMass(const double radius) const { return dens->enclosedMass(radius); }
    virtual coord::SymmetryType symmetry() const;
    virtual std::string name() const { return myName() + " " + dens->name(); }
    static std::string myName() { return "Tilted"; }
    virtual unsigned int size() const { return 1; }
    virtual PtrDensity component(unsigned int) const { return dens; }  // should check if index==0?

private:
    /// the instance of the actual density
    const PtrDensity dens;

    /// transformation between input/output (external) and intrinsic coordinate systems
    const coord::Orientation orientation;

    virtual double densityCar(const coord::PosCar &pos, double time) const
    { return dens->density(orientation.toRotated(pos), time); }

    virtual double densityCyl(const coord::PosCyl &pos, double time) const
    { return dens->density(orientation.toRotated(toPosCar(pos)), time); }

    virtual double densitySph(const coord::PosSph &pos, double time) const
    { return dens->density(orientation.toRotated(toPosCar(pos)), time); }

    virtual void evalmanyDensityCar(const size_t npoints, const coord::PosCar pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
    virtual void evalmanyDensityCyl(const size_t npoints, const coord::PosCyl pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
    virtual void evalmanyDensitySph(const size_t npoints, const coord::PosSph pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
};

/** Modifier of any potential profile tilting the coordinate axes by a triplet of Euler angles */
template<> class Tilted<BasePotential>: public BasePotentialCar, public BaseComposite<BasePotential> {
public:
    /// initialize from the given potential and a triplet of Euler angles
    Tilted(const PtrPotential& _pot, double alpha, double beta, double gamma) :
        pot(_pot), orientation(alpha, beta, gamma) {}

    virtual double totalMass() const { return pot->totalMass(); }
    virtual double enclosedMass(const double radius) const { return pot->enclosedMass(radius); }
    virtual coord::SymmetryType symmetry() const;
    virtual std::string name() const { return Tilted<BaseDensity>::myName() + " " + pot->name(); }
    virtual unsigned int size() const { return 1; }
    virtual PtrPotential component(unsigned int) const { return pot; }  // should check if index==0?

private:
    /// the instance of the actual potential
    const PtrPotential pot;

    /// transformation between input/output (external) and intrinsic coordinate systems
    const coord::Orientation orientation;

    virtual void evalCar(const coord::PosCar &pos,
        double* potential, coord::GradCar* deriv, coord::HessCar* deriv2, double time) const
    {
        pot->eval(orientation.toRotated(pos), potential, deriv, deriv2, time);
        if(deriv)  *deriv  = orientation.fromRotated(*deriv);
        if(deriv2) *deriv2 = orientation.fromRotated(*deriv2);
    }

    virtual double densityCar(const coord::PosCar &pos, double time) const
    { return pot->density(orientation.toRotated(pos), time); }

    virtual double densityCyl(const coord::PosCyl &pos, double time) const
    { return pot->density(orientation.toRotated(toPosCar(pos)), time); }

    virtual double densitySph(const coord::PosSph &pos, double time) const
    { return pot->density(orientation.toRotated(toPosCar(pos)), time); }

    virtual void evalmanyDensityCar(const size_t npoints, const coord::PosCar pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
    virtual void evalmanyDensityCyl(const size_t npoints, const coord::PosCyl pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
    virtual void evalmanyDensitySph(const size_t npoints, const coord::PosSph pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
};


/** Modifier of any density profile adding a rotation about the Z axis
    by an angle that varies smoothly with time (or is constant) */
template<> class Rotating<BaseDensity>: public BaseDensity, public BaseComposite<BaseDensity> {
public:
    /// initialize from the given potential and a time-dependent rotation angle given by a spline
    Rotating(const PtrDensity& _dens, const math::CubicSpline& _angle) :
        dens(_dens), angle(_angle) {}

    virtual double totalMass() const { return dens->totalMass(); }
    virtual double enclosedMass(const double radius) const { return dens->enclosedMass(radius); }
    virtual coord::SymmetryType symmetry() const;
    virtual std::string name() const { return myName() + " " + dens->name(); }
    static std::string myName() { return "Rotating"; }
    virtual unsigned int size() const { return 1; }
    virtual PtrDensity component(unsigned int) const { return dens; }

private:
    /// the instance of the actual density
    const PtrDensity dens;

    /// time-dependent rotation angle
    const math::CubicSpline angle;

    virtual double densityCar(const coord::PosCar &pos, double time) const;
    virtual double densityCyl(const coord::PosCyl &pos, double time) const
    { return dens->density(coord::PosCyl(pos.R, pos.z, pos.phi - angle(time)), time); }
    virtual double densitySph(const coord::PosSph &pos, double time) const
    { return dens->density(coord::PosSph(pos.r, pos.theta, pos.phi - angle(time)), time); }

    virtual void evalmanyDensityCar(const size_t npoints, const coord::PosCar pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
    virtual void evalmanyDensityCyl(const size_t npoints, const coord::PosCyl pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
    virtual void evalmanyDensitySph(const size_t npoints, const coord::PosSph pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
};

/** Modifier of any potential profile adding a rotation about the Z axis
    by an angle that varies smoothly with time (or is constant) */
template<> class Rotating<BasePotential>: public BasePotential, public BaseComposite<BasePotential> {
public:
    /// initialize from the given potential and a time-dependent rotation angle given by a spline
    Rotating(const PtrPotential& _pot, const math::CubicSpline& _angle) :
        pot(_pot), angle(_angle) {}

    virtual double totalMass() const { return pot->totalMass(); }
    virtual double enclosedMass(const double radius) const { return pot->enclosedMass(radius); }
    virtual coord::SymmetryType symmetry() const;
    virtual std::string name() const { return Rotating<BaseDensity>::myName() + " " + pot->name(); }
    virtual unsigned int size() const { return 1; }
    virtual PtrPotential component(unsigned int) const { return pot; }

private:
    /// the instance of the actual potential
    const PtrPotential pot;

    /// time-dependent rotation angle
    const math::CubicSpline angle;

    virtual void evalCar(const coord::PosCar &pos,
        double* potential, coord::GradCar* deriv, coord::HessCar* deriv2, double time) const;
    virtual void evalCyl(const coord::PosCyl &pos,
        double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2, double time) const;
    virtual void evalSph(const coord::PosSph &pos,
        double* potential, coord::GradSph* deriv, coord::HessSph* deriv2, double time) const;

    virtual double densityCar(const coord::PosCar &pos, double time) const;
    virtual double densityCyl(const coord::PosCyl &pos, double time) const
    { return pot->density(coord::PosCyl(pos.R, pos.z, pos.phi - angle(time)), time); }
    virtual double densitySph(const coord::PosSph &pos, double time) const
    { return pot->density(coord::PosSph(pos.r, pos.theta, pos.phi - angle(time)), time); }

    virtual void evalmanyDensityCar(const size_t npoints, const coord::PosCar pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
    virtual void evalmanyDensityCyl(const size_t npoints, const coord::PosCyl pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
    virtual void evalmanyDensitySph(const size_t npoints, const coord::PosSph pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
};


/** Modifier of any density profile adding a time-dependent modulation in amplitude and length scale */
template<> class Scaled<BaseDensity>: public BaseDensity, public BaseComposite<BaseDensity> {
public:
    /// initialize from the given density and two splines representing time-dependent amplitude and scale
    Scaled(const PtrDensity& _dens, const math::CubicSpline& _ampl, const math::CubicSpline& _scale) :
        dens(_dens), ampl(_ampl), scale(_scale) {}

    virtual double totalMass() const
    { return dens->totalMass() * ampl(0); }  // no way to specify time here, use t=0
    virtual double enclosedMass(const double radius) const
    { return dens->enclosedMass(radius / scale(0) ) * ampl(0); }  // same here - evaluated at t=0
    virtual coord::SymmetryType symmetry() const { return dens->symmetry(); }
    virtual std::string name() const { return myName() + " " + dens->name(); }
    static std::string myName() { return "Scaled"; }
    virtual unsigned int size() const { return 1; }
    virtual PtrDensity component(unsigned int) const { return dens; }  // should check if index==0?

private:
    /// the instance of the actual density
    const PtrDensity dens;

    /// time-dependent amplitude (mass normalization)
    const math::CubicSpline ampl;

    /// time-dependent length scale factor
    const math::CubicSpline scale;

    virtual double densityCar(const coord::PosCar &pos, double time) const
    {
        double s = 1 / scale(time);
        return s*s*s * ampl(time) * dens->density(coord::PosCar(pos.x * s, pos.y * s, pos.z * s), time);
    }

    virtual double densityCyl(const coord::PosCyl &pos, double time) const
    {
        double s = 1 / scale(time);
        return s*s*s * ampl(time) * dens->density(coord::PosCyl(pos.R * s, pos.z * s, pos.phi), time);
    }

    virtual double densitySph(const coord::PosSph &pos, double time) const
    {
        double s = 1 / scale(time);
        return s*s*s * ampl(time) * dens->density(coord::PosSph(pos.r * s, pos.theta, pos.phi), time);
    }

    virtual void evalmanyDensityCar(const size_t npoints, const coord::PosCar pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
    virtual void evalmanyDensityCyl(const size_t npoints, const coord::PosCyl pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
    virtual void evalmanyDensitySph(const size_t npoints, const coord::PosSph pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
};

/** Modifier of any potential profile adding a time-dependent modulation in amplitude and length scale */
template<> class Scaled<BasePotential>: public BasePotential, public BaseComposite<BasePotential> {
public:
    /// initialize from the given potential and two splines representing time-dependent amplitude and scale
    Scaled(const PtrPotential& _pot, const math::CubicSpline& _ampl, const math::CubicSpline& _scale) :
        pot(_pot), ampl(_ampl), scale(_scale) {}

    virtual double totalMass() const
    { return pot->totalMass() * ampl(0); }  // no way to specify time here, use t=0
    virtual double enclosedMass(const double radius) const
    { return pot->enclosedMass(radius / scale(0) ) * ampl(0); }  // same here - evaluated at t=0
    virtual coord::SymmetryType symmetry() const { return pot->symmetry(); }
    virtual std::string name() const { return Scaled<BaseDensity>::myName() + " " + pot->name(); }
    virtual unsigned int size() const { return 1; }
    virtual PtrPotential component(unsigned int) const { return pot; }

private:
    /// the instance of the actual potential
    const PtrPotential pot;

    /// time-dependent amplitude
    const math::CubicSpline ampl;

    /// time-dependent length scale factor
    const math::CubicSpline scale;

    virtual void evalCar(const coord::PosCar &pos,
        double* potential, coord::GradCar* deriv, coord::HessCar* deriv2, double time) const;

    virtual void evalCyl(const coord::PosCyl &pos,
        double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2, double time) const;

    virtual void evalSph(const coord::PosSph &pos,
        double* potential, coord::GradSph* deriv, coord::HessSph* deriv2, double time) const;

    virtual double densityCar(const coord::PosCar &pos, double time) const
    {
        double s = 1 / scale(time);
        return s*s*s * ampl(time) * pot->density(coord::PosCar(pos.x * s, pos.y * s, pos.z * s), time);
    }

    virtual double densityCyl(const coord::PosCyl &pos, double time) const
    {
        double s = 1 / scale(time);
        return s*s*s * ampl(time) * pot->density(coord::PosCyl(pos.R * s, pos.z * s, pos.phi), time);
    }

    virtual double densitySph(const coord::PosSph &pos, double time) const
    {
        double s = 1 / scale(time);
        return s*s*s * ampl(time) * pot->density(coord::PosSph(pos.r * s, pos.theta, pos.phi), time);
    }

    virtual void evalmanyDensityCar(const size_t npoints, const coord::PosCar pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
    virtual void evalmanyDensityCyl(const size_t npoints, const coord::PosCyl pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
    virtual void evalmanyDensitySph(const size_t npoints, const coord::PosSph pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
};

}  // namespace potential
