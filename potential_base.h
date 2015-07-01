#pragma once
#include "coord.h"

namespace potential{

/// \name   Base class for all potentials
///@{

    /** Abstract class defining the gravitational potential.

        It provides public non-virtual functions for computing potential and 
        up to two its derivatives in three standard coordinate systems:
        [Car]tesian, [Cyl]indrical, and [Sph]erical. 
        These three functions share the same name `eval`, i.e. are overloaded 
        on the type of input coordinates. 
        They internally call three protected virtual functions, named after 
        each coordinate system. These functions are implemented in derived classes.

        In addition, the public interface contains a separate triad of overloaded 
        non-virtual functions for computing density in each of three coordinate systems.
        Again they are implemented internally as protected virtual functions with 
        different names.
    */
    class BasePotential{
    public:
    /// \name  Data types
    ///{@ 
        /** defines the symmetry properties of the potential */
        enum SYMMETRYTYPE{ 
            ST_NONE = 0,       ///< no symmetry whatsoever
            ST_REFLECTION = 1, ///< reflection about origin (change of sign of all coordinates simultaneously)
            ST_PLANESYM = 2,   ///< reflection about principal planes (change of sign of any coordinate)
            ST_ZROTSYM = 4,    ///< rotation about z axis
            ST_SPHSYM = 8,     ///< rotation about arbitrary axis
            ST_TRIAXIAL = ST_REFLECTION | ST_PLANESYM,    ///< triaxial symmetry
            ST_AXISYMMETRIC = ST_TRIAXIAL | ST_ZROTSYM,   ///< axial symmetry
            ST_SPHERICAL = ST_AXISYMMETRIC | ST_SPHSYM,   ///< spherical symmetry
        };

    ///@}
    /// \name  Main public interface methods
    ///@{
    
        BasePotential() {};

        virtual ~BasePotential() {};

        /** Evaluate potential and up to two its derivatives in a specified coordinate system.
            \param[in]  pos is the position in the given coordinates.
            \param[out] potential - if not NULL, store the value of potential
                        in the variable addressed by this pointer.
            \param[out] deriv - if not NULL, store the gradient of potential
                        in the variable addressed by this pointer.
            \param[out] deriv2 - if not NULL, store the Hessian (matrix of second derivatives)
                        of potential in the variable addressed by this pointer.  */
        template<typename coordSysT>
        void eval(const coord::PosT<coordSysT> &pos,
            double* potential=0, 
            coord::GradT<coordSysT>* deriv=0, 
            coord::HessT<coordSysT>* deriv2=0) const;

        /** Evaluate density at the position in a specified coordinate system.
            Default implementation computes the density from Laplacian of the potential,
            but the derived classes may instead provide an explicit expression for it. */
        template<typename coordSysT>
        double density(const coord::PosT<coordSysT> &pos) const;
        
        /// returns symmetry type of this potential
        virtual SYMMETRYTYPE symmetry() const=0;
    
    ///@}
    protected:
    /// \name  Protected members: virtual methods for `eval` and `density` in different coordinate systems
    ///@{
#if 0
        /** universal conversion function: evaluate potential and possibly its derivatives
            in coordinate system specified by evalCS, and convert them to another
            coordinate system specified by outputCS. */
        template<typename evalCS, typename outputCS>
        void eval_and_convert(const coord::PosT<outputCS>& pos,
            double* potential, coord::GradT<outputCS>* deriv, coord::HessT<outputCS>* deriv2) const;
#endif

        /** evaluate potential and up to two its derivatives in cartesian coordinates;
            must be implemented in derived classes */
        virtual void eval_car(const coord::PosCar &pos,
            double* potential, coord::GradCar* deriv, coord::HessCar* deriv2) const=0;

        /** evaluate potential and up to two its derivatives in cylindrical coordinates */
        virtual void eval_cyl(const coord::PosCyl &pos,
            double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const=0;

        /** evaluate potential and up to two its derivatives in spherical coordinates */
        virtual void eval_sph(const coord::PosSph &pos,
            double* potential, coord::GradSph* deriv, coord::HessSph* deriv2) const=0;

        /** evaluate density at the position specified in cartesian coordinates */
        virtual double density_car(const coord::PosCar &pos) const;

        /** evaluate density at the position specified in cylindrical coordinates */
        virtual double density_cyl(const coord::PosCyl &pos) const;

        /** Evaluate density at the position specified in spherical coordinates */
        virtual double density_sph(const coord::PosSph &pos) const;

    ///@}
    /// \name  Copy constructor and assignment operators are not allowed
        BasePotential(const BasePotential& src);
        BasePotential& operator=(const BasePotential&);
    ///@}
    };  // class BasePotential
    
    // Template specializations for `BasePotential::eval` and `BasePotential::density` 
    // in particular coordinate systems 
    // (need to be declared outside the scope of class definition)
    
    /// Evaluate potential and up to two its derivatives in cartesian coordinates
    template<> inline void BasePotential::eval(const coord::PosCar &pos,
        double* potential, coord::GradCar* deriv, coord::HessCar* deriv2) const
    {  eval_car(pos, potential, deriv, deriv2); };

    /// Evaluate potential and up to two its derivatives in cylindrical coordinates
    template<> inline void BasePotential::eval(const coord::PosCyl &pos,
        double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const
    {  eval_cyl(pos, potential, deriv, deriv2); };

    /// Evaluate potential and up to two its derivatives in spherical coordinates
    template<> inline void BasePotential::eval(const coord::PosSph &pos,
        double* potential, coord::GradSph* deriv, coord::HessSph* deriv2) const
    {  eval_sph(pos, potential, deriv, deriv2); };

    /// Evaluate density at the position specified in cartesian coordinates
    template<> inline double BasePotential::density(const coord::PosCar &pos) const
    {  return density_car(pos); };

    /// Evaluate density at the position specified in cylindrical coordinates
    template<> inline double BasePotential::density(const coord::PosCyl &pos) const
    {  return density_cyl(pos); };

    /// Evaluate density at the position specified in spherical coordinates
    template<> inline double BasePotential::density(const coord::PosSph &pos) const
    {  return density_sph(pos); };


///@}
/// \name   Base classes for potentials that are easier to evaluate in a particular coordinate system
///@{

    /** Parent class for potentials that are evaluated in cartesian coordinates.
        It leaves the implementation of eval_car member function for cartesian coordinates undefined, 
        but provides the conversion from cartesian to cylindrical and spherical coordinates in eval_cyl and eval_sph. */
    class BasePotentialCar: public BasePotential, coord::ScalarFunction<coord::Car>{
    public:
        BasePotentialCar() : BasePotential() {}

    private:
        /** Evaluate potential and up to two its derivatives in cylindrical coordinates. */
        virtual void eval_cyl(const coord::PosCyl &pos,
            double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const;

        /** Evaluate potential and up to two its derivatives in spherical coordinates. */
        virtual void eval_sph(const coord::PosSph &pos,
            double* potential, coord::GradSph* deriv, coord::HessSph* deriv2) const;

        /** abstract function inherited from ScalarFunction interface */
        virtual void evaluate(const coord::PosCar& pos,
            double* value=0, coord::GradCar* deriv=0, coord::HessCar* deriv2=0) const
        { eval_car(pos, value, deriv, deriv2); }
    };  // class BasePotentialCar


    /** Parent class for potentials that are evaluated in cylindrical coordinates.
        It leaves the implementation of eval_cyl member function for cylindrical coordinates undefined, 
        but provides the conversion from cylindrical to cartesian and spherical coordinates. */
    class BasePotentialCyl: public BasePotential, coord::ScalarFunction<coord::Cyl>{
    public:
        BasePotentialCyl() : BasePotential() {}

    private:
        /** Evaluate potential and up to two its derivatives in cartesian coordinates. */
        virtual void eval_car(const coord::PosCar &pos,
            double* potential, coord::GradCar* deriv, coord::HessCar* deriv2) const;

        /** Evaluate potential and up to two its derivatives in spherical coordinates. */
        virtual void eval_sph(const coord::PosSph &pos,
            double* potential, coord::GradSph* deriv, coord::HessSph* deriv2) const;

        /** abstract function inherited from ScalarFunction interface */
        virtual void evaluate(const coord::PosCyl& pos,
            double* value=0, coord::GradCyl* deriv=0, coord::HessCyl* deriv2=0) const
        { eval_cyl(pos, value, deriv, deriv2); }        
    };  // class BasePotentialCyl


    /** Parent class for potentials that are evaluated in spherical coordinates.
        It leaves the implementation of eval_sph member function for spherical coordinates undefined, 
        but provides the conversion from spherical to cartesian and cylindrical coordinates. */
    class BasePotentialSph: public BasePotential, coord::ScalarFunction<coord::Sph>{
    public:
        BasePotentialSph() : BasePotential() {}

    private:
        /** Evaluate potential and up to two its derivatives in cartesian coordinates. */
        virtual void eval_car(const coord::PosCar &pos,
            double* potential, coord::GradCar* deriv, coord::HessCar* deriv2) const;

        /** Evaluate potential and up to two its derivatives in cylindrical coordinates. */
        virtual void eval_cyl(const coord::PosCyl &pos,
            double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const;

        /** abstract function inherited from ScalarFunction interface */
        virtual void evaluate(const coord::PosSph& pos,
            double* value=0, coord::GradSph* deriv=0, coord::HessSph* deriv2=0) const
        { eval_sph(pos, value, deriv, deriv2); }        
    };  // class BasePotentialSph

///@}
/// \name   A more specialized but still abstract class for spherically-symmetric potentials
///@{

    /** Parent class for analytic spherically-symmetric potentials. */
    class BasePotentialSphericallySymmetric: public BasePotentialSph{
    public:
        BasePotentialSphericallySymmetric() : BasePotentialSph() {}

        virtual SYMMETRYTYPE symmetry() const { return ST_SPHERICAL; }

    private:
        /** Evaluate potential and up to two its derivatives in spherical coordinates. */
        virtual void eval_sph(const coord::PosSph &pos,
            double* potential, coord::GradSph* deriv, coord::HessSph* deriv2) const;

    protected:
        /** Compute the potential and up to two its derivatives by radius; 
            should be implemented in derived classes. */
        virtual void eval_sph_rad(const coord::PosSph &pos,
            double* potential=0, double* deriv=0, double* deriv2=0) const = 0;
    };

///@}
/// \name   Convenience functions
///@{

    /** Shorthand for evaluating the value of potential at a given point */
    template<typename coordT>
    double Phi(const BasePotential& potential, const coordT& point) {
        double val;
        potential.eval(point, &val);
        return val;
    }

    /** Convenience function for evaluating total energy of a given position/velocity pair */
    template<typename coordT>
    double totalEnergy(const BasePotential& potential, const coord::PosVelT<coordT>& posvel);

    template<>
    inline double totalEnergy(const BasePotential& potential, const coord::PosVelCar& p)
    {  return Phi(potential, p) + 0.5*(p.vx*p.vx+p.vy*p.vy+p.vz*p.vz); }

    template<>
    inline double totalEnergy(const BasePotential& potential, const coord::PosVelCyl& p)
    {  return Phi(potential, p) + 0.5*(pow_2(p.vR)+pow_2(p.vz)+pow_2(p.vphi)); }

    template<>
    inline double totalEnergy(const BasePotential& potential, const coord::PosVelSph& p)
    {  return Phi(potential, p) + 0.5*(pow_2(p.vr)+pow_2(p.vtheta)+pow_2(p.vphi)); }


    /** Compute circular velocity at a given radius in equatorial plane */
    double v_circ(const BasePotential& potential, double radius);

///@}
}  // namespace potential
