#pragma once
#include "coord.h"

namespace potential{

    /** Abstract class defining the gravitational potential */
    class BasePotential{
    public:
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

        BasePotential() {};

        virtual ~BasePotential() {};

        /** Evaluate potential and up to two its derivatives in cartesian coordinates.
            \param[in]  pos is the position specified in cartesian coordinates.
            \param[out] potential - if not NULL, store the value of potential
                        in the variable addressed by this pointer.
            \param[out] deriv - if not NULL, store the gradient of potential
                        in the variable addressed by this pointer.
            \param[out] deriv2 - if not NULL, store the Hessian (matrix of second derivatives)
                        of potential in the variable addressed by this pointer.  */
        void eval(const coord::PosCar &pos,
            double* potential=0, coord::GradCar* deriv=0, coord::HessCar* deriv2=0) const
        {  eval_car(pos, potential, deriv, deriv2); };

        /** Evaluate potential and up to two its derivatives in cylindrical coordinates. */
        void eval(const coord::PosCyl &pos,
            double* potential=0, coord::GradCyl* deriv=0, coord::HessCyl* deriv2=0) const
        {  eval_cyl(pos, potential, deriv, deriv2); };

        /** Evaluate potential and up to two its derivatives in spherical coordinates. */
        void eval(const coord::PosSph &pos,
            double* potential=0, coord::GradSph* deriv=0, coord::HessSph* deriv2=0) const
        {  eval_sph(pos, potential, deriv, deriv2); };

        /** Evaluate density at the position specified in cartesian coordinates.
            Default implementation computes the density from Laplacian of the potential,
            but the derived classes may instead provide an explicit expression for it. */
        double density(const coord::PosCar &pos) const
        {  return density_car(pos); };

        /// Evaluate density at the position specified in cylindrical coordinates
        double density(const coord::PosCyl &pos) const
        {  return density_cyl(pos); };

        /// Evaluate density at the position specified in spherical coordinates
        double density(const coord::PosSph &pos) const
        {  return density_sph(pos); };

        /// returns symmetry type of this potential
        virtual SYMMETRYTYPE symmetry() const=0;

    protected:
        /** universal conversion function: evaluate potential and possibly its derivatives
            in coordinate system specified by evalT, and convert them to another
            coordinate system specified by outputT. */
        template<typename evalT, typename outputT>
        void eval_and_convert(const outputT& pos,
            double* potential, coord::GradT<outputT>* deriv, coord::HessT<outputT>* deriv2) const;

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

    };  // class BasePotential

    /** Parent class for potentials that are evaluated in cartesian coordinates.
        It leaves the implementation of eval_car member function for cartesian coordinates undefined, 
        but provides the conversion from cartesian to cylindrical and spherical coordinates in eval_cyl and eval_sph. */
    class BasePotentialCar: public BasePotential{
    public:
        BasePotentialCar() : BasePotential() {}

    private:
        /** Evaluate potential and up to two its derivatives in cylindrical coordinates. */
        virtual void eval_cyl(const coord::PosCyl &pos,
            double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const;

        /** Evaluate potential and up to two its derivatives in spherical coordinates. */
        virtual void eval_sph(const coord::PosSph &pos,
            double* potential, coord::GradSph* deriv, coord::HessSph* deriv2) const;

    };  // class BasePotentialCar


    /** Parent class for potentials that are evaluated in cylindrical coordinates.
        It leaves the implementation of eval_cyl member function for cylindrical coordinates undefined, 
        but provides the conversion from cylindrical to cartesian and spherical coordinates. */
    class BasePotentialCyl: public BasePotential{
    public:
        BasePotentialCyl() : BasePotential() {}

    private:
        /** Evaluate potential and up to two its derivatives in cartesian coordinates. */
        virtual void eval_car(const coord::PosCar &pos,
            double* potential, coord::GradCar* deriv, coord::HessCar* deriv2) const;

        /** Evaluate potential and up to two its derivatives in spherical coordinates. */
        virtual void eval_sph(const coord::PosSph &pos,
            double* potential, coord::GradSph* deriv, coord::HessSph* deriv2) const;

    };  // class BasePotentialCyl


    /** Parent class for potentials that are evaluated in spherical coordinates.
        It leaves the implementation of eval_sph member function for spherical coordinates undefined, 
        but provides the conversion from spherical to cartesian and cylindrical coordinates. */
    class BasePotentialSph: public BasePotential{
    public:
        BasePotentialSph() : BasePotential() {}

    private:
        /** Evaluate potential and up to two its derivatives in cartesian coordinates. */
        virtual void eval_car(const coord::PosCar &pos,
            double* potential, coord::GradCar* deriv, coord::HessCar* deriv2) const;

        /** Evaluate potential and up to two its derivatives in cylindrical coordinates. */
        virtual void eval_cyl(const coord::PosCyl &pos,
            double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const;

    };  // class BasePotentialSph


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

}  // namespace potential
