/** \file    particles_base.h 
    \brief   Base class for array of particles
    \author  Eugene Vasiliev
    \date    2010-2015
*/
#pragma once
#include "coord.h"
#include <vector>
#include <utility>
using std::size_t;

/** Classes and functions for manipulating arrays of particles */
namespace particles {

/// a "fat particle" type with auxiliary properties
struct ParticleAux : public coord::PosVelCar {
    double stellarMass;       ///< mass of the star (responsible for relaxation and mass segregation)
    double stellarRadius;     ///< radius of the star (responsible for loss-cone physics)
    ParticleAux(const coord::PosVelCar& pv, double m=0, double r=0) :
    coord::PosVelCar(pv), stellarMass(m), stellarRadius(r) {}
};


/** Helper class for converting between particle types (from SrcT to DestT).
    This is a templated class with essentially no generic implementation;
    the actual conversion is performed by one of the partially specialized
    template classes below, which are specialized w.r.t. particle type 
    (position/velocity, just position, or something else), but are still generic 
    w.r.t. coordinate conversion.
    Therefore, if one suddenly receives linker error about missing some obscure 
    templated Converter class, it means that one has tried a conversion which is 
    not supported (e.g. converting from position to position/velocity).
*/
template<typename SrcT, typename DestT>
struct Converter {
    DestT operator()(const SrcT& src);
};


/** An array of particles with masses.
    It is implemented as a separate structure instead of just a vector of pairs,
    because of the limitations of C++ that don't allow to template a typedef without a class,
    and to enable seamless conversion between compatible particle types and coordinate systems.
    \tparam ParticleT  is the particle type:
    it could be coord::PosT<CoordT> or coord::PosVelT<CoordT>, where `CoordT`
    is one of the three standard coordinate systems (coord::Car, coord::Cyl, coord::Sph).
    In other words, the particles in this array may have either  positions and masses,
    or positions, velocities and masses.
    The former usage is suitable for potential expansions, as they only need positions;
    seamless conversion ensures that one may supply position/velocity/mass arrays to routines
    that only need position/mass arrays, but not the other way round.
*/
template<typename ParticleT>
struct ParticleArray {

    /// templated typedef of a single particle
    typedef std::pair<ParticleT, double> ElemType;

    /// templated typedef of an array of particles
    typedef std::vector<ElemType> ArrayType;

    /// particles are stored in this array
    ArrayType data;

    ///  default empty constructor
    ParticleArray() {};

    /** a seamless conversion constructor from another point mass set 
        with a possibly different template argument.
        \tparam OtherParticleT is a particle type of the source ParticleArray.
    */
    template<typename OtherParticleT> ParticleArray(const ParticleArray<OtherParticleT> &src) {
        data.reserve(src.size());
        Converter<OtherParticleT, ParticleT> conv;  // this is the mighty thing
        for(size_t i=0; i<src.size(); i++)
            data.push_back(ElemType(conv(src.point(i)), src.mass(i)));
    }

    /// return the array size
    inline size_t size() const {
        return data.size(); }

    /// convenience function to add an element
    inline void add(const ParticleT &first, const double second) {
        data.push_back(ElemType(first, second)); }

    /// convenience shorthand for extracting array element
    inline ElemType& operator[](size_t index) {
        return data[index]; }

    /// convenience shorthand for extracting array element as a const reference
    inline const ElemType& operator[](size_t index) const {
        return data[index]; }

    /// convenience function for extracting the particle (without mass) from the array
    inline const ParticleT& point(size_t index) const {
        return data[index].first; }

    /// convenience function for extracting the mass of a particle from the array
    inline double mass(size_t index) const {
        return data[index].second; }

    /// return total mass of particles in the array
    inline double totalMass() const {
        double sum=0;
        for(size_t i=0; i<data.size(); i++)
            sum += data[i].second;
        return sum;
    }
};

/// more readable typenames for the three coordinate systems
typedef ParticleArray<coord::PosVelCar>  ParticleArrayCar;
typedef ParticleArray<coord::PosVelCyl>  ParticleArrayCyl;
typedef ParticleArray<coord::PosVelSph>  ParticleArraySph;
typedef ParticleArray<ParticleAux> ParticleArrayAux;


/// specializations of conversion operator for the case that both SrcT and DestT
/// are pos/vel/mass particle types in possibly different coordinate systems
template<typename SrcCoordT, typename DestCoordT>
struct Converter<coord::PosVelT<SrcCoordT>, coord::PosVelT<DestCoordT> > {
    coord::PosVelT<DestCoordT> operator()(const coord::PosVelT<SrcCoordT>& src) {
        return coord::toPosVel<SrcCoordT, DestCoordT>(src);
    }
};

/// specializations of conversion operator for the case that SrcT is pos/vel/mass 
/// and DestT is pos/mass particle type in possibly different coordinate systems
template<typename SrcCoordT, typename DestCoordT>
struct Converter<coord::PosVelT<SrcCoordT>, coord::PosT<DestCoordT> > {
    coord::PosT<DestCoordT> operator()(const coord::PosVelT<SrcCoordT>& src) {
        return coord::toPos<SrcCoordT, DestCoordT>(src);
    }
};

/// specializations of conversion operator for the case that both SrcT and DestT 
/// are pos/mass particle types in possibly different coordinate systems
template<typename SrcCoordT, typename DestCoordT>
struct Converter<coord::PosT<SrcCoordT>, coord::PosT<DestCoordT> > {
    coord::PosT<DestCoordT> operator()(const coord::PosT<SrcCoordT>& src) {
        return coord::toPos<SrcCoordT, DestCoordT>(src);
    }
};

/// specializations of conversion operator from a fat particle to an ordinary position
template<typename DestCoordT>
struct Converter<ParticleAux, coord::PosT<DestCoordT> > {
    coord::PosT<DestCoordT> operator()(const ParticleAux& src) {
        return coord::toPos<coord::Car, DestCoordT>(src);
    }
};

/// specializations of conversion operator from a fat particle to an ordinary position/velocity
template<typename DestCoordT>
struct Converter<ParticleAux, coord::PosVelT<DestCoordT> > {
    coord::PosVelT<DestCoordT> operator()(const ParticleAux& src) {
        return coord::toPosVel<coord::Car, DestCoordT>(src);
    }
};


}  // namespace