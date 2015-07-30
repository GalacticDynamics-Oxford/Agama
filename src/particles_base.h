/** \file    particles_base.h 
*/
#pragma once
#include "coord.h"
#include <vector>
#include <utility>

namespace particles {

/** An array of particles with positions, velocities and masses.
    It is implemented as a separate structure instead of just a vector of pairs,
    because of the limitations of C++ that don't allow to template a typedef without a class.
    \tparam CoordT is the coordinate system tag
*/
template<typename CoordT> struct PointMassSet {
    typedef std::pair< coord::PosVelT<CoordT>, double> ElemType;  ///< templated typedef of a single particle
    typedef std::vector< ElemType > Type;   ///< templated typedef of an array of particles
    Type data;           ///< particles are stored in this array
    ///  default empty constructor
    PointMassSet() {};
    /** a seamless conversion constructor from another point mass set with a possibly different template argument.
        \tparam OtherNumT is a coordinate system of the source PointMassSet */
    template<typename OtherCoordT> PointMassSet(const PointMassSet<OtherCoordT> &src) {
        data.reserve(src.size());
        for(size_t i=0; i<src.size(); i++)
            data.push_back(ElemType(coord::toPosVel<OtherCoordT,CoordT>(src[i].first), src[i].second));
    }
    inline ElemType& at(size_t index) { return data.at(index); }             // convenience shorthand
    inline const ElemType& at(size_t index) const { return data.at(index); } // convenience shorthand
    inline ElemType& operator[](size_t index) { return data[index]; }        // convenience shorthand
    inline const ElemType& operator[](size_t index) const { return data[index]; } // another shorthand
    inline size_t size() const { return data.size(); }  // convenience shorthand
    inline void add( const coord::PosVelT<CoordT> &first, const double second) /// convenience function to add an element
    { data.push_back(ElemType(first, second)); }
};

}  // namespace