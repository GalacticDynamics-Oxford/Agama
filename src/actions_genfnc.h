/** \file    actions_genfnc.h
    \brief   Generating functions for Torus mapping
    \author  Eugene Vasiliev
    \date    Feb 2016

*/
#pragma once
#include "actions_base.h"
#include "math_linalg.h"
#include <vector>

namespace actions {

/** Triple index of a single term in the generating function */
struct GenFncIndex {
    int mr, mz, mphi;
    GenFncIndex() : mr(0), mz(0), mphi(0) {}
    GenFncIndex(int _mr, int _mz, int _mphi) : mr(_mr), mz(_mz), mphi(_mphi) {}
};

/** Array of indices that represent all non-trivial terms in the generating function */
typedef std::vector<GenFncIndex> GenFncIndices;
typedef std::vector<Actions> GenFncDerivs;

/** Generating function that maps the true action/angles to toy action/angles */
class GenFnc: public BaseCanonicalMap{
public:
    GenFnc(const GenFncIndices& _indices, const double _values[], const Actions _derivs[]) :
        indices(_indices),
        values(_values, _values+indices.size()),
        derivs(_derivs, _derivs+indices.size()) {};
    virtual unsigned int numParams() const { return indices.size(); }
    virtual ActionAngles map(const ActionAngles& actAng) const;
private:
    const GenFncIndices indices;      ///< indices of terms in generating function
    const std::vector<double> values; ///< amplitudes of terms in generating function
    const GenFncDerivs derivs;        ///< amplitudes of derivatives dS/dJ_{r,z,phi}
};

/** Variant of generating function used during the fitting process.
    It converts true actions to toy actions at any of the pre-determined array of angles
    specified at its construction, for the given amplitudes of its terms;
    and also provides the derivatives of toy actions by these amplitudes.
    Unlike the general-purpose action map, it only operates on the restricted set of angles,
    which avoids repeated computation of trigonometric functions; on the other hand,
    the amplitudes of its terms are not fixed at construction, but provided at each call.
*/
class GenFncFit{
public:
    /** construct the object for the fixed indexing scheme,
        values of real actions, and array of toy angles */
    GenFncFit(const GenFncIndices& indices, const Actions& acts, const std::vector<Angles>& angs);

    /** number of terms in the generating function */
    unsigned int numParams() const { return indices.size(); }

    /** number of points in the array of angles */
    unsigned int numPoints() const { return angs.size(); }

    /** perform mapping from real actions to toy actions at the specific values of angles.
        \param[in]  indexAngle is the index of element in the pre-determined grid of angles;
        \param[in]  values     is the array of amplitudes of the generating function terms;
        \return  toy actions and angles.
    */
    ActionAngles toyActionAngles(const unsigned int indexAngle, const double values[]) const;

    /** compute derivatives of toy actions w.r.t generating function coefficients.
        \param[in]  indexAngle is the index of element in the pre-determined grid of angles;
        \param[in]  indexCoef  is the index of term in the generating function;
        \return  derivatives of each action by the amplitude of this term.
    */
    inline Actions deriv(const unsigned int indexAngle, const unsigned int indexCoef) const {
        double val = coefs(indexAngle, indexCoef);  // no range check performed!
        return Actions(
            val * indices[indexCoef].mr,
            val * indices[indexCoef].mz,
            val * indices[indexCoef].mphi);
    }
private:
    const GenFncIndices indices;    ///< indices of terms in generating function
    const Actions acts;             ///< values of real actions
    const std::vector<Angles> angs; ///< grid of toy angles
    math::Matrix<double> coefs;     ///< precomputed trigonometric functions at the grid of angles
};

}  // namespace actions
