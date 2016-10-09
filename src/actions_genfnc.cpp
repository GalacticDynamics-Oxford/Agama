#include "actions_genfnc.h"
#include "math_fit.h"
#include "math_core.h"
#include <cmath>

namespace actions{

namespace{ // internal

/** Helper class to be used in the iterative solution of a nonlinear system of equations
    that implicitly define the toy angles as functions of real angles and the derivatives
    of the generating function.
*/
class AngleFinder: public math::IFunctionNdimDeriv {
public:
    AngleFinder(const GenFncIndices& _indices, const GenFncDerivs& _dSby, const Angles& _ang) :
        indices(_indices), dSby(_dSby), ang(_ang) {}
    virtual unsigned int numVars() const { return 3; }
    virtual unsigned int numValues() const { return 3; }

    virtual void evalDeriv(const double vars[], double values[], double *derivs=0) const
    {
        if(values) {
            values[0] = vars[0] - ang.thetar;
            values[1] = vars[1] - ang.thetaz;
            values[2] = vars[2] - ang.thetaphi;
        }
        if(derivs) {
            derivs[0] = derivs[4] = derivs[8] = 1.;  // diagonal
            derivs[1] = derivs[2] = derivs[3] = derivs[5] = derivs[6] = derivs[7] = 0;  // off-diag
        }
        for(unsigned int i=0; i<indices.size(); i++) {
            double arg = indices[i].mr * vars[0] + indices[i].mz * vars[1] +
                indices[i].mphi * vars[2];    // argument of trig functions
            if(values) {
                double s = sin(arg);
                values[0] += s * dSby[i].Jr;
                values[1] += s * dSby[i].Jz;
                values[2] += s * dSby[i].Jphi;
            }
            if(derivs) {
                double c = cos(arg);
                derivs[0] += c * dSby[i].Jr   * indices[i].mr;
                derivs[1] += c * dSby[i].Jr   * indices[i].mz;
                derivs[2] += c * dSby[i].Jr   * indices[i].mphi;
                derivs[3] += c * dSby[i].Jz   * indices[i].mr;
                derivs[4] += c * dSby[i].Jz   * indices[i].mz;
                derivs[5] += c * dSby[i].Jz   * indices[i].mphi;
                derivs[6] += c * dSby[i].Jphi * indices[i].mr;
                derivs[7] += c * dSby[i].Jphi * indices[i].mz;
                derivs[8] += c * dSby[i].Jphi * indices[i].mphi;
            }
        }
    }
private:
    const GenFncIndices& indices; ///< indices of terms in generating function
    const GenFncDerivs& dSby;     ///< amplitudes of derivatives dS/dJ_{r,z,phi}
    const Angles ang;             ///< true angles
};

} // internal namespace

ActionAngles GenFnc::map(const ActionAngles& actAng) const
{
    // 1. compute toy angles from real angles, solving a non-linear 3d system of equations
    AngleFinder fnc(indices, derivs, actAng);
    double realAngles[3] = {actAng.thetar, actAng.thetaz, actAng.thetaphi};
    double toyAngles[3];
    math::findRootNdimDeriv(fnc, realAngles, 1e-6, 10, toyAngles);
    ActionAngles aa(actAng, Angles(math::wrapAngle(toyAngles[0]),
        math::wrapAngle(toyAngles[1]), math::wrapAngle(toyAngles[2])));
    // 2. compute toy actions from real actions and toy angles
    for(unsigned int i=0; i<indices.size(); i++) {
        double val = values[i] * cos(indices[i].mr * aa.thetar +
            indices[i].mz * aa.thetaz + indices[i].mphi * aa.thetaphi);
        aa.Jr  += val * indices[i].mr;
        aa.Jz  += val * indices[i].mz;
        aa.Jphi+= val * indices[i].mphi;
    }
    // prevent non-physical negative values
    aa.Jr = fmax(aa.Jr, 0);
    aa.Jz = fmax(aa.Jz, 0);
    return aa;
}

GenFncFit::GenFncFit(const GenFncIndices& _indices,
    const Actions& _acts, const std::vector<Angles>& _angs) :
    indices(_indices), acts(_acts), angs(_angs), coefs(angs.size(), indices.size())
{
    for(unsigned int indexAngle=0; indexAngle<angs.size(); indexAngle++)
        for(unsigned int indexCoef=0; indexCoef<indices.size(); indexCoef++)
            coefs(indexAngle, indexCoef) = cos(
                indices[indexCoef].mr * angs[indexAngle].thetar +
                indices[indexCoef].mz * angs[indexAngle].thetaz +
                indices[indexCoef].mphi * angs[indexAngle].thetaphi);
}

ActionAngles GenFncFit::toyActionAngles(unsigned int indexAngle, const double values[]) const
{
    ActionAngles aa(acts, angs[indexAngle]);
    for(unsigned int indexCoef=0; indexCoef<indices.size(); indexCoef++) {
        double val = values[indexCoef] * coefs(indexAngle, indexCoef);
        aa.Jr  += val * indices[indexCoef].mr;
        aa.Jz  += val * indices[indexCoef].mz;
        aa.Jphi+= val * indices[indexCoef].mphi;
    }
    // non-physical negative actions may appear,
    // which means that these values of parameters are unsuitable.
    return aa;
}

}  // namespace actions
