/** \file    galaxymodel_target.h
    \brief   Target objects for galaxy modelling
    \date    2017
    \author  Eugene Vasiliev

*/
#pragma once
#include "orbit.h"
#include "smart.h"
#include <string>

namespace galaxymodel{

/// \name  Base definitions
///@{

/// numerical type for storing the matrix elements (choose float to save memory)
typedef float StorageNumT;

class GalaxyModel;  // forward declaration

/** A Target object represents any possible constraint in the model.
    These could come from the self-consistency requirements for the density/potential pair,
    or from various kinematic requirements, velocity profiles, etc.
    A target consists of an array of required values for the constraints,
    an array of penalties for their violation (often related to measurement errors),
    and two methods for computing these constraints for a given galaxy model.
    The first one deals with orbit-based models, and returns an orbit runtime function
    that computes the values of these constraints for each orbit as it is being integrated.
    The second one deals with models based on a single- or multicomponent distribution function,
    and returns an array of values for each component of the DF.
*/
class BaseTarget {
public:
    virtual ~BaseTarget() {}

    /// human-readable name of this target object
    virtual const char* name() const = 0;

    /// textual representation of a given constraint
    virtual std::string elemName(unsigned int index) const = 0;

    /// total number of constraints
    virtual unsigned int numConstraints() const = 0;

    /// array of constraint values (size: numConstraints)
    virtual std::vector<double> constraints() const = 0;

    /// array of penalties for constraint violation (size: numConstraints)
    virtual std::vector<double> penalties() const = 0;

    /// construct a runtime function that collects the target-specific data during orbit integration
    /// \param[out] output is a pointer to the storage that will be filled by the runtime function;
    /// should point to an existing chunk of memory with size equal to numConstraints()
    /// \return  a new instance of a target-specific runtime function
    virtual orbit::PtrRuntimeFnc getOrbitRuntimeFnc(StorageNumT* output) const = 0;

    /// compute target-specific data (projection of a DF)
    /// \param[in] model  is the interface for computing the value(s) of a distribution function,
    /// possibly a multi-component DF
    /// \param[out] output is a pointer to the array where the DF projection will be stored:
    /// each DF component produces a contiguous array of numConstraints() output values;
    /// should be an existing chunk of memory with size numConstraints() * df.numValues()
    virtual void computeDFProjection(const GalaxyModel& model, StorageNumT* output) const = 0;
};

typedef shared_ptr<const BaseTarget> PtrTarget;

///@}
/// \name  Density discretization models
///@{

/** Choice of spatial discretization scheme for the density profile */
enum DensityGridType {
    /// Classical approach for spheroidal Schwarzschild models:
    /// radial shells divided into three panes, each pane - into several strips,
    /// and the density inside each resulting cell is approximated as a constant
    DG_CLASSIC_TOPHAT,

    /// Same as classical, but the density is specified at the grid nodes and
    /// interpolated tri-linearly within each cell
    DG_CLASSIC_LINEAR,

    /// Radial grid is the same as in the classic approach, but the angular dependence
    /// of the density is represented in terms of spherical-harmonic expansion,
    /// and the radial dependence of each term is a linearly-interpolated function
    DG_SPH_HARM,

    /// A grid in meridional plane (R,z) aligned with cylindrical coordinates,
    /// with the azimuthal dependence of the density represented by a Fourier expansion;
    /// the density is attributed to each cell (implicitly assumed to be constant within a cell)
    DG_CYLINDRICAL_TOPHAT,

    /// Same as the previous one, but each azimuthal Fourier term in the meridional plane (R,z)
    /// is bi-linearly interpolated within each cell
    DG_CYLINDRICAL_LINEAR,

    /// Default initialization value corresponds to an invalid choice
    DG_UNKNOWN = 999
};

struct DensityGridParams {
    DensityGridType type;    ///< type of the spatial grid, determines which subset of parameters to use
    unsigned int gridSizeR;  ///< number of radial grid points
    unsigned int gridSizez;  ///< number of grid points in z-direction
    double innerShellMass;   ///< fraction of total mass enclosed by the innermost radial shell
    double outerShellMass;   ///< same for the outermost
    unsigned int stripsPerPane;    ///< number of strips per pane in the classical scheme
    double axisRatioY, axisRatioZ; ///< axis ratios of the grid in the classical scheme
    unsigned int lmax;       ///< number of angular terms in spherical-harmonic expansion
    unsigned int mmax;       ///< number of angular terms in azimuthal-harmonic expansion

    /// assign default values to each member variable
    DensityGridParams() :
        type(DG_UNKNOWN),
        gridSizeR(20),
        gridSizez(20),
        innerShellMass(0.),  // auto-assign to   1/(gridSizeR+1)
        outerShellMass(0.),  // auto-assign to 1-1/(gridSizeR+1)
        stripsPerPane(2),
        axisRatioY(1.),
        axisRatioZ(1.),
        lmax(6),
        mmax(6)
    {}
};

class BaseDensityGrid;  // forward declaration

class TargetDensity: public BaseTarget {
    shared_ptr<const BaseDensityGrid> grid;
    std::vector<double> constraintValues;
    std::vector<double> constraintPenalties;
    std::vector<double> gridR, gridz;
    unsigned int lmax, mmax;
public:
    TargetDensity(const potential::BaseDensity& density, const DensityGridParams& params);

    virtual const char* name() const;

    virtual std::string elemName(unsigned int index) const;

    virtual unsigned int numConstraints() const { return constraintValues.size(); }

    virtual std::vector<double> constraints() const { return constraintValues; }

    virtual std::vector<double> penalties() const { return constraintPenalties; }

    virtual orbit::PtrRuntimeFnc getOrbitRuntimeFnc(StorageNumT* output) const;

    virtual void computeDFProjection(const GalaxyModel& model, StorageNumT* output) const;
};

///@}
/// \name  Kinematic discretization models
///@{

class TargetKinemJeans: public BaseTarget {
    math::PtrFunctionNdim grid;
    std::vector<double> constraintValues;
    std::vector<double> constraintPenalties;
public:
    TargetKinemJeans(
        const potential::BaseDensity& density,
        const potential::BasePotential& potential,
        double beta,
        unsigned int gridSizeR,
        unsigned int degree);

    virtual const char* name() const { return "KinemJeans"; }

    virtual std::string elemName(unsigned int index) const;

    virtual unsigned int numConstraints() const { return constraintValues.size(); }

    virtual std::vector<double> constraints() const { return constraintValues; }

    virtual std::vector<double> penalties() const { return constraintPenalties; }

    virtual orbit::PtrRuntimeFnc getOrbitRuntimeFnc(StorageNumT* output) const;

    virtual void computeDFProjection(const GalaxyModel& model, StorageNumT* output) const;
};

///@}

}  // namespace