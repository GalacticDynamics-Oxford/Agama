/** \file    galaxymodel_target.h
    \brief   Target objects for galaxy modelling
    \date    2017
    \author  Eugene Vasiliev

    This file defines the base class for all target objects.
    This is a header-only file.
*/
#pragma once
#include "orbit.h"
#include "smart.h"
#include "math_linalg.h"

namespace galaxymodel{

/// numerical type for storing the matrix elements (choose float to save memory)
typedef float StorageNumT;

struct GalaxyModel;  // forward declaration

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
class BaseTarget: public math::IFunctionNdimAdd {
public:

    /// human-readable name of this target object
    virtual const char* name() const = 0;

    /// textual representation of a given coefficient (must be in the range 0 <= index < numCoefs() )
    virtual std::string coefName(unsigned int index) const = 0;

    /// argument of addPoint() is a 6d point in the cartesian position/velocity space
    virtual unsigned int numVars() const { return 6; }

    /// number of values recorded internally for each element of the additive model
    /// (orbit or DF component), i.e., is the size of the intermediate datacube;
    /// needs not be the same as the number of output coefficients numValues()
    virtual unsigned int numValues() const = 0;

    /// number of values (coefficients) stored for each component of the additive model
    virtual unsigned int numCoefs() const { return numValues(); }  //  (default, may be overriden)

    /// allocate an empty matrix for internal storage of the datacube;
    /// its overall size is numValues(), but shape may vary between descendant classes
    virtual math::Matrix<double> newDatacube() const {
        return math::Matrix<double>(1, numValues(), 0.);
    }

    /** convert the intermediate datacube into array of output values;
        \param[in] datacube  is the matrix allocated by newDatacube() and filled by repeated calls
        to addPoint();
        it is allowed to be modified inside this routine, but is supposed to be discarded afterwards.
        \param[out]  output  must point to an existing array of length numCoefs(),
        which will be filled with suitably converted values from the datacube;
        default implementation is just to copy the internal datacube, but descendant classes
        need to implement a more complex (but still linear) transformation if numCoefs() != numValues()
    */
    virtual void finalizeDatacube(math::Matrix<double> &datacube, StorageNumT* output) const {
        const double* data = datacube.data();
        for(size_t i=0, size=numValues(); i<size; i++)
            output[i] = static_cast<StorageNumT>(data[i]);
    }

    /** the following method from IFunctionNdimAdd must be implemented in descendant classes:
        accumulate the contribution of the given point to the internal datacube, weighted with 'mult';
        \param[in]  point is the position and (optionally) velocity in cartesian coordinates;
        \param[in]  mult  is the weigth of the point in the output datacube;
        \param[in,out] datacube must point to an array of length numValues(), allocated by newDatacube()
    */
    virtual void addPoint(const double point[], const double mult, double datacube[]) const = 0;

    /** compute target-specific data (projection of a DF); NOT YET IMPLEMENTED!
        \param[in] model  is the interface for computing the value(s) of a distribution function,
        possibly a multi-component DF
        \param[out] output is a pointer to the array where the DF projection will be stored:
        each DF component produces a contiguous array of numCoefs() output values;
        should be an existing chunk of memory with size numCoefs() * df.numValues()
    */
    virtual void computeDFProjection(const GalaxyModel& model, StorageNumT* output) const = 0;

    /// compute the projections of the density onto all basis elements of the grid
    virtual std::vector<double> computeDensityProjection(const potential::BaseDensity& density) const = 0;
};

typedef shared_ptr<const BaseTarget> PtrTarget;


/// Orbit runtime function that collects the values of a given N-dimensional function
/// for each point on the trajectory, weighted by the amount of time spent at this point
class RuntimeFncTarget: public orbit::BaseRuntimeFnc {

    /// the function that collects some data for a given point
    /// (takes the position/velocity in Cartesian coordinates as input)
    const BaseTarget& target;

    /// where the data for this orbit will be ultimately stored (points to an external array)
    StorageNumT* output;

    /** intermediate storage for the data collected during orbit integration,
        weighted by the time chunk associated with each sub-step on the trajectory;
        internally accumulated in double precision, and at the end of integration normalized
        by the integration time and written in the output array converted to StorageNumT
    */
    math::Matrix<double> datacube;

    /// total integration time - will be used to normalize the collected data
    /// at the end of orbit integration
    double time;

public:
    /// number of points taken from the trajectory during each timestep of the ODE solver
    static const int NUM_SAMPLES_PER_STEP = 10;

    RuntimeFncTarget(orbit::BaseOrbitIntegrator& orbint, const BaseTarget& _target, StorageNumT* _output) :
        BaseRuntimeFnc(orbint), target(_target), output(_output),
        datacube(target.newDatacube()), time(0.) {}

    /// finalize data collection, normalize the array by the total integration time,
    /// and convert to the numerical type used in the output storage
    virtual ~RuntimeFncTarget()
    {
        target.finalizeDatacube(datacube, output);  // now output contains un-normalized values
        if(time==0) return;
        const StorageNumT invtime = static_cast<StorageNumT>(1./time);
        for(size_t i=0, size=target.numCoefs(); i<size; i++)
            output[i] *= invtime;
    }

    /// collect the data returned by the function for each point sub-sampled from the trajectory
    /// on the current timestep, and add it to the temporary storage array,
    /// weighted by the duration of the substep
    virtual bool processTimestep(double tbegin, double timestep)
    {
        time += timestep;
        double substep = timestep / NUM_SAMPLES_PER_STEP;  // duration of each sub-step
        double *dataptr = datacube.data();
        for(int s=0; s<NUM_SAMPLES_PER_STEP; s++) {
            // equally-spaced samples in time, offsets from the beginning of the current timestep
            double tsubstep = substep * (s+0.5);  
            double point[6];  // position and velocity in cartesian coordinates at the current sub-step
            orbint.getSol(tsubstep).unpack_to(point);
            target.addPoint(point, substep, dataptr);
        }
        return true;
    }
};

}  // namespace
