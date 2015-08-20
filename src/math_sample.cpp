#include "math_sample.h"
#include "math_core.h"
#include <stdexcept>
#include <cassert>
#include <cmath>

#include <iostream>
namespace math{

class Sampler{
public:
    /** Construct an N-dimensional sampler object and allocate bins */
    Sampler(const IFunctionNdim& fnc, const double xlower[], const double xupper[],
            const unsigned int numBins[]);

    /** Perform a number of samples from the distribution function with the current binning scheme */
    void runPass(unsigned int numSamples);

    /** Readjust the bins */
    void readjustBins();

    /** Return the integral of F over the entire volume, and its error estimate */
    void integral(double& value, double& error) const;

    /** Return the total number of function evaluations */
    unsigned int getNumCalls() const { return numCallsFnc; }
private:
    const IFunctionNdim& fnc;  ///< the N-dimensional probability distribution to work with
    const unsigned int Ndim;   ///< a shorthand for the number of dimensions
    double volume;             ///< the total N-dimensional volume to be surveyed
    unsigned int numCells;     ///< the total number of cells in the entire volume
    unsigned int numCallsFnc;  ///< count the number of function evaluations
    std::vector< std::vector<double> > binBoundaries; ///< boundaries of grid in each dimension
    /** array of sampling points drawn from the distribution.
        each row of the matrix contains N+2 numbers:
        N coordinates of the point, the value of the function,
        and the prior probability of the given sample (equal to the volume of the N-dimensional bin
        from which the point was sampled). */
    Matrix<double> samples;

    /** randomly sample an N-dimensional point, such that it has equal probability 
        of falling into any bin in each dimension, and its coordinates within the given bin
        have uniform probability distribution.
        \param[out] coords - array of point coordinates
        \param[out] binIndices - array of bin indices in each dimension
        \param[out] cellVol - the N-dimensional volume of the bin that contains the point.
    */
    void samplePoint(double coords[], unsigned int binIndices[], double& cellVol) const;
};

Sampler::Sampler(const IFunctionNdim& _fnc, const double xlower[], const double xupper[],
    const unsigned int numBinsDim[]) : fnc(_fnc), Ndim(fnc.numVars())
{
    binBoundaries.resize(Ndim);
    volume      = 1.0;
    numCells    = 1;
    numCallsFnc = 0;
    for(unsigned int d=0; d<Ndim; d++) {
        binBoundaries[d].resize(numBinsDim[d]+1);
        for(unsigned int i=0; i<=numBinsDim[d]; i++)  // initially all bins have equal widths
            binBoundaries[d][i] = ((numBinsDim[d]-i)*xlower[d] + i*xupper[d]) / numBinsDim[d];
        volume   *= xupper[d]-xlower[d];
        numCells *= numBinsDim[d];
    }
}

void Sampler::runPass(unsigned int numSamples)
{
    std::vector<double> coords(Ndim);
    std::vector<unsigned int> indices(Ndim);
    samples.resize(numSamples, Ndim+2);
    for(unsigned int i=0; i<numSamples; i++) {
        double vol, val;
        samplePoint(&coords.front(), &indices.front(), vol);
        for(unsigned int d=0; d<Ndim; d++)
            samples(i, d) = coords[d];
        fnc.eval(&coords.front(), &val);
        numCallsFnc++;
        samples(i, Ndim) = val;   // function value
        samples(i, Ndim+1) = vol; // volume of the bin from which the coordinates were sampled
    }
}

void Sampler::readjustBins()
{
    const unsigned int npoints = samples.numRows();
    assert(npoints>0);
    std::vector< std::pair<double,double> > projection(npoints);
    // work in each dimension separately
    for(unsigned int d=0; d<Ndim; d++)
    {
        std::cout << "D="<<d;
        // create a projection onto d-th coordinate axis
        for(unsigned int i=0; i<npoints; i++) {
            projection[i].first  = samples(i, d);
            projection[i].second = samples(i, Ndim) * samples(i, Ndim+1);  // fnc value times cell volume
        }

        // sort points by 1st value in pair (i.e., the d-th coordinate)
        std::sort(projection.begin(), projection.end());

        // compute the weighted sum, which is the integral of f over the volume (up to a const factor)
        double sum = 0;
        for(unsigned int i=0; i<npoints; i++)
            sum += projection[i].second;

        // divide the grid so that each bin contains equal fraction of the total weighted sum,
        // but no less than minbin points
        const unsigned int numBins = binBoundaries[d].size()-1;
        const unsigned int minBin  = std::min<unsigned int>(npoints/numBins, 10);
        assert(minBin>0);
        unsigned int pointInd = 0, binInd = 1, pointsInBin = 0;
        double partialSum = 0;
        while(pointInd<npoints) {
            partialSum += projection[pointInd].second;
            pointsInBin++;
            if( (partialSum > sum * (binInd*1.0/numBins) && pointsInBin>=minBin) ||
                pointInd >= npoints - minBin*(numBins-binInd) )
            {
                binBoundaries[d][binInd] = projection[pointInd].first;
                std::cout << " "<<binBoundaries[d][binInd];
                binInd++;
                pointsInBin = 0;
            }
            pointInd++;
        }
        std::cout << "\n";
    }
}

void Sampler::integral(double& value, double& error) const
{
    const double mult = numCells*1.0/samples.numRows();
    double mean = 0, vari = 0;
    for(unsigned int i=0; i<samples.numRows(); i++) {
        const double val = samples(i, Ndim);   // function value
        const double vol = samples(i, Ndim+1); // volume of the bin from which this value was sampled
        double result = val*vol*mult;
        double diff = result - mean;
        mean += diff / (i+1);
        vari += diff * (result - mean);
    }
    value = mean;
    error = sqrt(vari / (samples.numRows()-1) );
}

void Sampler::samplePoint(double coords[], unsigned int binIndices[], double& binVol) const
{
    binVol = 1.0;
    for(unsigned int d=0; d<Ndim; d++) {
        double rn = random();
        if(rn<0 || rn>=1) rn=0;
        rn *= binBoundaries[d].size()-1;
        // the integer part of the random number gives the bin index
        unsigned int b = static_cast<unsigned int>(floor(rn));
        binIndices[d] = b;
        rn -= b*1.0;  // the remainder gives the position inside the bin
        coords[d] = binBoundaries[d][b]*(1-rn) + binBoundaries[d][b+1]*rn;
        binVol *= (binBoundaries[d][b+1] - binBoundaries[d][b]);
    }
}

void sampleNdim(const IFunctionNdim& fnc, const double xlower[], const double xupper[], 
    const unsigned int numSamples, const unsigned int* numBinsInput,
    Matrix<double>& samples, int* numTrialPoints, double* integral, double* interror)
{
    // determine number of sample points and bins
    const double NUM_POINTS_PER_BIN = 10.;  // default # of sampling points per N-dimensional bin
    std::vector<unsigned int> numBins(fnc.numVars());  // number of bins per each dimension
    unsigned int totalNumBins = 1;  // total number of N-dimensional bins
    if(numBinsInput!=NULL) {  // use the provided values
        for(unsigned int d=0; d<fnc.numVars(); d++) {
            numBins[d] = numBinsInput[d];
            totalNumBins *= numBinsInput[d];
        }
        if(totalNumBins >= numSamples && totalNumBins >= powInt(2, fnc.numVars()))
            throw std::invalid_argument("sampleNdim: requested number of bins is too large");
    } else {
        const unsigned int numBinsPerDim = static_cast<unsigned int>( fmax(
            pow(numSamples*1.0/NUM_POINTS_PER_BIN, 1./fnc.numVars()), 1.) );
        numBins.assign(fnc.numVars(), numBinsPerDim);
        totalNumBins = powInt(numBinsPerDim, fnc.numVars());
    }
    Sampler sampler(fnc, xlower, xupper, &numBins.front());

    // first warmup run to collect statistics and adjust bins
    const unsigned int numWarmupSamples = std::max<unsigned int>(numSamples*0.1, 1000); //??
    sampler.runPass(numWarmupSamples);
    sampler.readjustBins();

    // second run to collect samples distributed more uniformly inside the bins
    const unsigned int numCollectSamples = std::max<unsigned int>(numSamples*5, 1000); //??
    sampler.runPass(numCollectSamples);
    sampler.readjustBins();

    // do the magic
    // (not yet implemented)

    // output
    samples.resize(0, 0);
    if(numTrialPoints!=NULL)
        *numTrialPoints = sampler.getNumCalls();
    if(integral!=NULL) {
        double err;
        sampler.integral(*integral, err);
        if(interror!=NULL)
            *interror = err;
    }
};

}  // namespace
