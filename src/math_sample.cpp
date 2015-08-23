#include "math_sample.h"
#include "math_core.h"
#include <stdexcept>
#include <cassert>
#include <cmath>
#include <set>
#include <algorithm>

#include <iostream>
namespace math{

/**      Definitions:
    d    is the index of spatial dimension (0 <= d <= Ndim-1);
    H    is the N-dimensional hypercube with coordinate boundaries xlower[d]..xupper[d], d=0..Ndim-1;
    V    is the volume of this hypercube = \f$ \prod_{d=0}^{Ndim-1} (xupper[d]-xlower[d]) \f$;
    x    is the point in H, represented as a vector of coordinates  x[d], d=0..Ndim-1;
    f(x) is the function to be sampled;
    B[d] is the binning scheme in each dimension: 
         for each d, B[d] an array of bin boundaries such that
         xlower[d] = B[d][0] < B[d][1] < B[d][2] < ... < B[d][K_d] = xupper[d],
         where K_d is the number of bins along d-th coordinate axis.
    C(x) is the cell in N-dimensional space that contains the point x;
         this cell is the intersection of bins in each coordinate axis d,
         where the index `b_d` of the bin in d-th dimension is assigned so that 
         B[d][b_d] <= x[d] < B[d][b_d+1]
         (in other words, bins enclose the point x in each coordinate).
    Vc(x) is the volume of this cell = \f$ \prod_{d=0}^{Ndim-1}  (B[d][b_d+1]-B[d][b_d]) \f$;
    Nc   is the total number of cells in the entire partitioning scheme of our hypercube,
         \f$ Nc = \prod_{d=0}^{Ndim-1} K_d \f$, and the sum of volumes of all cells is equal to V.    
    x_i, i=0..M-1  are the sampling points in H; they are distributed non-uniformly over H,
         so that the probability of drawing a sample with coordinates x is 1/w(x), where
    w(x) is the weighting function, which is initially assigned as w(x) = Vc(x) Nc / V;
         if all cells were of equal volume, then w(x) would be identically unity,
         and for unequal cells, w is proportional to the cell volume,
         After refinement iterations, w may differ from this original assignment.
    EI   is the estimate of the integral \f$ \int_{H} f(x) d^N x \f$,
         computed as \f$  EI = \sum_{i=0}^{M-1}  f(x_i) w(x_i)  \f$
         (i.e., as the weighted sum  f(x) w(x) over all sampling points x_i).
    EE   is the estimate of the error in that integral, computed as
         \f$  EE = \sqrt{ \sum_{i=0}^{M-1} [ f(x_i) w(x_i) - <f(x) w(x)> ]^2 } \f$.
    S    is the number of random samples to be drawn from the distribution function and returned
         as the result of sampling procedure; it is not the same as the total number of 
         evaluated samples M - the latter is typically considerably larger than S.
    y_k, k=0..S-1  are the returned sampling points.

    The Sampler class implements an algorithm inspired by VEGAS Monte Carlo integration method.
    It divides the entire hypercube into Nc cells by splitting each coordinate axis d 
    into K_d bins. The widths of bins are arranged in such a way as to make the integral 
    of f(x) over the volume of each cell as constant as possible. 
    Of course, due to the separable binning scheme there is only  ~ N K free parameters for
    K^N cells, so that this scheme works well only if the function itself is close to separable. 

    The volume of the hypercube is initially sampled by M points x_i distributed equally 
    between cells and uniformly within the volume of each cell; thus the contribution of each 
    sampling point to the total integral \f$  \int_{H} f(x) d^N x  \f$  is proportional to 
    the volume of cell that this point belongs to (the weight factor w_i).

    At first, all bins have equal widths, but this is not efficient for the sampling procedure; 
    thus a first warmup pass with a modest number of function evaluations is used to determine
    the bin boundaries in each dimension, in such a way that the marginalized distribution of f
    over all but one axes is equally divided into bins in that axis.
    Then a second run is performed with the unequal-sized bins, collecting the internal 
    sampling points in a more balanced way.

    The output samples are drawn from these internal sampling points in proportion
    to the magnitude of f(x_i) times the weight of the corresponding internal sample w_i.
    Even though the number of internal sampling points M should normally considerably exceed 
    the number of output samples S, it may still turn out that the weighted value f(x_i) w_i 
    is larger than the granularity of output samples. Then the cell that hosts this value 
    is scheduled for refinement.

    The refinement procedure takes one or several iterations to ensure that all f(x_i) w_i 
    are smaller than the weight of one output sample. In doing so, the existing internal 
    samples in problematic cells are augmented with additional ones, while decreasing 
    the weights of both original and new sampling points so that the integral remains unchanged.

*/
class Sampler{
public:
    /** Construct an N-dimensional sampler object and allocate bins */
    Sampler(const IFunctionNdim& fnc, const double xlower[], const double xupper[],
            const unsigned int numBins[]);

    /** Perform a number of samples from the distribution function with the current binning scheme,
        and computes the estimate of integral EI (stored internally) */
    void runPass(const unsigned int numSamples);

    /** Readjust the bin widths using the collected samples */
    void readjustBins();

    /** Make sure that the number of internal samples is enough to draw 
        the requested number of output samples, and if not, run a refinement loop */
    void ensureEnoughSamples(const unsigned int numSamples);

    /** Draw a requested number of output samples from the already computed array of internal samples */
    void drawSamples(const unsigned int numSamples, Matrix<double>& samples) const;

    /** Return the integral of F over the entire volume, and its error estimate */
    void integral(double& value, double& error) const {
        value = integValue;
        error = integError;
    }

    /** Return the total number of function evaluations */
    unsigned int numCalls() const { return numCallsFnc; }

private:
    const IFunctionNdim& fnc;    ///< the N-dimensional function to work with        [ f(x) ]
    const unsigned int Ndim;     ///< a shorthand for the number of dimensions
    double volume;               ///< the total N-dimensional volume to be surveyed  [ V ]
    unsigned int numCells;       ///< the total number of cells in the entire volume [ Nc ]
    unsigned int numCallsFnc;    ///< count the number of function evaluations
    std::vector< std::vector<double> >
        binBoundaries;           ///< boundaries of grid in each dimension           [ B[d][b] ]
    Matrix<double> sampleCoords; ///< array of sampling points drawn from the distribution,
    ///< each i-th row of the matrix contains N coordinates of the point             [ x_i[d] ]
    std::vector<double> weightedFncValues;  ///< array of weighted function values  f(x_i) w(x_i),
    ///< where initially w = Vc(x) * Nc / V, i.e., proportional to the volume of 
    ///< the N-dimensional cell from which the point was sampled,
    ///< and later w may be reduced if this cell gets refined
    double integValue;           ///< estimate of the integral of f(x) over H        [ EI ]
    double integError;           ///< estimate of the error in the integral          [ EE ]
    typedef long unsigned int CellEnum; ///< the way to enumerate all cells, should be a large enough type

    /** randomly sample an N-dimensional point, such that it has equal probability 
        of falling into each cell, and its location within the given cell
        has uniform probability distribution.
        \param[out] coords - array of point coordinates;                             [ x[d] ]
        \return  the weight of this point w(x), which is proportional to
        the N-dimensional volume Vc(x) of the cell that contains the point.          [ w(x) ]
    */
    double samplePoint(double coords[]) const;

    /** randomly sample an N-dimensional point inside a given cell;
        \param[in]  cellInd is the index of cell that the point should lie in;
        \param[out] coords is the array of point coordinates;
        \return  the weight of this point (same as for `samplePoint()` ).
    */
    double samplePointFromCell(CellEnum cellInd, double coords[]) const;

    /** return the value of function f(x) for the given coordinates, 
        and check that it is non-negative and finite (otherwise throw an exception). */
    double evalFnc(const double coords[]);

    /** return the index of the N-dimensional cell containing a given point */
    CellEnum cellIndex(const double coords[]) const;
};

Sampler::Sampler(const IFunctionNdim& _fnc, const double xlower[], const double xupper[],
    const unsigned int numBinsDim[]) : fnc(_fnc), Ndim(fnc.numVars())
{
    binBoundaries.resize(Ndim);
    volume      = 1.0;
    numCells    = 1;
    numCallsFnc = 0;
    integValue  = integError = NAN;
    for(unsigned int d=0; d<Ndim; d++) {
        binBoundaries[d].resize(numBinsDim[d]+1);
        for(unsigned int i=0; i<=numBinsDim[d]; i++)  // initially all bins have equal widths
            binBoundaries[d][i] = ((numBinsDim[d]-i)*xlower[d] + i*xupper[d]) / numBinsDim[d];
        volume   *= xupper[d]-xlower[d];
        numCells *= numBinsDim[d];
    }
}

double Sampler::samplePoint(double coords[]) const
{
    double binVol = 1.0;
    for(unsigned int d=0; d<Ndim; d++) {
        double rn = random();
        if(rn<0 || rn>=1) rn=0;
        rn *= binBoundaries[d].size()-1;
        // the integer part of the random number gives the bin index
        unsigned int b = static_cast<unsigned int>(floor(rn));
        rn -= b*1.0;  // the remainder gives the position inside the bin
        coords[d] = binBoundaries[d][b]*(1-rn) + binBoundaries[d][b+1]*rn;
        binVol *= (binBoundaries[d][b+1] - binBoundaries[d][b]);
    }
    return binVol;
}

double Sampler::samplePointFromCell(CellEnum cellInd, double coords[]) const
{
    assert(cellInd<numCells);
    double binVol = 1.0;
    for(unsigned int d=Ndim; d>0; d--) {
        unsigned int b = cellInd % (binBoundaries[d-1].size()-1);
        cellInd /= (binBoundaries[d-1].size()-1);
        double rn = random();
        coords[d-1] = binBoundaries[d-1][b]*(1-rn) + binBoundaries[d-1][b+1]*rn;
        binVol *= (binBoundaries[d-1][b+1] - binBoundaries[d-1][b]);
    }
    return binVol;
}

Sampler::CellEnum Sampler::cellIndex(const double coords[]) const
{
    CellEnum cellInd = 0;
    for(unsigned int d=0; d<Ndim; d++) {
        cellInd *= binBoundaries[d].size()-1;
        cellInd += binSearch(coords[d], &binBoundaries[d].front(), binBoundaries[d].size());
    }
    return cellInd;
}

double Sampler::evalFnc(const double coords[])
{
    double val;
    fnc.eval(coords, &val);
    if(val<0 || !isFinite(val))
        throw std::runtime_error("Error in sampleNdim: function value is negative or not finite");
    numCallsFnc++;
    return val;
}

void Sampler::runPass(const unsigned int numSamples)
{
    sampleCoords.resize(numSamples, Ndim);
    weightedFncValues.resize(numSamples);
    const double sampleWeight = numCells * 1. / numSamples;  // multiplier for sample weight
    Averager avg;
    for(unsigned int i=0; i<numSamples; i++) {
        double* coords = &(sampleCoords(i, 0));     // address of 0th element in i-th matrix row
        // randomly assign coords and record the weight of this point, proportional to
        // the volume of the cell from which the coordinates were sampled
        double weight  = samplePoint(coords) * sampleWeight; 
        double wval    = evalFnc(coords) * weight;  // function value times the weight coefficient
        weightedFncValues[i] = wval;
        avg.add(wval);
    }
    integValue = avg.mean() * numSamples;
    integError = sqrt(avg.disp() * numSamples);
    std::cout << "Integral = "<<integValue<<" +- "<<integError<<"\n";
}

void Sampler::readjustBins()
{
    const unsigned int npoints = weightedFncValues.size();
    assert(sampleCoords.numRows() == npoints);
    assert(npoints>0);
    std::vector< std::pair<double,double> > projection(npoints);
    // work in each dimension separately
    for(unsigned int d=0; d<Ndim; d++)
    {
        std::cout << "D="<<d;
        // create a projection onto d-th coordinate axis
        for(unsigned int i=0; i<npoints; i++) {
            projection[i].first  = sampleCoords(i, d);
            projection[i].second = weightedFncValues[i];  // fnc value times weight
        }

        // sort points by 1st value in pair (i.e., the d-th coordinate)
        std::sort(projection.begin(), projection.end());

        // divide the grid so that each bin contains equal fraction of the total sum
        // but no less than minbin points
        const unsigned int numBins = binBoundaries[d].size()-1;
        const unsigned int minBin  = std::min<unsigned int>(npoints/numBins, 10);
        assert(minBin>0);
        unsigned int pointInd = 0, binInd = 1, pointsInBin = 0;
        double partialSum = 0;
        while(pointInd<npoints) {
            partialSum += projection[pointInd].second;
            pointsInBin++;
            if( (partialSum > integValue * (binInd*1.0/numBins) && 
                 pointsInBin>=minBin && binInd<numBins) ||
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

void Sampler::ensureEnoughSamples(const unsigned int numOutputSamples)
{
    // initial number of internal samples per cell, doubled on each iteration
    double samplesPerCell = weightedFncValues.size() * 1.0 / numCells;
    // multiplier for weights of newly-sampled points, which is halved on each iteration
    double sampleWeight = numCells * 1. / sampleCoords.numRows();
    int nIter=0;  // safeguard against infinite loop
    do{
        // list of cells that need refinement
        std::set<CellEnum> cellsForRefinement;
        const unsigned int npoints = weightedFncValues.size();
        assert(sampleCoords.numRows() == npoints);   // number of internal samples already taken
        // maximum allowed value of f(x)*w(x), which is the weight of one output sample
        const double maxWeight = integValue / (numOutputSamples+1e-6);

        // determine if any of our sampled points are too heavy for the requested number of output points
        unsigned int numOverflows = 0;
        for(unsigned int i=0; i<npoints; i++) {
            if(weightedFncValues[i] > maxWeight) {
                cellsForRefinement.insert(cellIndex(&sampleCoords(i, 0)));
                numOverflows++;
            }
        }
        if(cellsForRefinement.empty())
            return;   // no further action necessary

        // some cells need refinement, which means that we put more samples into these cells,
        // while decreasing the weights of existing samples in these cells.
        Averager avg;  // accumulators for recomputing the integral and its error

        // first decrease the weights of all samples that belong to the cells deemed for refinement
        for(unsigned int i=0; i<npoints; i++) {
            if(cellsForRefinement.find(cellIndex(&sampleCoords(i, 0))) != cellsForRefinement.end())
                weightedFncValues[i] *= 0.5;  // with each refinement iteration, weights are halved
            // update the estimate of integral using all existing points (not only the resampled ones)
            avg.add(weightedFncValues[i]);
        }

        // next add new samples into each cell that was marked for refinement
        sampleWeight *= 0.5;
        unsigned int numRefinedCells = 0;
        for(std::set<CellEnum>::const_iterator iterCell = cellsForRefinement.begin();  
            iterCell != cellsForRefinement.end(); ++iterCell) 
        {
            CellEnum indexCell = *iterCell;
            // since # of samplesPerCell may not be an integer number,
            // we randomly choose this number for each cell to provide the correct average value
            int samplesPerThisCell = static_cast<int>(floor(samplesPerCell+random()));
            for(int i=0; i<samplesPerThisCell; i++) {
                sampleCoords.resize(sampleCoords.numRows()+1, Ndim);
                double* coords = &(sampleCoords(sampleCoords.numRows()-1, 0));  // taking the entire row
                double weight  = samplePointFromCell(indexCell, coords) * sampleWeight;
                assert(cellIndex(coords) == indexCell);
                double wval    = evalFnc(coords) * weight;  // function value times the weight coefficient
                weightedFncValues.push_back(wval);
                avg.add(wval);
            }
            numRefinedCells++;
        }
        samplesPerCell *= 2;
        
        integValue = avg.mean() * sampleCoords.numRows();
        integError = sqrt(avg.disp() * sampleCoords.numRows());
        std::cout << "Integral = "<<integValue<<" +- "<<integError<<
        " after refining "<<numRefinedCells<<
        " cells because of "<<numOverflows<<" overweight samples\n";
    } while(++nIter<10);
    throw std::runtime_error("Error in sampleNdim: refinement procedure did not converge in 10 iterations");
}
    
void Sampler::drawSamples(const unsigned int numOutputSamples, Matrix<double>& outputSamples) const
{
    outputSamples.resize(numOutputSamples, Ndim);
    const unsigned int npoints = weightedFncValues.size();
    assert(sampleCoords.numRows() == npoints);   // number of internal samples already taken
    double partialSum = 0;        // accumulates the sum of f(x_i) w(x_i) for i=0..{current value}
    const double sampleWeight =   // difference in accumulated sum between two output samples
        integValue / (numOutputSamples+1e-6);
    // the tiny addition above ensures that the last output sample coincides with the last internal sample
    unsigned int outputIndex = 0;
    for(unsigned int i=0; i<npoints && outputIndex<numOutputSamples; i++) {
        assert(weightedFncValues[i] <= sampleWeight);  // has been guaranteed by ensureEnoughSamples()
        partialSum += weightedFncValues[i];
        if(partialSum >= (outputIndex+1)*sampleWeight) {
            for(unsigned int d=0; d<Ndim; d++)
                outputSamples(outputIndex, d) = sampleCoords(i, d);
            outputIndex++;
        }
    }
    outputSamples.resize(outputIndex, Ndim);
    std::cout << outputIndex << " points sampled out of " << numCallsFnc << " internal ones\n";
}

void sampleNdim(const IFunctionNdim& fnc, const double xlower[], const double xupper[], 
    const unsigned int numSamples, const unsigned int* numBinsInput,
    Matrix<double>& samples, int* numTrialPoints, double* integral, double* interror)
{
    // determine number of sample points and bins
    const double NUM_POINTS_PER_BIN = 10.;  // default # of sampling points per N-dimensional bin
    std::vector<unsigned int> numBins(fnc.numVars());  // number of bins per each dimension
    unsigned int totalNumBins = 1;  // total number of N-dimensional bins
    if(numBinsInput!=NULL) {        // use the provided values
        for(unsigned int d=0; d<fnc.numVars(); d++) {
            numBins[d] = numBinsInput[d];
            totalNumBins *= numBinsInput[d];
        }
        if( totalNumBins >= std::max<unsigned int>(numSamples, 10000) && 
            totalNumBins >= powInt(2, fnc.numVars()))
            throw std::invalid_argument("sampleNdim: requested number of bins is too large");
    } else {   // nothing provided for the number of bins - determine automatically
        const unsigned int numBinsPerDim = static_cast<unsigned int>( fmax(
            pow(numSamples*1.0/NUM_POINTS_PER_BIN, 1./fnc.numVars()), 1.) );
        numBins.assign(fnc.numVars(), numBinsPerDim);
        totalNumBins = powInt(numBinsPerDim, fnc.numVars());
    }
    Sampler sampler(fnc, xlower, xupper, &numBins.front());

    // first warmup run to collect statistics and adjust bins
    const unsigned int numWarmupSamples = std::max<unsigned int>(numSamples*0.2, 10000);
    sampler.runPass(numWarmupSamples);
    sampler.readjustBins();

    // second run to collect samples distributed more uniformly inside the bins
    const unsigned int numCollectSamples = std::max<unsigned int>(numSamples*(fnc.numVars()+1), 10000);
    sampler.runPass(numCollectSamples);

    // make sure that no sampling point has too large weight, 
    // if no then seed additional samples
    sampler.ensureEnoughSamples(numSamples);

    // finally, draw the required number of output samples from the internal ones
    sampler.drawSamples(numSamples, samples);

    // statistics
    if(numTrialPoints!=NULL)
        *numTrialPoints = sampler.numCalls();
    if(integral!=NULL) {
        double err;
        sampler.integral(*integral, err);
        if(interror!=NULL)
            *interror = err;
    }
};

}  // namespace
