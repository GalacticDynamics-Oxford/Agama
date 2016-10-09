#include "math_sample.h"
#include "math_core.h"
#include "math_spline.h"
#include "utils.h"
#include <stdexcept>
#include <cassert>
#include <cmath>
#include <map>
#include <algorithm>

namespace math{

namespace {  // internal namespace for Sampler class

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

    The number of bins per dimension is a power of two and is determined by the following scheme:
    at the first (warmup) pass, all dimensions have only one bin (i.e. all samples are distributed
    uniformly along each coordinate).
    Then the collected sampling points are used to build a projection of `f` onto each axis 
    (i.e., ignoring the values of all other coordinates). Each axis is then divided into equal number 
    of bins with varying widths, so that the integral of `f` over each bin is approximately equal. 
    In this way, the number of cells created can be very large (numBins^Ndim), which is undesirable.
    Therefore, the bins are merged, according to the following rule: at each iteration, 
    we determine the dimension with the least variation in widths of adjacent bins 
    (signalling that the function values do not strongly depend on this coordinate), 
    and halve the number of bins in this dimension, merging even with odd bins.
    This is repeated until the total number of cells becomes manageable.
    The rebinning procedure can be performed more than once, but after each one a new sample
    collection pass is required.

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
    /** Construct an N-dimensional sampler object */
    Sampler(const IFunctionNdim& fnc, const double xlower[], const double xupper[]);

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
    /// the way to enumerate all cells, should be a large enough type
    typedef long unsigned int CellEnum;

    /// correspondence between cell index and a numerical value
    typedef std::map<CellEnum, double> CellMap;

    /// the N-dimensional function to work with                          [ f(x) ]
    const IFunctionNdim& fnc;
    
    /// a shorthand for the number of dimensions
    const unsigned int Ndim;
    
    /// the total N-dimensional volume to be surveyed                    [ V ]
    double volume;
    
    /// the total number of cells in the entire volume                   [ Nc ]
    CellEnum numCells;

    /// count the number of function evaluations
    unsigned int numCallsFnc;

    /// boundaries of grid in each dimension                             [ B[d][b] ]
    std::vector< std::vector<double> > binBoundaries;

    /// array of sampling points drawn from the distribution,
    /// each i-th row of the matrix contains N coordinates of the point  [ x_i[d] ]
    Matrix<double> sampleCoords;
    
    /// array of weighted function values  f(x_i) w(x_i),  where initially
    /// w = Vc(x) * Nc / V, i.e., proportional to the volume of the N-dimensional cell 
    /// from which the point was sampled, and later w may be reduced if this cell gets refined
    std::vector<double> weightedFncValues;

    /// default average number of sampling points per (unrefined) cell,
    /// equal to numCells / numSamples;  is modified for cells that undergo refinement
    double defaultSamplesPerCell;

    /// average number of sampling points per cell that has been refined
    CellMap samplesPerCell;

    /// estimate of the integral of f(x) over H                          [ EI ]
    double integValue;

    /// estimate of the error in the integral                            [ EE ]
    double integError;

    /** randomly sample an N-dimensional point, such that it has equal probability 
        of falling into each cell, and its location within the given cell
        has uniform probability distribution.
        \param[out] coords - array of point coordinates;                      [ x[d] ]
        \return  the weight of this point w(x), which is proportional to
        the N-dimensional volume Vc(x) of the cell that contains the point.   [ w(x) ]
    */
    double samplePoint(double coords[]) const;

    /** randomly sample an N-dimensional point inside a given cell;
        \param[in]  cellInd is the index of cell that the point should lie in;
        \param[out] coords is the array of point coordinates;
        \return  the weight of this point (same as for `samplePoint()` ).
    */
    double samplePointFromCell(CellEnum cellInd, double coords[]) const;

    /** evaluate the value of function f(x) for the points from the sampleCoords array,
        parallelizing the loop and guarding against exceptions */
    void evalFncLoop(unsigned int indexOfFirstPoint, unsigned int count);

    /** obtain the bin boundaries in each dimension for the given cell index */
    void getBinBoundaries(CellEnum indexCell, double lowerCorner[], double upperCorner[]) const;

    /** return the index of the N-dimensional cell containing a given point */
    CellEnum cellIndex(const double coords[]) const;

    /** return the index of cell containing the given sampling point */
    CellEnum cellIndex(const unsigned int indPoint) const {
        return cellIndex(&sampleCoords(indPoint, 0));
    }

    /** refine a cell by adding more sampling points into it, 
        while decreasing the weights of existing points, their list being provided in the last argument */
    void refineCellByAddingSamples(CellEnum indexCell,
        unsigned int indexAddSamples, unsigned int numAddSamples,
        const std::vector<unsigned int>& listOfPointsInCell);

    /** update the estimate of integral and its error, using all collected samples */
    void computeIntegral();
};

/// limit the total number of cells so that each cell has, on average, at least that many sampling points
static const unsigned int MIN_SAMPLES_PER_CELL = 5;

/// maximum number of bins in each dimension (MUST be a power of two)
static const unsigned int MAX_BINS_PER_DIM = 16;

Sampler::Sampler(const IFunctionNdim& _fnc, const double xlower[], const double xupper[]) :
    fnc(_fnc), Ndim(fnc.numVars())
{
    volume      = 1.0;
    numCells    = 1;
    numCallsFnc = 0;
    integValue  = integError = NAN;
    binBoundaries.resize(Ndim);
    for(unsigned int d=0; d<Ndim; d++) {
        binBoundaries[d].resize(2);
        binBoundaries[d][0] = xlower[d];
        binBoundaries[d][1] = xupper[d];
        volume   *= xupper[d]-xlower[d];
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
        binVol   *= (binBoundaries[d][b+1] - binBoundaries[d][b]);
    }
    return binVol;
}

double Sampler::samplePointFromCell(CellEnum indexCell, double coords[]) const
{
    assert(indexCell<numCells);
    double binVol      = 1.0;
    for(unsigned int d = Ndim; d>0; d--) {
        unsigned int b = indexCell % (binBoundaries[d-1].size()-1);
        indexCell     /= (binBoundaries[d-1].size()-1);
        double rn      = random();
        coords[d-1]    = binBoundaries[d-1][b]*(1-rn) + binBoundaries[d-1][b+1]*rn;
        binVol        *= (binBoundaries[d-1][b+1] - binBoundaries[d-1][b]);
    }
    return binVol;
}

void Sampler::getBinBoundaries(CellEnum indexCell, double lowerCorner[], double upperCorner[]) const
{
    for(unsigned int d   = Ndim; d>0; d--) {
        unsigned int b   = indexCell % (binBoundaries[d-1].size()-1);
        indexCell       /= (binBoundaries[d-1].size()-1);
        lowerCorner[d-1] = binBoundaries[d-1][b];
        upperCorner[d-1] = binBoundaries[d-1][b+1];
    }
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

void Sampler::evalFncLoop(unsigned int first, unsigned int count)
{
    if(count==0) return;
    bool badValueOccured = false;
    std::string errorMsg;
    // loop over assigned points and compute the values of function (possibly in parallel)
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic,256)
    for(int i=0; i<(int)count; i++) {
        double val;
        try{
            fnc.eval(&(sampleCoords(i+first, 0)), &val);
        }
        // guard against possible exceptions, since they must not leave the OpenMP block
        catch(std::exception& e) {
            errorMsg = e.what();
            val = NAN;
        }
        if(val<0 || !isFinite(val))
            badValueOccured = true;
        weightedFncValues[i+first] *= val;
    }
#else
    for(unsigned int i=0; i<count; i++) {
        double val;
        fnc.eval(&(sampleCoords(i+first, 0)), &val);
        if(val<0 || !isFinite(val)) {
            badValueOccured = true;
            i=count;
        }
        weightedFncValues[i+first] *= val;
    }
#endif
    numCallsFnc += count;
    if(badValueOccured)
        throw std::runtime_error("Error in sampleNdim: " + 
            (errorMsg.empty() ? "function value is negative or not finite" : errorMsg));
}

void Sampler::runPass(const unsigned int numSamples)
{
    sampleCoords=math::Matrix<double>(numSamples, Ndim);    // preallocate space for both arrays
    weightedFncValues.resize(numSamples);
    defaultSamplesPerCell = numSamples * 1. / numCells;
    samplesPerCell.clear();

    // first assign the coordinates of the sampled points and their weight coefficients
    for(unsigned int i=0; i<numSamples; i++) {
        double* coords = &(sampleCoords(i, 0)); // address of 0th element in i-th matrix row
        // randomly assign coords and record the weight of this point, proportional to
        // the volume of the cell from which the coordinates were sampled (fnc is not yet called)
        weightedFncValues[i] = samplePoint(coords) / defaultSamplesPerCell; 
    }
    // next compute the values of function at these points
    evalFncLoop(0, numSamples);

    // update the estimate of integral
    computeIntegral();
}

void Sampler::computeIntegral()
{
    const unsigned int numSamples = weightedFncValues.size();
    assert(sampleCoords.rows() == numSamples);
    Averager avg;
    for(unsigned int i=0; i<numSamples; i++)
        avg.add(weightedFncValues[i]);
    integValue = avg.mean() * numSamples;
    integError = sqrt(avg.disp() * numSamples);
    utils::msg(utils::VL_DEBUG, "sampleNdim",
        "Integral value="+utils::toString(integValue)+" +- "+utils::toString(integError)+
        " using "+utils::toString(numCallsFnc)+" function calls");
}

void Sampler::readjustBins()
{
    const unsigned int numSamples = weightedFncValues.size();
    assert(sampleCoords.rows() == numSamples);
    assert(numSamples>0);

    // draw bin boundaties in each dimension separately
    const unsigned int MIN_SAMPLES_PER_BIN =
        std::max<unsigned int>(MIN_SAMPLES_PER_CELL*2,
        std::min<unsigned int>(100, numSamples/MAX_BINS_PER_DIM/10));
    numCells = 1;
    std::vector<std::pair<double,double> > projection(numSamples);
    std::vector<double> cumSumValues(numSamples);
    std::vector<std::vector<double> > binIntegrals(Ndim);
    for(unsigned int d=0; d<Ndim; d++) {
        // create a projection onto d-th coordinate axis
        for(unsigned int i=0; i<numSamples; i++) {
            projection[i].first  = sampleCoords(i, d);
            projection[i].second = weightedFncValues[i];  // fnc value times weight
        }

        // sort points by 1st value in pair (i.e., the d-th coordinate)
        std::sort(projection.begin(), projection.end());

        // replace the point value by the cumulative sum of all values up to this one
        double cumSum = 0;
        for(unsigned int i=0; i<numSamples; i++) {
            cumSumValues[i] = (cumSum += projection[i].second);
        }

        std::vector<double> newBinBoundaries(2);
        std::vector<unsigned int> newBinIndices(2);
        newBinBoundaries[0] = binBoundaries[d].front();
        newBinBoundaries[1] = binBoundaries[d].back();
        newBinIndices[0] = 0;
        newBinIndices[1] = numSamples-1;
        unsigned int nbins = 1;
        do{
            // split each existing bin in two halves
            for(unsigned int b=0; b<nbins; b++) {
                // locate the center of the bin (in terms of cumulative weight)
                double cumSumCenter = (b+0.5) * cumSum / nbins;
                unsigned int indLeft = binSearch(cumSumCenter, &cumSumValues.front(), numSamples);
                assert(indLeft<numSamples-1);
                // determine the x-coordinate that splits the bin into two equal halves
                double binHalf = linearInterp(cumSumCenter,
                    cumSumValues[indLeft], cumSumValues[indLeft+1],
                    projection[indLeft].first, projection[indLeft+1].first);
                newBinIndices.insert(newBinIndices.begin() + b*2+1, indLeft);
                newBinBoundaries.insert(newBinBoundaries.begin() + b*2+1, binHalf);
            }
            // check if bins are large enough
            nbins = newBinBoundaries.size()-1;  // now twice as large as before
            bool valid = true;
            for(unsigned int b=0; b<nbins; b++)
                valid &= newBinIndices[b+1] - newBinIndices[b] >= MIN_SAMPLES_PER_BIN;
            if(valid)  // commit results
                binBoundaries[d] = newBinBoundaries;
            else       // discard this level of refinement because some bins contain too few points
                break;
        } while(nbins<MAX_BINS_PER_DIM);
        numCells *= binBoundaries[d].size()-1;
        binIntegrals[d].assign(binBoundaries[d].size()-1, cumSum/(binBoundaries[d].size()-1));
    }

    // now the total number of cells is probably quite large;
    // we reduce it by halving the number of bins in a dimension that demonstrates the least
    // variation in bin widths, repeatedly until the total number of cells becomes reasonably small
    const CellEnum maxNumCells = std::max<unsigned int>(1, numSamples / MIN_SAMPLES_PER_CELL);
    while(numCells > maxNumCells) {
        // determine the dimension with the lowest variation in widths of adjacent bins
        unsigned int dimToMerge = Ndim;
        double minOverhead = INFINITY;  // minimum variation among all dimensions
        for(unsigned int d=0; d<Ndim; d++) {
            unsigned int nbins = binBoundaries[d].size()-1;
            if(nbins>1) {
                double overhead = 0;
                for(unsigned int b=1; b<nbins; b+=2) {
                    double width1 = binBoundaries[d][b] - binBoundaries[d][b-1];
                    double width2 = binBoundaries[d][b+1] - binBoundaries[d][b];
                    overhead += fmax(binIntegrals[d][b-1]/width1, binIntegrals[d][b]/width2) * (width1+width2);
                }
                if(overhead < minOverhead) {
                    dimToMerge = d;
                    minOverhead = overhead;
                }
            }
        }
        assert(dimToMerge<Ndim);  // it cannot be left unassigned,
        // since we must still have at least one dimension with more than one bin
        
        // merge pairs of adjacent bins in the given dimension
        unsigned int newNumBins = (binBoundaries[dimToMerge].size()-1) / 2;
        assert(newNumBins>=1);
        // erase every other boundary, i.e. between 0 and 1, between 2 and 3, and so on
        for(unsigned int i=0; i<newNumBins; i++) {
            binIntegrals[dimToMerge][i] = fmax(
                binIntegrals[dimToMerge][i]/(binBoundaries[dimToMerge][i+1] - binBoundaries[dimToMerge][i]),
                binIntegrals[dimToMerge][i+1]/(binBoundaries[dimToMerge][i+2] - binBoundaries[dimToMerge][i+1])) *
                (binBoundaries[dimToMerge][i+2] - binBoundaries[dimToMerge][i]);
            binIntegrals[dimToMerge].erase(binIntegrals[dimToMerge].begin()+i+1);
            binBoundaries[dimToMerge].erase(binBoundaries[dimToMerge].begin()+i+1);
        }
        
        numCells /= 2;
    }
    if(utils::verbosityLevel >= utils::VL_VERBOSE) {
        for(unsigned int d=0; d<Ndim; d++) {
            std::string text = "bins for D=" + utils::toString(d) + ':';
            for(unsigned int k=0; k<binBoundaries[d].size(); k++)
                text += ' ' + utils::toString(binBoundaries[d][k]);
            utils::msg(utils::VL_VERBOSE, "sampleNdim", text);
        }
    }
}

// put more samples into a cells, while decreasing the weights of existing samples in it
void Sampler::refineCellByAddingSamples(CellEnum indexCell,
    unsigned int indexAddSamples, unsigned int numAddSamples,
    const std::vector<unsigned int>& listOfPointsInCell)
{
    assert(numAddSamples>0);
    double refineFactor = 1. + numAddSamples * 1. / listOfPointsInCell.size();

    // retrieve the average number of samples per cell for this cell
    double samplesPerThisCell = defaultSamplesPerCell;
    CellMap::iterator iter = samplesPerCell.find(indexCell);
    if(iter != samplesPerCell.end())  // this cell has already been refined before
        samplesPerThisCell = iter->second;
    samplesPerThisCell *= refineFactor;

    // update the list of (non-default) average number of samples per cell
    if(iter != samplesPerCell.end())
        iter->second = samplesPerThisCell;  // update info
    else   // has not yet been refined - append to the list
        samplesPerCell.insert(std::make_pair(indexCell, samplesPerThisCell));

    // decrease the weights of all existing samples that belong to this cell
    for(unsigned int i=0; i<listOfPointsInCell.size(); i++)
        weightedFncValues[ listOfPointsInCell[i] ] /= refineFactor;

    // assign coordinates for newly sampled points, but don't evaluate function yet --
    // this will be performed once all cells have been refined
    for(unsigned int i=0; i<numAddSamples; i++) {
        double* coords = &(sampleCoords(indexAddSamples+i, 0));  // taking the entire row
        weightedFncValues[indexAddSamples+i] = 
            samplePointFromCell(indexCell, coords) / samplesPerThisCell;
        assert(cellIndex(coords) == indexCell);
    }
}

void Sampler::ensureEnoughSamples(const unsigned int numOutputSamples)
{
    int nIter=0;  // safeguard against infinite loop
    do{
        const unsigned int numSamples = weightedFncValues.size();
        assert(sampleCoords.rows() == numSamples);   // number of internal samples already taken
        // maximum allowed value of f(x)*w(x), which is the weight of one output sample
        // (this number is not constant because the estimate of integValue is adjusted after each iteration)
        const double maxWeight = integValue / (numOutputSamples+1e-6);

        // list of cells that need refinement, along with their refinement factors R
        // ( the ratio of the largest sample weight to maxWeight, which determines how many
        // new samples we need to place into this cell: R = (N_new + N_existing) / N_existing )
        CellMap cellsForRefinement;

        unsigned int numOverweightSamples=0, numCellsForRefinement=0;
        // determine if any of our sampled points are too heavy for the requested number of output points
        for(unsigned int indexPoint=0; indexPoint<numSamples; indexPoint++) {
            double refineFactor = weightedFncValues[indexPoint] / maxWeight;
            if(refineFactor > 1) {  // encountered an overweight sample
                CellEnum indexCell = cellIndex(indexPoint);
                CellMap::iterator iter = cellsForRefinement.find(indexCell);
                if(iter == cellsForRefinement.end())  // append a new cell
                    cellsForRefinement.insert(std::make_pair(indexCell, refineFactor));
                else if(iter->second < refineFactor)
                    iter->second = refineFactor;   // update the required refinement factor for this cell
                ++numOverweightSamples;
            }
        }
        if(cellsForRefinement.empty())
            return;   // no further action necessary

        // compile the list of samples belonging to each cell to be refined
        std::map<CellEnum, std::vector<unsigned int> > samplesInCell;
        for(unsigned int indexPoint=0; indexPoint<numSamples; indexPoint++) {
            CellEnum indexCell = cellIndex(indexPoint);
            CellMap::const_iterator iter = cellsForRefinement.find(indexCell);
            if(iter != cellsForRefinement.end())
                samplesInCell[iter->first].push_back(indexPoint);
        }

        // loop over cells to be refined and assign the number of additional samples for each cell
        std::map<CellEnum, unsigned int> numAddSamples;
        unsigned int numAddSamplesTotal = 0;
        for(CellMap::const_iterator iter = cellsForRefinement.begin();
            iter != cellsForRefinement.end(); ++iter)
        {
            CellEnum indexCell  = iter->first;
            double refineFactor = iter->second*1.25;  // safety margin
            assert(refineFactor>1);
            // ensure that we add at least one new sample (increase refineFactor if needed)
            unsigned int numAddSamplesThisCell = 
                std::max<unsigned int>(1, samplesInCell[indexCell].size() * (refineFactor-1));
            numAddSamples[indexCell] = numAddSamplesThisCell;
            numAddSamplesTotal += numAddSamplesThisCell;
            ++numCellsForRefinement;
        }

        // reserve space in the array of sample coords while preserving the existing values
        Matrix<double> newSampleCoords(numSamples + numAddSamplesTotal, Ndim);
        std::copy(sampleCoords.data(), sampleCoords.data()+sampleCoords.size(), newSampleCoords.data());
        sampleCoords = newSampleCoords;
        weightedFncValues.resize(numSamples + numAddSamplesTotal);

        // assign coordinates and weight factors for new samples
        unsigned int indexAddSamples = numSamples;
        for(CellMap::const_iterator iter = cellsForRefinement.begin();
            iter != cellsForRefinement.end(); ++iter)
        {
            CellEnum indexCell  = iter->first;
            unsigned int numAddSamplesThisCell = numAddSamples[indexCell];
            refineCellByAddingSamples(indexCell,
                indexAddSamples, numAddSamplesThisCell, samplesInCell[indexCell]);
            indexAddSamples += numAddSamplesThisCell;
        }
        assert(indexAddSamples = numSamples + numAddSamplesTotal);

        // then evaluate the function for all new samples
        utils::msg(utils::VL_DEBUG, "sampleNdim",
            "Iteration #"+utils::toString(nIter)+": refining "+utils::toString(numCellsForRefinement)+
            " cells because of "+utils::toString(numOverweightSamples)+" overweight samples"
            " by making further "+utils::toString(numAddSamplesTotal)+" function calls");
        evalFncLoop(numSamples, numAddSamplesTotal);

        // update the integral estimate
        computeIntegral();
    } while(++nIter<16);
    throw std::runtime_error(
        "Error in sampleNdim: refinement procedure did not converge in 16 iterations");
}

void Sampler::drawSamples(const unsigned int numOutputSamples, Matrix<double>& outputSamples) const
{
    outputSamples=math::Matrix<double>(numOutputSamples, Ndim);
    const unsigned int npoints = weightedFncValues.size();
    assert(sampleCoords.rows() == npoints);   // number of internal samples already taken
    double partialSum = 0;        // accumulates the sum of f(x_i) w(x_i) for i=0..{current value}
    const double outputWeight =   // difference in accumulated sum between two output samples
        integValue / (numOutputSamples+1e-6);
    // the tiny addition above ensures that the last output sample coincides with the last internal sample
    unsigned int outputIndex = 0;
    for(unsigned int i=0; i<npoints && outputIndex<numOutputSamples; i++) {
        assert(weightedFncValues[i] <= outputWeight);  // has been guaranteed by ensureEnoughSamples()
        partialSum += weightedFncValues[i];
        if(partialSum >= (outputIndex+1) * outputWeight) {
            for(unsigned int d=0; d<Ndim; d++)
                outputSamples(outputIndex, d) = sampleCoords(i, d);
            outputIndex++;
        }
    }
    assert(outputIndex == numOutputSamples);
}

#if 0
/// this is a potentially more efficient implementation, but currently is considered unstable
class NewSampler {
public:
    /** Construct an N-dimensional sampler object */
    NewSampler(const IFunctionNdim& fnc, const double xlower[], const double xupper[],
        const unsigned int numOutputSamples);

    /** Create the internal array of sampling points sufficiently large for the requested output size */
    void run();

    /** Draw a requested number of output samples from the array of internal samples */
    void drawSamples(Matrix<double>& samples) const;

    /** Return the integral of F over the entire volume, and its error estimate */
    void integral(double& value, double& error) const {
        value = integValue;
        error = integError;
    }

    /** Return the total number of function evaluations */
    unsigned int numCalls() const { return numCallsFnc; }

private:

    /// A single cell of the tree
    struct Cell {
        /// index of the parent cell (or -1 for the root cell)
        int parentIndex;
        /// the dimension along which the cell is split, or -1 if this is a leaf cell
        int splitDim;
        /// coordinate of the boundary between two child cells (if it is split, otherwise NAN)
        double coord;
        /// probability of choosing this cell among all leaf cells
        double prob;
        /// index of the first child cell (if it is split, otherwise -1);
        /// the second child cell has index childIndex+1
        int childIndex;
        /// index of the first sample point belonging to this cell,
        /// or -1 if none exist or the cell is split (is not a leaf one)
        int headPointIndex;

        Cell() :
            parentIndex(-1), splitDim(-1), coord(NAN), prob(1),
            childIndex(-1), headPointIndex(-1) {}
    };

    /// A single sampling point
    /// (contains everything except its coordinates, which are stored elsewhere)
    struct Point {
        /// value of the function F at this point
        double fncValue;
        /// weight of this sample
        double weight;

        Point() : fncValue(NAN), weight(0) {}
    };

    /// the N-dimensional function to work with                          [ f(x) ]
    const IFunctionNdim& fnc;

    /// a shorthand for the number of dimensions
    const unsigned int Ndim;

    /// coordinates of the bounding box of the root cell (the entire region)
    const std::vector<double> xlower, xupper;

    /// required number of output samples
    const unsigned int numOutputSamples;

    /// list of cells in the tree
    std::vector<Cell> cells;

    /// list of sampling points (except their coords which are stored in pointCoords)
    std::vector<Point> points;

    /// index of the successor point belonging to the same cell, or -1 if this is the last one
    /// in this cell (changes if the cell is split, or more points are added to this cell)
    std::vector<int> nextPoint;

    /// array of sampling points drawn from the distribution,
    /// each i-th row of the matrix contains N coordinates of the point  [ x_i[d] ]
    Matrix<double> pointCoords;

    /// count the number of function evaluations
    unsigned int numCallsFnc;

    /// estimate of the integral of f(x) over H                          [ EI ]
    double integValue;

    /// estimate of the error in the integral                            [ EE ]
    double integError;

    /// maximum allowed mass for the internal sampling points is given by
    /// the mass of each output sample: integValue / numOutputSamples
    //double outputSampleWeight;
    int numActivePoints;

    /// analyze the sampling points inside a leaf cell and perform one of the following actions:
    /// - if all sampling points have weights less than the maximum allowed value, do nothing;
    /// - otherwise, decide whether to split the cell, and if yes, along which dimension;
    ///   -- if splitting the cell, rearrange the sampling points by calling splitCell();
    ///   -- if not splitting, decrease the weights of existing points in the cell and
    ///      add more points (assign their coordinates, but don't evaluate the function yet).
    void processCell(const int cellIndex, double&);

    void addPoints(const int numAdd);

    /// split a cell and repartition the existing sampling points in this cell between
    /// its children, keeping track of all associated indices/pointers
    /// \param[in]  cellIndex  is the cell to split;
    /// \param[in]  dim  is the index of dimension along which to split;
    /// \param[in]  boundary  is the coordinate along the selected dimension
    /// that will be the new boundary between the two child cells.
    void splitCell(const int cellIndex, const unsigned int splitDim,
        const double splitCoord, const double splitProb);

    void assembleListOfPointsInCell(const int cellIndex,
        std::vector<int> &stack, double &maxFncValue) const;

    /// determine the cell boundaries by recursively traversing the tree upwards
    void getCellBoundaries(const int cellIndex, double xlower[], double xupper[]) const;

    /// evaluate the value of function f(x) for the points from the pointCoords array,
    /// parallelizing the loop and guarding against exceptions
    void evalFncLoop(unsigned int indexOfFirstPoint, unsigned int count);

    /// update the estimate of integral and its error, using all collected samples
    double computeResult();

};

static const double oversamplingFactor = 1.25;

static const unsigned int minNumPointsInCell = 16;

NewSampler::NewSampler(const IFunctionNdim& _fnc, const double _xlower[], const double _xupper[],
    unsigned int _numOutputSamples) :
    fnc(_fnc), Ndim(fnc.numVars()),
    xlower(_xlower, _xlower+Ndim), xupper(_xupper, _xupper+Ndim),
    numOutputSamples(_numOutputSamples),
    cells(1), numCallsFnc(0), numActivePoints(0)
{}

double NewSampler::computeResult()
{
    const unsigned int numPoints = points.size();
    Averager avg;
    double maxSampleWeight = 0;
    for(unsigned int i=0; i<numPoints; i++)
        if(points[i].weight != 0) {
            double sampleWeight = points[i].fncValue * points[i].weight;
            avg.add(sampleWeight);
            maxSampleWeight = fmax(maxSampleWeight, sampleWeight);
        }
    numActivePoints = avg.count();
    integValue = avg.mean();
    integError = sqrt(avg.disp() / numActivePoints);
    double refineFactor = maxSampleWeight * numOutputSamples / (numActivePoints * integValue);
    // maximum allowed value of f(x)*w(x), which is the weight of one output sample
    // (it is not constant because the estimate of integValue is adjusted after each iteration)
//    outputSampleWeight = integValue / numOutputSamples;
    std::cout <<  "Integral value= " << integValue << " +- " << integError <<
        " using " << numCallsFnc << " points (" << numActivePoints << " active)"
        " with " << cells.size() << " cells; refineFactor=" << refineFactor << std::endl;
    return refineFactor;
}

void NewSampler::evalFncLoop(unsigned int first, unsigned int count)
{
    if(count==0) return;
    // loop over assigned points and compute the values of function (in parallel)
    bool badValueOccured = false;
    std::string errorMsg;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic,256)
#endif
    for(int i=0; i<(int)count; i++) {
        double val;
        try{
            fnc.eval(&(pointCoords(i+first, 0)), &val);
        }
        // guard against possible exceptions, since they must not leave the OpenMP block
        catch(std::exception& e) {
            errorMsg = e.what();
            val = NAN;
        }
        if(val<0 || !isFinite(val))
            badValueOccured = true;
        points[i+first].fncValue = val;
    }
    numCallsFnc += count;
    if(badValueOccured)
        throw std::runtime_error("Error in sampleNdim: " + 
            (errorMsg.empty() ? "function value is negative or not finite" : errorMsg));
/*    for(unsigned int i=first; i<first+count; i++) {
        for(unsigned int d=0; d<Ndim; d++)
            strm << pointCoords(i, d) << ' ';
        strm << points[i].fncValue << ' ' << points[i].weight << '\n';
    }    */
}

void NewSampler::getCellBoundaries(const int cellIndex, double cellXlower[], double cellXupper[]) const
{
    // stage 0: store the bounding box of the root cell
    for(unsigned int d=0; d<Ndim; d++) {
        cellXlower[d] = xlower[d];
        cellXupper[d] = xupper[d];
    }
    if(cellIndex==0)
        return;
    // stage 1: walk up the tree to the root cell, storing all intermediate cell indices on the stack
    std::vector<int> stack;
    int index = cellIndex;
    do {
        stack.push_back(index);
        index = cells[index].parentIndex;
    } while(index>0);
    assert(index == 0);
    // stage 2: walk down the tree and shrink the bounding box on each subdivision
    while(!stack.empty()) {
        int next   = stack.back();
        stack.pop_back();
        bool lower = next == cells[index].childIndex;
        bool upper = next == cells[index].childIndex + 1;
        assert(lower || upper);
        if(lower)
            cellXupper[ cells[index].splitDim ] = cells[index].coord;
        else
            cellXlower[ cells[index].splitDim ] = cells[index].coord;
        index = next;
    }
}

void NewSampler::addPoints(const int numAdd)
{
    // allocate room for additional samples
    unsigned int numExisting = pointCoords.rows();
    assert(points.size() == numExisting);
    pointCoords.conservativeResize(numExisting + numAdd, Ndim);
    points.resize(numExisting + numAdd);

    // add more points sampled uniformly in the cell: assign their coordinates,
    // but do not evaluate the function yet - this will be performed at the end of the pass
    std::vector<double> cellXlower(Ndim), cellXupper(Ndim), coords(Ndim);
    for(unsigned int pointIndex = numExisting; pointIndex < numExisting+numAdd; pointIndex++)
    {
        for(unsigned int d=0; d<Ndim; d++) {
            cellXlower[d] = xlower[d];
            cellXupper[d] = xupper[d];
            coords[d] = random();
            //quasiRandomHalton(pointIndex+42, PRIMES[d]);
        }
        // locate the leaf cell to put the point into
        int cellIndex = 0, dim;
        while((dim = cells[cellIndex].splitDim) >= 0)
        {
            double probLower = cells[cells[cellIndex]  .childIndex].prob / cells[cellIndex].prob;
            // NB: probUpper = cells[cells[cellIndex+1].childIndex].prob / cells[cellIndex].prob,
            // and probLower + probUpper = 1 by construction
            if(coords[dim] < probLower) {  // add the point to the lower half
                cellXupper[dim] = cells[cellIndex].coord;
                coords[dim] = coords[dim] / probLower;
                cellIndex   = cells[cellIndex].childIndex;
            } else {  // add the point to the upper half
                cellXlower[dim] = cells[cellIndex].coord;
                coords[dim] = (coords[dim] - probLower) / (1 - probLower);
                cellIndex   = cells[cellIndex].childIndex + 1;
            }
        }
        // assign the actual point coords within this cell
        double cellVolume = 1;
        for(unsigned int d=0; d<Ndim; d++) {
            pointCoords(pointIndex, d) = coords[d] * (cellXupper[d]-cellXlower[d]) + cellXlower[d];
            cellVolume *= (cellXupper[d]-cellXlower[d]);
        }
        points[pointIndex].weight = cellVolume / cells[cellIndex].prob;
        // update the linked list
//        nextPoint[pointIndex] = cells[cellIndex].headPointIndex; // or -1 if the cell was empty
//        cells[cellIndex].headPointIndex = pointIndex;  // this is the new head of the linked list
    }
}

void NewSampler::splitCell(const int cellIndex, const unsigned int splitDim,
    const double coord, const double probLower)
{
    int childIndex = cells.size();  // the two new cells will be added at the end of the existing list
    cells.resize(childIndex + 2);
    cells[childIndex  ].parentIndex = cellIndex;
    cells[childIndex+1].parentIndex = cellIndex;
    cells[childIndex  ].prob        = cells[cellIndex].prob * probLower;
    cells[childIndex+1].prob        = cells[cellIndex].prob * (1-probLower);
    cells[cellIndex   ].splitDim    = splitDim;
    cells[cellIndex   ].coord       = coord;
    cells[cellIndex   ].childIndex  = childIndex;
    int pointIndex = cells[cellIndex].headPointIndex;
    assert(pointIndex>=0);  // it must have some points, otherwise why split?
    cells[cellIndex].headPointIndex = -1;  // all its points are moved to child cells
    // traverse the list of points belonged to the cellIndex, and distribute them between child cells
    do {
        // memorize the successor point index
        // of the original list of points of the cellIndex that is being split
        int next = nextPoint[pointIndex];
        // determine which of the two child cells this points belongs to
        int child = pointCoords(pointIndex, splitDim) < coord ? childIndex : childIndex+1;
        if(cells[child].headPointIndex < 0) {
            // this is the first point in the child cell,
            // thus it will be marked as the end of the list (no successors)
            nextPoint[pointIndex] = -1;
        } else {
            // this child cell already contained points:
            // set the index of successor point to be equal to the previous head of the list
            // (i.e. we reverse the original order of points)
            nextPoint[pointIndex] = cells[child].headPointIndex;
        }
        // assign the new head of the list of points belonging to this child cell
        cells[child].headPointIndex = pointIndex;
        // move to the successor point of the original list
        pointIndex = next;
    } while(pointIndex>=0);
}

void NewSampler::processCell(const int cellIndex, double& newRefFactor)
{
    // assemble the list of points belonging to this cell
    double maxFncValueCell = 0;
    std::vector<int> stack;
    int pointIndex = cells.at(cellIndex).headPointIndex;
    assert(pointIndex >= 0);
    while(pointIndex >= 0) {
        maxFncValueCell = fmax(maxFncValueCell, points[pointIndex].fncValue);
        stack.push_back(pointIndex);
        pointIndex = nextPoint[pointIndex];
    }    
    const unsigned int numPointsInCell = stack.size();
    assert(numPointsInCell >= minNumPointsInCell);
    
    // obtain the cell boundaries and volume
    std::vector<double> cellXlower(Ndim), cellXupper(Ndim);
    getCellBoundaries(cellIndex, &cellXlower[0], &cellXupper[0]);
    double cellVolume = 1;
    for(unsigned int d=0; d<Ndim; d++)
        cellVolume *= cellXupper[d] - cellXlower[d];
    
    double refineFactor = maxFncValueCell * cellVolume / cells[cellIndex].prob *
        numOutputSamples / (integValue * numActivePoints);
//    std::cout << cellIndex << ' ' << numPointsInCell << ' ' << refineFactor;

    // consider splitting only if the cell contains enough points
    if(numPointsInCell >= 2*minNumPointsInCell) {

    // sorted list of particle coordinates and fnc values, constructed for each dimension in turn
    std::vector<std::pair<double,double> > projection(numPointsInCell);
    // maximum of function values among all points lying to the right of the current one
    std::vector<double> maxFncValueUpper(numPointsInCell);
    // the index of dimension best suitable for splitting
    int splitDim = -1;
    // the coordinate along the chosen dimension where the cell should be split
    double splitCoord = NAN, splitProb = NAN;

    for(unsigned int d=0; d<Ndim; d++) {
        const double mult = cellVolume / (cellXupper[d] - cellXlower[d]) / cells[cellIndex].prob *
            numOutputSamples / (integValue * numActivePoints);
        
        // create a projection onto d-th coordinate axis
        for(unsigned int p=0; p<numPointsInCell; p++) {
            projection[p].first  = pointCoords(stack[p], d);
            projection[p].second = points[stack[p]].fncValue;
        }

        // sort points by 1st value in pair (i.e., the d-th coordinate)
        std::sort(projection.begin(), projection.end());

        // for each point in the sorted list, record the maximum fnc value among all points
        // which lie to the right of the current one (including the current point itself),
        // i.e. have indices greater than the current one.
        maxFncValueUpper[numPointsInCell-1] = projection[numPointsInCell-1].second;
        for(unsigned int p=numPointsInCell-1; p>0; p--)
            maxFncValueUpper[p-1] = fmax(maxFncValueUpper[p], projection[p-1].second);

        // now scan the sorted list in the opposite direction, and for each point
        // record the maximum fnc value among all points lying strictly to the left of it.
        double maxFncValueLower = 0;
        for(unsigned int p = 0; p < numPointsInCell-minNumPointsInCell; p++) {
            double coord   = projection[p].first;
            maxFncValueLower = fmax(maxFncValueLower, projection[p].second);
            if( p < minNumPointsInCell /*||
                coord - cellXlower[d] < (cellXupper[d] - cellXlower[d])*0.125 ||
                cellXupper[d] - coord < (cellXupper[d] - cellXlower[d])*0.125*/ )
                continue;
            double refLower  = mult * (coord - cellXlower[d]) * maxFncValueLower;
            double refUpper  = mult * (cellXupper[d] - coord) * maxFncValueUpper[p];
            double refFact   = refLower + refUpper;
            // update the choice of best splitting point
            if( refFact < refineFactor ) {
                refineFactor = refFact;
                splitDim     = d;
                splitCoord   = coord;
                splitProb    = refLower / refFact;
            }
        }
    }
    if(splitDim >= 0 && refineFactor >= 0.5) {
//        std::cout << " => " << refineFactor << " (d="<<splitDim<<") c>"<<cells.size()<<"\n";
        splitCell(cellIndex, splitDim, splitCoord, splitProb);
        // the two new child cells will be appended at the end of cells array,
        // and will be traversed again later in this pass
        return;
    }
    }
    
    newRefFactor = fmax(refineFactor, newRefFactor);
/*    if(numPointsInCell >= 2*minNumPointsInCell)
        std::cout << " => " << refineFactor << " finish"<<std::endl;
    else
        std::cout << " finish"<<std::endl;

    Averager avg;
    for(unsigned int p=0; p<numPointsInCell; p++)
        avg.add(points[stack[p]].fncValue);
    strm << cellIndex;
    for(unsigned int d=0; d<Ndim; d++)
        strm << '\t' << cellXlower[d] << ' ' << cellXupper[d];
    strm << '\t' << cellVolume << ' ' << cells[cellIndex].prob << ' ' << refineFactor << '\t' <<
        numPointsInCell << ' ' << avg.mean() << ' ' << sqrt(avg.disp()) << '\n';
*/
}

void NewSampler::run()
{
    const unsigned int numPointsInit = std::max<unsigned int>(minNumPointsInCell, numOutputSamples);

    // collect the initial batch of points in the root cell
    addPoints(numPointsInit);
//    strm.open("iter0.points");
    evalFncLoop(0, numPointsInit);
//    strm.close();
    double refineFactor = computeResult();
    if(refineFactor <= 1)
        return;

    int nIter=0;  // safeguard against infinite loop
    do{
        // number of internal samples already taken
        const unsigned int numPoints = points.size();
        assert(pointCoords.rows() == numPoints);
        
        // rebuild the tree: first start with one cell that contains all points in sequence
        cells.assign(1, Cell());
        cells[0].headPointIndex = 0;
        nextPoint.resize(numPoints);
        for(unsigned int p=0; p<numPoints; p++)
            nextPoint[p] = p+1;
        nextPoint.back() = -1;

        double newRefFactor = 0;
//        strm.open((std::string("iter")+char(nIter+48)+".cells").c_str());
        for(unsigned int cellIndex=0; cellIndex<cells.size(); cellIndex++) {
            processCell(cellIndex, newRefFactor);
        }
//        strm.close();
        
        int numAdd = numActivePoints * newRefFactor * oversamplingFactor;
        std::cout << "New ref factor="<<newRefFactor<<"\n";
        if(newRefFactor+1 < refineFactor) {  // discard old points
            for(unsigned int p=0; p<numPoints; p++)
                points[p].weight = 0;
        }
        
        if(numAdd>0) {
            std::cout << "Adding " << numAdd << " points in " << cells.size() << " cells" << std::flush;
            addPoints(numAdd);
            std::cout << ", collecting samples" << std::endl;
//            strm.open((std::string("iter")+char(nIter+49)+".points").c_str());
            evalFncLoop(numPoints, numAdd);
//            strm.close();
            refineFactor = computeResult();
            if(refineFactor <= 1)
                return;
        } else
            return;  // done!
    } while(++nIter < 10);
    throw std::runtime_error(
        "Error in sampleNdim: refinement procedure did not converge in 10 iterations");
}

void NewSampler::drawSamples(Matrix<double>& outputSamples) const
{
    // number of internal samples already taken
    const unsigned int numPoints = points.size();
    assert(pointCoords.rows() == numPoints);
    // accumulates the sum of f(x_i) w(x_i) for i=0..{current value}
    double partialSum = 0;        
    unsigned int outputIndex = 0;
    outputSamples.resize(numOutputSamples, Ndim);
    const double outputSampleWeight = integValue / numOutputSamples;
    for(unsigned int i=0; i<numPoints && outputIndex < numOutputSamples; i++) {
        double sampleWeight = points[i].fncValue * points[i].weight / numActivePoints;
        assert(sampleWeight <= outputSampleWeight);  // has been guaranteed by run()
        partialSum += sampleWeight;
        if(partialSum >= (outputIndex+0.5) * outputSampleWeight) {
            for(unsigned int d=0; d<Ndim; d++)
                outputSamples(outputIndex, d) = pointCoords(i, d);
            outputIndex++;
        }
    }
    if(outputIndex != numOutputSamples)
        outputSamples.conservativeResize(outputIndex, Ndim);
}
#endif
}  // unnamed namespace

void sampleNdim(const IFunctionNdim& fnc, const double xlower[], const double xupper[], 
    const unsigned int numSamples,
    Matrix<double>& samples, int* numTrialPoints, double* integral, double* interror)
{
    if(fnc.numValues() != 1)
        throw std::invalid_argument("sampleNdim: function must provide one value");
#if 1
    Sampler sampler(fnc, xlower, xupper);

    // first warmup run (actually, two) to collect statistics and adjust bins
    const unsigned int numWarmupSamples = std::max<unsigned int>(numSamples*0.2, 10000);
    sampler.runPass(numWarmupSamples);   // first pass without any pre-existing bins;
    sampler.readjustBins();              // they are initialized after collecting some samples.
    sampler.runPass(numWarmupSamples*4); // second pass with already assigned binning scheme;
    sampler.readjustBins();              // reinitialize bins with better statistics.

    // second run to collect samples distributed more uniformly inside the bins
    const unsigned int numCollectSamples = std::max<unsigned int>(numSamples*(fnc.numVars()+1), 10000);
    sampler.runPass(numCollectSamples);

    // make sure that no sampling point has too large weight, if no then seed additional samples
    sampler.ensureEnoughSamples(numSamples);

    // finally, draw the required number of output samples from the internal ones
    sampler.drawSamples(numSamples, samples);
#else
    NewSampler sampler(fnc, xlower, xupper, numSamples);
    sampler.run();
    sampler.drawSamples(samples);
#endif
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
