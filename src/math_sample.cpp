#include "math_sample.h"
#include "math_random.h"
#include "utils.h"
#include <stdexcept>
#include <cassert>
#include <cmath>
#include <map>
#include <algorithm>
#include <fstream>

#ifndef _MSC_VER
#include <alloca.h>
#else
#include <malloc.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace math{

namespace {  // internal namespace for Sampler class

// somewhat speed up the sampling procedure at the expense of extra 8 bytes per point
#define CACHE_CELL_INDEX

class Sampler {

public:
    /** Construct an N-dimensional sampler object */
    Sampler(const IFunctionNdim& fnc, const double xlower[], const double xupper[],
        const size_t numOutputSamples, const SampleMethod method,
        Matrix<double>& outputSamples, std::vector<double>& outputWeights,
        double* error, size_t* numEval, PRNGState* state);

    /** Perform actual work and output results in the requested form */
    void run();

private:

    /// signed integral type large enough to enumerate all cells in the tree
    typedef ptrdiff_t CellEnum;

    /// same for all points in the array of samples
    typedef ptrdiff_t PointEnum;

    /// A single cell of the tree
    struct Cell {
        /// index of the first sample point belonging to this cell,
        /// or -1 if none exist or the cell is split (is not a leaf one)
        PointEnum headPointIndex;

        /// index of the parent cell (or -1 for the root cell)
        CellEnum parentIndex;

        /// index of the first child cell (if it is split, otherwise -1);
        /// the second child cell has index childIndex+1
        CellEnum childIndex;

        /// the dimension along which the cell is split, or -1 if this is a leaf cell
        int16_t splitDim;

        /// number of times the root cell needs to be split to reach this cell
        int16_t splitLevel;

        /// number of times the cell has gone through refinement (doubling the number of samples)
        int16_t refineLevel;

        Cell() :
            headPointIndex(-1), parentIndex(-1), childIndex(-1),
            splitDim(-1), splitLevel(0), refineLevel(0) {}
    };

    /// the N-dimensional function to work with
    const IFunctionNdim& fnc;

    /// a shorthand for the number of dimensions
    const int Ndim;

    /// minimum allowed number of points in any cell (depends on Ndim)
    const PointEnum minNumPointsInCell;

    /// coordinates of the bounding box of the root cell (the entire region)
    const std::vector<double> xlower, xupper;

    /// volume of the entire region (root cell)
    double volume;

    /// required number of output samples
    const size_t numOutputSamples;

    /// mode of operation
    const SampleMethod method;

    /// output variable which will be filled with the coordinates of sampled points
    Matrix<double>& outputSamples;

    /// output variable which will be filled with the corresponding weights of points
    std::vector<double>& outputWeights;

    /// output variable which will store the error estimate of the integral (if not NULL)
    double* error;

    /// output variable which will store the number of all internally collected samples (if not NULL)
    size_t* numEval;

    /// random seed/state, updated in the process of sampling
    PRNGState* randomState;

    /// actual storage for the random state if not provided externally
    PRNGState randomStateInternal;

    /// number of iterative refinements carried out so far
    int numIter;

    /// upper limit for the number of iterations in the refinement loop
    static const int maxNumIter = 50;

    /// initial number of samples: should be a power of two if using adaptive refinement
    PointEnum numInitSamples;

    /// list of cells in the tree  (size: Ncells)
    std::vector<Cell> cells;

    /// index of the successor point belonging to the same cell, or -1 if this is the last one
    /// in this cell (changes if the cell is split, or more points are added to this cell)
    /// (size: Npoints)
    std::vector<PointEnum> nextPoint;

#ifdef CACHE_CELL_INDEX
    /// keep track of the cell index for each point
    /// (even though it can be reconstructed from coordinates at extra cost)
    std::vector<CellEnum> cellIndexOfPoint;
#endif

    /// flattened 2d array of sampling points drawn from the distribution,
    /// each i-th row of the matrix contains N coordinates of the point
    /// (size: Npoints x Ndim)
    std::vector<double> pointCoords;

    /// array of function values at sampling points  (size: Npoints)
    std::vector<double> fncValues;

    /// quasi-random number generators for each coordinate, initialized with a random scrambling
    /// to avoid repetition when the same sampling routine is called twice
    std::vector<QuasiRandomSobol> qrng;

    /// estimate of the integral of f(x) over H, divided by the volume
    double integValue;

    /// estimate of the error in the integral, divided by the volume
    double integError;

    /** list of cells that need to be populated with more points on this iteration:
        the first element of the pair is the index of the cell,
        the second element is the cumulative number of points to be added to all cells
        up to and including this one (so that the last element of this array contains
        the total number of points to be added, and for each cell we know the indices
        of new points in the global array that belong to this cell).
        This list is re-build at each iteration while scanning all cells of the tree.
    */
    std::vector<std::pair<CellEnum, PointEnum> > cellsQueue;

    /// catch Ctrl-Break keypress and terminate the process immediately
    utils::CtrlBreakHandler cbrk;


    /** update the estimate of integral and its error, using all collected samples;
        \return refineFactor (the ratio between maximum sample weight and the output sample weight)
    */
    double computeResult();

    /** evaluate the value of function f(x) for the points whose coordinates are stored 
        in the pointCoords array, parallelizing the loop and guarding against exceptions.
        The function values will be stored in corresponding elements of the fncValues array.
    */
    void evalFncLoop(PointEnum firstPointIndex, PointEnum lastPointIndex);

    /** determine the cell boundaries by recursively traversing the tree upwards.
        \param[in]  cellIndex is the index of the cell;
        \param[out] xlower, xupper must point to existing arrays of length Ndim,
        which will be filled with the coordinates of this cell's boundaries.
    */
    void getCellBoundaries(CellEnum cellIndex, double xlower[], double xupper[]) const;

    /** determine the index of the leaf cell that contains the given point */
    CellEnum getCellIndex(PointEnum pointIndex) const;

    /** append the given cell to the queue of cells that will be populated with new points;
        the number of points in this cell will be doubled.
        This routine is called only when the number of points already sampled from this cell
        is equal to minNumPointsInCell (approximately; the actual number may be off by a few).
    */
    void addCellToQueue(CellEnum cellIndex);

    /** assign coordinates to the new points inside the given cell;
        \param[in]  cellIndex  is the cell that the new points will belong to;
        \param[in]  firstPointIndex  is the index of the first new point in the pointCoords array;
        the array must have been expanded before calling this routine;
        \param[in]  lastPointIndex  is the index of the last new point that will be assigned
        to this cell; the list of points belonging to this cell will be expanded accordingly
        (i.e. the elements of the `nextPoint' array will be assigned; the array itself
        must have been expanded before calling this routine).
        \note May be called in parallel for many cells.
    */
    void addPointsToCell(CellEnum cellIndex, PointEnum firstPointIndex, PointEnum lastPointIndex);

    /** choose the best dimension and boundary to split a given cell, 
        based on the variation of the function between two halves of the cell in each dimension.
        \return a pair consisting of the suggested split dimension and the value of
        the corresponding coordinate that separates the left and right halves of the cell.
    */
    std::pair<int16_t, double> decideHowToSplitCell(CellEnum cellIndex) const;

    /** split a cell and repartition the existing sampling points in this cell between
        its children, keeping track of all associated indices/pointers.
        \param[in]  cellIndex  is the cell to split;
        \param[in]  splitDim   is the index of dimension along which to split;
        \param[in]  boundary   is the absolute coordinate along the selected dimension
        that will be the new boundary between the two child cells, used to split the list
        of points between the child cells.
    */
    void splitCell(CellEnum cellIndex, int16_t splitDim, double boundary);

    /** analyze the given cell and perform some action (only for leaf cells that have no child cells
        but contain some points):
        compute the maximum weight of all points belonging to this cell, and if it exceeds
        the weight of one output sample, then either split this cell in a suitable place,
        while repartitioning its existing points among the two child cells
        (if it contained enough points already to make a motivated choice),
        or add this cell to the queue of cells that will be populated with new points.
    */
    void processCell(CellEnum cellIndex);

    /** Draw a requested number of equal-weighted output samples from the array of internal samples */
    void getEqualWeightedSamples();

    /** Return all collected internal samples and their associated weights */
    void getAllSamples() const;

    /** Kahan-Babuska-Neumaier compensated summation */
    inline void addCompensated(const double& value, double& accum, double& extra)
    {
        double tmp = value + accum;
        // in the more general version of the method, the condition should be fabs(value) > fabs(accum)
        // but in our case we are dealing with nonnegative numbers, so abs can be skipped
        extra += value > accum ? (value - tmp) + accum : (accum - tmp) + value;
        accum = tmp;
    }
};

Sampler::Sampler(const IFunctionNdim& _fnc, const double _xlower[], const double _xupper[],
    const size_t _numOutputSamples, const SampleMethod _method,
    Matrix<double>& _outputSamples, std::vector<double>& _outputWeights,
    double* _error, size_t* _numEval, PRNGState* state)
:
    fnc(_fnc),
    Ndim(fnc.numVars()),
    minNumPointsInCell(32UL << Ndim),  // calibrated by numerous experiments up to Ndim=6
    xlower(_xlower, _xlower+Ndim),
    xupper(_xupper, _xupper+Ndim),
    numOutputSamples(_numOutputSamples),
    method(_method),
    outputSamples(_outputSamples),
    outputWeights(_outputWeights),
    error(_error),
    numEval(_numEval),
    randomState(state ? state : &randomStateInternal),
    randomStateInternal(state ? 0 : random() * (1UL << 31)),  // create if not provided externally
    numIter(0),
    cells(1),  // create the root cell
    integValue(0),
    integError(0)
{
   if(Ndim == 0)
        throw std::invalid_argument("sampleNdim: number of dimensions must be positive");
    if(Ndim >= QuasiRandomSobol::MAX_DIM)
        throw std::invalid_argument("sampleNdim: number of dimensions must be less than " +
            utils::toString(QuasiRandomSobol::MAX_DIM));
    if(fnc.numValues() != 1)
        throw std::invalid_argument("sampleNdim: function must provide one value");
    if(numOutputSamples == 0)
        throw std::invalid_argument("sampleNdim: number of requested samples should be positive");
    if( method != SM_RETURN_EQUAL_WEIGHT_SAMPLES &&
        method != SM_RETURN_ALL_SAMPLES &&
        method != SM_RETURN_ALL_SAMPLES_DISABLE_REFINEMENT &&
        method != SM_USE_PRNG )
        throw std::invalid_argument("sampleNdim: invalid method");

    volume = 1;
    for(int d=0; d<Ndim; d++) {
        if(xupper[d] > xlower[d])
            volume *= xupper[d] - xlower[d];
        else
            throw std::invalid_argument(
            "sampleNdim: upper boundary of sampling region must be larger than the lower boundary");
    }

    if((method & SM_USE_PRNG) == 0) {
        // ensure that the QRNG can produce at least 16x more points than requested
        int numBits = std::min<int>(std::max<int>(ceil(log2(numOutputSamples)) + 4, 32), 53);
        for(int d=0; d<Ndim; d++)
            qrng.push_back(QuasiRandomSobol(d, randomState, numBits));
#ifdef _OPENMP
        // clone the above created objects to have separate instances in each thread
        for(int i=1, numThreads = omp_get_max_threads(); i<numThreads; i++)
            for(int d=0; d<Ndim; d++)
                qrng.push_back(qrng[d]);
#endif
    }
}

double Sampler::computeResult()
{
    integValue = 0;
    double extraBits = 0;
    double maxSampleWeight = 0;
    double sumSquareWeights = 0;
    size_t numPoints = 0;
    for(CellEnum cellIndex = 0; cellIndex < static_cast<CellEnum>(cells.size()); cellIndex++) {
        const double cellWeight = 1.0 / (numInitSamples << cells[cellIndex].refineLevel);
        for(PointEnum pointIndex = cells[cellIndex].headPointIndex;
            pointIndex >= 0;
            pointIndex = nextPoint[pointIndex])
        {
            double sampleWeight = fncValues[pointIndex] * cellWeight;
            addCompensated(sampleWeight, integValue, extraBits);
            sumSquareWeights += pow_2(sampleWeight);
            maxSampleWeight = std::max(maxSampleWeight, sampleWeight);
            numPoints++;
        }
    }
    assert(numPoints == fncValues.size());
    integValue += extraBits;
    integError = sqrt(fmax(0, sumSquareWeights - pow_2(integValue) / numPoints));
    // maximum allowed value of f(x) * cellWeight is the weight of one output sample
    // (integValue/numOutputSamples); if it is larger, we need to do another iteration
    double refineFactor = maxSampleWeight * numOutputSamples / integValue;
    FILTERMSG(utils::VL_VERBOSE, "sampleNdim",
        "Iteration #" + utils::toString(numIter) +  ": "
        "integral value= " + utils::toString(volume * integValue) +
        " +- " + utils::toString(volume * integError) +
        " using " + utils::toString(numPoints) + " points"
        " with " + utils::toString(cells.size()) + " cells;"
        " refineFactor=" + utils::toString(refineFactor));
    if(integValue <= 0)
        throw std::runtime_error("sampleNdim: function is identically zero inside the region");
    return refineFactor;
}

void Sampler::evalFncLoop(PointEnum firstPointIndex, PointEnum lastPointIndex)
{
    if(firstPointIndex>=lastPointIndex) return;
    bool badValueOccured = false;
    std::string errorMsg;
    // loop over assigned points and compute the values of function (in parallel);
    // compute the function values for a block of points at once;
    // operations on different blocks may be OpenMP-parallelized
    const unsigned int blocksize = 1024;
    PointEnum nblocks = (lastPointIndex - firstPointIndex - 1) / blocksize + 1;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for(PointEnum b=0; b<nblocks; b++) {
        if(badValueOccured || cbrk.triggered())
            continue;
        PointEnum pointIndex = firstPointIndex + b * blocksize;
        PointEnum npoints = std::min<PointEnum>(blocksize, lastPointIndex - pointIndex);
        try {
            fnc.evalmany(npoints, &pointCoords[pointIndex * Ndim], &fncValues[pointIndex]);
        }
        // guard against possible exceptions, since they must not leave the OpenMP parallel section
        catch(std::exception& e) {
            errorMsg = e.what();
            badValueOccured = true;
        }
        for(PointEnum i=0; i<npoints; i++) {
            double val = fncValues[pointIndex + i];
            if(val<0 || !isFinite(val))
                badValueOccured = true;
        }
    }
    if(badValueOccured)
        throw std::runtime_error("Error in sampleNdim: " + 
            (errorMsg.empty() ? "function value is negative or not finite" : errorMsg));
}

void Sampler::getCellBoundaries(CellEnum cellIndex, double cellXlower[], double cellXupper[]) const
{
    for(int d=0; d<Ndim; d++) {
        cellXlower[d] = 0;
        cellXupper[d] = 1;
    }
    // scan all parent cells and shrink the boundaries along the relevant dimension on each step,
    // until we reach the root cell
    CellEnum index = cellIndex;
    while(index>0) {
        CellEnum parentIndex = cells[index].parentIndex;
        int splitDim         = cells[parentIndex].splitDim;
        bool lower =  index == cells[parentIndex].childIndex;
        bool upper =  index == cells[parentIndex].childIndex + 1;
        assert(lower || upper);
        if(lower) {
            cellXlower[splitDim] *= 0.5;
            cellXupper[splitDim] *= 0.5;
        } else {
            cellXlower[splitDim] = 0.5 * (1 + cellXlower[splitDim]);
            cellXupper[splitDim] = 0.5 * (1 + cellXupper[splitDim]);
        }
        index = parentIndex;
    }
    // now this array contains relative coordinates of the given cell within the topmost cell;
    // convert it to absolute coordinates
    for(int d=0; d<Ndim; d++) {
        cellXlower[d] = xlower[d] + (xupper[d]-xlower[d]) * cellXlower[d];
        cellXupper[d] = xlower[d] + (xupper[d]-xlower[d]) * cellXupper[d];
    }
}

Sampler::CellEnum Sampler::getCellIndex(PointEnum pointIndex) const
{
#ifdef CACHE_CELL_INDEX
    return cellIndexOfPoint[pointIndex];
#else
    double *relCoord = static_cast<double*>(alloca(Ndim * sizeof(double)));  // normalized to [0..1]
    for(int d=0; d<Ndim; d++) {
        relCoord[d] = (pointCoords[pointIndex * Ndim + d] - xlower[d]) / (xupper[d] - xlower[d]);
    }
    CellEnum cellIndex = 0;
    while(cells[cellIndex].childIndex > 0) {
        int d = cells[cellIndex].splitDim;
        if(relCoord[d] < 0.5) {  // left half
            cellIndex = cells[cellIndex].childIndex;
            relCoord[d] *= 2;
        } else {  // right half
            cellIndex = cells[cellIndex].childIndex + 1;
            relCoord[d] = relCoord[d] * 2 - 1;
        }
    }
    return cellIndex;
#endif
}

void Sampler::addCellToQueue(CellEnum cellIndex)
{
    // each round of refinement doubles the number of points in cell
    cells[cellIndex].refineLevel++;

    // schedule this cell for adding more points in the next iteration;
    // the coordinates of these new points will be assigned later, once this queue is completed.
    PointEnum numPrev = cellsQueue.empty() ? 0 : cellsQueue.back().second;
    cellsQueue.push_back(std::pair<CellEnum, PointEnum>(cellIndex, minNumPointsInCell + numPrev));
}

void Sampler::addPointsToCell(CellEnum cellIndex, PointEnum firstPointIndex, PointEnum lastPointIndex)
{
    // obtain the cell boundaries into a temporary stack-allocated array
    double *cellXlower = static_cast<double*>(alloca(2*Ndim * sizeof(double)));
    double *cellXupper = cellXlower+Ndim;
    getCellBoundaries(cellIndex, cellXlower, cellXupper);
    PointEnum nextPointInList = cells[cellIndex].headPointIndex;  // or -1 if the cell was empty
    if(method & SM_USE_PRNG) {
        // seed for the PRNG sequence in this cell; constructed independently of the thread index
        PRNGState cellState = hash(&lastPointIndex, 1, *randomState);
        for(PointEnum pointIndex = firstPointIndex; pointIndex < lastPointIndex; pointIndex++) {
            // assign coordinates of the new point
            for(int d=0; d<Ndim; d++)
                pointCoords[pointIndex * Ndim + d] = cellXlower[d] +
                    (cellXupper[d] - cellXlower[d]) * random(&cellState);
#ifdef CACHE_CELL_INDEX
            cellIndexOfPoint[pointIndex] = cellIndex;
#endif
            // update the linked list of points in the cell
            nextPoint[pointIndex] = nextPointInList;
            nextPointInList = pointIndex;  // this is the new head of the linked list
        }
    } else {  // use QRNG instead of PRNG
#ifdef _OPENMP
        // when using the Sobol QRNG with multiple threads, we use thread-local clones of generators
        // to enable faster production of consecutive numbers that relies on caching of the internal state.
        int qrngIndex = omp_get_thread_num() * Ndim;  // start of the block of generators for this thread
        assert(qrngIndex + Ndim <= (int)qrng.size());
#else
        int qrngIndex = 0;
#endif
        for(PointEnum pointIndex = firstPointIndex; pointIndex < lastPointIndex; pointIndex++) {
            // assign coordinates of the new point
            for(int d=0; d<Ndim; d++)
                pointCoords[pointIndex * Ndim + d] = cellXlower[d] +
                    (cellXupper[d] - cellXlower[d]) * qrng[d + qrngIndex](pointIndex);
#ifdef CACHE_CELL_INDEX
            cellIndexOfPoint[pointIndex] = cellIndex;
#endif
            // update the linked list of points in the cell
            nextPoint[pointIndex] = nextPointInList;
            nextPointInList = pointIndex;  // this is the new head of the linked list
        }
    }
    cells[cellIndex].headPointIndex = nextPointInList;  // store the new head of the list for this cell
}

std::pair<int16_t, double> Sampler::decideHowToSplitCell(CellEnum cellIndex) const
{
    // allocate temporary array on stack, to store the cell boundaries
    // and the 1d histograms of the function in each dimension
    const int NUM_BINS_ENTROPY = 16;
    double *cellXlower = static_cast<double*>(alloca( (2+NUM_BINS_ENTROPY) * Ndim * sizeof(double)));
    double *cellXupper = cellXlower + Ndim;    // space within the array allocated above
    double *histogram  = cellXlower + 2*Ndim;
    std::fill(histogram, histogram + Ndim*NUM_BINS_ENTROPY, 0.);
    getCellBoundaries(cellIndex, cellXlower, cellXupper);
    double sumCell = 0;
    PointEnum pointIndex = cells[cellIndex].headPointIndex;
    assert(pointIndex >= 0);
    // loop over the list of points belonging to this cell
    while(pointIndex >= 0) {
        double fval = fncValues[pointIndex];
        for(int dim = 0; dim < Ndim; dim++) {
            double relCoord = (pointCoords[pointIndex * Ndim + dim] - cellXlower[dim]) /
                (cellXupper[dim] - cellXlower[dim]);
            int bin = static_cast<int>(relCoord * NUM_BINS_ENTROPY);
            histogram[dim * NUM_BINS_ENTROPY + bin] += fval;
        }
        sumCell += fval;
        pointIndex = nextPoint[pointIndex];
    }
    // We aim to split the dimension in which the function varies most significantly,
    // but it appears beneficial to allow for some wiggle room in deciding:
    // if two candidate dimensions for splitting have similar amount of variation,
    // choose between them randomly rather than always picking up the best one.
    // Thus we first estimate the entropy for each dimension d, summing -F_{i,d} log(F_{i,d}), where
    // F_{i,d} is the sum of function values in i-th bin normalized to the total sum in the cell...
    double minEntropy = INFINITY;
    double* entropy = histogram;  // reuse the same temporarily allocated space for the new quantity
    for(int dim = 0; dim < Ndim; dim++) {
        double entropyDim = 0.;
        for(int bin = 0; bin < NUM_BINS_ENTROPY; bin++)
            if(histogram[dim * NUM_BINS_ENTROPY + bin] > 0)
                entropyDim -= histogram[dim * NUM_BINS_ENTROPY + bin] / sumCell
                        * log(histogram[dim * NUM_BINS_ENTROPY + bin] / sumCell);
        entropy[dim] = entropyDim;
        minEntropy = std::min(minEntropy, entropyDim);
    }
    // ...then select the dimensions with entropy close to the minimum (within some tolerance)...
    int *splitDims = static_cast<int*>(alloca(Ndim * sizeof(int)));
    int numSplits = 0;
    const double  THRESHOLD_DIFFERENCE_FROM_MIN_ENTROPY = 0.05;
    minEntropy += THRESHOLD_DIFFERENCE_FROM_MIN_ENTROPY;
    for(int dim=0; dim<Ndim; dim++) {
        if(entropy[dim] <= minEntropy)
            splitDims[numSplits++] = dim;
    }
    // ...and finally select among all suitable dimensions,
    // use the cell index scrambled with a hash function to randomize the choice
    assert(numSplits>0);
    int splitDim = splitDims[hash(NULL, 0, cellIndex) % numSplits];
    // split in the given direction into two equal halves
    double cellCenter = 0.5 * (cellXlower[splitDim] + cellXupper[splitDim]);
    return std::pair<int16_t, double>(splitDim, cellCenter);
}

void Sampler::splitCell(CellEnum cellIndex, int16_t splitDim, const double boundary)
{
    // the two new cells will be added at the end of the existing list
    CellEnum childIndex = cells.size();
    cells.resize(childIndex + 2);
    cells[childIndex  ].parentIndex = cellIndex;
    cells[childIndex+1].parentIndex = cellIndex;
    cells[childIndex  ].splitLevel  = cells[cellIndex].splitLevel + 1;
    cells[childIndex+1].splitLevel  = cells[cellIndex].splitLevel + 1;
    cells[childIndex  ].refineLevel = cells[cellIndex].refineLevel;  // not changed upon splitting,
    cells[childIndex+1].refineLevel = cells[cellIndex].refineLevel;  // only upon refinement
    cells[cellIndex   ].splitDim    = splitDim;
    cells[cellIndex   ].childIndex  = childIndex;
    PointEnum pointIndex = cells[cellIndex].headPointIndex;
    assert(pointIndex >= 0);  // it must have some points, otherwise why split?
    cells[cellIndex].headPointIndex = -1;  // all its points are moved to child cells
    // traverse the list of points belonged to the cellIndex, and distribute them between child cells
    do {
        // memorize the successor point index
        // of the original list of points of the cellIndex that is being split
        PointEnum next = nextPoint[pointIndex];
        // determine which of the two child cells this points belongs to
        CellEnum child = childIndex + (pointCoords[pointIndex * Ndim + splitDim] < boundary ? 0 : 1);
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
#ifdef CACHE_CELL_INDEX
        cellIndexOfPoint[pointIndex] = child;
#endif
        // assign the new head of the list of points belonging to this child cell
        cells[child].headPointIndex = pointIndex;
        // move to the successor point of the original list
        pointIndex = next;
    } while(pointIndex>=0);
}

void Sampler::processCell(CellEnum cellIndex)
{
    if(cells[cellIndex].headPointIndex < 0)
        return;  // this is a non-leaf cell

    double largestFncValueCell = 0;
    for(PointEnum pointIndex = cells[cellIndex].headPointIndex;
        pointIndex >= 0;
        pointIndex = nextPoint[pointIndex])
    {
        largestFncValueCell = std::max(fncValues[pointIndex], largestFncValueCell);
    }

    const double cellWeight = 1.0 / (numInitSamples << cells[cellIndex].refineLevel);
    double refineFactor = largestFncValueCell * cellWeight * numOutputSamples / (integValue - integError);

    if(refineFactor < 1)
        return;  // nothing to do for this cell, at least in this iteration

    // each splitting halves (on average) the number of points, each round of refinement doubles it
    int16_t numHalvings = cells[cellIndex].splitLevel - cells[cellIndex].refineLevel;
    assert(numHalvings >= 0);
    // number of points already sampled in this cell (on average; the actual number may be off by a few)
    PointEnum numPointsInCell = numInitSamples >> numHalvings;
    if(numPointsInCell >= 2*minNumPointsInCell) {
        std::pair<int16_t, double> split = decideHowToSplitCell(cellIndex);  // splitDim, coord
        splitCell(cellIndex, split.first, split.second);
    } else {
        assert(numPointsInCell == minNumPointsInCell);
        addCellToQueue(cellIndex);
    }
}

void Sampler::getEqualWeightedSamples()
{
    const PointEnum numPoints = fncValues.size();  // total number of internal samples
    assert(pointCoords.size() == static_cast<size_t>(numPoints) * Ndim);
    outputSamples = Matrix<double>(numOutputSamples, Ndim);
    double* outputData = outputSamples.data();
    outputWeights.assign(numOutputSamples, integValue * volume / numOutputSamples);
    // construct a random permutation of internal samples to erase the original order
    std::vector<size_t> permutation(numPoints);
    getRandomPermutation(numPoints, &permutation.front(), randomState);
    // loop over internal samples in the permuted order and pick up some of them for output
    double accum = 0, extra = 0;  // accumulates the sum of f(x_i) w(x_i) for i=0..{current value}
    size_t outputIndex = 0;
    for(PointEnum permutedIndex=0; permutedIndex < numPoints; permutedIndex++) {
        PointEnum pointIndex = permutation[permutedIndex];
        CellEnum cellIndex = getCellIndex(pointIndex);
        double sampleWeight = fncValues[pointIndex] /
            (numInitSamples << cells[cellIndex].refineLevel);
        assert(sampleWeight * numOutputSamples <= integValue);  // has been guaranteed by run()
        addCompensated(sampleWeight, accum, extra);
        if((accum + extra) * numOutputSamples >= integValue * (outputIndex + 0.5)) {
            std::copy(pointCoords.begin() +  pointIndex    * Ndim,
                      pointCoords.begin() + (pointIndex+1) * Ndim,
                      outputData + outputIndex * Ndim);
            outputIndex++;
        }
    }
    // even though we summed the weights in a different order than in computeResult(),
    // the use of compensated summation should ensure that the result is the same to machine precision
    assert(outputIndex == numOutputSamples && accum + extra == integValue); 
}

void Sampler::getAllSamples() const
{
    const PointEnum numPoints = fncValues.size();  // total number of internal samples
    assert(pointCoords.size() == static_cast<size_t>(numPoints) * Ndim);
    outputSamples = Matrix<double>(numPoints, Ndim);
    std::copy(pointCoords.begin(), pointCoords.end(), outputSamples.data());
    outputWeights.resize(numPoints);
    for(PointEnum pointIndex=0; pointIndex < numPoints; pointIndex++) {
        outputWeights[pointIndex] = fncValues[pointIndex] * volume /
            (numInitSamples << cells[getCellIndex(pointIndex)].refineLevel);
    }
}

void Sampler::run()
{
    // first iteration: sample uniformly in a single root cell;
    // the number of initial samples is a power of two not exceeding the number of output samples,
    // but at least twice the minimum number of points in cell
    numInitSamples = (method & SM_DISABLE_REFINEMENT) ? numOutputSamples :
        minNumPointsInCell << std::max<int>(1, log2(numOutputSamples / minNumPointsInCell));
    pointCoords.resize(numInitSamples * Ndim);
    fncValues.resize(numInitSamples);
    nextPoint.resize(numInitSamples);
#ifdef CACHE_CELL_INDEX
    cellIndexOfPoint.resize(numInitSamples);
#endif
    addPointsToCell(0, 0, numInitSamples);
    evalFncLoop(0, numInitSamples);
    double refineFactor = computeResult();

    while(!cbrk.triggered() && 
        (method & SM_DISABLE_REFINEMENT) == 0 &&
        refineFactor > 1 &&
        numIter++ < maxNumIter)
    {
        // Loop over all cells and check if there are enough sample points in the cell;
        // if not, either split the cell in two halves or schedule this cell for adding more points later.
        // The newly added cells (after splitting) are appended to the end of the cell list,
        // therefore the list is gradually growing, and this loop must be done sequentially.
        cellsQueue.clear();
        for(CellEnum cellIndex=0; cellIndex < static_cast<CellEnum>(cells.size()); cellIndex++)
            processCell(cellIndex);
        assert(!cellsQueue.empty());
        // find out how many new samples do we need to add, and extend the relevant arrays
        PointEnum numPointsExisting = fncValues.size();
        PointEnum numPointsOverall  = numPointsExisting + cellsQueue.back().second;
        pointCoords.resize(numPointsOverall * Ndim);
        fncValues.resize(numPointsOverall);
        nextPoint.resize(numPointsOverall);
#ifdef CACHE_CELL_INDEX
        cellIndexOfPoint.resize(numPointsOverall);
#endif
        // assign the coordinates of new samples (each cell is processed independently, may parallelize)
        CellEnum numNewCells = cellsQueue.size();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for(CellEnum queueIndex=0; queueIndex < numNewCells; queueIndex++)
            addPointsToCell( cellsQueue[queueIndex].first,
                numPointsExisting + (queueIndex==0 ? 0 : cellsQueue[queueIndex-1].second),
                numPointsExisting + cellsQueue[queueIndex].second);
        FILTERMSG(utils::VL_VERBOSE, "sampleNdim", "Iteration #" + utils::toString(numIter) +
            ": #cells=" + utils::toString(cells.size()) +
            ", #refined cells=" + utils::toString(cellsQueue.size()) +
            ", #new points=" + utils::toString(cellsQueue.back().second));
        // now compute the function values for the newly added samples (also in parallel)
        evalFncLoop(numPointsExisting, numPointsOverall);
        // estimate the integral value and check if we have enough samples
        refineFactor = computeResult();
    }

    if(cbrk.triggered())
        throw std::runtime_error(cbrk.message());
    if((method & SM_DISABLE_REFINEMENT) == 0 && refineFactor > 1)
        throw std::runtime_error("Error in sampleNdim: refinement procedure did not converge");
    // statistics
    if(error)
        *error = integError * volume;
    if(numEval)
        *numEval = fncValues.size();
    // output in the requested form
    if(method & SM_RETURN_ALL_SAMPLES)
        getAllSamples();
    else
        getEqualWeightedSamples();
}

}  // unnamed namespace

void sampleNdim(
    const IFunctionNdim& F, const double xlower[], const double xupper[], const size_t numSamples,
    const SampleMethod method, Matrix<double>& samples, std::vector<double>& weights,
    double* error, size_t* numEval, PRNGState* state)
{
    Sampler(F, xlower, xupper, numSamples, method, samples, weights, error, numEval, state).run();
}

}  // namespace
