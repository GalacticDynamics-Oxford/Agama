#include "math_sample.h"
#include "math_core.h"  // for Averager
#include "math_random.h"
#include "utils.h"
#include <stdexcept>
#include <cassert>
#include <cmath>
#include <map>
#include <algorithm>
#ifndef _MSC_VER
#include <alloca.h>
#else
#include <malloc.h>
#endif

/// choose between pseudo-random (PRNG) and quasi-random (QRNG) number generators:
/// the latter is generally preferred, but can be disabled if necessary (e.g. in Makefile.local)
//#define DISABLE_QRNG

namespace math{

namespace {  // internal namespace for Sampler class

/// maximum number of points initially sampled from the root cell
static const int maxInitSamples = 1048576;

/// minimum allowed number of points in any cell
static const int minNumPointsInCell = 256;

/// number of bins in the 1-d histogram of function values (projection in each dimension) in each cell,
/// used to estimate the entropy and ultimately decide the dimension to split a cell;
/// should be ~ sqrt(minNumPointsInCell)
static const int numBinsEntropy = 16;

/// limit the number of iterations in the recursive refinement loop
static const int maxNumIter = 50;

class Sampler {
public:
    /** Construct an N-dimensional sampler object */
    Sampler(const IFunctionNdim& fnc, const double xlower[], const double xupper[],
        const size_t numOutputSamples);

    /** Create the internal array of sampling points sufficiently large for the requested output size */
    void run();

    /** Draw a requested number of output samples from the array of internal samples */
    void drawSamples(Matrix<double>& samples) const;

    /** Return the integral of F over the entire volume, and its error estimate */
    void integral(double& value, double& error) const {
        value = volume * integValue;
        error = volume * integError;
    }

    /** Return the total number of function evaluations */
    size_t numCalls() const { return fncValues.size(); }

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
        int splitDim;

        /// weight of any sample point in the cell
        double weight;

        Cell() :
            headPointIndex(-1), parentIndex(-1), childIndex(-1), splitDim(-1), weight(0) {}
    };

    /// the N-dimensional function to work with
    const IFunctionNdim& fnc;

    /// a shorthand for the number of dimensions
    const int Ndim;

    /// coordinates of the bounding box of the root cell (the entire region)
    const std::vector<double> xlower, xupper;

    /// volume of the entire region (root cell)
    double volume;

    /// required number of output samples
    const size_t numOutputSamples;

    /// list of cells in the tree  (size: Ncells)
    std::vector<Cell> cells;

    /// index of the successor point belonging to the same cell, or -1 if this is the last one
    /// in this cell (changes if the cell is split, or more points are added to this cell)
    /// (size: Npoints)
    std::vector<PointEnum> nextPoint;

    /// flattened 2d array of sampling points drawn from the distribution,
    /// each i-th row of the matrix contains N coordinates of the point
    /// (size: Npoints x Ndim)
    std::vector<double> pointCoords;

    /// array of function values at sampling points  (size: Npoints)
    std::vector<double> fncValues;

#ifndef DISABLE_QRNG
    /// random offsets in each coordinate, added (mod 1) to the values returned by the quasi-random
    /// number generator, to avoid repetition when the same sampling routine is called twice
    std::vector<double> offsets;
#endif

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

    /** split a cell and repartition the existing sampling points in this cell between
        its children, keeping track of all associated indices/pointers.
        \param[in]  cellIndex  is the cell to split;
        \param[in]  splitDim   is the index of dimension along which to split;
        \param[in]  boundary   is the absolute coordinate along the selected dimension
        that will be the new boundary between the two child cells, used to split the list
        of points between the child cells.
    */
    void splitCell(CellEnum cellIndex, int splitDim, double boundary);

    /** choose the best dimension and boundary to split a given cell and perform the division;
        the dimension along which the function varies most significantly will be split.
    */
    void decideHowToSplitCell(CellEnum cellIndex);

    /** append the given cell to the queue of cells that will be populated with new points;
        the number of points in this cell will be doubled.
    */
    void addCellToQueue(CellEnum cellIndex);

    /** determine the cell boundaries by recursively traversing the tree upwards.
        \param[in]  cellIndex is the index of the cell;
        \param[out] xlower, xupper must point to existing arrays of length Ndim,
        which will be filled with the coordinates of this cell's boundaries.
    */
    void getCellBoundaries(CellEnum cellIndex, double xlower[], double xupper[]) const;

#if 0
    /** determine the index of the leaf cell that contains the given point */
    CellEnum getCellIndex(PointEnum pointIndex) const;
#endif

    /** assign coordinates to the new points inside the given cell;
        \param[in]  cellIndex  is the cell that the new points will belong to;
        \param[in]  firstPointIndex  is the index of the first new point in the pointCoords array;
        the array must have been expanded before calling this routine;
        \param[in]  lastPointIndex  is the index of the last new point that will be assigned
        to this cell; the list of points belonging to this cell will be expanded accordingly
        (i.e. the elements of the `nextPoint' array will be assigned; the array itself
        must have been expanded before calling this routine).
        May be called in parallel for many cells.
    */
    void addPointsToCell(CellEnum cellIndex, PointEnum firstPointIndex, PointEnum lastPointIndex);

    /** analyze the given cell and perform some action (only for leaf cells that have no child cells
        but contain some points):
        compute the maximum weight of all points belonging to this cell, and if it exceeds
        the weight of one output sample, then either split this cell in a suitable place,
        while repartitioning its existing points among the two child cells
        (if it contained enough points already to make a motivated choice),
        or add this cell to the queue of cells that will be populated with new points.
    */
    void processCell(CellEnum cellIndex);

    /** evaluate the value of function f(x) for the points whose coordinates are stored 
        in the pointCoords array, parallelizing the loop and guarding against exceptions.
        The function values will be stored in corresponding elements of the fncValues array.
    */
    void evalFncLoop(PointEnum firstPointIndex, PointEnum lastPointIndex);

    /** update the estimate of integral and its error, using all collected samples;
        return refineFactor (the ratio between maximum sample weight and the output sample weight)
    */
    double computeResult();
};

Sampler::Sampler(const IFunctionNdim& _fnc, const double _xlower[], const double _xupper[],
    size_t _numOutputSamples) :
    fnc(_fnc),
    Ndim(fnc.numVars()),
    xlower(_xlower, _xlower+Ndim),
    xupper(_xupper, _xupper+Ndim),
    numOutputSamples(_numOutputSamples),
    cells(1),  // create the root cell
#ifndef DISABLE_QRNG
    offsets(Ndim),
#endif
    integValue(0),
    integError(0)
{
#ifndef DISABLE_QRNG
    if(Ndim > MAX_PRIMES)  // this is only a limitation of the quasi-random number generator
        throw std::runtime_error("sampleNdim: more than "+utils::toString(MAX_PRIMES)+
            " dimensions is not supported");
    for(int d=0; d<Ndim; d++)
        offsets[d] = random();
#endif
    volume = 1;
    for(int d=0; d<Ndim; d++) {
        if(xupper[d] > xlower[d])
            volume *= xupper[d] - xlower[d];
        else
            throw std::runtime_error(
            "sampleNdim: upper boundary of sampling region must be larger than the lower boundary");
    }
}

double Sampler::computeResult()
{
    Averager avg;
    // declare the accumulator variable as volatile to PREVENT auto-vectorization:
    // the summation needs to be done exactly in the same order here and in drawSamples()
    volatile double integ  = 0;
    double maxSampleWeight = 0;
    for(CellEnum cellIndex = 0; cellIndex < static_cast<CellEnum>(cells.size()); cellIndex++) {
        for(PointEnum pointIndex = cells[cellIndex].headPointIndex;
            pointIndex >= 0;
            pointIndex = nextPoint[pointIndex])
        {
            double sampleWeight = fncValues[pointIndex] * cells[cellIndex].weight;
            avg.add(sampleWeight);
            integ += sampleWeight;
            maxSampleWeight = std::max(maxSampleWeight, sampleWeight);
        }
    }
    const size_t numPoints = fncValues.size();
    assert(numPoints == avg.count());
    integValue = integ;  // equal to avg.mean() * numPoints to within roundoff errors
    integError = sqrt(avg.disp() * numPoints);
    // maximum allowed value of f(x)*w(x) is the weight of one output sample
    // (integValue/numOutputSamples); if it is larger, we need to do another iteration
    double refineFactor = maxSampleWeight * numOutputSamples / integValue;
    FILTERMSG(utils::VL_VERBOSE, "sampleNdim",
        "Integral value= " + utils::toString(volume * integValue) +
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
    // loop over assigned points and compute the values of function (in parallel)
    bool badValueOccured = false;
    std::string errorMsg;
    // compute the function values for a block of points at once;
    // operations on different blocks may be OpenMP-parallelized
    const unsigned int blocksize = 1024;
    PointEnum nblocks = (lastPointIndex - firstPointIndex - 1) / blocksize + 1;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for(PointEnum b=0; b<nblocks; b++) {
        if(badValueOccured)
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

#if 0
Sampler::CellEnum Sampler::getCellIndex(PointEnum pointIndex) const
{
    double *relcoord = static_cast<double*>(alloca(Ndim * sizeof(double)));
    for(int d=0; d<Ndim; d++) {
        relcoord[d] = (pointCoords[pointIndex * Ndim + d] - xlower[d]) / (xupper[d] - xlower[d]);
    }
    CellEnum cellIndex = 0;
    while(cells[cellIndex].childIndex > 0) {
        int d = cells[cellIndex].splitDim;
        if(relcoord[d] < 0.5) {
            cellIndex = cells[cellIndex].childIndex;
            relcoord[d] *= 2;
        } else {
            cellIndex = cells[cellIndex].childIndex + 1;
            relcoord[d] = relcoord[d] * 2 - 1;
        }
    }
    return cellIndex;
}
#endif

void Sampler::addCellToQueue(CellEnum cellIndex)
{
    // get the number of samples in the cell (same number of new samples will be added)
    double numPoints = 1. / cells[cellIndex].weight;
    for(CellEnum index = cellIndex; index>0; index = cells[index].parentIndex)
        numPoints *= 0.5;   // each division halves the cell volume and hence the number of points

    // halve the weight of each sample point in this cell
    cells[cellIndex].weight *= 0.5;

    // schedule this cell for adding more points in the next iteration;
    // the coordinates of these new points will be assigned later, once this queue is completed.
    PointEnum numPrev = cellsQueue.empty() ? 0 : cellsQueue.back().second;
    cellsQueue.push_back(std::pair<CellEnum, PointEnum>(cellIndex,
        static_cast<PointEnum>(numPoints) + numPrev));
}

void Sampler::addPointsToCell(CellEnum cellIndex, PointEnum firstPointIndex, PointEnum lastPointIndex)
{
    // obtain the cell boundaries into a temporary stack-allocated array
    double *cellXlower = static_cast<double*>(alloca(2*Ndim * sizeof(double)));
    double *cellXupper = cellXlower+Ndim;
    getCellBoundaries(cellIndex, cellXlower, cellXupper);
    math::PRNGState state = lastPointIndex;     // initial seed for the PRNG
#ifndef DISABLE_QRNG
    // Assign point coordinates using quasi-random numbers, but randomize the sequence of these numbers.
    // These quasi-random numbers, or low-discrepancy sequences, have a property that each element of
    // the sequence is reasonably distant from any other element (very close pairs are less likely than
    // for purely (pseudo-)random numbers). But the order of these QR numbers in their sequence is not so
    // random, which may be problematic because of the way we select the output points in drawSamples().
    // Namely, after the sampling procedure has produced enough number of samples, we construct
    // the cumulative probability distribution, summing the probabilities of all points with indices
    // smaller than the given one. Then we slice through this cumulative distribution at equal intervals,
    // and pick up samples which are closest to the given slice, putting them into the output array.
    // The problem may happen when the coordinates (and hence the value of the probability function that
    // we are working with) are correlated with their indices in a regular way. This would be the case
    // for the original quasi-random sequence, and hence we shuffle the order of QR numbers in the list
    // of sampled points randomly.
    // An additional technical detail is that we use either stack- or heap- allocated temporary array.
    PointEnum count = lastPointIndex-firstPointIndex;
    std::vector<size_t> permBuffer;  // temporary storage, might not be needed (see below)
    size_t *perm;   // will point to the actual array of permutation indices
    if(count > 2*minNumPointsInCell) {
        // the length of the permutation list is too large, allocate it on the heap (freed automatically)
        // (this usually happens only for the initial coordinate assignment within the root cell)
        permBuffer.resize(count);
        perm = &permBuffer[0];
    } else
        // the permutation list is small enough to be allocated on the stack (no need to free explicitly)
        // (should be the case for all subsequent calls to this routine)
        perm = static_cast<size_t*>(alloca(count * sizeof(size_t)));
    getRandomPermutation(count, perm, &state);  // assign the actual permutation list, no matter where it's stored
#endif
    PointEnum nextPointInList = cells[cellIndex].headPointIndex;  // or -1 if the cell was empty
    for(PointEnum pointIndex = firstPointIndex; pointIndex < lastPointIndex; pointIndex++) {
        // assign coordinates of the new point
        for(int d=0; d<Ndim; d++) {
            pointCoords[ pointIndex * Ndim + d ] = cellXlower[d] +
                (cellXupper[d] - cellXlower[d]) *
#ifndef DISABLE_QRNG
                fmod(quasiRandomHalton(firstPointIndex + perm[pointIndex-firstPointIndex], PRIMES[d]) +
                     offsets[d], 1);
#else
                random(&state);
#endif
        }
        // update the linked list of points in the cell
        nextPoint[pointIndex] = nextPointInList;
        nextPointInList = pointIndex;  // this is the new head of the linked list
    }
    cells[cellIndex].headPointIndex = nextPointInList;  // store the new head of the list for this cell
}

void Sampler::splitCell(CellEnum cellIndex, int splitDim, const double boundary)
{
    CellEnum childIndex = cells.size();  // the two new cells will be added at the end of the existing list
    cells.resize(childIndex + 2);
    cells[childIndex  ].parentIndex = cellIndex;
    cells[childIndex+1].parentIndex = cellIndex;
    cells[childIndex  ].weight      = cells[cellIndex].weight;
    cells[childIndex+1].weight      = cells[cellIndex].weight;
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
        CellEnum child = pointCoords[pointIndex * Ndim + splitDim] < boundary ? childIndex : childIndex+1;
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

void Sampler::decideHowToSplitCell(CellEnum cellIndex)
{
    // allocate temporary array on stack, to store the cell boundaries
    // and the histogram of the projection of the function in each dimension
    double *cellXlower = static_cast<double*>(alloca( (2+numBinsEntropy) * Ndim * sizeof(double)));
    double *cellXupper = cellXlower + Ndim;    // space within the array allocated above
    double *histogram  = cellXlower + 2*Ndim;
    std::fill(histogram, histogram + Ndim*numBinsEntropy, 0.);
    getCellBoundaries(cellIndex, cellXlower, cellXupper);

    PointEnum pointIndex   = cells[cellIndex].headPointIndex;
    assert(pointIndex >= 0);
    // loop over the list of points belonging to this cell
    while(pointIndex >= 0) {
        double fval = fncValues[pointIndex];
        for(int dim = 0; dim < Ndim; dim++) {
            double relCoord = (pointCoords[pointIndex * Ndim + dim] - cellXlower[dim]) /
                (cellXupper[dim]-cellXlower[dim]);
            int bin = static_cast<int>(relCoord * numBinsEntropy);
            histogram[dim * numBinsEntropy + bin] += fval;
        }
        pointIndex = nextPoint[pointIndex];
    }

    // Compute a crude estimate of the (un-normalized) entropy in each dimension d,
    // summing -F_{i,d} log(F_{i,d}), where F_{i,d} is the sum of function values in i-th bin;
    // the dimension in which the entropy is minimal (the function varies most significantly) will be split
    int splitDim = -1;
    double minEntropy = INFINITY;
    for(int dim = 0; dim < Ndim; dim++) {
        double entropy = 0.;
        for(int bin = 0; bin < numBinsEntropy; bin++)
            if(histogram[dim * numBinsEntropy + bin] > 0)
                entropy -= histogram[dim * numBinsEntropy + bin]
                    *  log(histogram[dim * numBinsEntropy + bin]);
        if(entropy < minEntropy) {
            minEntropy = entropy;
            splitDim = dim;
        }
    }
    assert(splitDim>=0);

    // split in the given direction into two equal halves
    double coord = 0.5 * (cellXlower[splitDim] + cellXupper[splitDim]);
    splitCell(cellIndex, splitDim, coord);
}

void Sampler::processCell(CellEnum cellIndex)
{
    // keep track of the top two function values in the cell;
    // the difference between the largest and the second-largest is used
    // to estimate the possible largest value when even more samples are taken
    double largestFncValueCell = 0, secondLargestFncValueCell = 0;
    size_t numPointsInCell = 0;
    for(PointEnum pointIndex = cells[cellIndex].headPointIndex;
        pointIndex >= 0;
        pointIndex = nextPoint[pointIndex])
    {
        numPointsInCell++;
        double fncValue = fncValues[pointIndex];
        if(fncValue > largestFncValueCell) {
            secondLargestFncValueCell = largestFncValueCell;
            largestFncValueCell = fncValue;
        } else if(fncValue > secondLargestFncValueCell)
            secondLargestFncValueCell = fncValue;
    }
    if(numPointsInCell==0)
        return;  // this is a non-leaf cell

    double refineFactor = cells[cellIndex].weight * numOutputSamples / integValue *
        (largestFncValueCell * 2 - secondLargestFncValueCell);

    if(refineFactor < 1)
        return;

    if(numPointsInCell > 2*minNumPointsInCell)
        decideHowToSplitCell(cellIndex);
    else
        addCellToQueue(cellIndex);
}

void Sampler::run()
{
    utils::CtrlBreakHandler cbrk;  // catch Ctrl-Break keypress

    // first iteration: sample uniformly in a single root cell
    PointEnum numInitSamples = std::min<PointEnum>(maxInitSamples,
        (1 + numOutputSamples / minNumPointsInCell) * minNumPointsInCell);
    cells[0].weight = 1. / numInitSamples;
    pointCoords.resize(numInitSamples * Ndim);
    fncValues.resize(numInitSamples);
    nextPoint.resize(numInitSamples);
    addPointsToCell(0, 0, numInitSamples);
    evalFncLoop(0, numInitSamples);
    double refineFactor = computeResult();
    if(refineFactor <= 1)
        return;

    int nIter = 1;
    do{
        if(cbrk.triggered())
            throw std::runtime_error(cbrk.message());
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
        // assign the coordinates of new samples (each cell is processed independently, may parallelize)
        CellEnum numNewCells = cellsQueue.size();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for(CellEnum queueIndex=0; queueIndex < numNewCells; queueIndex++)
            addPointsToCell( cellsQueue[queueIndex].first,
                numPointsExisting + (queueIndex==0 ? 0 : cellsQueue[queueIndex-1].second),
                numPointsExisting + cellsQueue[queueIndex].second);
        FILTERMSG(utils::VL_VERBOSE, "sampleNdim", "Iteration #" + utils::toString(nIter) +
            ": #cells=" + utils::toString(cells.size()) +
            ", #refined cells=" + utils::toString(cellsQueue.size()) +
            ", #new points=" + utils::toString(cellsQueue.back().second));
        // now compute the function values for the newly added samples (also in parallel)
        evalFncLoop(numPointsExisting, numPointsOverall);
        // estimate the integral value and check if we have enough samples
        double refineFactor = computeResult();
        if(refineFactor <= 1)
            return;
    } while(++nIter < maxNumIter);
    throw std::runtime_error(
        "Error in sampleNdim: refinement procedure did not converge");
}

void Sampler::drawSamples(Matrix<double>& outputSamples) const
{
    // number of internal samples already taken
    const size_t numPoints = fncValues.size();
    assert(pointCoords.size() == numPoints * Ndim);
    const double outputSampleWeight = integValue / numOutputSamples;
    volatile double partialSum = 0;  // accumulates the sum of f(x_i) w(x_i) for i=0..{current value}
    size_t outputIndex = 0;
    outputSamples = math::Matrix<double>(numOutputSamples, Ndim);
    // construct a random permutation of internal samples to erase the original order
    std::vector<size_t> permutation(numOutputSamples);
    getRandomPermutation(numOutputSamples, &permutation.front());
    // loop over internal samples and pick up some of them for output
    for(CellEnum cellIndex = 0; cellIndex < static_cast<CellEnum>(cells.size()); cellIndex++) {
        PointEnum pointIndex = cells[cellIndex].headPointIndex;
        while(pointIndex >= 0) {
            double sampleWeight = fncValues[pointIndex] * cells[cellIndex].weight;
            assert(sampleWeight <= outputSampleWeight);  // has been guaranteed by run()
            partialSum += sampleWeight;
            if(partialSum >= (outputIndex+0.5) * outputSampleWeight) {
                for(int d=0; d<Ndim; d++)
                    outputSamples(permutation[outputIndex], d) = pointCoords[pointIndex * Ndim + d];
                outputIndex++;
            }
            pointIndex = nextPoint[pointIndex];
        }
    }
    assert(outputIndex == numOutputSamples);
}

}  // unnamed namespace

void sampleNdim(const IFunctionNdim& fnc, const double xlower[], const double xupper[], 
    const size_t numSamples,
    Matrix<double>& samples, size_t* numTrialPoints, double* integral, double* interror)
{
    if(fnc.numValues() != 1)
        throw std::invalid_argument("sampleNdim: function must provide one value");
    Sampler sampler(fnc, xlower, xupper, numSamples);
    sampler.run();
    sampler.drawSamples(samples);
    // statistics
    if(numTrialPoints!=NULL)
        *numTrialPoints = sampler.numCalls();
    if(integral!=NULL) {
        double err;
        sampler.integral(*integral, err);
        if(interror!=NULL)
            *interror = err;
    }
}

}  // namespace
