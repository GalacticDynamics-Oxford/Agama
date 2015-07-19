#include "math_grid.h"
#include "mathutils.h"
#include <cmath>
#include <cassert>
#include <stdexcept>

namespace mathutils {

/* ----------- generation of unequally spaced grids --------------- */

// Creation of grid with exponentially increasing cells
class GridSpacingFinder: public IFunction {
public:
    GridSpacingFinder(double _dynrange, int _nnodes) : dynrange(_dynrange), nnodes(_nnodes) {};
    virtual void eval_deriv(const double A, double* result=0, double* =0, double* =0) const {
        assert(result!=NULL);
        *result = (A==0) ? nnodes-dynrange :
            (exp(A*nnodes)-1)/(exp(A)-1) - dynrange;
    }
    virtual int numDerivs() const { return 0; }
private:
    double dynrange;
    int nnodes;
};

void createNonuniformGrid(size_t nnodes, double xmin, double xmax, bool zeroelem, std::vector<double>& grid)
{   // create grid so that x_k = B*(exp(A*k)-1)
    double A, B, dynrange=xmax/xmin;
    grid.resize(nnodes);
    int indexstart=zeroelem?1:0;
    if(zeroelem) {
        grid[0] = 0;
        nnodes--;
    }
    if(fcmp(static_cast<double>(nnodes), dynrange, 1e-6)==0) { // no need for non-uniform grid
        for(size_t i=0; i<nnodes; i++)
            grid[i+indexstart] = xmin+(xmax-xmin)*i/(nnodes-1);
        return;
    }
    // solve for A:  dynrange = (exp(A*nnodes)-1)/(exp(A)-1)
    GridSpacingFinder F(dynrange, nnodes);
    // first localize the root coarsely, to avoid overflows in root solver
    double Amin=0, Amax=0;
    double step=1;
    while(step>10./nnodes)
        step/=2;
    if(dynrange>nnodes) {
        while(Amax<10 && F.value(Amax)<=0)
            Amax+=step;
        Amin = Amax-step;
    } else {
        while(Amin>-10 && F.value(Amin)>=0)
            Amin-=step;
        Amax = Amin+step;
    }
    A = findRoot(F, Amin, Amax, 1e-4);
    B = xmin / (exp(A)-1);
    for(size_t i=0; i<nnodes; i++)
        grid[i+indexstart] = B*(exp(A*(i+1))-1);
    grid[nnodes-1+indexstart] = xmax;
}

/// creation of a grid with minimum guaranteed number of input points per bin
static void makegrid(std::vector<double>::iterator begin, std::vector<double>::iterator end, double startval, double endval)
{
    double step=(endval-startval)/(end-begin-1);
    while(begin!=end){
        *begin=startval;
        startval+=step;
        begin++;
    }
    *(end-1)=endval;  // exact value
}

void createAlmostUniformGrid(const std::vector<double> &srcpoints, size_t minbin, size_t& gridsize, std::vector<double>& grid)
{
    if(srcpoints.size()==0)
        throw std::invalid_argument("Error in creating a grid: input points array is empty");
    gridsize = std::max<size_t>(2, std::min<size_t>(gridsize, static_cast<size_t>(srcpoints.size()/minbin)));
    grid.resize(gridsize);
    std::vector<double>::iterator gridbegin=grid.begin(), gridend=grid.end();
    std::vector<double>::const_iterator srcbegin=srcpoints.begin(), srcend=srcpoints.end();
    std::vector<double>::const_iterator srciter;
    std::vector<double>::iterator griditer;
    bool ok=true, directionBackward=false;
    int numChangesDirection=0;
    do{
        makegrid(gridbegin, gridend, *srcbegin, *(srcend-1));
        ok=true; 
        // find the index of bin with the largest number of points
        int largestbin=-1;
        size_t maxptperbin=0;
        for(srciter=srcbegin, griditer=gridbegin; griditer!=gridend-1; griditer++) {
            size_t ptperbin=0;
            while(srciter+ptperbin!=srcend && *(srciter+ptperbin) < *(griditer+1)) 
                ptperbin++;
            if(ptperbin>maxptperbin) {
                maxptperbin=ptperbin;
                largestbin=griditer-grid.begin();
            }
            srciter+=ptperbin;
        }
        // check that all bins contain at least minbin srcpoints
        if(!directionBackward) {  // forward scan
            srciter = srcbegin;
            griditer = gridbegin;
            while(ok && griditer!=gridend-1) {
                size_t ptperbin=0;
                while(srciter+ptperbin!=srcend && *(srciter+ptperbin) < *(griditer+1)) 
                    ptperbin++;
                if(ptperbin>=minbin)  // ok, move to the next one
                {
                    griditer++;
                    srciter+=ptperbin;
                } else {  // assign minbin points and decrease the available grid interval from the front
                    if(griditer-grid.begin() < largestbin) { 
                        // bad bin is closer to the grid front; move gridbegin forward
                        while(ptperbin<minbin && srciter+ptperbin!=srcend) 
                            ptperbin++;
                        if(srciter+ptperbin==srcend)
                            directionBackward=true; // oops, hit the end of array..
                        else {
                            srcbegin=srciter+ptperbin;
                            gridbegin=griditer+1;
                        }
                    } else {
                        directionBackward=true;
                    }   // will restart scanning from the end of the grid
                    ok=false;
                }
            }
        } else {  // backward scan
            srciter = srcend-1;
            griditer = gridend-1;
            while(ok && griditer!=gridbegin) {
                size_t ptperbin=0;
                while(srciter+1-ptperbin!=srcbegin && *(srciter-ptperbin) >= *(griditer-1))
                    ptperbin++;
                if(ptperbin>=minbin)  // ok, move to the previous one
                {
                    griditer--;
                    if(srciter+1-ptperbin==srcbegin)
                        srciter=srcbegin;
                    else
                        srciter-=ptperbin;
                } else {  // assign minbin points and decrease the available grid interval from the back
                    if(griditer-grid.begin() <= largestbin) { 
                        // bad bin is closer to the grid front; reset direction to forward
                        directionBackward=false;
                        numChangesDirection++;
                        if(numChangesDirection>10) {
//                            my_message(FUNCNAME, "grid creation seems not to converge?");
                            return;  // don't run forever but would not fulfill the minbin condition
                        }
                    } else {
                        // move gridend backward
                        while(ptperbin<minbin && srciter-ptperbin!=srcbegin) 
                            ptperbin++;
                        if(srciter-ptperbin==srcbegin) {
                            directionBackward=false;
                            numChangesDirection++;
                            if(numChangesDirection>10) {
//                                my_message(FUNCNAME, "grid creation seems not to converge?");
                                return;  // don't run forever but would not fulfill the minbin condition
                            }
                        } else {
                            srcend=srciter-ptperbin+1;
                            gridend=griditer;
                        }
                    }
                    ok=false;
                }
            }
        }
    } while(!ok);
}

}  // namespace
