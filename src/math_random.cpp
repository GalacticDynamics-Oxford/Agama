#include "math_random.h"
#include "math_core.h"  // for sincos
#include <vector>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

// a couple of bit-twiddling functions
#if _cplusplus >= 201907L
#include <bit>
#endif

// count the number of bits that are set to 1
static inline int popcount(uint64_t x) {
#ifdef __GNUC__
    return __builtin_popcountll(x);
#elif _cplusplus >= 201907L
    return std::popcount(x);
#else   // bit twiddling hack
    x -= ((x >> 1) & 0x5555555555555555ULL);
    x = (x & 0x3333333333333333ULL) + (x >> 2 & 0x3333333333333333ULL);
    return ((x + (x >> 4)) & 0xF0F0F0F0F0F0F0FULL) * 0x101010101010101ULL >> 56;
#endif
}

// find the position of the least-significant zero bit
static inline int countr_one(uint64_t x) {
#ifdef __GNUC__
    return __builtin_ffsll(~x) - 1;
#elif _cplusplus >= 201907L
    return std::countr_one(x);
#else   // naive approach; compilers are smart enough to optimize it
    int l = 0;
    while((x & (1ULL << l)))
        l++;
    return l;
#endif
}

namespace math{

namespace {

/// The "xoroshiro128+" pseudo-random number generator, supposed to be very fast and good quality.
/// Written in 2016 by David Blackman and Sebastiano Vigna

/// return the next random number from the sequence, and update the state
inline uint64_t xoroshiro128plus_next(uint64_t state[2]) {
    const uint64_t s0 = state[0];
    uint64_t s1 = state[1];
    const uint64_t result = s0 + s1;  // take the random number from the current state
    // update the state using a few bit-shifts and xor operators
    s1 ^= s0;
    state[0] = ((s0 << 55) | (s0 >> 9)) ^ s1 ^ (s1 << 14); // a, b
    state[1] =  (s1 << 36) | (s1 >> 28); // c
    return result;
}

/// Jump function for the generator. It is equivalent to 2^64 calls to xoroshiro128plus_next();
/// it can be used to generate 2^64 non-overlapping subsequences for parallel computations.
inline void xoroshiro128plus_jump(uint64_t state[2]) {
    static const uint64_t JUMP[] = { 0xbeac0467eba5facb, 0xd86b048b86aa9922 };
    uint64_t s0 = 0;
    uint64_t s1 = 0;
    for(int i = 0; i < 2; i++)
        for(int b = 0; b < 64; b++) {
            if (JUMP[i] & 1ull << b) {
                s0 ^= state[0];
                s1 ^= state[1];
            }
            xoroshiro128plus_next(state);
        }
    state[0] = s0;
    state[1] = s1;
}

/// another (simpler) PRNG of the same family: xorshift64*

/// return the next number from the sequence, and update the 8-byte state vector
inline uint64_t xorshift64star_next(uint64_t *state)
{
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545f4914f6cdd1d; /*scramble output*/
}

/// yet another simple PRNG: splitmix64
inline uint64_t splitmix64_next(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

/// storage for state vectors of PRNGs (separately for each thread)
struct RandGenStorage{
    // in the case of OpenMP, we have as many independent pseudo-random number generators
    // as there are threads, and each thread uses its own state (seed), to avoid race condition
    // and maintain deterministic output
    int maxThreads;
    std::vector<uint64_t> state;  /// two 64-bit integers per thread
    RandGenStorage() :
#ifdef _OPENMP
        maxThreads(std::max(1, omp_get_max_threads())),
#else
        maxThreads(1),
#endif
        state(maxThreads*2)
    {
        randomize(42);  // set some nontrivial initial seeds (anything except zero is fine)
    }

    /// set the initial seed values for all threads
    void randomize(uint64_t seed) {
        if(!seed)
            seed = (uint64_t)time(NULL);
        for(int i=0; i<maxThreads; i++) {
            // take the initial seed (for 0th thread) or copy the seed value from the previous thread...
            state[i*2]   = i>0 ? state[i*2-2] : splitmix64_next(&seed);
            state[i*2+1] = i>0 ? state[i*2-1] : splitmix64_next(&seed);
            // ...and fast-forward 2^64 elements in the sequence
            xoroshiro128plus_jump(&state[i*2]);
        }
    }
};

// global instance of random number generator -- created at program startup and destroyed
// at program exit. Note that the order of initialization of different modules is undefined,
// thus no other static variable initializer may use the random() function without arguments.
// Moving the initializer into the first call of random() is not a remedy either,
// since it may already be called from a parallel section and will not determine
// the number of threads correctly.
static RandGenStorage randgen;

}  // namespace

void randomize(unsigned int seed)
{
    randgen.randomize(seed);
}

uint64_t randint(PRNGState* state)
{
    if(state == NULL) {
        // use a thread-local state vector
#ifdef _OPENMP
        state = &randgen.state[std::min(omp_get_thread_num(), randgen.maxThreads-1)];
#else
        state = &randgen.state[0];
#endif
        return xoroshiro128plus_next(state);
    } else {
        return xorshift64star_next(state);
    }
}

/// not a PRNG by itself, but a function that thoroughly scrambles the input bit sequence:
/// adaptation of MurmurHash64A written by Austin Appleby
uint64_t hash(const void* data, int len /*length of data in 8-byte chunks*/, unsigned int seed)
{
    const uint64_t m = 0xc6a4a7935bd1e995;  // magic number for scrambling
    uint64_t h = (len + m) ^ seed;
    for(int i=0; i<len; i++) {
        uint64_t k = static_cast<const uint64_t*>(data)[i];
        k *= m;
        k ^= k >> 47;
        k *= m;
        h ^= k;
        h *= m;
    }
    h ^= h >> 47;
    h *= m;
    h ^= h >> 47;
    return h;
}

// generate 2 random numbers with normal distribution, using the Box-Muller approach
void getNormalRandomNumbers(double& num1, double& num2, PRNGState* state)
{
    double u, v, p1 = random(state), p2 = random(state);
    if(p1>0)
        p1 = sqrt(-2*log(p1));
    sincos(2*M_PI * p2, u, v);
    num1 = p1 * u;
    num2 = p1 * v;
}

void getRandomUnitVector(double vec[3], PRNGState* state)
{
    double costh =  random(state) * 2 - 1;
    double sinth = sqrt(1 - pow_2(costh)), sinphi, cosphi;
    sincos(2*M_PI * random(state), sinphi, cosphi);
    vec[0] = sinth * cosphi;
    vec[1] = sinth * sinphi;
    vec[2] = costh;
}

double getRandomPerpendicularVector(const double vec[3], double vper[3], PRNGState* state)
{
    double phi = random(state), sinphi, cosphi;
    sincos(2*M_PI * phi, sinphi, cosphi);
    if(vec[1] != 0 || vec[2] != 0) {  // input vector has a nontrivial projection in the y-z plane
        // a combination of two steps:
        // (1) obtain one perpendicular vector as a cross product of v and e_x;
        // (2) rotate it about the vector v by angle phi, using the Rodriguez formula.
        double vmag = sqrt(pow_2(vec[0]) + pow_2(vec[1]) + pow_2(vec[2]));
        double norm = 1 / sqrt(pow_2(vec[1]) + pow_2(vec[2])) / vmag;
        vper[0] = norm * (sinphi * (pow_2(vec[0]) - pow_2(vmag)) );
        vper[1] = norm * (sinphi * vec[0] * vec[1] - cosphi * vmag * vec[2]);
        vper[2] = norm * (sinphi * vec[0] * vec[2] + cosphi * vmag * vec[1]);
        return vmag;
    } else if(vec[0] != 0) {  // degenerate case - a vector directed in the x plane
        vper[0] = 0;
        vper[1] = cosphi;
        vper[2] = sinphi;
        return fabs(vec[0]);
    } else {  // even more degenerate case of a null vector - create a random isotropic vector
        double costh = random(state)*2-1;
        double sinth = sqrt(1-pow_2(costh));
        vper[0] = sinth * cosphi;
        vper[1] = sinth * sinphi;
        vper[2] = costh;
        return 0;
    }
}

void getRandomRotationMatrix(double mat[9], PRNGState* state)
{
    // the algorithm of Arvo(1992)
    double sinth, costh, sinphi, cosphi;
    sincos(2*M_PI * random(state), sinth,  costh );
    sincos(2*M_PI * random(state), sinphi, cosphi);
    double
    mu = random(state) * 2,
    nu = sqrt(mu),
    vx = sinphi * nu,
    vy = cosphi * nu,
    vz = sqrt(2-mu),
    st = sinth,
    ct = costh,
    sx = vx*ct - vy*st,
    sy = vx*st + vy*ct;
    mat[0] = vx*sx-ct;
    mat[1] = vx*sy-st;
    mat[2] = vx*vz;
    mat[3] = vy*sx+st;
    mat[4] = vy*sy-ct;
    mat[5] = vy*vz;
    mat[6] = vz*sx;
    mat[7] = vz*sy;
    mat[8] = 1-mu;
}

void getRandomPermutation(size_t count, size_t output[], PRNGState* state)
{
    // Fisher-Yates-Durstenfeld algo
    for(size_t i=0; i<count; i++) {
        size_t j = std::min(static_cast<size_t>(random(state) * (i+1)), i);
        output[i] = output[j];
        output[j] = i;
    }
}

QuasiRandomHalton::QuasiRandomHalton(uint8_t dimension, PRNGState* state)
{
    static const uint8_t PRIMES[MAX_DIM] = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53 };
    if(dimension >= MAX_DIM)
        throw std::runtime_error("QuasiRandomHalton: dimension is too large");
    base = PRIMES[dimension];
    invbase = 1./base;
    // init permutations (the algorithm of Owen 2017; arXiv:1706.02808)
    int permSize = ceil(54 / log2(base)) - 1;
    permutations.resize(permSize * base);
    remainders.resize(permSize);
    double fac = invbase;
    for(int i=0; i<permSize; i++) {
        getRandomPermutation(base, &(permutations[i*base]), state);
        // table of remainders for each iteration, which considerably reduces
        // the number of iterations when the index is not too large
        for(int j=0; j<=i; j++)
            remainders[j] += fac * permutations[i*base];
        fac *= invbase;
    }
}

double QuasiRandomHalton::operator()(size_t index) const
{
    double value = 0, fac = invbase;
    unsigned int k = 0;
    while(index > 0) {
        value += fac * permutations[k * base + index % base];
        index /= base;
        fac   *= invbase;
        k++;
    }
    // index becomes zero rather quickly unless the starting value is very large,
    // so instead of running the loop for the entire set of permutations (k=0..remainders.size()-1),
    // we retrieve the precomputed remainders summing up 0th elements of permutations
    // for all subsequent iterations
    return value + remainders[k];
}

QuasiRandomSobol::QuasiRandomSobol(uint8_t dimension, PRNGState* prngState, uint8_t _numBits) :
    numBits(_numBits),
    scale(pow(0.5, numBits)),
    offset(randint(prngState) & ((1ULL << numBits) - 1)),  // random integer restricted to numbits
    count(0),
    state(offset)
{
    static const int DEGREES[MAX_DIM] = { 1, 3, 7, 11, 13, 19, 25, 37, 41, 47, 55, 59, 61, 67, 91, 97 };
    static const int INIT_NUMBERS[MAX_DIM][6] = {
        {0},
        {1},
        {1, 3},
        {1, 3, 1},
        {1, 1, 1},
        {1, 1, 3,  3},
        {1, 3, 5, 13},
        {1, 1, 5,  5, 17},
        {1, 1, 5,  5,  5},
        {1, 1, 7, 11, 19},
        {1, 1, 5,  1,  1},
        {1, 1, 1,  3, 11},
        {1, 3, 5,  5, 31},
        {1, 3, 3,  9,  7, 49},
        {1, 1, 1, 15, 21, 21},
        {1, 3, 1, 13, 27, 49} };
    if(dimension >= MAX_DIM)
        throw std::runtime_error("QuasiRandomSobol: dimension is too large");
    if(numBits > MAX_BITS)
        throw std::runtime_error("QuasiRandomSobol: number of bits is too large");
    if(dimension == 0) {
        directionNumbers.assign(numBits, 1);
    } else {
        directionNumbers.assign(numBits, 0);
        int p = DEGREES[dimension];
        int m = 0, pp = p;
        while(pp >>= 1)
            m++; // determine the width of p in bits
        for(int b=0; b<m; b++)
            directionNumbers[b] = INIT_NUMBERS[dimension][b];
        // fill remaining elements using Bratley & Fox (1988) algorithm 659
        for(int b=m; b<numBits; b++) {
            uint64_t z = directionNumbers[b-m];
            for(int k=0; k<m; k++) {
                if((p >> (m-k-1)) & 1)
                    z ^= (2<<k) * directionNumbers[b-k-1];
            }
            directionNumbers[b] = z;
        }
    }
    for(uint8_t b=0; b<numBits; b++)  // multiply the direction numbers by successive powers of 2
        directionNumbers[b] <<= numBits-1-b;

    // generate a lower triangular matrix with size numbits*numbits containing random 0/1 values
    // except diagonal elements, which are set to 1;
    // the bit sequence in each row is represented by one integer, most significant bit is on the left
    uint64_t scrambles[MAX_BITS];
    for(uint8_t b=0; b<numBits; b++) {
        scrambles[b]  = randint(prngState);
        // keep only b upper bits in each row (i.e. create a lower triangular matrix)
        scrambles[b] &= ((1ULL << numBits) - 1) >> (numBits-1-b) << (numBits-1-b);
        // set diagonal elements to 1
        scrambles[b] |= 1ULL << (numBits-1-b);
    }
    // now scramble the direction numbers (same algorithm as in scipy; Matousek 1998)
    for(uint8_t b=0; b<numBits; b++) {
        uint64_t dirNum = directionNumbers[b], newDirNum = 0;
        for(uint8_t p=0; p<numBits; p++)
            newDirNum += uint64_t(popcount(scrambles[numBits-1-p] & dirNum) & 1) << p;
        directionNumbers[b] = newDirNum;
    }
}

double QuasiRandomSobol::operator()(size_t index) const
{
    // check that the index does not exceed the number of available distinct values (2^numBits)
    if(index >> numBits)
        return NAN;
    // if the requested index is different from the number of already completed elements,
    // we need to replace the cached state with the freshly prepared one
    // (fortunately, this "fast-forward" operation has fixed cost independent of index).
    if(index != count) {
        uint64_t gray = index ^ (index >> 1);  // Gray code for index
        state = offset;
        for(uint8_t b=0; b<numBits; b++) {
            state ^= ((gray >> b) & 1) * directionNumbers[b];
        }
    }
    // find the location of the least significant zero bit in index
    int l = countr_one(index);
    // return the current state
    double result = state * scale;
    // update cached state in preparation for the next element
    state ^= directionNumbers[l];
    count = index + 1;
    return result;
}

}  // namespace math
