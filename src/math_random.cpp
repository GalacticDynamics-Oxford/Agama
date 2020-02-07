#include "math_random.h"
#include "math_core.h"  // for sincos
#include <vector>
#include <cmath>
#include <ctime>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace math{

namespace {

/// 2^-64, conversion factor from integer to double random numbers
static const double TWOMINUS64 = 1./18446744073709551616.;


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

double random(PRNGState* state)
{
    if(state == NULL) {
        // use a thread-local state vector
#ifdef _OPENMP
        state = &randgen.state[std::min(omp_get_thread_num(), randgen.maxThreads-1)];
#else
        state = &randgen.state[0];
#endif
        return xoroshiro128plus_next(state) * TWOMINUS64;  /*convert to the range [0:1) */
    } else {
        return xorshift64star_next(state) * TWOMINUS64;
    }
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
    // Fisher-Yates algo
    for(size_t i=0; i<count; i++) {
        size_t j = std::min(static_cast<size_t>(random(state) * (i+1)), i);
        output[i] = output[j];
        output[j] = i;
    }
}

double quasiRandomHalton(size_t ind, unsigned int base)
{
    double val = 0, fac = 1., invbase = 1./base;
    while(ind > 0) {
        fac *= invbase;
        val += fac * (ind % base);
        ind /= base;
    }
    return val;
}

// adaptation of MurmurHash64A written by Austin Appleby
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

}  // namespace math
