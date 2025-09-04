/** \file   math_random.h
    \brief  pseudo- and quasi-random number generators
    \date   2015-2025
    \author Eugene Vasiliev
*/
#pragma once
#include <cstddef>    // for NULL
#include <stdint.h>   // for uint64_t
#include <vector>

namespace math{

/** state vector for a pseudo-random number generator (PRNG) */
typedef uint64_t PRNGState;

/** initialize the pseudo-random number generator with the given value,
    or a completely arbitrary value (depending on system time) if seed==0.
    In a multi-threaded case, each thread is initialized with a different seed.
    At startup (without any call to randomize()), the random seed always has the same value.
*/
void randomize(unsigned int seed=0);

/** generate an integer pseudo-random number in the range [0, 2^64-1].
    \param[in,out]  state  is the internal state of the PRNG, which gets updated after each call;
    can be any number except zero.
    If not provided (NULL), the internal global state is used, which is an array with the number
    of elements equal to the number of OpenMP threads; each thread thus has access to a separate
    PRNG instance, seeded with a different number at the beginning of the program or after randomize().
    \return the pseudo-random number.
*/
uint64_t randint(PRNGState* state=NULL);

/** generate a floating-point pseudo-random number in the range [0,1).
    \param[in,out]  state  is the internal state of the PRNG, which gets updated after each call;
    can be any number except zero.
    If not provided (NULL), the internal global state is used, which is an array with the number
    of elements equal to the number of OpenMP threads; each thread thus has access to a separate
    PRNG instance, seeded with a different number at the beginning of the program or after randomize().
    \return the pseudo-random number.
*/
inline double random(PRNGState* state=NULL)
{
    /// 2^-64, conversion factor from the full range of 64-bit integers to doubles in the range [0:1)
    const double TWOMINUS64 = 1./18446744073709551616.;
    return randint(state) * TWOMINUS64;
}

/** generate two uncorrelated random numbers from the standard normal distribution.
    \param[out]  num1, num2 will contain the output -- two normally distributed random numbers.
    \param[in,out]  state  is the internal state of the PRNG, which gets updated after each call.
*/
void getNormalRandomNumbers(/*output*/ double& num1, double& num2, /*input/output*/ PRNGState* state=NULL);

/** construct a random vector, uniformly distributed on the unit sphere in 3d.
    \param[out]  vec  is an array filled with 3 components of the unit vector.
    \param[in,out]  state  is the internal state of the PRNG, which gets updated after each call.
*/
void getRandomUnitVector(/*output*/ double vec[3], /*input/output*/ PRNGState* state=NULL);

/** construct a random unit vector perpendicular to the given vector.
    \param[in]   vec  is an array of 3 cartesian coordinates of the input vector;
    \param[out]  vper is an array filled with three components of the random vector
    perpendicular to the input vector;
    \param[in,out]  state  is the internal state of the PRNG, which gets updated after each call.
    \return  the magnitude of the input vector as a by-product.
*/
double getRandomPerpendicularVector(const double vec[3], /*output*/ double vper[3],
    /*input/output*/ PRNGState* state=NULL);

/** construct a random 3d rotation matrix (uniformly distributed random rotation angle
    about an axis specified by a random vector uniformly distributed on the unit sphere).
    \param[out]  mat  is an array of 9 numbers representing a 3x3 rotation matrix.
    \param[in,out]  state  is the internal state of the PRNG, which gets updated after each call.
*/
void getRandomRotationMatrix(/*output*/ double mat[9], /*input/output*/ PRNGState* state=NULL);

/** create a sequence of integers ranging from 0 to count-1 arranged in a random order.
    \param[in]  count  is the number of elements in the permutation;
    \param[out] output is the array to be filled by the permutation indices
    (must be an existing array of sufficient length)
    \param[in,out]  state  is the internal state of the PRNG, which gets updated after each call.
*/
void getRandomPermutation(size_t count, size_t output[], /*input/output*/ PRNGState* state=NULL);

/** Construct the initial state of a PRNG from the provided input bitstream.
    The purpose of this function is to produce a predictable but well-mixed output --
    if it is called with the same input, it will always produce the same result,
    but a difference even in a single bit will likely produce very different result.
    It can be used in a dynamically load-balanced multi-threaded context, such as orbit
    integration, when one needs a stream of pseudo-random numbers which doesn't depend on
    the thread index, only on some input data, and one can provide an input entropy source.
    The output of this function can be used as optional input for several routines in this module,
    which generate random numbers with specific distributions from the input values
    uniformly distributed in [0,1).
    \param[in]  data  is the array with some data serving as the source of 'entropy'.
    \param[in]  len   is the length of input array, measured in 8-byte chunks
    (could be float, double, int64, ...).
    \param[in]  seed  additional source of bits, essentially augmenting the input array by 1 element.
    \return  well-scrambled hash of input data, which can be used with any of the above routines.
*/
PRNGState hash(const void* data, int len, unsigned int seed=0);

/** Generator of quasirandom numbers from the Halton sequence with a randomly initialized scrambling.
    Sequences with different dimension indices are independent and have mutually low discrepancy,
    while sequences with the same dimension but different scrambling seeds are not.
    Once initialized, the () operator returns the quasi-random number with the given index.
*/
class QuasiRandomHalton {
    uint8_t base;    ///< prime number that depends on dimension
    double invbase;  ///< inverse of the above
    std::vector<size_t> permutations;   ///< scrambling set
    std::vector<double> remainders;     ///< table for speeding up computations for low indices
public:
    static const uint8_t MAX_DIM = 16;  ///< maximum number of dimensions

    /** Create the generator for the given dimension and random seed.
        \param[in] dimension  is the index of dimension (starting from zero), should be < MAX_DIM.
        \param[in,out] state  is the internal state of a PRNG that is used to initialize the scrambling,
        which gets updated after the call; if NULL, the internal global state is used.
    */
    QuasiRandomHalton(uint8_t dimension, /*input/output*/ PRNGState* state=NULL);

    /** Return the quasi-random number between 0 and 1 from this sequence with a given index;
        the value depends deterministically on index, dimension and random seed used to initialize
        the scrambling, i.e., subsequent calls with the same index return the same value.
    */
    double operator()(size_t index) const;
};

/** Generator of quasirandom numbers from the Sobol sequence with a randomly initialized scrambling.
    Sequences with different dimension indices are independent and have mutually low discrepancy,
    while sequences with the same dimension but different scrambling seeds are not.
    Once initialized, the () operator returns the quasi-random number with the given index.
    This generator is somewhat faster than Halton (when called with sequentially increasing indices),
    and produces sequences that are more equally distributed between the lower and upper halves
    of the interval, but its statistical properties are best realized when the number of points
    is a power of two. It is also not thread-safe due to caching of the internal state.
*/
class QuasiRandomSobol {
    const uint8_t numBits;  ///< number of bits generated by the sequence before it starts to repeat
    const double scale;     ///< 2^(-numbits)
    std::vector<uint64_t> directionNumbers;  ///< magic constants (including random scrambling)
    const uint64_t offset;  ///< random offset of the initial point
    mutable uint64_t count; ///< index of next point; generation is faster if done sequentially
    mutable uint64_t state; ///< cached state for optimized generation of count+1'th point
public:
    static const uint8_t MAX_DIM = 16;   ///< maximum number of dimensions
    static const uint8_t MAX_BITS = 53;  ///< maximum number of bits produced by the generator

    /** Create the generator for the given dimension and random seed.
        \param[in] dimension  is the index of dimension (starting from zero), should be < MAX_DIM.
        \param[in,out] state  is the internal state of a PRNG that is used to initialize the scrambling,
        which gets updated after the call; if NULL, the internal global state is used.
        \param[in] numBits  is the number of bits produced by the generator;
        larger values enable longer non-repeating sequences, up to the limit of double precision,
        but make the generator slower to construct and to access out-of-order; default is 32.
    */
    QuasiRandomSobol(uint8_t dimension, /*input/output*/ PRNGState* state=NULL, uint8_t numBits=32);

    /** Return the quasi-random number between 0 and 1 from this sequence with a given index;
        the value depends deterministically on index, dimension and random seed used to initialize
        the scrambling, i.e., subsequent calls with the same index return the same value.
        If index exceeds 2^numbits, the sequence would repeat itself, so a NAN is returned instead.
        Note that the subsequent calls with indices incremented by one are ~10x faster than
        the general case of arbitrary-order access, thanks to an optimized computation method
        for sequential numbers, which relies on caching of the current state.
        \warning  Due to caching, this method is not safe to call from multiple threads in parallel;
        one should instead create and use thread-local copies of this generator object cloned from
        the same master generator (to ensure identical scrambling constants initialized from a PRNG).
    */
    double operator()(size_t index) const;
};

}  // namespace
