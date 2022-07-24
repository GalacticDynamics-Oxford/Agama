/** \file   math_random.h
    \brief  pseudo- and quasi-random number generators
    \date   2015-2019
    \author Eugene Vasiliev
*/
#pragma once
#include <cstddef>    // for NULL
#include <stdint.h>   // for uint64_t

namespace math{

/** state vector for a pseudo-random number generator (PRNG) */
typedef uint64_t PRNGState;

/** initialize the pseudo-random number generator with the given value,
    or a completely arbitrary value (depending on system time) if seed==0.
    In a multi-threaded case, each thread is initialized with a different seed.
    At startup (without any call to randomize()), the random seed always has the same value.
*/
void randomize(unsigned int seed=0);

/** generate a pseudo-random number in the range [0,1).
    \param[in,out]  state  is the internal state of the PRNG, which gets updated after each call.
    If not provided (NULL), the internal global state is used, which is an array with the number
    of elements equal to the number of OpenMP threads; each thread thus has access to a separate
    PRNG instance, seeded with a different number at the beginning of the program or after randomize().
    \return the pseudo-random number.
*/
double random(PRNGState* state=NULL);

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

/** return a quasirandom number from the Halton sequence.
    \param[in]  index  is the index of the number (should be >0, or better > ~10-20);
    \param[in]  base   is the base of the sequence, must be a prime number;
    if one needs an N-dimensional vector of ostensibly independent quasirandom numbers,
    one may call this function N times with different prime numbers as bases.
    \return  a number between 0 and 1.
    \note that that the numbers get increasingly more correlated as the base increases,
    thus it is not recommended to use more than ~6 dimensions unless index spans larger enough range.
*/
double quasiRandomHalton(size_t index, unsigned int base);

/** first ten prime numbers, may be used as bases in `quasiRandomHalton` */
static const int MAX_PRIMES = 10;  // not that there aren't more!
static const int PRIMES[MAX_PRIMES] = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29 };

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

/// an overloaded version which takes an array of doubles as input
inline PRNGState hash(const double* data, int len, unsigned int seed=0) {
    return hash(static_cast<const void*>(data), len, seed); }

}  // namespace
