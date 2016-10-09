#!/usr/bin/python

### shows a pretty checkerboard picture - samples drawn from the given function
import agama, numpy, matplotlib.pyplot as plt

# the user-defined function must take a single argument which is a 2d array MxN,
# where N is the dimension of the space and M is the number of points where
# the function should be evaluated simultaneously (for performance reasons),
# i.e., it should operate with columns of the input array x[:,0], x[:,1], etc.
def fnc(x):
    return numpy.maximum(0, numpy.sin(11*numpy.pi*x[:,0]) * numpy.sin(15*numpy.pi*x[:,1]))

valI,errI,_ = agama.integrateNdim(fnc, 2, maxeval=50000)
exact =  (2/numpy.pi)**2 * 83./165
print "N-dimensional integration: result =", valI, "+-", errI, " (exact value:", exact, ")"

arr,valS,errS,_ = agama.sampleNdim(fnc, 50000, [-1,1], [0,2])
print "N-dimensional sampling: result =", valS, "+-", errS
plt.plot(arr[:,0], arr[:,1], ',')
plt.show()

if abs(valI-exact)<errI and abs(valS-exact)<errS and errI<1e-3 and errS<1e-3:
    print "\033[1;32mALL TESTS PASSED\033[0m"
else:
    print "\033[1;31mSOME TESTS FAILED\033[0m"