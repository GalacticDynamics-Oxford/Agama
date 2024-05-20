#!/usr/bin/python

"""
Illustrates various kinds of splines provided by Agama and
tests the methods for computing their roots and extrema.
"""
import numpy, sys
# if the module has been installed to the globally known directory, just import it
try: import agama
except ImportError:  # otherwise load the shared library from the parent folder
    import sys
    sys.path += ['../']
    try: import agama
    except ImportError as ex: sys.exit("\033[1;31mFAILED TO IMPORT AGAMA: %s\033[0m" % ex)
numpy.random.seed(42)

def test(kind):
    """
    test the accuracy of roots and extrema returned by various kinds of splines
    """
    # the number and placement of x-grid nodes is random, as are the spline values or amplitudes
    N = int(numpy.random.random()*100+10)
    xval = numpy.cumsum(numpy.random.random(size=N)+0.1) - numpy.random.random()*N*0.5
    if isinstance(kind, int):
        ampl = numpy.random.random(size=N+kind-1)-0.5
        spl = agama.Spline(xval, ampl=ampl)
    else:
        yval = numpy.random.random(size=N)-0.5
        if kind=='quintic':
            # assign derivatives from an auxiliary cubic spline
            der = agama.Spline(xval, yval)(xval, 1)
            spl = agama.Spline(xval, yval, der=der)
        elif kind=='regcubic':
            spl = agama.Spline(xval, yval, reg=True)
        else:
            spl = agama.Spline(xval, yval)

    ok = True

    y_root = numpy.random.random()-0.5
    t_root = spl.roots(y_root)
    if len(t_root) > 1 and numpy.any(t_root[1:] <= t_root[:-1]):
        print('%s: roots not monotonically increasing' % spl)
        ok = False
    if kind==0:
        if not numpy.all(numpy.unique(numpy.hstack((t_root, xval))) == xval):
            print('%s: roots not among grid nodes' % spl)
            ok = False
    elif len(t_root)>0:
        diff_root = spl(t_root) - y_root
        # due to finite precision of x coordinates, the spline value may not be exactly zero;
        # check the nearby points to assess the tolerance threshold
        roundoff = numpy.maximum(2 * abs(spl(numpy.nextafter(t_root, 1e10)) - y_root - diff_root), 2e-14)
        if numpy.any(abs(diff_root) > roundoff):
            print('%s: value at root %g exceeds threshold' % (spl, max(abs(diff_root))))
            print(numpy.column_stack((t_root, diff_root, roundoff, diff_root / roundoff)))
            ok = False

    t_extr = spl.extrema()
    if (len(t_extr) < 2 or numpy.any(t_extr[1:] <= t_extr[:-1]) or
        t_extr[0] != xval[0] or t_extr[-1] > xval[-1]):
        print('%s: extrema not monotonically increasing' % spl)
        ok = False
    if kind==0 or kind==1:
        change_sign = (ampl[1:-1]-ampl[:-2]) * (ampl[2:]-ampl[1:-1]) <= 0
        expected = numpy.hstack((xval[0], xval[1:-2+kind][change_sign], xval[len(ampl)-1]))
        if len(expected) != len(t_extr) or numpy.any(t_extr != expected):
            print('%s: extrema not among grid nodes' % spl)
            print('grid nodes: %s' % xval)
            print('expected: %s' % expected)
            print('received: %s' % t_extr)
            ok = False
    else:
        t_extr = t_extr[(t_extr>xval[0]) & (t_extr<xval[-1])]
        f_extr = spl(t_extr)
        f_left = spl(numpy.maximum(t_extr-1e-6, xval[0]))
        f_right= spl(numpy.minimum(t_extr+1e-6, xval[-1]))
        if kind != 'regcubic' and not numpy.all((f_extr-f_left) * (f_right-f_extr) <= 0):
            print('%s: reported extrema not consistent with finite-differences' % spl)
            print(numpy.column_stack((t_extr, f_left, f_extr, f_right)))
            ok = False
        diff_extr = spl(t_extr, 1)
        # due to finite precision of x coordinates, the derivative may not be exactly zero;
        # check the nearby points to assess the tolerance threshold
        roundoff = numpy.maximum(2 * abs(spl(numpy.nextafter(t_extr, 1e10), 1) - diff_extr), 2e-13)
        if numpy.any(abs(diff_extr) > roundoff):
            print('%s: derivative at extrema %g exceeds threshold' % (spl, max(abs(diff_extr))))
            print(numpy.column_stack((t_extr, diff_extr, roundoff, diff_extr / roundoff)))
            ok = False

    if not ok:
        import matplotlib.pyplot as plt
        t = numpy.linspace(min(xval), max(xval), 10000)
        plt.plot(t, spl(t), c='b')
        plt.plot(xval, spl(xval), 'o', ms=3, mew=0, c='b')
        plt.plot(t_root, spl(t_root), 'x', ms=4, c='m')
        plt.plot(t_extr, spl(t_extr), 'o', mew=0, ms=3, c='r')
        plt.show()
    return ok

ok = True
for i in range(10000):
    ok &= test(0)
    ok &= test(1)
    ok &= test(2)
    ok &= test(3)
    ok &= test('cubic')
    ok &= test('regcubic')
    ok &= test('quintic')

if len(sys.argv) > 1:
    import matplotlib.pyplot as plt
    xval = numpy.array([-0.5, -0.125, 0.75, 1.25, 1.5])
    cubr = agama.Spline(xval, [-0.5, 1, 1, 0, 0], reg=True)
    quin = agama.Spline(xval, [-0.5, 1-2**-6, 1-2**-6, 0, 0], der=[0, 1, -1, 0, 0])
    bsp0 = agama.Spline(xval, ampl=[-0.5, 1+2**-6, 0.5, 0])
    bsp1 = agama.Spline(xval, ampl=[-0.5, 0.875, 1, 0, 0])
    bsp2 = agama.Spline(xval, ampl=[-0.5, 1, 1, 1.0625, 0, 0])
    bsp3 = agama.Spline(xval, ampl=[-0.5,-0.5,1.25,0.875,1.125,-0.0625,0])
    splines = (cubr, quin, bsp0, bsp1, bsp2, bsp3)
    colors = ('b', 'g', 'r', 'c', 'm', 'y')
    t = numpy.linspace(min(xval), max(xval), 1001)
    for spline, color in zip(splines, colors):
        def check(x, exempt):
            return '' if spline==exempt or max(abs(x)) < 2e-14 else '\033[1;31m**\033[0m'  # a fairly lousy tolerance!.. :(
        plt.plot(t, spline(t), color=color, label=str(spline).replace(' with 5 nodes on [-0.5:1.5]', ''))
        t_extr = spline.extrema()
        d_extr = spline(t_extr, 1)
        print('\033[1;33m%s\033[0m' % spline)
        print('\033[1;37mextrema               deriv\033[0m%s' % check(d_extr[1:-1], bsp1))
        for te, de in zip(t_extr, d_extr):
            print('%-21.17g %.3g' % (te, de))
        plt.plot(t_extr, spline(t_extr), 's', ms=3, mew=0, color=color)
        t_root = spline.roots(1.0)
        f_root = spline(t_root)-1.0
        print('\033[1;37mroots=1               value-1\033[0m%s' % check(f_root, bsp0))
        for tr, fr in zip(t_root, f_root):
            print('%-21.17g %.3g' % (tr, fr))
        plt.plot(t_root, spline(t_root), 'x', ms=5, color=color)
        t_root = spline.roots(0.0)
        f_root = spline(t_root)
        print('\033[1;37mroots=0               value\033[0m%s' % check(f_root, bsp0))
        for tr, fr in zip(t_root, f_root):
            print('%-21.17g %.3g' % (tr, fr))
        plt.plot(t_root, spline(t_root), '+', ms=5, color=color)
    plt.legend(loc='lower center', frameon=False)
    plt.show()
else:
    print('Run the script with a non-empty command-line argument to show a plot')

if ok:
    print("\033[1;32mALL TESTS PASSED\033[0m")
else:
    print("\033[1;31mSOME TESTS FAILED\033[0m")
