#!/usr/bin/python
import numpy
from matplotlib.pyplot import *
from pygama import *
sin=numpy.sin
cos=numpy.cos

# draw N points from a separable triaxial Gaussian density profile with dispersions sx, sy, sz
def sampleTriaxialGaussian(N, sx, sy, sz):
    return numpy.column_stack((
        numpy.random.normal(0, sx, N), numpy.random.normal(0, sy, N), numpy.random.normal(0, sz, N) ))

# draw an ellipse with axes Sxp, Syp, rotated ccw from y' axis by angle eta
def drawEllipse(Sxp, Syp, eta):
    angles = numpy.linspace(0, 1.75*numpy.pi, 100)
    xp = Sxp * cos(angles)
    yp = Syp * sin(angles)
    plot(-xp * sin(eta) - yp * cos(eta), xp * cos(eta) - yp * sin(eta), 'r-')
    plot([0, -Sxp*sin(eta)], [0,  Sxp*cos(eta)], 'r--')
    plot([0, -Syp*cos(eta)], [0, -Syp*sin(eta)], 'r--')


# original orientation angles
trueTheta = numpy.random.random()*numpy.pi
truePhi   = numpy.random.random()*numpy.pi
trueChi   = numpy.random.random()*numpy.pi
# original axes of the ellipsoid
trueSx    = 1.5
trueSy    = 1.0
trueSz    = 0.5

# main code
points = sampleTriaxialGaussian(20000, trueSx, trueSy, trueSz)
prjmat = makeProjectionMatrix(trueTheta, truePhi, trueChi)
prjpoi = numpy.dot(prjmat, points.T).T

trueSxp, trueSyp, trueEta = getProjectedEllipsoid(trueSx, trueSy, trueSz, trueTheta, truePhi, trueChi)
print "theta=%f, phi=%f, chi=%f" % (trueTheta, truePhi, trueChi)
print "sigma_x'=%f, sigma_y'=%f, psi=%f" % (trueSxp, trueSyp, trueEta - trueChi)
print "Position angle of x-axis: %f, y-axis: %f, z-axis: %f, proj.long axis: %f" % \
    (numpy.arctan2( sin(truePhi), -cos(truePhi)*cos(trueTheta) ) + trueChi,
     numpy.arctan2(-cos(truePhi), -sin(truePhi)*cos(trueTheta) ) + trueChi,
     trueChi, trueEta)
p = trueSy  / trueSx
q = trueSz  / trueSx
qp= trueSyp / trueSxp
u = trueSxp / trueSx
print "max({p=%f}, {q/q'=%f}) <= {u=%f} <= min({p/q'=%f}, 1)" % (p, q/qp, u, p/qp)

for i in range(0,10):
    la,mu,nu = numpy.random.random(3)
    q = la * qp
    p = mu + (1-mu) * q
    u = nu * min(p/qp, 1) + (1-nu) * max(p, q/qp)
    Sx = trueSx if i==0 else trueSxp / u
    Sy = trueSy if i==0 else Sx * p
    Sz = trueSz if i==0 else Sx * q
    solTheta, solPhi, solChi  = getViewingAngles(trueSxp, trueSyp, trueEta, Sx, Sy, Sz)
    print "sigma_x=%f, sigma_y=%f, sigma_z=%f => theta=%f, phi=%f, chi=%f" % \
        (Sx, Sy, Sz, solTheta, solPhi, solChi)

for i in range(0,10):
    theta = trueTheta if i==0 else numpy.random.random()*numpy.pi
    phi   = truePhi   if i==0 else numpy.random.random()*numpy.pi
    chi   = trueChi   if i==0 else numpy.random.random()*numpy.pi
    print "theta=%.6f, phi=%.6f, chi=%.6f =>" % (theta, phi, chi),
    try:
        solSx, solSy, solSz = getIntrinsicShape(trueSxp, trueSyp, trueEta, theta, phi, chi)
        print "sigma_x=%f, sigma_y=%f, sigma_z=%f" % (solSx, solSy, solSz),
        if solSy>solSx or solSz>solSy: print "**"
        else: print ""
    except Exception as e:
        print "impossible (%s)" % e

figure(figsize=(5,5))
plot(prjpoi[:,0], prjpoi[:,1], '.', ms=2, c='gray')
drawEllipse(1*trueSxp, 1*trueSyp, trueEta)
drawEllipse(2*trueSxp, 2*trueSyp, trueEta)
axes = numpy.dot(prjmat, numpy.array([[1,0,0],[0,1,0],[0,0,1]]))
plot([0,2*prjmat[0,0]], [0,2*prjmat[1,0]], 'g-o')
text( 2.5*prjmat[0,0],   2.5*prjmat[1,0],  'x', fontsize=20, ha='center', va='center')
plot([0,2*prjmat[0,1]], [0,2*prjmat[1,1]], 'g-o')
text( 2.5*prjmat[0,1],   2.5*prjmat[1,1],  'y', fontsize=20, ha='center', va='center')
plot([0,2*prjmat[0,2]], [0,2*prjmat[1,2]], 'g-o')
text( 2.5*prjmat[0,2],   2.5*prjmat[1,2],  'z', fontsize=20, ha='center', va='center')
xlim(-4,4)
ylim(-4,4)
show()
