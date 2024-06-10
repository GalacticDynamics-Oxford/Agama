#!/usr/bin/python
__docstring__ = """
This example program shows an interactive plot illustrating the projection and
deprojection of a triaxial ellipsoid, viewed at an arbitrary orientation.
The axis lengths Sx,Sy,Sz of the ellipsoid may be provided as command-line
arguments, otherwise they have default values 1.5, 1.0, 0.7.
The orientation of the ellipsoid relative to the observer is controlled by
three viewing angles -- Euler rotation angles of the transformation between
the intrinsic coordinate system associated with the object and the image plane.
The angles can be adjusted by mouse movement: when the left button is pressed,
horizontal movement changes the angle alpha of the rotation in the equatorial
plane of the ellipsoid, and vertical movement changes the inclination angle beta;
when the right button is pressed, the movement changes the rotation angle gamma
in the image plane.
The ellipsoid is drawn by a triangulated surface, coloured according to the
intrinsic coordinates x,y,z: red for large |x| - at the long-axis endpoints,
green for large |y| - intermediate axis, and blue for large |z| - short axis;
additionally, the brightness shows the scattered and reflected light coming
from the direction of observer -- parts of the surface that are parallel
to the image plane are brighter.
The intrinsic coordinate axes x,y,z are shown by red, green and blue lines:
solid if they point towards the observer, dashed if behind the image plane.
The position angles (PA) of the projected principal axes x,y,z are printed
in the top left corner (measured counter-clockwise from the vertical axis/north).
The projected shape is shown by black ellipse, and its axis lengths and
orientation in the image plane are also displayed in the top left corner.
Right part of the plot shows four possible deprojections of this ellipse
for the fixed assumed axis lengths of the triaxial ellipsoid; they differ by
the inferred viewing angles, and the correct one is highlighted in green.
Alternatively, the intrinsic shape of the ellipsoid can be inferred from the
observed ellipse and the assumed viewing angles, except when it is projected
along one if its principal planes; it is shown in the bottom of the left panel.
"""
import sys, numpy, agama, traceback
import matplotlib.pyplot as plt, matplotlib.patches, matplotlib.collections, matplotlib.backend_bases
sin=numpy.sin
cos=numpy.cos
pi=numpy.pi

# orientation angles
alpha = numpy.random.randint(0,180)*pi/180
beta  = numpy.random.randint(0, 90)*pi/180
gamma = 0
# axes of the ellipsoid
Sx    = 1.5  # default values
Sy    = 1.0
Sz    = 0.7
if len(sys.argv)>=4:  # may be overriden in the command line
    Sx = float(sys.argv[1])
    Sy = float(sys.argv[2])
    Sz = float(sys.argv[3])
    if 2<Sx or Sx<Sy or Sy<Sz or Sz<0:
        raise ValueError("Axes must be sorted in order of increase")

# lighting and shading
ampScattered = 0.4
ampDirection = 0.5
ampSpecular  = 0.5
powSpecular  = 8

def getColor(xyz):
    return 0.3 + xyz**2 * numpy.array([0.5 * Sx**-2, 0.3 * Sy**-2, 0.9 * Sz**-2])

# grid for ray tracing
Ngrid  = round(90*Sx)
gridn  = numpy.linspace(-Sx, Sx, Ngrid+1)
gridc  = 0.5*(gridn[1:]+gridn[:-1])
X      = numpy.tile  (gridc, len(gridc))
Y      = numpy.repeat(gridc, len(gridc))

# polygons for triangulated surface plotting
points = numpy.array([[0,1,0],[1,0,0],[0,0,1],[-1,0,0],[0,0,-1],[0,-1,0],[0.967,0,0.255],[0.665,0.665,0.339],[0.828,0.561,0],[0.561,0,0.828],[0.339,0.665,0.665],[0.967,0.255,0],[0.31,0.899,0.31],[0.899,0.31,0.31],[0.561,0.828,0],[0.255,0,0.967],[0.665,0.339,0.665],[0.828,0,0.561],[0.31,0.31,0.899],[0.255,0.967,0],[-0.255,0,0.967],[-0.339,0.665,0.665],[0,0.561,0.828],[-0.828,0,0.561],[-0.665,0.665,0.339],[0,0.255,0.967],[-0.31,0.899,0.31],[-0.31,0.31,0.899],[0,0.828,0.561],[-0.967,0,0.255],[-0.665,0.339,0.665],[-0.561,0,0.828],[-0.899,0.31,0.31],[0,0.967,0.255],[-0.967,0,-0.255],[-0.665,0.665,-0.339],[-0.828,0.561,0],[-0.561,0,-0.828],[-0.339,0.665,-0.665],[-0.967,0.255,0],[-0.31,0.899,-0.31],[-0.899,0.31,-0.31],[-0.561,0.828,0],[-0.255,0,-0.967],[-0.665,0.339,-0.665],[-0.828,0,-0.561],[-0.31,0.31,-0.899],[-0.255,0.967,0],[0.339,0.665,-0.665],[0,0.561,-0.828],[0.665,0.665,-0.339],[0,0.255,-0.967],[0.31,0.899,-0.31],[0.31,0.31,-0.899],[0,0.828,-0.561],[0.665,0.339,-0.665],[0.899,0.31,-0.31],[0,0.967,-0.255],[0.339,-0.665,0.665],[0.828,-0.561,0],[0.665,-0.665,0.339],[0.31,-0.899,0.31],[0.561,-0.828,0],[0.31,-0.31,0.899],[0.665,-0.339,0.665],[0.255,-0.967,0],[0.899,-0.31,0.31],[0.967,-0.255,0],[-0.339,-0.665,0.665],[0,-0.828,0.561],[-0.665,-0.339,0.665],[0,-0.967,0.255],[-0.31,-0.31,0.899],[-0.31,-0.899,0.31],[0,-0.561,0.828],[-0.665,-0.665,0.339],[-0.899,-0.31,0.31],[0,-0.255,0.967],[-0.665,-0.665,-0.339],[-0.561,-0.828,0],[-0.665,-0.339,-0.665],[-0.255,-0.967,0],[-0.899,-0.31,-0.31],[-0.31,-0.899,-0.31],[-0.828,-0.561,0],[-0.339,-0.665,-0.665],[-0.31,-0.31,-0.899],[-0.967,-0.255,0],[0.339,-0.665,-0.665],[0,-0.828,-0.561],[0.828,0,-0.561],[0.665,-0.339,-0.665],[0,-0.967,-0.255],[0.31,-0.31,-0.899],[0.561,0,-0.828],[0.31,-0.899,-0.31],[0,-0.561,-0.828],[0.665,-0.665,-0.339],[0.255,0,-0.967],[0.899,-0.31,-0.31],[0,-0.255,-0.967],[0.967,0,-0.255]])
faces  = numpy.array([[44,38,35],[41,35,36],[41,44,35],[7,14,8],[55,94,90],[51,4,98],[13,11,6],[17,13,6],[13,7,8],[11,13,8],[46,51,49],[38,46,49],[44,46,38],[46,44,37],[29,32,23],[27,21,22],[25,27,22],[63,15,9],[61,69,58],[10,12,7],[12,14,7],[47,57,0],[42,47,26],[35,42,36],[93,94,98],[53,51,98],[48,53,55],[94,53,98],[53,94,55],[51,53,49],[53,48,49],[50,48,55],[14,50,8],[50,14,52],[48,50,52],[86,43,37],[43,46,37],[43,4,51],[46,43,51],[44,45,37],[45,41,34],[41,45,44],[101,1,11],[11,1,6],[13,16,7],[16,13,17],[16,17,9],[16,10,7],[41,39,34],[39,32,29],[39,41,36],[32,39,36],[31,20,72],[20,31,27],[25,20,27],[70,31,72],[31,70,23],[31,30,27],[30,21,27],[30,31,23],[32,30,23],[17,64,9],[64,63,9],[63,64,58],[73,79,75],[12,19,14],[14,19,52],[57,19,0],[19,57,52],[28,10,22],[28,12,10],[21,28,22],[28,21,26],[54,38,49],[48,54,49],[54,48,52],[57,54,52],[38,40,35],[40,57,47],[40,42,35],[42,40,47],[54,40,38],[40,54,57],[21,24,26],[24,42,26],[42,24,36],[30,24,21],[24,32,36],[24,30,32],[4,100,98],[100,93,98],[100,43,86],[43,100,4],[56,101,11],[56,50,55],[56,55,90],[101,56,90],[56,11,8],[50,56,8],[25,18,15],[16,18,10],[10,18,22],[18,25,22],[15,18,9],[18,16,9],[87,3,29],[3,87,34],[39,3,34],[3,39,29],[2,25,15],[2,20,25],[68,70,72],[70,68,75],[68,73,75],[73,68,69],[76,87,29],[76,70,75],[76,29,23],[70,76,23],[94,91,90],[91,93,88],[93,91,94],[67,1,101],[1,67,6],[71,73,69],[71,69,61],[83,78,79],[78,83,85],[81,79,73],[81,71,5],[71,81,73],[81,83,79],[92,81,5],[83,81,92],[28,33,12],[19,33,0],[33,19,12],[33,28,26],[33,47,0],[47,33,26],[89,95,88],[95,89,92],[83,89,85],[89,83,92],[93,96,88],[100,96,93],[96,89,88],[89,96,85],[85,96,86],[96,100,86],[78,84,79],[79,84,75],[84,76,75],[76,84,87],[87,82,34],[82,45,34],[84,82,87],[82,84,78],[45,80,37],[80,78,85],[80,86,37],[80,85,86],[82,80,45],[80,82,78],[20,77,72],[2,77,20],[63,77,15],[77,2,15],[97,91,88],[95,97,88],[71,65,5],[65,71,61],[65,92,5],[65,95,92],[67,66,6],[66,17,6],[66,64,17],[66,67,59],[68,74,69],[69,74,58],[74,68,72],[77,74,72],[74,63,58],[74,77,63],[99,67,101],[97,99,91],[99,101,90],[91,99,90],[67,99,59],[99,97,59],[62,97,95],[65,62,95],[97,62,59],[62,65,61],[66,60,64],[60,66,59],[64,60,58],[62,60,59],[60,61,58],[60,62,61]], dtype=int)
points[:,0] *= Sx
points[:,1] *= Sy
points[:,2] *= Sz
# vertices of all triangular faces
fvert  = numpy.vstack([points[f] for f in faces]).reshape(len(faces), 3, 3)
fcent  = numpy.mean(fvert, axis=1)
fcolor = getColor(fcent)
fveca  = fvert[:,0]-fvert[:,1]
fvecb  = fvert[:,2]-fvert[:,1]
# normal vectors of all triangular faces
fnorm  = numpy.column_stack((
    fveca[:,1]*fvecb[:,2] - fveca[:,2]*fvecb[:,1],
    fveca[:,2]*fvecb[:,0] - fveca[:,0]*fvecb[:,2],
    fveca[:,0]*fvecb[:,1] - fveca[:,1]*fvecb[:,0] ))
fnorm /= (numpy.sum(fnorm**2, axis=1)**0.5)[:,None]

def getEllipse(Sxp, Syp, eta):
    angles = numpy.linspace(0, 2*pi, 49)
    xp = Sxp * numpy.hstack((cos(angles), 0, 0, 0))
    yp = Syp * numpy.hstack((sin(angles), 0, 1, 0))
    return numpy.column_stack((xp * sin(eta) + yp * cos(eta), xp * cos(eta) - yp * sin(eta)))

def clip(x):
    return numpy.maximum(0, (1-1/(1+x**5))**0.2)   # soft clipping into the interval [0..1]

def traceEllipsoid(alpha, beta, gamma):
    R = agama.makeRotationMatrix(alpha, beta, gamma)
    # ray tracing - find the (smallest) Z coordinate of ellipsoid for each point in the X,Y grid
    Q = numpy.einsum('ij,j,kj->ik', R, [Sx**-2, Sy**-2, Sz**-2], R)
    a =  Q[2,2]
    b = (Q[0,2]+Q[2,0]) * X + (Q[1,2]+Q[2,1]) * Y
    c =  Q[0,0] * X**2 + (Q[0,1]+Q[1,0]) * X*Y + Q[1,1] * Y**2 - 1
    d = 0.25*b*b - a*c
    Z = -0.5*b/a - numpy.abs(d**0.5/a)
    xyz = numpy.dot(numpy.column_stack((X,Y,Z)), R)
    pcolor   = getColor(xyz)
    pnorm    = xyz * numpy.array([Sx**-2, Sy**-2, Sz**-2])
    pnorm   /= (numpy.sum(pnorm**2, axis=1)**0.5)[:,None]
    angle    = numpy.maximum(0, -numpy.dot(pnorm, R[2]))
    intensity= ampScattered + ampDirection * angle + ampSpecular * angle**powSpecular
    return numpy.column_stack((clip(pcolor * intensity[:,None]),
        numpy.isfinite(Z))).reshape(len(gridc), len(gridc), 4)

def drawproj():
    Sxp, Syp, eta = agama.getProjectedEllipse(Sx, Sy, Sz, alpha, beta, gamma)
    R = agama.makeRotationMatrix(alpha, beta, gamma)
    prjfaces = numpy.dot(fvert.reshape(-1,3), R.T).reshape(-1, 3, 3)
    angle    = numpy.maximum(0, -numpy.dot(fnorm, R[2]))
    intensity= ampScattered + ampDirection * angle + ampSpecular * angle**powSpecular
    prjcolor = clip(fcolor * intensity[:,None])
    meanz    = numpy.mean(prjfaces[:,:,2], axis=1)
    order    = numpy.argsort(-meanz)
    figfaces.set_paths([matplotlib.patches.Polygon(f[:,0:2], closed=True) for f in prjfaces[order]])
    figfaces.set_facecolor(prjcolor[order])
    figfaces.set_edgecolor('gray')
    figarrowx.set_xy(numpy.vstack((Sx*R[0:2,0], 2*R[0:2,0])))
    figarrowy.set_xy(numpy.vstack((Sy*R[0:2,1], 2*R[0:2,1])))
    figarrowz.set_xy(numpy.vstack((Sz*R[0:2,2], 2*R[0:2,2])))
    figarrowx.set_zorder(-10 if R[2,0]>0 else 10)
    figarrowy.set_zorder(-10 if R[2,1]>0 else 10)
    figarrowz.set_zorder(-10 if R[2,2]>0 else 10)
    figarrowx.set_linestyle('dotted' if R[2,0]>0 else 'solid')
    figarrowy.set_linestyle('dotted' if R[2,1]>0 else 'solid')
    figarrowz.set_linestyle('dotted' if R[2,2]>0 else 'solid')
    figlabelx.set_x(2.2*R[0,0])
    figlabelx.set_y(2.2*R[1,0])
    figlabely.set_x(2.2*R[0,1])
    figlabely.set_y(2.2*R[1,1])
    figlabelz.set_x(2.2*R[0,2])
    figlabelz.set_y(2.2*R[1,2])
    figellips.set_xy(getEllipse(Sxp, Syp, eta))
    PAprojx = numpy.arctan2(R[0,0], R[1,0])
    PAprojy = numpy.arctan2(R[0,1], R[1,1])
    PAprojz = numpy.arctan2(R[0,2], R[1,2])
    figlabela.set_text('alpha=%.2f, beta=%.2f, gamma=%.2f' % (alpha*180/pi, beta*180/pi, gamma*180/pi) +
        '\nPAprojx=%.2f, PAprojy=%.2f, PAprojz=%.2f' % (PAprojx*180/pi, PAprojy*180/pi, PAprojz*180/pi) +
        '\nmajor=%.3f, minor=%.3f, PAmajor=%.2f' % (Sxp, Syp, eta*180/pi))
    try:
        getSx, getSy, getSz = agama.getIntrinsicShape(Sxp, Syp, eta, alpha, beta, gamma)
        figlabeld.set_text('deprojected A=%.9f, B=%.9f, C=%.9f' % (getSx, getSy, getSz))
        figlabeld.set_color('k' if abs(getSx-Sx)+abs(getSy-Sy)+abs(getSz-Sz) < 1e-8 else 'r')
    except Exception as e:
        traceback.print_exc()
        figlabeld.set_text(str(e))
        figlabeld.set_color('r')
    # plot four possible deprojections
    try:
        cax.cla()
        cax.set_axis_off()
        getang = agama.getViewingAngles(Sxp, Syp, eta, Sx, Sy, Sz)
        images = []
        arrowf = []
        arrowb = []
        trueR  = R  # rotation matrix corresponding to the true orientation
        for i in range(4):
            getAlpha, getBeta, getGamma = getang[i]
            getSxp, getSyp, geteta = agama.getProjectedEllipse(Sx, Sy, Sz, getAlpha, getBeta, getGamma)
            R = agama.makeRotationMatrix(getAlpha, getBeta, getGamma)
            # check which orientation is equivalent to the true one, i.e. when the rotation matrix is
            # "almost" the same as the true one (up to a simultaneous change of sign in two axes),
            # the product of R and inverse of true R is a diagonal matrix with +-1 on diagonal and det=1.
            RtR = numpy.dot(trueR.T, R)
            tru = numpy.linalg.norm(numpy.abs(RtR) - numpy.eye(3)) < 1e-8
            off = numpy.array([ ((i%2)*2-1)*Sx, ((i//2)*2-1)*Sx ])
            sign = numpy.where(R[2]<=0, 1, -1)
            arrowf += [matplotlib.patches.Polygon(numpy.vstack((Sx*R[0:2,0], 2*R[0:2,0])) * sign[0] + off,
                edgecolor='r', fill=False, closed=False, linestyle='solid')]
            arrowb += [matplotlib.patches.Polygon(numpy.vstack((Sx*R[0:2,0], 2*R[0:2,0])) *-sign[0] + off,
                edgecolor='r', fill=False, closed=False, linestyle='dotted')]
            arrowf += [matplotlib.patches.Polygon(numpy.vstack((Sy*R[0:2,1], 2*R[0:2,1])) * sign[1] + off,
                edgecolor='g', fill=False, closed=False, linestyle='solid')]
            arrowb += [matplotlib.patches.Polygon(numpy.vstack((Sy*R[0:2,1], 2*R[0:2,1])) *-sign[1] + off,
                edgecolor='g', fill=False, closed=False, linestyle='dotted')]
            arrowf += [matplotlib.patches.Polygon(numpy.vstack((Sz*R[0:2,2], 2*R[0:2,2])) * sign[2] + off,
                edgecolor='b', fill=False, closed=False, linestyle='solid')]
            arrowb += [matplotlib.patches.Polygon(numpy.vstack((Sz*R[0:2,2], 2*R[0:2,2])) *-sign[2] + off,
                edgecolor='b', fill=False, closed=False, linestyle='dotted')]
            cax.add_artist(matplotlib.patches.Ellipse(off, getSxp*2, getSyp*2, angle=(pi/2-eta)*180/pi,
                fill=False, edgecolor='k', clip_on=False))
            cax.text(off[0]+Sx, off[1]+Sx,
                'alpha=%.9f\nbeta=%.9f\ngamma=%.9f' % (getAlpha*180/pi, getBeta*180/pi, getGamma*180/pi),
                color='g' if tru else 'k' if abs(getSxp-Sxp)+abs(getSyp-Syp)+abs(sin(geteta-eta)) < 1e-8 else 'r',
                ha='left', va='top', fontsize=12)
            images += [traceEllipsoid(getAlpha, getBeta, getGamma)]
        im = numpy.hstack(numpy.hstack(numpy.array(images).reshape(2, 2, len(gridc), len(gridc), 4)))
        cax.imshow(im, extent=[-2*Sx, 2*Sx, -2*Sx, 2*Sx], origin='lower', interpolation='nearest', aspect='auto', zorder=0)
        arrowsf = matplotlib.collections.PatchCollection(arrowf, match_original=True)
        arrowsf.set_zorder(1)
        arrowsf.set_clip_on(False)
        arrowsb = matplotlib.collections.PatchCollection(arrowb, match_original=True)
        arrowsb.set_zorder(-1)
        arrowsb.set_clip_on(False)
        cax.add_artist(arrowsf)
        cax.add_artist(arrowsb)
        cax.set_xlim( 2*Sx,-2*Sx)
        cax.set_ylim(-2*Sx, 2*Sx)
    except Exception as e:
        traceback.print_exc()
        cax.text(0, 0, str(e), color='r', ha='center', va='center')
    # repaing the whole figure
    plt.draw()

def onmousepress(event):
    if event.xdata is None or event.button is None or event.inaxes!=ax: return
    global mousex, mousey
    mousex = event.xdata
    mousey = event.ydata

def onmousemove(event):
    if event.xdata is None or event.button is None or event.inaxes!=ax: return
    global alpha, beta, gamma, mousex, mousey
    if event.button==1:  # change inclination and/or intrinsic rotation
        deltax = (event.xdata - mousex) * pi/2
        deltay = (event.ydata - mousey) * pi/2
        alpha += deltax * cos(gamma) - deltay * sin(gamma)
        beta  -= deltay * cos(gamma) + deltax * sin(gamma)
    if event.button==3:  # change rotation in the image plane
        gamma += numpy.arctan2(event.xdata, event.ydata) - numpy.arctan2(mousex, mousey)
    # normalize angles to the default range
    while beta > pi: beta -= 2*pi
    if beta < 0: beta = -beta; alpha += pi; gamma += pi;
    while alpha >   pi: alpha -= 2*pi
    while alpha <= -pi: alpha += 2*pi
    while gamma >   pi: gamma -= 2*pi
    while gamma <= -pi: gamma += 2*pi
    mousex = event.xdata
    mousey = event.ydata
    drawproj()

def new_home_button(self, *args, **kwargs):
    # reset orientation to default when home button is pressed
    global alpha, beta, gamma
    alpha = 0; beta = 0; gamma = 0;
    drawproj()
    home_button(self, *args, **kwargs)

home_button = matplotlib.backend_bases.NavigationToolbar2.home
matplotlib.backend_bases.NavigationToolbar2.home = new_home_button
fig = plt.figure(figsize=(15, 7.5), dpi=80)
ax  = plt.axes([0.04, 0.08, 0.45, 0.9])
cax = plt.axes([0.52, 0.08, 0.45, 0.9])
ax.set_xlim(3, -3)
ax.set_ylim(-3, 3)
ax.set_xlabel('X')
ax.set_ylabel('Y')
figfaces  = matplotlib.collections.PatchCollection([matplotlib.patches.Polygon([[1,0], [0,0], [0,1]])])
figarrowx = matplotlib.patches.Polygon([[0,0], [0,0]], closed=False, color='r')
figarrowy = matplotlib.patches.Polygon([[0,0], [0,0]], closed=False, color='g')
figarrowz = matplotlib.patches.Polygon([[0,0], [0,0]], closed=False, color='b')
figellips = matplotlib.patches.Polygon([[0,0], [0,0]], closed=False, color='k', fill=False, zorder=9)
ax.add_artist(figfaces)
ax.add_patch(figarrowx)
ax.add_patch(figarrowy)
ax.add_patch(figarrowz)
ax.add_patch(figellips)
figlabelx = ax.text(0, 0, 'x',  ha='center', va='center', fontsize=12, color='r')
figlabely = ax.text(0, 0, 'y',  ha='center', va='center', fontsize=12, color='g')
figlabelz = ax.text(0, 0, 'z',  ha='center', va='center', fontsize=12, color='b')
figlabela = ax.text(2.9, 2.9, '', ha='left', va='top',    fontsize=12)
figlabeld = ax.text(2.9,-2.9, '', ha='left', va='bottom', fontsize=12)
fig.canvas.mpl_connect("button_press_event", onmousepress)
fig.canvas.mpl_connect("motion_notify_event", onmousemove)
drawproj()
print(__docstring__)
plt.show()
