#!/usr/bin/python
from __future__ import print_function
import numpy, sys, re, matplotlib, matplotlib.pyplot as plt
from matplotlib.patches import Polygon,Circle
from matplotlib.colors import LogNorm
from matplotlib.collections import PatchCollection
import pygama as agama

if len(sys.argv)<=1:
    print("Provide file name")
    exit()
filename  = sys.argv[1]
zooming   = False
patchcoll = []
panels    = []

radiolabels = ('Data values', 'Data errors', 'Model errors')
def clickradio(button):
    for p, patch in enumerate(patchcoll):
        if button == buttons[0]:
            patch.set_cmap('jet')
            data = ghm[:,(3+2*p if p<=2 else 9+2*p)] if p!=0 else numpy.log(ghm[:,3]/ghm[:,2])
            patch.set_array(data)
            if p>=3:
                patch.set_clim([-0.1,0.1])  # color scale for h3,h4 and beyond
            #elif p==1:
            #    patch.set_clim(-30,30)
            #elif p==2:
            #    patch.set_clim(60,120)
            else:
                patch.set_clim(min(data), max(data))
        elif button == buttons[1]:
            patch.set_cmap('PuBu')
            patch.set_array(ghm[:,10+2*p])
            patch.set_clim(0, max(ghm[:,10+2*p]))
            #patch.set_clim(0, 0.05 if p<=3 else 0.02)
        else:
            patch.set_cmap('RdBu')
            patch.set_array((modelghm[:,p] - ghm[:,9+2*p]) / ghm[:,10+2*p])
            patch.set_clim([-2,2])
    plt.draw()

def plotApertures(axis, title):
    patches = PatchCollection([Polygon(p, True) for p in poly], alpha=1.0, picker=0.0, edgecolor='gray')
    patchcoll.append(patches)
    panels.append(axis)
    axis.add_collection(patches)
    axis.set_xlim(viewxmin, viewxmax)
    axis.set_ylim(viewymin, viewymax)
    axis.text(0.05, 0.9, title, fontsize=16, transform=axis.transAxes)

def onclick(event):
    if event.artist in buttons:
        clickradio(event.artist)
        return
    toolbar = plt.get_current_fig_manager().toolbar
    if toolbar.mode!='':
        print("You clicked on something, but toolbar is in mode {:s}.".format(toolbar.mode))
        return
    else:
        HasPoly, IndPoly = event.artist.contains(event.mouseevent)
        if HasPoly and len(IndPoly["ind"]) == 1:
            ind = IndPoly["ind"][0]
            losvdplot.cla()
            # plot the Gauss-Hermite approximation
            gorig = ghm[ind,4]  # original amplitude (from LOSVD itself)
            gamma = ghm[ind,3]  # amplitude used in the model (recomputed from the density profile)
            v_0   = ghm[ind,5]
            sigma = ghm[ind,7]
            losvdplot.plot(vfinegrid,
                agama.GaussHermite(gamma, v_0, sigma, coefs=ghm[ind,9::2], xarr=vfinegrid), 'r--')
            # plot only the Gaussian approximation (no higher moments)
            losvdplot.plot(vfinegrid, agama.GaussHermite(gamma, v_0, sigma, coefs=None, xarr=vfinegrid),
                'purple', lw=0.75)[0].set_dashes([2,2])
            # plot the interpolated function defined by B-spline amplitudes in the 'ind'-th row,
            # shading the error estimate of LOSVD.
            # Note: rescale the amplitude to match the one from density profile
            if numpy.all(los[ind,1::2]==0): losvdplot.plot(vfinegrid,
                agama.bsplInt(degree, vgrid, los[ind, 0::2], vfinegrid) * gamma/gorig,
                color='b', alpha=0.33, lw=2)
            else: losvdplot.fill_between(vfinegrid,
                agama.bsplInt(degree, vgrid, los[ind, 0::2]-los[ind, 1::2], vfinegrid) * gamma/gorig,
                agama.bsplInt(degree, vgrid, los[ind, 0::2]+los[ind, 1::2], vfinegrid) * gamma/gorig,
                facecolor='b', alpha=0.33, lw=0)
            # plot the model LOSVD
            if not modellos is None:
                losvdplot.plot(vfinegrid, \
                    agama.bsplInt(degree, vgridm, modellos[ind], vfinegrid), 'g')[0].set_dashes([5,2])
            losvdplot.set_xlim(vfinegrid[0], vfinegrid[-1])
            # print some useful info
            print("Aperture #%i centered at x=%f, y=%f: v_0=%f +- %f  sigma=%f +- %f " % \
                (ind, ghm[ind,0], ghm[ind,1], ghm[ind,5], ghm[ind,6], ghm[ind,7], ghm[ind,8]), end='')
            for i in range(len(ghm[ind,9::2])):
                print("h%i=%f +- %f " % (i, ghm[ind,9+2*i], ghm[ind,10+2*i]), end='')
                if not modelghm is None:
                    err = (modelghm[ind,i]-ghm[ind,9+2*i]) / ghm[ind,10+2*i]
                    if err < -1.:  print("[\033[1;31m %f \033[0m] " % modelghm[ind,i], end='')
                    elif err> 1.:  print("[\033[1;34m %f \033[0m] " % modelghm[ind,i], end='')
                    else:  print("[ %f ] " % modelghm[ind,i], end='')
            print('')
            # highlight the selected polygon in all panels (make its boundary thicker)
            lw = numpy.ones(len(poly))
            lw[ind] = 3.
            for p in patchcoll: p.set_linewidths(lw)
            plt.draw()
        else: print("Can't understand where you clicked:", HasPoly, IndPoly)

# read polygon file, where each block of several lines separated by empty line is stored to a separate array
with open(filename+".pol") as dfile:
    poly = [numpy.array([float(a) for a in b.split()]).reshape(-1,2) for b in dfile.read().split("\n\n")]
viewxmin = min([numpy.amin(p[:,0]) for p in poly])
viewxmax = max([numpy.amax(p[:,0]) for p in poly])
viewymin = min([numpy.amin(p[:,1]) for p in poly])
viewymax = max([numpy.amax(p[:,1]) for p in poly])

# read the first line of LOSVD amplitudes file, which defines the grid in velocity and the degree of B-spline
with open(filename+".los") as lfile:
    header = lfile.readline()
    degree = int(re.search('Degree: (\d+)', header).group(1))
    vgrid = numpy.array([float(a) for a in re.search('Grid: (.*)$', header).group(1).split()])
    vfinegrid = numpy.linspace(vgrid[0], vgrid[-1], 201)
# read the remaining file into a 2d array: rows are apertures,
# columns - amplitudes of B-splines in each aperture and their error estimates
los = numpy.loadtxt(filename+".los")

# read the file with Gauss-Hermite moments
ghm = numpy.loadtxt(filename+".ghm")
ghorder = (ghm.shape[1]-11)//2
if numpy.all(ghm[:,4]==0): ghm[:,4] = ghm[:,3]
# read the model LOSVDs
if len(sys.argv)>=3:
    modelfilename = sys.argv[2]
    with open(modelfilename) as lfile:
        header = lfile.readline()
        #degree = int(re.search('Degree: (\d+)', header).group(1))
        vgridm = numpy.array([float(a) for a in re.search('Grid: (.*)$', header).group(1).split()])
    modellos = numpy.loadtxt(modelfilename)
    modelghm = agama.ghmoments(matrix=modellos.reshape(-1), gridv=vgridm, degree=degree, \
        ghorder=ghorder, ghexp=ghm[:,(3,5,7)]).reshape(modellos.shape[0],-1)
    # compute chi^2 in the model
    print("Chi^2 = %f" % numpy.sum(((modelghm-ghm[:,9::2]) / ghm[:,10::2])**2))
else:
    modellos = None
    modelghm = None

fig = plt.figure(figsize=(12,8))
plotApertures(plt.axes([0.25, 0.5, 0.24, 0.5]), r'$\Sigma$')
plotApertures(plt.axes([0.50, 0.5, 0.24, 0.5], sharex=panels[0], sharey=panels[0]), r'$\langle v \rangle$')
plotApertures(plt.axes([0.75, 0.5, 0.24, 0.5], sharex=panels[0], sharey=panels[0]), r'$\sigma$')
plotApertures(plt.axes([0.00, 0.0, 0.24, 0.5], sharex=panels[0], sharey=panels[0]), r'$h_3$')
plotApertures(plt.axes([0.25, 0.0, 0.24, 0.5], sharex=panels[0], sharey=panels[0]), r'$h_4$')
if ghorder>=6:
    plotApertures(plt.axes([0.50, 0.0, 0.24, 0.5], sharex=panels[0], sharey=panels[0]), r'$h_5$')
    plotApertures(plt.axes([0.75, 0.0, 0.24, 0.5], sharex=panels[0], sharey=panels[0]), r'$h_6$')
radioplot  =  plt.axes([0.00, 0.5, 0.24, 0.1])
losvdplot  =  plt.axes([0.00, 0.6, 0.24, 0.4])
buttons = [
    Circle((0.5,0),0.4,picker=True,color='#60f080'), \
    Circle((1.5,0),0.4,picker=True,color='#ffe000'), \
    Circle((2.5,0),0.4,picker=True,color='#ff6080') ]
for b in buttons: radioplot.add_artist(b)
radioplot.text(1.5, 0, filename, ha='center', va='center')
radioplot.set_xlim(0,3)
radioplot.set_ylim(-.5,.5)
clickradio(buttons[0])
for axis,patch in zip(panels,patchcoll):
    fig.colorbar(patch, ax=axis, orientation='horizontal')
fig.canvas.mpl_connect('pick_event', onclick)
plt.show()
