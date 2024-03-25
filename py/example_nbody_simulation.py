#!/usr/bin/python
"""
This example demonstrates the use of Agama as a source of external gravitational potential
in several N-body simulation codes: GyrfalcON (from NEMO), Gadget4, Arepo.
The first of these codes uses the agama.so shared library directly as a plug-in, whereas
the other two need some patches and recompilation; this script attempts to do this automatically.
In additional, a "restricted N-body simulation" method that uses the built-in tools from Agama
can be used as a faster (but more approximate) alternative: it evolves the collection of orbits
in the moving potential of the cluster on a prescribed trajectory in the host galaxy, updating
this cluster potential periodically to account for the mass loss due to tidal stripping.
This method is also illustrated by the script "example_tidal_stream.py".
The host galaxy potential mimics the Milky Way with a rotating bar, and is created separately
using the script "example_mw_potential_hunter24.py". The cluster is placed on an orbit within
the bar region, which leads to the stripping of ~1/3 of its mass by the end of the simulation
(which takes a few minutes).
"""
import sys, agama, numpy, matplotlib.pyplot as plt

# use the "N-body" units, in which G=1:
# length = 1 kpc, velocity = 1 km/s, mass = 232500 Msun

# create an isolated star cluster
c_pot = agama.Potential(type='plummer', scaleRadius=0.01, mass=1e6 / 232500)
c_df  = agama.DistributionFunction(type='quasispherical', potential=c_pot)
c_gm  = agama.GalaxyModel(c_pot, c_df)
N     = 10000
xv, m = c_gm.sample(N)
# shift it to some initial point in the Galaxy
c_center = numpy.hstack([2.0, 0, 0, 0, -100, 50])
xv   += c_center
sim_time = 0.25  # total simulation time in units of kpc/(km/s) = 0.98 Gyr

potfile = 'MWPotentialHunter24_rotating.ini'
try:
    g_pot = agama.Potential(potfile)
except RuntimeError:
    print('You need to create the barred Milky Way potential by running the script "example_mw_potential_hunter24.py"')
    exit(1)
print('''Initial conditions for a cluster in the galaxy are created, what to do next?
  1 - show the orbit of the cluster center as a test particle
  2 - run the simulation of a disrupting cluster using the Agama-only "restricted N-body" approach
  3 - run the simulation using the GyrfalcON code (from existing NEMO installation)
  4 - run the simulation using the Gadget4 code (downloading and compiling if necessary)
  5 - run the simulation using the Arepo code (downloading and compiling if necessary)
  any other key - exit''')

def runSingleOrbit():
    o = agama.orbit(potential=g_pot, ic=c_center, time=sim_time, trajsize=500)[1]
    plt.figure(figsize=(10,10))
    plt.plot(o[:,0], o[:,1], 'k')
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.show()


def runRestrictedNbody():
    plt.ion()
    plt.figure(figsize=(10,10))
    ax = plt.axes([0.08, 0.08, 0.9, 0.9])
    num_intervals = 32
    num_subint = 16
    interval = sim_time / num_intervals
    time_center, orbit_center = agama.orbit(potential=g_pot, ic=c_center, time=sim_time,
        trajsize=num_intervals * num_subint + 1)
    cpot = c_pot
    snap = xv.copy()
    print('Playing animation as the simulation progresses...')
    for i in range(num_intervals+1):
        # determine which particles remain bound to the satellite
        c_bound = c_pot.potential(snap[:,0:3] - orbit_center[i*num_subint, 0:3]) + 0.5 * numpy.sum((snap[:,3:6] - orbit_center[i*num_subint, 3:6])**2, axis=1) < 0
        time = i * interval
        ax.cla()
        ax.scatter(snap[:,0], snap[:,1], c=c_bound, cmap='bwr_r', vmin=0, vmax=1, s=2, linewidths=0)
        ax.plot(orbit_center[0:i*num_subint, 0], orbit_center[0:i*num_subint, 1], color='k')
        ax.text(0.01, 0.99, 'time=%.4f, bound fraction=%.3f' % (time, numpy.sum(c_bound)*1./N), ha='left', va='top', transform=ax.transAxes)
        grid = numpy.linspace(-2.5, 2.5, 101)
        xyz = numpy.column_stack([numpy.repeat(grid,len(grid)), numpy.tile(grid,len(grid)), numpy.zeros(len(grid)**2)])
        den = g_pot.density(xyz, t=time).reshape(len(grid), len(grid)).T
        ax.contour(grid, grid, numpy.log10(den), levels=numpy.linspace(3, 7, 17), cmap='earth_r', zorder=2)
        ax.set_xlim(min(grid), max(grid))
        ax.set_ylim(min(grid), max(grid))
        plt.draw()
        plt.pause(0.01)
        if i == num_intervals: continue
        # evolve the cluster for some time:
        # initialize the time-dependent total potential (host + moving sat) on this time interval
        t_pot = agama.Potential(g_pot,
                agama.Potential(potential=cpot, center=numpy.column_stack((time_center, orbit_center))))
        # compute the trajectories of all particles moving in the combined potential of the host galaxy and the moving satellite
        snap = numpy.vstack(agama.orbit(ic=snap, potential=t_pot, time=interval, timestart=time, trajsize=1, accuracy=1e-5)[:,1])
        # update the potential of the satellite (using a spherical monopole approximation)
        cpot = agama.Potential(type='multipole', particles=(snap[:,0:3] - orbit_center[(i+1)*num_subint, 0:3], m), symmetry='s')
    filename = 'example_nbody_simulation_last.nemo'
    agama.writeSnapshot(filename, (snap,m), 'nemo')
    print('Saved the final snapshot to %s' % filename)
    plt.ioff()
    plt.show()


def runGyrfalcon():
    import os, subprocess
    path = 'gyrfalcon'
    if not os.path.isdir(path):
        os.mkdir(path)
    os.chdir(path)
    infile = 'IC.nemo'
    print('*** Writing the initial conditions into %s' % infile)
    agama.writeSnapshot(infile, (xv, m), 'nemo')
    outfile = 'output.nemo'
    endfile = 'output_last.nemo'
    if os.path.isfile(outfile):
        os.remove(outfile)
    if os.path.isfile(endfile):
        os.remove(endfile)
    runstr = 'gyrfalcON %s %s eps=0.001 kmax=14 Nlev=3 fea=0.5 step=0.001953125 tstop=0.25 accname=agama accfile=%s' % (infile, outfile, '../'+potfile)
    print('*** Launching the simulation (end time: %g):\n%s' % (sim_time, runstr))
    sub = subprocess.check_call(runstr, shell=True)
    if os.path.isfile(outfile):
        print('*** Done! You may play the simulation movie using "glnemo %s"' % outfile)
        # extract and show the last snapshot
        subprocess.check_call('s2s %s %s times=%g' % (outfile, endfile, sim_time), shell=True)
        showSnapshot(endfile)
    else:
        print('No output files produced, try running the above command manually to see the error messages')


def runArepoOrGadget(code):
    import os, platform, subprocess, zipfile, shutil, multiprocessing
    if sys.version_info.major == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve
    path = code
    if os.path.isdir(path):
        print('*** Using the existing installation of %s' % code)
    else:
        print('*** Downloading %s' % code)
        filename = code + '.zip'
        # download a specific version of the codes, which is known to work;
        # one may use the same approach with the latest version or a private branch, but the patch might need to be adjusted
        if code == 'arepo':
            url = 'https://gitlab.mpcdf.mpg.de/vrs/arepo/-/archive/73d6bc4821daece021f02c0ae5da834c1d05b3c7/arepo-73d6bc4821daece021f02c0ae5da834c1d05b3c7.zip'
        elif code == 'gadget4':
            url = 'https://gitlab.mpcdf.mpg.de/vrs/gadget4/-/archive/6134834915dda11edbf76bf34e1d3e2736867c60/gadget4-6134834915dda11edbf76bf34e1d3e2736867c60.zip'
        urlretrieve(url, filename)
        if not os.path.isfile(filename):
            raise RuntimeError('Cannot find downloaded file %s' % filename)
        with zipfile.ZipFile(filename, 'r') as f:  # unpack the archive, manually setting the file attributes
            for info in f.infolist():
                extr = f.extract(info)
                attr = info.external_attr >> 16
                if attr:
                    os.chmod(extr, attr)
        os.remove(filename)  # remove the downloaded archive
        folder = url[url.rfind('/')+1:-4]
        os.rename(folder, path)  # simplify the folder name, removing the tag id

    # apply the patch file if necessary
    with open(path + '/Template-Config.sh', 'r') as f:
        lines = ''.join(f.readlines())
        if 'AGAMA' in lines:
            print('*** Code is already patched')
        else:
            print('*** Patching %s' % code)
            subprocess.check_call('patch -r -u -N -d %s -p 1 < example_nbody_simulation_%s.patch' % (path, code), shell=True)
            # another quirk is that sometimes one needs to link the executable against the Python shared library,
            # which is used by agama.so but not always loaded automatically.
            # For this, we take EXE_FLAGS from the Makefile.local
            exe_flags = ''
            with open('../Makefile.local', 'r') as mf:
                for line in mf.readlines():
                    if line.startswith('EXE_FLAGS'):
                        ind = line.find('-Wl,-rpath,')
                        if ind>0:
                            exe_flags = line[ind:].strip()
            if exe_flags:
                with open(path + '/Makefile', 'r') as mf:
                    lines = mf.readlines()
                    for ind, line in enumerate(lines):
                        if line.startswith('LIBS += agama.so'):
                            print('*** Adding extra link flags to %s/Makefile: "%s"' % (path, exe_flags))
                            lines[ind] = line.replace('agama.so', 'agama.so ' + exe_flags)
                with open(path + '/Makefile', 'w') as mf:
                    mf.write(''.join(lines))

    # compile
    os.chdir(path)
    executable = code[0].upper() + code[1:]
    if not os.path.isfile(executable):
        if not os.path.isfile('Makefile.systype'):
            with open('Makefile.systype', 'w') as f:
                if platform.uname()[0]=='Darwin':
                    systype = 'Darwin'
                else:
                    systype = 'Generic-gcc' if code == 'gadget4' else 'Ubuntu'
                f.write('SYSTYPE="%s"\n' % systype)
        # macros written into Config.sh before compilation
        macros = ['SELFGRAVITY', 'EXTERNALGRAVITY', 'EXTERNALGRAVITY_AGAMA']
        if code == 'gadget4':
            macros += ['GADGET2_HEADER']
        elif code == 'arepo':
            macros += ['GRAVITY_NOT_PERIODIC']
        with open('Config.sh', 'w') as f:
            f.write('\n'.join(macros)+'\n')
        # create a symlink to agama.so
        if not os.path.exists('agama.so'):
            os.symlink(os.path.abspath('../../agama.so'), 'agama.so')
        print('*** I am going to compile the executable %s, please adjust the files Config.sh, Makefile, Makefile.systype as needed and press any key when ready...' % executable)
        input()
        # initiate compilation
        subprocess.check_call('make', shell=True)

    # final preparations
    infile = 'IC.snap'
    print('*** Writing the initial conditions into %s' % infile)
    agama.writeSnapshot(infile, (xv, m), 'gadget')

    with open('agama_potential.ini', 'w') as f:
        f.write('[Potential]\nfile=../%s\n' % potfile)

    shutil.copyfile('../example_nbody_simulation_%s.param' % code, 'param.txt')

    numproc = max(1, min(8, multiprocessing.cpu_count()))  # determine the number of processor cores for running MPI

    runstr = 'mpirun -np %i ./%s param.txt' % (numproc, executable)
    print('*** Launching the simulation (end time: %g):\n%s' % (sim_time, runstr))
    sub = subprocess.check_call(runstr + ' | grep Time:', shell=True)

    # create a list of output files for easy loading into GLNemo
    if os.path.isdir('output'):
        outputfiles = sorted(['output/'+name for name in os.listdir('output') if name.startswith('snapshot')])
    else:
        outputfiles = []
    if outputfiles:
        with open('output_files.txt', 'w') as f:
            f.write('\n'.join(outputfiles) + '\n')
        print('*** Done! You may play the simulation movie using "glnemo output_files.txt"')
        showSnapshot(outputfiles[-1])
    else:
        print('No output files produced, try running the above command manually to see the error messages')


def showSnapshot(filename):
    pos = agama.readSnapshot(filename)[0]
    plt.figure(figsize=(10,10))
    plt.axes([0.08, 0.08, 0.9, 0.9])
    plt.scatter(pos[:,0], pos[:,1], c='b', s=2, linewidths=0)
    grid = numpy.linspace(-2.5, 2.5, 101)
    xyz = numpy.column_stack([numpy.repeat(grid,len(grid)), numpy.tile(grid,len(grid)), numpy.zeros(len(grid)**2)])
    den = g_pot.density(xyz, t=sim_time).reshape(len(grid), len(grid)).T
    plt.contour(grid, grid, numpy.log10(den), levels=numpy.linspace(3, 7, 17), cmap='earth_r')
    plt.xlim(min(grid), max(grid))
    plt.ylim(min(grid), max(grid))
    plt.show()

if sys.version_info.major == 2:
    input = raw_input
if len(sys.argv)>1:
    choice = sys.argv[1]
    print(choice+'\n')
else:
    choice = input()
if choice == '1': runSingleOrbit()
if choice == '2': runRestrictedNbody()
if choice == '3': runGyrfalcon()
if choice == '4': runArepoOrGadget('gadget4')
if choice == '5': runArepoOrGadget('arepo')
