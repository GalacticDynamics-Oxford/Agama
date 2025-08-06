#!/usr/bin/python
"""
This program constructs orbit-based Schwarzschild models
for systems with arbitrary geometry, density profiles and number of components.
The parameters of a model are provided in an ini file; this program parses the file,
constructs initial conditions, runs orbit integration, solves the optimization problem
to find orbit weights, and finally creates an N-body representation of the model -
separately for each density component defined in the input file.
The model is also stored in a numpy binary file, and the orbit library is also exported
as a text file (initial conditions, weights and integration times).
See an example of input file in ../data/schwarzschild_axisym.ini
"""
import agama, numpy, os, sys, re
try:
    from ConfigParser import RawConfigParser  # python 2
except ImportError:
    from configparser import RawConfigParser  # python 3

def createModel(iniFileName):
    """
    parse the ini file and construct the initial data for the entire model:
    all density, potentials, and Schwarzschild model components,
    with initial conditions for the orbit library and relevant density/kinematic targets.
    """
    model = type('Model', (), {})
    ini = RawConfigParser()
    ini.read(iniFileName)
    sec = ini.sections()
    sec_den = dict()   # list of density components
    sec_pot = dict()   # same for potential
    sec_comp= dict()   # list of model components (each one may include several density objects)
    Omega   = 0
    for s in sec:
        if s.lower().startswith('density'):
            sec_den [s.lower()]  = dict(ini.items(s))
        if s.lower().startswith('potential'):
            sec_pot [s.lower()] = dict(ini.items(s))
        if s.lower().startswith('component'):
            sec_comp[s.lower()] = dict(ini.items(s))
        if s.lower()=='global':   # pattern speed
            for name, value in ini.items(s):
                if name.lower() == 'omega':
                    Omega = float(value)

    # construct all density and potential objects
    den = dict()
    pot = dict()
    for name,value in sec_den.items():
        den[name] = agama.Density(**value)
    for name,value in sec_pot.items():
        if 'density' in value:
            listdens = list()
            for den_str in filter(None, re.split(r'[,\s]', value['density'].lower())):
                if den_str in sec_den:
                    listdens.append(den[den_str])
            if len(listdens) == 1:
                value['density'] = listdens[0]
            elif len(listdens) > 1:
                value['density'] = agama.Density(*listdens)
            # else there are no user-defined density sections
        print("Creating "+name)
        pot[name] = agama.Potential(**value)
        pot[name].export("model_"+name)
    model.density = den
    if len(pot) == 0:
        raise ValueError("No potential components defined")
    if len(pot) == 1:
        model.potential = list(pot.values())[0]
    else:
        model.potential = agama.Potential(*pot.values())

    # determine corotation radius in case of nonzero pattern speed
    if Omega!=0:
        try:
            from scipy.optimize import brentq
            print("Omega=%.3g => corotation radius: %.3g" % (Omega,
                brentq(lambda r: model.potential.force(r,0,0)[0]/r+Omega**2, 1e-10, 1e10, rtol=1e-4)))
        except Exception as e:
            print("Omega=%.3g, corotation radius unknown: %s" % (Omega, str(e)))

    # construct all model components
    if len(sec_comp) == 0:
        raise ValueError("No model components defined")
    model.components = dict()
    for name,value in sec_comp.items():
        targets = list()
        if not 'density' in value:
            raise ValueError("Component "+name+" does not have an associated density")
        listdens = list()
        for den_str in filter(None, re.split(r'[,\s]', value['density'].lower())):
            if den_str in sec_den:
                listdens.append(den[den_str])
            elif den_str in sec_pot:
                listdens.append(pot[den_str])
            else:
                raise ValueError("Unknown density component: "+den_str+" in "+name)
        if len(listdens) == 1:
            density = listdens[0]
        elif len(listdens) > 1:
            density = agama.Density(*listdens)
        else:
            raise ValueError("No density components in "+name)
        print("Creating density target for "+name)
        # pick up only the parameters corresponding to a Density*** target
        targetDensityParams = dict([param for param in value.items() if param[0].lower() in
            ('type', 'gridr', 'gridz', 'lmax', 'mmax', 'stripsperpane', 'axisratioy', 'axisratioz')])
        targets.append(agama.Target(**targetDensityParams))
        if 'kinemgrid' in value:
            targetKinemParams = {
                "type": 'KinemShell',
                "gridr":  eval(value['kinemgrid']),  # an array or an expression, e.g. numpy.linspace(...)
                "degree": int(value['kinemdegree']) }
            print("Creating kinematic target for "+name)
            targets.append(agama.Target(**targetKinemParams))
        if 'numorbits' in value:
            icoptions = { 'n': int(value['numorbits']), 'potential': model.potential }
            if 'icbeta'  in value:
                icoptions['beta' ] = float(value['icbeta'])
            if 'ickappa' in value:
                icoptions['kappa'] = float(value['ickappa'])
            print("Creating initial conditions for %i orbits in %s" % (icoptions['n'], name))
            ic, weightprior = density.sample(**icoptions)
        else:
            raise ValueError("No orbits defined in "+name)
        if 'inttime' in value:
            inttime = float(value['inttime']) * model.potential.Tcirc(ic)
        else:
            raise ValueError("No integration time defined in "+name)
        comp = type('Component', (),
            {"density": density,
             "ic": ic,
             "weightprior": weightprior,
             "inttime": inttime,
             "targets": targets,
             "Omega": Omega} )
        if 'beta' in value:
            # beta can be a single number, an array, or a callable function
            beta = eval(value['beta'])
            if not 'kinemgrid' in value:
                raise ValueError("Anisotropy parameter beta provided without a kinematic grid in "+name)
            if callable(beta):
                # replace a callable function with its result, namely the beta values at grid radii
                if targetKinemParams['degree'] != 1:
                    # otherwise the length of gridr does not match the number of B-spline basis functions
                    raise ValueError("Specifying anisotropy profile beta(r) is only allowed when kinemdegree=1")
                comp.beta = beta(targetKinemParams['gridr'])
            else:  # otherwise it can be a single number or a list/array of appropriate length, which we check
                beta = numpy.array(beta)
                try:
                    maxbeta = max(numpy.zeros(len(targets[-1]) // 2) + beta)
                except Exception:  # likely incorrect length of the array
                    raise ValueError("Anisotropy parameter beta should be a single number or an array of length %i" %
                        (len(targets[-1]) // 2))
                if maxbeta >= 1:
                    raise ValueError("Anisotropy parameter beta must be less than 1")
                comp.beta = beta
        if 'nbody' in value:
            comp.nbody = int(value['nbody'])
        model.components[name] = comp
    return model

def runComponent(comp, pot):
    """
    run the orbit integration, optimization, and construction of an N-body model
    for a given component of the Schwarzschild model
    """
    if hasattr(comp, 'nbody'):  # record orbit trajectories
        result = agama.orbit(potential=pot, ic=comp.ic, time=comp.inttime,
            Omega=comp.Omega, targets=comp.targets, dtype=object)
        traj = result[-1]
    else:
        result = agama.orbit(potential=pot, ic=comp.ic, time=comp.inttime,
            Omega=comp.Omega, targets=comp.targets)
    if isinstance(result, numpy.ndarray):  # in case that only one output was requested (e.g. trajectory),
        result = (result,)                 # the orbit() function returns it rather than a tuple of one element
    # targets[0] is density, targets[1], if provided, is kinematics
    matrix = list()
    rhs    = list()
    rpenl  = list()
    mass   = comp.density.totalMass()
    # density constraints
    matrix.append(result[0].T)
    rhs.   append(comp.targets[0](comp.density))
    avgrhs = mass/len(rhs[0])  # typical magnitude of density constraints
    rpenl. append(numpy.ones_like(rhs[0]) / avgrhs)
    # kinematic constraints
    if len(comp.targets) == 2 and hasattr(comp, 'beta'):
        numrow = len(comp.targets[1]) // 2
        matrix.append((result[1][:,0:numrow] * 2*(1-comp.beta) - result[1][:,numrow:2*numrow]).T)
        rhs.   append(numpy.zeros(numrow))
        rpenl. append(numpy.ones(numrow))
    # total mass constraint
    matrix.append(numpy.ones((len(comp.ic), 1)).T)
    rhs.   append(numpy.array([mass]))
    rpenl. append(numpy.array([numpy.inf]))
    # regularization (penalty for unequal weights)
    avgweight = mass / len(comp.ic)
    xpenq   = numpy.ones(len(comp.ic)) / avgweight**2 / len(comp.ic) * 0.1

    # solve the linear equation for weights
    weights = agama.solveOpt(matrix=matrix, rhs=rhs, rpenl=rpenl, xpenq=xpenq )

    # check for any outstanding constraints
    for t in range(len(matrix)):
        delta = matrix[t].dot(weights) - rhs[t]
        norm  = numpy.where(rhs[t]==0, 1e-6, 1e-3 * abs(rhs[t]))
        for c, d in enumerate(delta):
            if abs(d) > norm[c]:
                print("Constraint %i:%i not satisfied: %s, val=%.4g, dif=%.4g" %
                (t, c, comp.targets[t][c], rhs[t][c], d))
    print("Entropy: %f, # of useful orbits: %i / %i" %
        ( -sum(weights * numpy.log(weights+1e-100)) / mass + numpy.log(avgweight),
        len(numpy.where(weights >= avgweight)[0]), len(comp.ic)))

    # create an N-body model if needed
    if hasattr(comp, 'nbody'):
        status, particles = agama.sampleOrbitLibrary(comp.nbody, traj, weights)
        if not status: raise RuntimeError("Failed to produce output N-body model")
        comp.nbodymodel = particles

    # output
    comp.weights = weights
    comp.densitydata = result[0]
    if len(comp.targets) >= 2:  comp.kinemdata = result[1]

if __name__ == '__main__':
    # read parameters from the INI file
    if len(sys.argv)<=1:
        print("Provide the ini file name as the command-line argument")
        exit()
    if not os.path.isfile(sys.argv[1]):
        print("File "+sys.argv[1]+" does not exist!")
        exit()
    model = createModel(sys.argv[1])
    for name, comp in model.components.items():
        print("Running "+name)
        runComponent(comp, model.potential)
        print("Done with "+name)
        if hasattr(comp, 'nbody'):  # export N-body model
            agama.writeSnapshot("model_"+name+".nbody", comp.nbodymodel)
        # write out the complete model as a numpy binary archive
        args = {'ic': comp.ic, 'inttime': comp.inttime, 'weights': comp.weights}
        if hasattr(comp, 'densitydata'):  args['densitydata'] = comp.densitydata
        if hasattr(comp, 'kinemdata'):    args['kinemdata']   = comp.kinemdata
        try: numpy.savez_compressed("model_"+name+".data", **args)
        except Exception as e: print(e)
        # write out the initial conditions and weights as a text file
        numpy.savetxt("model_"+name+".orb",
            numpy.column_stack((comp.ic, comp.weights, comp.weightprior, comp.inttime)),
            header='x y z vx vy vz weight prior inttime', fmt="%8g")
