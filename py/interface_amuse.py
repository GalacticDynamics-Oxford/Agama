'''
This is the plugin for AMUSE N-body simulation framework.

Agama is a module that provides methods for computing gravitational potential and forces corresponding
to a variety of static density profiles (either analytic or constructed from N-body snapshots,
in the latter case creating a smooth non-parametric representation of the N-body model).
The potential solvers are taken from the SMILE code [Vasiliev, 2013], and this module can be used
as an external potential in the Bridge scheme (see example 'example_amuse.py')

The potential is constructed using
>>> from amuse.community.agama.interface import Agama
>>> pot = Agama(type='type', other_params=...)
where type may be either one of the known analytic potential models (see a full list in doc/reference.pdf),
or, more likely, one of the general-purpose potential approximations ('Multipole' or 'CylSpline',
the former is based on spherical-harmonic expansion and is suitable for not too flattened density profiles,
and the latter is more appropriate for highly flattened density profiles without a cusp in the center).
In the case of type='Multipole' or type='CylSpline', one needs to provide additionally the name of density
profile (e.g., density='Dehnen'), or the array of particles that are used to construct a smooth density
profile from an N-body snapshot (e.g., particles=new_plummer_model(10000) ).
The default parameters controlling the accuracy of potential approximation are suitable in most cases,
but sometimes need to be adjusted (e.g., lmax=10 or symmetry='Axisymmetric').
'''

from amuse.community import CodeInterface, LiteratureReferencesMixIn, legacy_function, LegacyFunctionSpecification
from amuse.community.interface.gd import InCodeComponentImplementation, GravitationalDynamicsInterface, GravityFieldInterface, GravityFieldCode, GravitationalDynamics
from amuse.units import nbody_system, constants

class AgamaInterface(CodeInterface, LiteratureReferencesMixIn, GravitationalDynamicsInterface, GravityFieldInterface):
    '''
    Agama is a library for galaxy modelling; among other features, it provides methods
    for computing potential, density and accelerations for a wide range of models of stellar systems.
    The Agama class provides methods for computing potential and accelerations at given coordinates.
    It is initialized with a list of named parameters defining the potential:
    >>> Agama([converter], type='Dehnen', [mass=10|mass], [scaleradius=0.42|length], ...)
    type is obligatory, and if it is one of the potential expansions (Multipole or CylSpline),
    then also  density='...'  or  particles=...  is necessary to define the actual model.
    See a complete list of arguments in doc/reference.pdf,
    or check the python interface for the Agama library itself:
    >>> import agama
    >>> help(agama.Potential)

    .. [#] Vasiliev E., 2019, MNRAS, 482, 1525
    '''

    include_headers = ['worker_code.h']

    def __init__(self, **keyword_arguments):
        CodeInterface.__init__(self, name_of_the_worker='agama_worker', **keyword_arguments),
        LiteratureReferencesMixIn.__init__(self)

    @legacy_function
    def set_params():
        function = LegacyFunctionSpecification()
        function.addParameter('nparams', dtype='int32',  direction=function.LENGTH)
        function.addParameter('params',  dtype='string', direction=function.IN)
        function.must_handle_array = True
        function.result_type = 'int32'
        return function

    @legacy_function
    def set_time_step():
        function = LegacyFunctionSpecification()
        function.addParameter('time_step', dtype='float64', direction=function.IN,
            description = 'Set the episode length')
        function.result_type = 'int32'
        return function

    @legacy_function
    def set_num_threads():
        function = LegacyFunctionSpecification()
        function.addParameter('num_threads', dtype='int32', direction=function.IN,
            description = 'Number of OpenMP threads')
        function.result_type = 'int32'
        return function

    @legacy_function
    def get_gravitating_mass():
        function = LegacyFunctionSpecification()
        function.addParameter('index_of_the_particle', dtype='int32', direction=function.IN,
            description = 'Index of the particle')
        function.addParameter('gravitating_mass', dtype='float64', direction=function.OUT,
            description = 'Get the contribution of this particle to the total potential')
        function.can_handle_array = True
        function.result_type = 'int32'
        return function

    @legacy_function
    def set_gravitating_mass():
        function = LegacyFunctionSpecification()
        function.addParameter('index_of_the_particle', dtype='int32', direction=function.IN,
            description = 'Index of the particle')
        function.addParameter('gravitating_mass', dtype='float64', direction=function.IN,
            description = 'Set the contribution of this particle to the total potential')
        function.can_handle_array = True
        function.result_type = 'int32'
        return function


class Agama(GravitationalDynamics, GravityFieldCode):

    def __init__(self, unit_converter = None, number_of_workers = None, **options):
        GravitationalDynamics.__init__(self, AgamaInterface(**options), unit_converter, **options)
        self.unit_converter = unit_converter
        if number_of_workers is not None:
            self.overridden().set_num_threads(number_of_workers)
        dimensional_params = {
            'mass'             : nbody_system.mass,
            'rscale'           : nbody_system.length,
            'scaleradius'      : nbody_system.length,
            'scaleradius2'     : nbody_system.length,
            'scaleheight'      : nbody_system.length,
            'innercutoffradius': nbody_system.length,
            'outercutoffradius': nbody_system.length,
            'rmin'             : nbody_system.length,
            'rmax'             : nbody_system.length,
            'zmin'             : nbody_system.length,
            'zmax'             : nbody_system.length,
            'densitynorm'      : nbody_system.mass / nbody_system.length**3,
            'rho0'             : nbody_system.mass / nbody_system.length**3,
            'surfacedensity'   : nbody_system.mass / nbody_system.length**2,
            'sigma0'           : nbody_system.mass / nbody_system.length**2,
            'v0'               : nbody_system.speed,
            'omega'            : nbody_system.time**-1,

            'mbh'              : nbody_system.mass,
            'binary_sma'       : nbody_system.length,
            'timetotal'        : nbody_system.time,
            'timeinit'         : nbody_system.time,
            'episodelength'    : nbody_system.time,
            'outputinterval'   : nbody_system.time,
            'captureradius'    : nbody_system.length,
            'captureradius1'   : nbody_system.length,
            'captureradius2'   : nbody_system.length,
            'speedoflight'     : nbody_system.speed,
        }
        params=[]
        # check if speed of light was provided by user, if not then put the universally accepted value
        speed_of_light_provided = False
        black_hole_provided = False
        for k,v in options.items():
            param_name=str(k)
            if param_name.lower() in dimensional_params.keys():
                if unit_converter is not None:
                    v=unit_converter.as_converter_from_si_to_nbody().from_source_to_target(v)
                params.append(param_name+'='+str(v.value_in(dimensional_params[param_name.lower()])))
                if param_name.lower() == 'speedoflight': speed_of_light_provided = True
                if param_name.lower() == 'mbh': black_hole_provided = True
            elif param_name=='particles':
                self.particles.add_particles(v)
                #print('added %i particles'%len(v))
            elif param_name=='channel_type' or param_name=='redirection':
                pass  # do not confuse Agama with the AMUSE-specific parameters
            else:
                params.append(param_name+'='+str(v))
        if black_hole_provided and not speed_of_light_provided and unit_converter is not None:
            params.append('speedOfLight='+str(unit_converter.to_nbody(constants.c).value_in(nbody_system.speed)))
        #print('setting params')
        self.overridden().set_params(params)
        handler=self.get_handler('PARTICLES')
        handler.add_setter('particles', 'set_gravitating_mass', names = ('gravitating_mass',))
        handler.add_getter('particles', 'get_gravitating_mass', names = ('gravitating_mass',))

    def define_converter(self, handler):
        if not self.unit_converter is None:
            handler.set_converter(self.unit_converter.as_converter_from_si_to_nbody())

    def define_methods(self, handler):
        GravitationalDynamics.define_methods(self, handler)
        handler.add_method( 'set_time_step',
            ( nbody_system.time, ),
            ( handler.ERROR_CODE,)
        )
        handler.add_method( 'get_gravity_at_point',
            ( nbody_system.length,) * 4,
            ( nbody_system.acceleration,) * 3 + (handler.ERROR_CODE,)
        )
        handler.add_method( 'get_potential_at_point',
            ( nbody_system.length,) * 4,
            ( nbody_system.potential, handler.ERROR_CODE )
        )
        handler.add_method( 'set_gravitating_mass',
            ( handler.NO_UNIT, nbody_system.mass ),
            ( handler.ERROR_CODE, )
        )
        handler.add_method( 'get_gravitating_mass',
            ( handler.NO_UNIT, ),
            ( nbody_system.mass, handler.ERROR_CODE )
        )

    def define_state(self, handler):
        GravitationalDynamics.define_state(self, handler)
        GravityFieldCode.define_state(self, handler)
