from amuse.community import *
from amuse.community.interface.gd import *


class AgamaInterface(CodeInterface, LiteratureReferencesMixIn, GravityFieldInterface):
    """
    Agama is a library for galaxy modelling; among other features, it provides methods
    for computing potential, density and accelerations for a wide range of models of stellar systems.
    The Agama class provides methods for computing potential and accelerations at given coordinates.
    It is initialized with a list of named parameters defining the potential:
    >>> Agama([converter], type="Dehnen", [mass="10|mass"], [scalerad="0.42|length"], ...)
    type is obligatory, and if it is one of the potential expansions (Multipole or CylSpline),
    then also  density="..."  or  points=...  is necessary to define the actual model.
    See a complete list of arguments in readme_agama.pdf

    .. [#] Vasiliev E., 2013, MNRAS, 434, 3174
    """

    include_headers = ['worker_code.h']

    def __init__(self, **keyword_arguments):
        CodeInterface.__init__(self, name_of_the_worker="agama_worker", **keyword_arguments),
        LiteratureReferencesMixIn.__init__(self)

    @legacy_function
    def initialize_code():
        function = LegacyFunctionSpecification()
        function.addParameter('nparams', dtype='int32',  direction=function.LENGTH)
        function.addParameter('params',  dtype='string', direction=function.IN)
        function.must_handle_array = True
        function.result_type = 'int32'
        return function

    @legacy_function
    def set_particles():
        function = LegacyFunctionSpecification()
        function.addParameter('pointx',  dtype='float64',direction=function.IN)
        function.addParameter('pointy',  dtype='float64',direction=function.IN)
        function.addParameter('pointz',  dtype='float64',direction=function.IN)
        function.addParameter('pointm',  dtype='float64',direction=function.IN)
        function.addParameter('nparams', dtype='int32',  direction=function.LENGTH)
        function.must_handle_array = True
        function.result_type = 'int32'
        return function

    @legacy_function
    def cleanup_code():
        function = LegacyFunctionSpecification()
        function.result_type = 'int32'
        return function

class Agama(InCodeComponentImplementation, GravityFieldCode):

    def __init__(self, unit_converter = None, **options):
        self.unit_converter = unit_converter
        InCodeComponentImplementation.__init__(self,  AgamaInterface(**options), **options)
        dimensional_params = {
            'mass'        : nbody_system.mass,
            'scalerad'    : nbody_system.length,
            'scalerad2'   : nbody_system.length,
            'rmax'        : nbody_system.length,
            'splinermin'  : nbody_system.length,
            'splinermax'  : nbody_system.length,
            'splinezmin'  : nbody_system.length,
            'splinezmax'  : nbody_system.length,
            'treecodeeps' : nbody_system.length
        }
        params=[]
        for k,v in options.items():
            param_name=str(k)
            if param_name.lower() in dimensional_params.keys():
                if unit_converter is not None:
                    v=unit_converter.as_converter_from_si_to_nbody().from_source_to_target(v)
                params.append(param_name+"="+str(v.value_in(dimensional_params[param_name.lower()])))
            elif param_name=="points":
                if unit_converter is not None:
                    self.overridden().set_particles(
                        unit_converter.as_converter_from_si_to_nbody().from_source_to_target(v.x).value_in(nbody_system.length),
                        unit_converter.as_converter_from_si_to_nbody().from_source_to_target(v.y).value_in(nbody_system.length),
                        unit_converter.as_converter_from_si_to_nbody().from_source_to_target(v.z).value_in(nbody_system.length),
                        unit_converter.as_converter_from_si_to_nbody().from_source_to_target(v.mass).value_in(nbody_system.mass) )
                else:
                    self.overridden().set_particles(
                        v.x.value_in(nbody_system.length),
                        v.y.value_in(nbody_system.length),
                        v.z.value_in(nbody_system.length),
                        v.mass.value_in(nbody_system.mass) )
            else:
                params.append(param_name+"="+str(v))
        self.overridden().initialize_code(params)

    def define_converter(self, object):
        if not self.unit_converter is None:
            object.set_converter(self.unit_converter.as_converter_from_si_to_nbody())

    def define_methods(self, handler):
        handler.add_method(
            'get_gravity_at_point',
            (
                nbody_system.length,
                nbody_system.length,
                nbody_system.length,
                nbody_system.length,
            ),
            (
                nbody_system.acceleration,
                nbody_system.acceleration,
                nbody_system.acceleration,
                handler.ERROR_CODE
            )
        )
        handler.add_method(
            'get_potential_at_point',
            (
                nbody_system.length,
                nbody_system.length,
                nbody_system.length,
                nbody_system.length,
            ),
            (
                nbody_system.potential,
                handler.ERROR_CODE
            )
        )
