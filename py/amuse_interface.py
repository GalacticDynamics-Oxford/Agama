"""
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
"""

from amuse.community import *
from amuse.community.interface.gd import *


class AgamaInterface(CodeInterface, LiteratureReferencesMixIn, GravityFieldInterface):
    """
    Agama is a library for galaxy modelling; among other features, it provides methods
    for computing potential, density and accelerations for a wide range of models of stellar systems.
    The Agama class provides methods for computing potential and accelerations at given coordinates.
    It is initialized with a list of named parameters defining the potential:
    >>> Agama([converter], type="Dehnen", [mass="10|mass"], [scaleradius="0.42|length"], ...)
    type is obligatory, and if it is one of the potential expansions (Multipole or CylSpline),
    then also  density="..."  or  particles=...  is necessary to define the actual model.
    See a complete list of arguments in doc/reference.pdf,
    or check the python interface for the Agama library itself:
    >>> import agama
    >>> help(agama.Potential)

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
            elif param_name=="particles":
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
