#!/usr/bin/python

try:
    from amuse.community.agama.interface import Agama
    from amuse.units import *
    from amuse.test.amusetest import TestWithMPI
    from amuse.ic.plummer import new_plummer_model
    from numpy.random import seed
except ImportError:
    import sys
    sys.exit("\033[1;33mSKIPPED DUE TO AMUSE IMPORT ERROR\033[0m")

class AgamaInterfaceTests(TestWithMPI):

    def test1(self):
        M=10.
        a=2.
        x=1.
        y=2.
        z=3.
        r=(x*x+y*y+z*z)**0.5
        instance = Agama(type="Dehnen",
            mass=M|generic_unit_system.mass, scaleRadius=a|generic_unit_system.length,
            channel_type="sockets", redirection="none")
        result=instance.get_potential_at_point(
            0.|generic_unit_system.length,
            x |generic_unit_system.length,
            y |generic_unit_system.length,
            z |generic_unit_system.length)
        self.assertAlmostEqual(result.value_in(
            generic_unit_system.length**2/generic_unit_system.time**2), -M/(r+a), places=14)
        result=instance.get_gravity_at_point(
            0.|generic_unit_system.length,
            x |generic_unit_system.length,
            y |generic_unit_system.length,
            z |generic_unit_system.length)
        self.assertAlmostEqual(result[0].value_in(
            generic_unit_system.length/generic_unit_system.time**2), -M/(r+a)**2 * x/r, places=14)
        self.assertAlmostEqual(result[0],result[2]/3, places=14)

        scaleM=2.345678|units.MSun
        scalea=7.654321|units.parsec
        converter=nbody_system.nbody_to_si(scaleM, scalea)
        instance = Agama(converter, type="Dehnen", mass=M*scaleM, scaleradius=a*scalea,
            channel_type="sockets", redirection="none")
        result=instance.get_gravity_at_point(
            0.|generic_unit_system.length, x*scalea, y*scalea, z*scalea)
        scale=M/(r+a)**2 *scaleM/scalea**2*constants.G
        self.assertAlmostEqual(result[0], scale * x/r, places=10)
        instance.stop()

    def test2(self):
        seed(1)
        particles=new_plummer_model(10000)
        instance = Agama(type="Multipole", particles=particles, symmetry="none",
            channel_type="sockets", redirection="none")
        result=instance.get_potential_at_point(
            0.|generic_unit_system.length,
            0.|generic_unit_system.length,
            0.|generic_unit_system.length,
            0.|generic_unit_system.length)
        self.assertLess(abs(result/(-16./3/3.1416|
            generic_unit_system.length**2/generic_unit_system.time**2) - 1), 0.05)

        scaleM=1234|units.MSun
        scaleR=5.6 |units.parsec
        converter=nbody_system.nbody_to_si(scaleM,scaleR)
        particles=new_plummer_model(20000, convert_nbody=converter)
        instance = Agama(converter, type="Multipole", particles=particles, symmetry="none",
            channel_type="sockets", redirection="none")
        x=3.|units.parsec
        y=4.|units.parsec
        z=5.|units.parsec
        r=(x*x+y*y+z*z)**0.5
        result=instance.get_gravity_at_point(0|units.parsec, x, y, z)
        expected=-constants.G*scaleM/(r*r+(3*3.1416/16*scaleR)**2)**1.5*x
        self.assertLess(abs(result[0]/expected-1), 0.03)
        self.assertLess(abs(result[2]/result[0]-z/x), 0.03)
        instance.stop()

if __name__ == '__main__':
    try:
        test = AgamaInterfaceTests('test1')
        test.test1()
        test.test2()
        print("\033[1;32mALL TESTS PASSED\033[0m")
    except Exception as e:
        print(str(e)+"\n\033[1;31mSOME TESTS FAILED\033[0m")
