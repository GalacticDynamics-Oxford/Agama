#include <iostream>
#include "potential_factory.h"
#include "utils_config.h"

potential::PtrPotential pot;  // single instance of potential
particles::PointMassArray<coord::PosCar> points;  // array of particles that are used to compute potential

/// store particle positions and masses used to initialize potential
int set_particles(double x[], double y[], double z[], double m[], int n)
{
    if(!x || !y || !z || !m || n<=0) return -1;
    points.data.resize(n);
    for(int i=0; i<n; i++) {
        points.data[i].first.x=x[i];
        points.data[i].first.y=y[i];
        points.data[i].first.z=z[i];
        points.data[i].second =m[i];
    }
    return 0;
}

/// create the instance of potential
int initialize_code(int nparams, char** params)
{
    if(nparams<=0) return -2;
    if(pot) return -3;
    try{
        utils::KeyValueMap args(nparams, params);
        if(!points.data.empty())
            pot = potential::createPotential(args, points, units::ExternalUnits());
        else
            pot = potential::createPotential(args);
    }
    catch(std::exception& e) {
        std::cerr << e.what() << '\n';
        return -1;
    }
    return 0;
}

/// destroy the instance of potential
int cleanup_code()
{
    if(!pot) return -1;
    pot.reset();  // not necessary?
    return 0;
}

/// compute accelerations at given points: x,y,z are input coordinates, ax,ay,az are output accelerations 
int get_gravity_at_point(double /*eps*/[],
    double x[], double y[], double z[],
    double ax[], double ay[], double az[], int npoints)
{
    if(!pot) return -1;
    for(int i=0; i<npoints; i++) {
        coord::GradCar grad;
        pot->eval(coord::PosCar(x[i], y[i], z[i]), NULL, &grad);
        ax[i]=-grad.dx;
        ay[i]=-grad.dy;
        az[i]=-grad.dz;
    }
    return 0;
}

/// compute potential at given points: x,y,z are input coordinates, p is output potential
int get_potential_at_point(double /*eps*/[],
    double x[], double y[], double z[],
    double p[], int npoints)
{
    if(!pot) return -1;
    for(int i=0; i<npoints; i++) {
        p[i]=pot->value(coord::PosCar(x[i], y[i], z[i]));
    }
    return 0;
}
