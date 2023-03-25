#include "potential_factory.h"
#include "raga_core.h"
#include "utils.h"
#include "utils_config.h"
#include <iostream>
#include <fstream>
#include <stdint.h>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace{
raga::RagaCore core;  // the mighty thing...
}
#define MSG        { if(utils::verbosityLevel >= utils::VL_DEBUG) std::cout << "-{Agama} " << __FUNCTION__ << "\n"; }
#define MSGS(text) { if(utils::verbosityLevel >= utils::VL_DEBUG) std::cout << "-{Agama} " << __FUNCTION__ << ": " << text << "\n"; }
#define MSGE(text) { std::cout << "!{Agama} " << __FUNCTION__ << ": " << text << "\n"; }

/// set the number of OpenMP threads
int32_t set_num_threads(int32_t num_threads)
{
    MSGS(num_threads)
#ifdef _OPENMP
    omp_set_num_threads(num_threads);
#endif
    return 0;
}

int32_t set_params(int argc, char** argv)
{
    try{
        utils::KeyValueMap params = utils::KeyValueMap(argc, argv);
        core.init(params);
        MSGS(params.dumpSingleLine()+" potential="+std::string(core.ptrPot?core.ptrPot->name():"none"))
        return 0;
    }
    catch(std::exception& e) {
        MSGE(e.what())
        return -1;
    }
}

int32_t initialize_code()
{
    MSG
    return 0;
}

/// destroy the instance of potential
int32_t cleanup_code()
{
    if(!core.ptrPot) return -1;
    core.ptrPot.reset();  // not necessary?
    MSG
    return 0;
}

/// compute accelerations at given points: x,y,z are input coordinates, ax,ay,az are output accelerations 
int32_t get_gravity_at_point(/*input*/ double * /*eps*/, double *x, double *y, double *z,
    /*output*/ double *ax, double *ay, double *az, /*input*/ int npoints)
{
    if(!core.ptrPot) return -1;
    for(int i=0; i<npoints; i++) {
        coord::GradCar grad;
        core.ptrPot->eval(coord::PosCar(x[i], y[i], z[i]), NULL, &grad, NULL, /*time*/ core.paramsRaga.timeCurr);
        ax[i]=-grad.dx;
        ay[i]=-grad.dy;
        az[i]=-grad.dz;
    }
    return 0;
}

/// compute potential at given points: x,y,z are input coordinates, p is output potential
int32_t get_potential_at_point(/*input*/double * /*eps*/, double *x, double *y, double *z,
    /*output*/ double *p, /*input*/ int npoints)
{
    if(!core.ptrPot) return -1;
    for(int i=0; i<npoints; i++) {
        p[i]=core.ptrPot->value(coord::PosCar(x[i], y[i], z[i]), /*time*/ core.paramsRaga.timeCurr);
    }
    return 0;
}


int32_t new_particle(/*output*/ int32_t *index,
    /*input*/ double mass, double x, double y, double z,
    double vx, double vy, double vz, double radius)
{
    if(core.particles.size()==0) {MSG}
    *index = core.particles.size();
    core.particles.add(particles::ParticleAux(coord::PosVelCar(x, y, z, vx, vy, vz), mass, radius), mass);
    return 0;
}

int32_t delete_particle(int32_t) { return -1; /*NOT IMPLEMENTED*/ }

int32_t commit_particles()
{
    try{
        MSGS(core.particles.size())
        core.initPotentialFromParticles();
    }
    catch(std::exception& e) {
        MSGE(e.what())
        return -1;
    }
    return 0;
}

int32_t commit_parameters() { MSG return 0; }

int32_t recommit_particles() { return commit_particles(); }

int32_t recommit_parameters() { return commit_parameters(); }

int32_t synchronize_model() { MSG return 0; }

int32_t evolve_model(double time)
{
    MSGS(time)
    if(core.particles.size() == 0) {   // no need to evolve anything, just set the current time
        core.paramsRaga.timeCurr = time;
        return 0;
    }
    try {
        while(core.paramsRaga.timeCurr < time) {
            double timestep = time - core.paramsRaga.timeCurr;
            if(core.paramsRaga.episodeLength > 0 && core.paramsRaga.episodeLength < timestep)
                timestep = core.paramsRaga.episodeLength;
            MSGS("doEpisode start at t="+utils::toString(core.paramsRaga.timeCurr)+
                 ", end at t="+utils::toString(core.paramsRaga.timeCurr+timestep))
            core.doEpisode(timestep);
        }
        return 0;
    }
    catch(std::exception& e) {
        MSGE(e.what())
        return -1;
    }
}

int32_t get_state(/*input*/ int32_t index,
    /*output*/ double *mass, double *x, double *y, double *z,
    double *vx, double *vy, double *vz, double *radius)
{
    if(index==0) {MSG}
    if(index >= (int32_t)core.particles.size()) return -1;
    *x = core.particles[index].first.x;
    *y = core.particles[index].first.y;
    *z = core.particles[index].first.z;
    *vx= core.particles[index].first.vx;
    *vy= core.particles[index].first.vy;
    *vz= core.particles[index].first.vz;
    *mass   = core.particles[index].first.stellarMass;
    *radius = core.particles[index].first.stellarRadius;
    return 0;
}

int32_t set_state(/*input*/ int32_t index,
    double mass, double x, double y, double z,
    double vx, double vy, double vz, double radius)
{
    if(index==0) {MSG}
    if(index >= (int32_t)core.particles.size()) return -1;
    // when modifying the stellar mass, scale the gravitational mass by the same factor
    core.particles[index].second *= mass / core.particles[index].first.stellarMass;
    core.particles[index].first = coord::PosVelCar(x, y, z, vx, vy, vz);
    core.particles[index].first.stellarMass = mass;
    core.particles[index].first.stellarRadius = radius;
    return 0;
}

int32_t get_gravitating_mass(int32_t index, /*output*/ double *mass)
{
    if(index==0) {MSG}
    if(index >= (int32_t)core.particles.size()) return -1;
    *mass = core.particles[index].second;
    return 0;
}

int32_t set_gravitating_mass(int32_t index, double mass)
{
    if(index==0) {MSG}
    if(index >= (int32_t)core.particles.size()) return -1;
    core.particles[index].second = mass;
    return 0;
}

int32_t get_mass(int32_t index, /*output*/ double *mass)
{
    if(index==0) {MSG}
    if(index >= (int32_t)core.particles.size()) return -1;
    *mass = core.particles[index].first.stellarMass;
    return 0;
}

int32_t set_mass(int32_t index, double mass)
{
    if(index==0) {MSG}
    if(index >= (int32_t)core.particles.size()) return -1;
    // when modifying the stellar mass, scale the gravitational mass by the same factor
    core.particles[index].second *= mass / core.particles[index].first.stellarMass;
    core.particles[index].first.stellarMass = mass;
    return 0;
}

int32_t get_radius(int32_t index, /*output*/ double *radius)
{
    if(index==0) {MSG}
    if(index >= (int32_t)core.particles.size()) return -1;
    *radius = core.particles[index].first.stellarRadius;
    return 0;
}

int32_t set_radius(int32_t index, double radius)
{
    if(index==0) {MSG}
    if(index >= (int32_t)core.particles.size()) return -1;
    core.particles[index].first.stellarRadius = radius;
    return 0;
}

int32_t get_position(int32_t index, /*output*/ double *x, double *y, double *z)
{
    if(index==0) {MSG}
    if(index >= (int32_t)core.particles.size()) return -1;
    *x = core.particles[index].first.x;
    *y = core.particles[index].first.y;
    *z = core.particles[index].first.z;
    return 0;
}

int32_t set_position(int32_t index, double x, double y, double z)
{
    if(index==0) {MSG}
    if(index >= (int32_t)core.particles.size()) return -1;
    core.particles[index].first.x = x;
    core.particles[index].first.y = y;
    core.particles[index].first.z = z;
    return 0;
}

int32_t get_velocity(int32_t index, /*output*/ double *vx, double *vy, double *vz)
{
    if(index==0) {MSG}
    if(index >= (int32_t)core.particles.size()) return -1;
    *vx = core.particles[index].first.vx;
    *vy = core.particles[index].first.vy;
    *vz = core.particles[index].first.vz;
    return 0;
}

int32_t set_velocity(int32_t index, double vx, double vy, double vz)
{
    if(index==0) {MSG}
    if(index >= (int32_t)core.particles.size()) return -1;
    core.particles[index].first.vx = vx;
    core.particles[index].first.vy = vy;
    core.particles[index].first.vz = vz;
    return 0;
}

int32_t get_eps2(double*) { return -1; /*NOT SUPPORTED*/ }
int32_t set_eps2(double ) { return -1; /*NOT SUPPORTED*/ }

int32_t get_acceleration(int32_t index, /*output*/ double *ax, double *ay, double *az)
{
    if(index >= (int32_t)core.particles.size()) return -1;
    coord::GradCar grad;
    core.ptrPot->eval(core.particles[index].first, NULL, &grad, NULL, /*time*/ core.paramsRaga.timeCurr);
    *ax=-grad.dx;
    *ay=-grad.dy;
    *az=-grad.dz;
    return 0;
}

int32_t set_acceleration(int32_t, double, double, double) { return -1; /*NOT SUPPORTED*/ }

int32_t get_potential(int32_t index, /*output*/ double* potential)
{
    if(index >= (int32_t)core.particles.size()) return -1;
    *potential = core.ptrPot->value(core.particles[index].first, /*time*/ core.paramsRaga.timeCurr);
    return 0;
}

int32_t get_begin_time(double *time) { *time = core.paramsRaga.timeCurr; MSGS(*time) return 0; }

int32_t set_begin_time(double  time) { core.paramsRaga.timeCurr = time; MSGS(time) return 0; }

int32_t get_time(double *time) { *time = core.paramsRaga.timeCurr; MSGS(*time) return 0; }

int32_t get_time_step(double *timestep) { *timestep = core.paramsRaga.episodeLength; MSGS(*timestep) return 0; }

int32_t set_time_step(double timestep)  { core.paramsRaga.episodeLength = timestep; MSGS(timestep) return 0; }

int32_t get_total_mass(double *mass) { *mass = core.particles.totalMass(); MSGS(*mass) return 0; }

int32_t get_kinetic_energy(double *E)   { *E = core.Ekin; MSGS(*E) return 0; }

int32_t get_potential_energy(double *E) { *E = core.Epot; MSGS(*E) return 0; }

int32_t get_total_radius(double *rmax)
{
    *rmax = NAN;
    for(size_t i=0; i<core.particles.size(); i++)
        *rmax = fmax(*rmax, sqrt(
            pow_2(core.particles[i].first.x) +
            pow_2(core.particles[i].first.y) +
            pow_2(core.particles[i].first.z)));
    return 0;
}

int32_t get_center_of_mass_position(double *x, double *y, double *z)
{
    double sumx=0, sumy=0, sumz=0, summ=0;
    for(size_t i=0; i<core.particles.size(); i++) {
        sumx += core.particles[i].second * core.particles[i].first.x;
        sumy += core.particles[i].second * core.particles[i].first.y;
        sumz += core.particles[i].second * core.particles[i].first.z;
        summ += core.particles[i].second;
    }
    *x = sumx/summ;
    *y = sumy/summ;
    *z = sumz/summ;
    return 0;
}

int32_t get_center_of_mass_velocity(double *vx, double *vy, double *vz)
{
    double sumx=0, sumy=0, sumz=0, summ=0;
    for(size_t i=0; i<core.particles.size(); i++) {
        sumx += core.particles[i].second * core.particles[i].first.vx;
        sumy += core.particles[i].second * core.particles[i].first.vy;
        sumz += core.particles[i].second * core.particles[i].first.vz;
        summ += core.particles[i].second;
    }
    *vx = sumx/summ;
    *vy = sumy/summ;
    *vz = sumz/summ;
    return 0;
}

int32_t get_number_of_particles(int32_t *count)
{
    *count=core.particles.size();
    MSG return 0;
}

int32_t get_index_of_first_particle(int32_t *ind0)
{
    *ind0=0;
    return core.particles.size()>0 ? 0 : -1;
}

int32_t get_index_of_next_particle(int32_t ind, int32_t *ind1)
{
    *ind1=ind+1;
    return ind+1 < (int32_t)core.particles.size() ? 0 : -1;
}
