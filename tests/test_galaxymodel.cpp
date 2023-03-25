/** \name   test_galaxymodel.cpp
    \author Eugene Vasiliev
    \date   2021

    This program examines the accuracy of computation of various DF moments and derived distributions:
    intrinsic and projected moments (density, mean velocity, second moment of velocity),
    projected DF (integrated along the line of sight and some of the velocity components with errors),
    intrinsic and projected velocity distribution functions represented by B-splines.
    Many of these quantities can be computed in more than one way, so we check that the different
    methods are in agreement; moreover, for DFs with some internal symmetries (spherical isotropic
    or just spherical) there are additional checks (e.g. that the mixed velocity moments are zero),
    and some computations are compared between different orientations of the coordinate system.
    Finally, we compare the results with analytic expressions for the spherical isotropic Plummer model.
    There are three test cases - isotropic Plummer, spherical anisotropic Plummer, and flattened and
    rotating DoublePowerLaw DF (all in a spherical potential, but this shouldn't be a limiting factor).
*/
#include "galaxymodel_base.h"
#include "potential_analytic.h"
#include "df_halo.h"
#include "df_spherical.h"
#include "math_specfunc.h"
#include "math_core.h"
#include "utils.h"
#include <cmath>
#include <iostream>
#include <fstream>

bool test(double a, double b, double eps, const char* label)
{
    double err = fabs(a-b) / fmax(fabs(a), fabs(b));
    bool ok = err < eps;
    if(utils::verbosityLevel >= utils::VL_DEBUG || !ok) {
        std::cout << label << ": rel.err.=" << err;
        if(ok) std::cout << " < " << eps << "\n";
        else   std::cout << " > " << eps << " \033[1;31m**\033[0m\n";
    }
    return ok;
}

///---- analytic expressions for the Plummer model ----///

// DF integrated along Z when both R and velocity components are fixed
double projectedDF_Plummer_fixed_v(double R, const coord::VelCar& vel)
{
    double L = pow_2(vel.vx) + pow_2(vel.vy) + pow_2(vel.vz);
    double U = 2 / sqrt(R*R+1), Q = L / U, A = sqrt(1-Q) / M_SQRT2;
    if(Q>=1) return 0;
    return M_SQRT2*2/7 /pow_3(M_PI) * U * U * sqrt(U) * (
        math::ellintK(A) * (Q+1) * (Q * (Q+1) * 18 + 2) +
        math::ellintE(A) * Q * (6 * Q * Q - 40) -
        (Q>0 ? math::ellintP(M_PI/2, A, 0.5*(1/Q-1)) * Q * Q * (Q+1) * 21 : 0) );
}

// DF integrated along Z and over Gaussian uncertainties in any velocity component (when they are small)
double projectedDF_Plummer_with_errors(double R, const coord::VelCar& vel, const coord::VelCar& velerr)
{
    // integrate numerically over Gaussian uncertainties in vx,vy,vz assuming these are small
    // (only over the components with nonzero errors)
    const int N = 7,  // number of quadrature points in each dimension is 2 N^2 + 1
    Nx = velerr.vx ? N : 0, Ny = velerr.vy ? N : 0, Nz = velerr.vz ? N : 0;
    double sum = 0;
    for(int ix=-Nx*Nx; ix<=Nx*Nx; ix++) {
        double dx = Nx==0 ? 0 : 1.0 * ix / Nx;
        double wx = Nx==0 ? 1 : 1/M_SQRTPI/Nx;
        for(int iy=-Ny*Ny; iy<=Ny*Ny; iy++) {
            double dy = Ny==0 ? 0 : 1.0 * iy / Ny;
            double wy = Ny==0 ? 1 : 1/M_SQRTPI/Ny;
            for(int iz=-Nz*Nz; iz<=Nz*Nz; iz++) {
                double dz = Nz==0 ? 0 : 1.0 * iz / Nz;
                double wz = Nz==0 ? 1 : 1/M_SQRTPI/Nz;
                sum += projectedDF_Plummer_fixed_v(R, coord::VelCar(
                    vel.vx + velerr.vx * M_SQRT2 * dx,
                    vel.vy + velerr.vy * M_SQRT2 * dy,
                    vel.vz + velerr.vz * M_SQRT2 * dz)) *
                    exp( - dx*dx - dy*dy - dz*dz) * wx * wy * wz;
            }
        }
    }
    return sum;
}

// DF integrated along Z and over vx,vy, when R and vz are fixed
double projectedDF_Plummer_marginalized_vxvy(double R, const coord::VelCar& vel)
{
    double L = pow_2(vel.vz), U = 2 / sqrt(R*R+1), Q = L / U, A = sqrt(1-Q) / M_SQRT2;
    if(Q>=1) return 0;
    return M_SQRT2*4/105 /pow_2(M_PI) * pow_3(U) * sqrt(U) * (
        math::ellintE(A) * (12 + Q*Q * (144 - 10*Q*Q)) -
        math::ellintK(A) * 2*(Q+1) * (3 + Q * (6 + Q * (30 + 20*Q))) +
        (Q>0 ? math::ellintP(M_PI/2, A, 0.5*(1/Q-1)) * Q*Q*Q * (Q+1) * 45 : 0) );
}

/// the dispatcher function calling one of the previous implementations
double projectedDF_Plummer(double R, const coord::VelCar& vel, const coord::VelCar& velerr)
{
    if(velerr.vx==INFINITY && velerr.vy==INFINITY) {
        if(velerr.vz == INFINITY)
            return 1 / M_PI / pow_2(R*R+1);   // surface density of the Plummer model
        if(velerr.vz >= 10)                   // very large error: again return surface density,
            return 1 / M_PI / pow_2(R*R+1) *  // multiplied by value of Gaussian at a point
                exp(-0.5 * pow_2(vel.vz / velerr.vz)) / (M_SQRTPI * M_SQRT2 * velerr.vz);
        if(velerr.vz > 1)
            return NAN;   // intermediate case of neither large nor small error is not implemented
        if(velerr.vz == 0)
            return projectedDF_Plummer_marginalized_vxvy(R, vel);
        // otherwise integrate over vz_error, assuming it is small
        const int Nz = 7;
        double sum = 0;
        for(int iz=-Nz*Nz; iz<=Nz*Nz; iz++) {
            double dz = 1.0 * iz / Nz;
            sum += projectedDF_Plummer_marginalized_vxvy(R,
                coord::VelCar(0, 0, vel.vz + velerr.vz * M_SQRT2 * dz)) *
                exp( - dz*dz) / M_SQRTPI/Nz;
        }
        return sum;
    } else {
        // integrate over small errors in all velocity components
        return projectedDF_Plummer_with_errors(R, vel, velerr);
    }
}

/// second moment of velocity at the given 3d radius (i.e. isotropic velocity dispersion)
double vel2mom_Plummer(double r)
{
    return 1 / sqrt(r*r+1) / 6;
}

/// second moment of vlos at the given projected radius
double projectedvel2mom_Plummer(double R)
{
    return 3*M_PI/64 / sqrt(R*R+1);
}

//---- test suite ----//
bool test(const galaxymodel::GalaxyModel& model, const std::string& name,
    bool havetruedens = true, bool havetruevel = true, bool isotropic = true, bool spherical = true)
{
    bool ok = true;

    coord::PosCar pos(0.8, 0.6, 0.4);
    coord::PosProj posproj(pos);  // keep only x,y coordinates
    double R = sqrt(pos.x*pos.x + pos.y*pos.y), r = sqrt(R*R + pos.z*pos.z);
    coord::VelCar vel(0.5, 0.4, 0.3),   // value of the velocity and various choices for its uncertainty
        noerr(0, 0, 0),
        noerrx(0, 25, INFINITY),
        noerrz(INFINITY, INFINITY, 0),
        smallerr(0.04, 0.08, 0.12),
        smallerrz(INFINITY, INFINITY, 0.2),
        largeerr(30, 25, 20),
        largeerrxy(30, 25, INFINITY),
        largeerrz(INFINITY, INFINITY, 20),
        inferr(INFINITY, INFINITY, INFINITY);
    coord::Orientation ori(1, 2, 3);  // some random orientation

    // for the spherical isotropic Plummer model, we have the following analytic expressions
    double denstrue  = 0.75 / M_PI * pow(1+r*r, -2.5),  //pot->density(pos),
        densprojtrue = projectedDF_Plummer(R, vel, inferr),
        velnorm      = v_circ(model.potential, r),  // characteristic value of velocity
        vel2true     = vel2mom_Plummer(r),
        vel2projtrue = projectedvel2mom_Plummer(R);

    //-------------------------------//
    std::cout << "Intrinsic moments\n";

    double densmom, densrot, densproj, densprojrot;
    coord::VelCar  velmom,  velrot,  velproj,  velprojrot;
    coord::Vel2Car vel2mom, vel2rot, vel2proj, vel2projrot;
    if(havetruedens) {
        // first compare just the density, which is supposed to be cheaper but less accurate
        galaxymodel::computeMoments(model, pos, &densmom, NULL, NULL);
        ok &= test(densmom, denstrue, 1e-3, "moments() density-only vs. analytic");
    }
    // then compute density and two velocity moments, which use more points but are more accurate
    galaxymodel::computeMoments(model, pos, &densmom, &velmom, &vel2mom);
    if(havetruedens)
        ok &= test(densmom, denstrue, 2e-4, "moments() density vs. analytic");
    if(havetruevel) {
        ok &= test(vel2mom.vx2, vel2true, 1e-4, "moments() velocity vx^2 vs. analytic");
        ok &= test(vel2mom.vy2, vel2true, 1e-4, "moments() velocity vy^2 vs. analytic");
        ok &= test(vel2mom.vz2, vel2true, 1e-4, "moments() velocity vz^2 vs. analytic");
    }
    if(isotropic) {
        // mixed moments should be zero in the isotropic case (add a constant value to comparison)
        ok &= test(vel2mom.vxvy + vel2true, vel2true, 1e-4, "moments() velocity vx vy = 0");
        ok &= test(vel2mom.vxvz + vel2true, vel2true, 1e-4, "moments() velocity vx vz = 0");
        ok &= test(vel2mom.vyvz + vel2true, vel2true, 1e-4, "moments() velocity vy vz = 0");
    }
    if(spherical) {
        // mean velocity should be zero in the spherical non-rotating case
        ok &= test(velmom.vx + velmom.vy + velmom.vz + velnorm, velnorm, 1e-4, "moments() mean velocity = 0");
    }

    // in the spherical case, rotating the observed reference frame should have no effect
    if(spherical) {
        galaxymodel::computeMoments(model, pos, &densrot, NULL, &vel2rot, false, ori);
        ok &= test(densmom, densrot, 2e-4, "moments() density in rotated vs. non-rotated frame");
        ok &= test(vel2mom.vx2, vel2rot.vx2, 1e-4, "moments() velocity vx^2 in rotated vs. non-rotated frame");
        ok &= test(vel2mom.vy2, vel2rot.vy2, 1e-4, "moments() velocity vy^2 in rotated vs. non-rotated frame");
        ok &= test(vel2mom.vz2, vel2rot.vz2, 1e-4, "moments() velocity vz^2 in rotated vs. non-rotated frame");
        // mixed moments - all together for simplicity
        ok &= test(vel2mom.vxvy + vel2mom.vxvz + vel2mom.vyvz + vel2true,
                   vel2rot.vxvy + vel2rot.vxvz + vel2rot.vyvz + vel2true,
                   1e-3, "moments() mixed velocity moments are zero in rotated vs. non-rotated frame");
    }

    // now rotate BOTH the point at which the moments are computed AND the observed reference frame:
    // the density should not change, while 1st and 2nd moments transform according to known rules
    galaxymodel::computeMoments(model, ori.toRotated(pos), &densrot, &velrot, &vel2rot, false, ori);
    ok &= test(densmom, densrot, 1e-4, "moments() density in rotated vs. original frame");
    // for velocity comparison, we add a constant value to avoid comparing zero to zero in non-rotating models
    coord::VelCar  velnonrot  = ori.fromRotated(velrot);   // transform back to the initial point
    ok &= test(velmom.vx + velnorm, velnonrot.vx + velnorm, 1e-4, "moments() velocity vx in rotated vs. original frame");
    ok &= test(velmom.vy + velnorm, velnonrot.vy + velnorm, 1e-4, "moments() velocity vy in rotated vs. original frame");
    ok &= test(velmom.vz + velnorm, velnonrot.vz + velnorm, 1e-4, "moments() velocity vz in rotated vs. original frame");
    // second moments (again add a constant offset to mixed moments, which may be close to zero)
    coord::Vel2Car vel2nonrot = ori.fromRotated(vel2rot);
    ok &= test(vel2mom.vx2, vel2nonrot.vx2, 1e-4, "moments() velocity vx^2 in rotated vs. original frame");
    ok &= test(vel2mom.vy2, vel2nonrot.vy2, 1e-4, "moments() velocity vy^2 in rotated vs. original frame");
    ok &= test(vel2mom.vz2, vel2nonrot.vz2, 1e-4, "moments() velocity vz^2 in rotated vs. original frame");
    ok &= test(vel2mom.vxvy + vel2true, vel2nonrot.vxvy + vel2true, 1e-4, "moments() velocity vx vy in rotated vs. original frame");
    ok &= test(vel2mom.vxvz + vel2true, vel2nonrot.vxvz + vel2true, 1e-4, "moments() velocity vx vz in rotated vs. original frame");
    ok &= test(vel2mom.vyvz + vel2true, vel2nonrot.vyvz + vel2true, 1e-4, "moments() velocity vy vz in rotated vs. original frame");


    //-------------------------------//
    std::cout << "Projected moments\n";

    if(havetruedens) {
        // again first compare just the projected density, which is cheaper but less accurate
        galaxymodel::computeMoments(model, posproj, &densproj, NULL, NULL);
        ok &= test(densproj, densprojtrue, 1e-3, "projected moments() density-only vs. analytic");
    }
    galaxymodel::computeMoments(model, posproj, &densproj, &velproj, &vel2proj);
    if(havetruevel) {
        ok &= test(vel2proj.vx2, vel2projtrue, 1e-4, "projected moments() velocity vx^2 vs. analytic");
        ok &= test(vel2proj.vy2, vel2projtrue, 1e-4, "projected moments() velocity vy^2 vs. analytic");
        ok &= test(vel2proj.vz2, vel2projtrue, 1e-4, "projected moments() velocity vz^2 vs. analytic");
        // mixed moments should be zero in the isotropic case (add a constant value to comparison)
        ok &= test(vel2proj.vxvy + vel2projtrue, vel2projtrue, 1e-4, "projected moments() velocity vx vy = 0");
    }
    if(true) {
        // mixed moments involving vz should be zero in projection even if the DF is not isotropic
        ok &= test(vel2proj.vxvz + vel2projtrue, vel2projtrue, 1e-4, "projected moments() velocity vx vz = 0");
        ok &= test(vel2proj.vyvz + vel2projtrue, vel2projtrue, 1e-4, "projected moments() velocity vy vz = 0");
    }
    if(spherical) {
        // mean velocity should be zero in the spherical non-rotating case
        ok &= test(velproj.vx + velproj.vy + velproj.vz + velnorm, velnorm, 1e-4, "projected moments() mean velocity = 0");
    }

    // in the spherical case, rotating the observed reference frame should have no effect
    galaxymodel::computeMoments(model, posproj, &densprojrot, &velprojrot, &vel2projrot, false, ori);
    if(spherical) {
        ok &= test(densproj, densprojrot, 2e-4, "projected moments() density in rotated vs. non-rotated frame");
        ok &= test(vel2proj.vx2, vel2projrot.vx2, 3e-4, "projected moments() velocity vx^2 in rotated vs. non-rotated frame");
        ok &= test(vel2proj.vy2, vel2projrot.vy2, 3e-4, "projected moments() velocity vy^2 in rotated vs. non-rotated frame");
        ok &= test(vel2proj.vz2, vel2projrot.vz2, 3e-4, "projected moments() velocity vz^2 in rotated vs. non-rotated frame");
        // mixed moments - all together for simplicity
        ok &= test(vel2proj.vxvy + vel2proj.vxvz + vel2proj.vyvz + vel2true,
                   vel2projrot.vxvy + vel2projrot.vxvz + vel2projrot.vyvz + vel2true,
                   1e-3, "projected moments() mixed velocity moments are zero in rotated vs. non-rotated frame");
    }


    //--------------------------//
    std::cout << "Projected DF\n";

    // test projectedDF in different orientations - should give identical results for spherical models
    double projdfface, projdfrot, projdf_largeerr, projdf_largeerrxy, projdf_largeerrz,
        projdf_inferr, projdf_noerrx, projdf_noerrz, projdf_smallerrz;
    // no errors in velocity
    if(spherical) {
        galaxymodel::computeProjectedDF(model, posproj, vel, noerr, &projdfface);  // face-on
        galaxymodel::computeProjectedDF(model, posproj, vel, noerr, &projdfrot, false, ori); // rotated
        ok &= test(projdfface, projdfrot, 1e-6, "projected df(), no error, rotated vs. non-rotated");
        if(havetruevel)
            ok &= test(projdfface, projectedDF_Plummer(R, vel, noerr), 1e-6, "projected df(), no error, vs. analytic");
    }

    // small errors
    if(spherical) {
        galaxymodel::computeProjectedDF(model, posproj, vel, smallerr, &projdfface);
        galaxymodel::computeProjectedDF(model, posproj, vel, smallerr, &projdfrot, false, ori);
        ok &= test(projdfface, projdfrot, 1e-4, "projected df(), small error, rotated vs. non-rotated");
        // swapping the velocity components and their respective errors in the isotropic case has no difference
        if(isotropic) {
            double dfswap;
            galaxymodel::computeProjectedDF(model, posproj,
                coord::VelCar(vel.vx, vel.vz, vel.vy), coord::VelCar(smallerr.vx, smallerr.vz, smallerr.vy), &dfswap);
            ok &= test(projdfface, dfswap, 1e-4, "projected df(), small error, swapped x<->y axes");
        }
        if(havetruevel)
            ok &= test(projdfface, projectedDF_Plummer(R, vel, smallerr), 1e-3, "projected df(), small error, vs. analytic");
    }

    // large errors - this one seems to require a larger-than-default maxNumEval to reach the desired accuracy
    galaxymodel::computeProjectedDF(model, posproj, vel, largeerr, &projdf_largeerr,  false, coord::Orientation(), 1e-3, 2e5);
    if(spherical) {
        galaxymodel::computeProjectedDF(model, posproj, vel, largeerr, &projdfrot, false, ori);
        ok &= test(projdf_largeerr, projdfrot, 1e-4, "projected df(), large error, rotated vs. non-rotated");
    }

    // large errors in vX,vY and infinite in vz
    galaxymodel::computeProjectedDF(model, posproj, vel, largeerrxy, &projdf_largeerrxy);

    // infinite errors in vX,vY and large in vz
    galaxymodel::computeProjectedDF(model, posproj, vel, largeerrz, &projdf_largeerrz);
    if(spherical) {
        galaxymodel::computeProjectedDF(model, posproj, vel, largeerrz, &projdfrot, false, ori);
        ok &= test(projdf_largeerrz, projdfrot, 1e-3, "projected df(), large error in vz / infinite in vx,vy, rotated vs. non-rotated");
        if(havetruedens)
             ok &= test(projdf_largeerrz, projectedDF_Plummer(R, vel, largeerrz), 1e-3,
                "projected df(), large error in vz / infinite in vx,vy, vs. analytic");
    }

    // infinite errors in all velocity components - result is equivalent to projected density
    galaxymodel::computeProjectedDF(model, posproj, vel, inferr, &projdf_inferr);
    galaxymodel::computeProjectedDF(model, posproj, vel, inferr, &projdfrot, false, ori);
    if(spherical) {
        ok &= test(projdf_inferr, projdfrot, 1e-3, "projected df(), infinite error, rotated vs. non-rotated");
    }
    if(havetruedens)
        ok &= test(projdf_inferr, projectedDF_Plummer(R, vel, inferr), 1e-3, "projected df(), infinite error, vs. analytic");

    // infinite errors in vX,vY and no error or a small error in vZ,
    // equivalent to projected VDF f_z(v_z) or a VDF convolved with a Gaussian
    galaxymodel::computeProjectedDF(model, posproj, vel, noerrz,    &projdf_noerrz);
    galaxymodel::computeProjectedDF(model, posproj, vel, smallerrz, &projdf_smallerrz);
    if(havetruevel) {
        ok &= test(projdf_noerrz,    projectedDF_Plummer(R, vel, noerrz),    2e-4, "projected df() f_z(v), no error, vs. analytic");
        ok &= test(projdf_smallerrz, projectedDF_Plummer(R, vel, smallerrz), 1e-3, "projected df() f_z(v), small error, vs. analytic");
    }

    // no error in vX, infinite error in vZ, and very large (not infinite due to API limitations)
    // error in vY (passed as noerrvx.vy); the result is then renormalized to approximate an infinite error in vY
    galaxymodel::computeProjectedDF(model, posproj, vel, noerrx, &projdf_noerrx);

    // large errors in some velocity components are almost identical to infinite errors after appropriate renormalization
    projdf_noerrx     *= exp(0.5 *  pow_2(vel.vy / noerrx.vy)) * (M_SQRTPI * M_SQRT2 * noerrx.vy);
    projdf_largeerrxy *= exp(0.5 * (pow_2(vel.vx / largeerrxy.vx) + pow_2(vel.vy / largeerrxy.vy))) *
        (2 * M_PI * largeerrxy.vx * largeerrxy.vy);
    projdf_largeerrz  *= exp(0.5 *  pow_2(vel.vz / largeerrz.vz)) * (M_SQRTPI * M_SQRT2 * largeerrz.vz);
    projdf_largeerr   *= exp(0.5 * (pow_2(vel.vx / largeerr.vx) + pow_2(vel.vy / largeerr.vy) + pow_2(vel.vz / largeerr.vz))) *
        (2 * M_SQRT2 * M_PI * M_SQRTPI * largeerr.vx * largeerr.vy * largeerr.vz);

    // check that the renormalization works
    ok &= test(projdf_largeerr,   projdf_inferr, 1e-3, "projected df() large error vs. infinite error");
    ok &= test(projdf_largeerrxy, projdf_inferr, 1e-3, "projected df() large error in vx,vy vs. infinite error");
    ok &= test(projdf_largeerrz,  projdf_inferr, 1e-3, "projected df() large error in vz vs. infinite error");

    // even if we don't have true surface density, the result should agree with projected moments
    ok &= test(projdf_inferr, densproj, 1e-3, "projected df(), infinite error, vs. projected moments() density");
    ok &= test(projdfrot,  densprojrot, 1e-3, "projected df(), infinite error, vs. projected moments() density, rotated");


    //---------------------------//
    std::cout << "Intrinsic VDF\n";

    std::vector<double> gridv;
    const int gridsize=15;
    const int N=3;  // degree of B-splines
    std::vector<double> amplvX, amplvY, amplvZ, amplvXrot, amplvYrot, amplvZrot;
    double densvdf, densvdfrot, densvdfproj, densvdfprojrot;

    galaxymodel::computeVelocityDistribution<N>(model, pos, gridsize, gridv,
        &densvdf, &amplvX, &amplvY, &amplvZ);
    math::BsplineInterpolator1d<N> vdfi(gridv);
    double vmin = gridv.front(), vmax = gridv.back();

    // the density should match the one returned by computeMoments()
    ok &= test(densvdf, densmom, 2e-4, "vdf() density vs. moments() density");
    if(havetruedens)
        ok &= test(densvdf, denstrue, 1e-6, "vdf() density vs. analytic");

    // the VDF should be normalized to unity
    ok &= test(vdfi.integrate(vmin, vmax, amplvX), 1, 1e-13, "vdf() f_x(v) normalized to unity");
    ok &= test(vdfi.integrate(vmin, vmax, amplvY), 1, 1e-13, "vdf() f_y(v) normalized to unity");
    ok &= test(vdfi.integrate(vmin, vmax, amplvZ), 1, 1e-13, "vdf() f_z(v) normalized to unity");

    // the mean velocity in VDF matches the first moment returned by computeMoments()
    ok &= test(vdfi.integrate(vmin, vmax, amplvX, 1) + velnorm, velmom.vx + velnorm, 1e-4, "vdf() mean vx vs. moments()");
    ok &= test(vdfi.integrate(vmin, vmax, amplvY, 1) + velnorm, velmom.vy + velnorm, 1e-4, "vdf() mean vy vs. moments()");
    ok &= test(vdfi.integrate(vmin, vmax, amplvZ, 1) + velnorm, velmom.vz + velnorm, 1e-4, "vdf() mean vz vs. moments()");

    // the dispersion of VDF matches the second moment returned by computeMoments()
    ok &= test(vdfi.integrate(vmin, vmax, amplvX, 2), vel2mom.vx2, 2e-4, "vdf() mean vx^2 vs. moments()");
    ok &= test(vdfi.integrate(vmin, vmax, amplvY, 2), vel2mom.vy2, 2e-4, "vdf() mean vy^2 vs. moments()");
    ok &= test(vdfi.integrate(vmin, vmax, amplvZ, 2), vel2mom.vz2, 2e-4, "vdf() mean vz^2 vs. moments()");
    // mixed moments are not compared, since VDF does not provide any information about them

    // in the spherical [non-rotating] case, the x- and y-VDFs should be symmetric w.r.t. sign change
    if(spherical) {
        ok &= test(vdfi.interpolate(vel.vx, amplvX), vdfi.interpolate(-vel.vx, amplvX), 1e-5, "vdf() f_x(v)==f_x(-v)");
        ok &= test(vdfi.interpolate(vel.vy, amplvY), vdfi.interpolate(-vel.vy, amplvY), 1e-5, "vdf() f_y(v)==f_y(-v)");
    }
    // and the z-VDF should be symmetric in any case (in the face-on orientation)
    if(true)
        ok &= test(vdfi.interpolate(vel.vz, amplvZ), vdfi.interpolate(-vel.vz, amplvZ), 1e-13, "vdf() f_z(v)==f_z(-v)");

    // compare the VDF in a rotated frame for a rotated point with the corresponding moments
    galaxymodel::computeVelocityDistribution<N>(model, ori.toRotated(pos), gridv, gridv, gridv,
        &densvdfrot, &amplvXrot, &amplvYrot, &amplvZrot,  false, ori);
    ok &= test(densvdfrot, densrot, 1e-4, "vdf() density vs. moments() density, rotated");

    // the mean velocity in VDF matches the first moment returned by computeMoments()
    ok &= test(vdfi.integrate(vmin, vmax, amplvXrot, 1) + velnorm, velrot.vx + velnorm, 1e-4, "vdf() mean vx vs. moments(), rotated");
    ok &= test(vdfi.integrate(vmin, vmax, amplvYrot, 1) + velnorm, velrot.vy + velnorm, 1e-4, "vdf() mean vy vs. moments(), rotated");
    ok &= test(vdfi.integrate(vmin, vmax, amplvZrot, 1) + velnorm, velrot.vz + velnorm, 1e-4, "vdf() mean vz vs. moments(), rotated");

    // the dispersion of VDF matches the second moment returned by computeMoments()
    ok &= test(vdfi.integrate(vmin, vmax, amplvXrot, 2), vel2rot.vx2, 2e-4, "vdf() mean vx^2 vs. moments(), rotated");
    ok &= test(vdfi.integrate(vmin, vmax, amplvYrot, 2), vel2rot.vy2, 2e-4, "vdf() mean vy^2 vs. moments(), rotated");
    ok &= test(vdfi.integrate(vmin, vmax, amplvZrot, 2), vel2rot.vz2, 2e-4, "vdf() mean vz^2 vs. moments(), rotated");

    std::ofstream strm;
    if(utils::verbosityLevel >= utils::VL_VERBOSE) {
        std::vector<double> gridvel = math::createUniformGrid(101, vmin, vmax);
        strm.open("test_galaxymodel_"+name+".dat");
        strm << "#v\tf_x(v)  f_y(v)  f_z(v)\trotated:f_x f_y f_z\n";
        for(size_t i=0; i<gridvel.size(); i++) {
            strm << utils::pp(gridvel[i], 7) + '\t' +
            utils::pp(vdfi.interpolate(gridvel[i], amplvX), 7) + ' ' +
            utils::pp(vdfi.interpolate(gridvel[i], amplvY), 7) + ' ' +
            utils::pp(vdfi.interpolate(gridvel[i], amplvZ), 7) + '\t'+
            utils::pp(vdfi.interpolate(gridvel[i], amplvXrot), 7) + ' ' +
            utils::pp(vdfi.interpolate(gridvel[i], amplvYrot), 7) + ' ' +
            utils::pp(vdfi.interpolate(gridvel[i], amplvZrot), 7) + '\n';
        }
    }

    // in the spherical isotropic case, VDF for a rotated point should be the same
    if(isotropic) {
        math::blas_daxpy(-1, amplvX, amplvXrot);
        math::blas_daxpy(-1, amplvY, amplvYrot);
        math::blas_daxpy(-1, amplvZ, amplvZrot);
        double ampldiff = sqrt((math::blas_dnrm2(amplvXrot) + math::blas_dnrm2(amplvYrot) + math::blas_dnrm2(amplvZrot))/3);
        ok &= test(densvdfrot, densvdf, 1e-5, "vdf() density rotated vs. nonrotated");
        ok &= test(1+ampldiff, 1, 1e-5, "vdf() f_{x,y,z} rotated vs. non-rotated");
    }

    // in the spherical isotorpic case, all three VDFs should be identical,
    // i.e. the norm of their difference is close to 0
    if(isotropic) {
        math::blas_daxpy(-1, amplvX, amplvY);
        math::blas_daxpy(-1, amplvX, amplvZ);
        double ampldiff = sqrt((math::blas_dnrm2(amplvY) + math::blas_dnrm2(amplvZ))/2);
        ok &= test(1+ampldiff, 1, 1e-5, "vdf() isotropic f_x==f_y==f_z");
    }


    //---------------------------//
    std::cout << "Projected VDF\n";

    galaxymodel::computeVelocityDistribution<N>(model, posproj, gridsize, gridv,
        &densvdfproj, &amplvX, &amplvY, &amplvZ);
    math::BsplineInterpolator1d<N> vdfp(gridv);
    vmin = gridv.front(), vmax = gridv.back();

    // the density should match the one returned by computeMoments()
    ok &= test(densvdfproj, densproj, 1e-4, "projected vdf() density vs. projected moments() density");
    if(havetruedens)
        ok &= test(densvdfproj, densprojtrue, 2e-5, "projected vdf() density vs. analytic");

    // the VDF should be normalized to unity
    ok &= test(vdfp.integrate(vmin, vmax, amplvX), 1, 1e-13, "projected vdf() f_x(v) normalized to unity");
    ok &= test(vdfp.integrate(vmin, vmax, amplvY), 1, 1e-13, "projected vdf() f_y(v) normalized to unity");
    ok &= test(vdfp.integrate(vmin, vmax, amplvZ), 1, 1e-13, "projected vdf() f_z(v) normalized to unity");

    // the mean velocity in VDF matches the first moment returned by computeMoments()
    ok &= test(vdfp.integrate(vmin, vmax, amplvX, 1) + velnorm, velproj.vx + velnorm, 1e-4, "projected vdf() mean vx vs. projected moments()");
    ok &= test(vdfp.integrate(vmin, vmax, amplvY, 1) + velnorm, velproj.vy + velnorm, 1e-4, "projected vdf() mean vy vs. projected moments()");
    ok &= test(vdfp.integrate(vmin, vmax, amplvZ, 1) + velnorm, velproj.vz + velnorm, 1e-4, "projected vdf() mean vz vs. projected moments()");

    // the dispersion of VDF matches the second moment returned by computeMoments()
    ok &= test(vdfp.integrate(vmin, vmax, amplvX, 2), vel2proj.vx2, 2e-4, "projected vdf() mean vx^2 vs. projected moments()");
    ok &= test(vdfp.integrate(vmin, vmax, amplvY, 2), vel2proj.vy2, 2e-4, "projected vdf() mean vy^2 vs. projected moments()");
    ok &= test(vdfp.integrate(vmin, vmax, amplvZ, 2), vel2proj.vz2, 2e-4, "projected vdf() mean vz^2 vs. projected moments()");

    // in the spherical [non-rotating] case, the x- and y-VDFs should be symmetric w.r.t. sign change
    if(spherical) {
        ok &= test(vdfp.interpolate(vel.vx, amplvX), vdfp.interpolate(-vel.vx, amplvX), 2e-5, "projected vdf() f_x(v)==f_x(-v)");
        ok &= test(vdfp.interpolate(vel.vy, amplvY), vdfp.interpolate(-vel.vy, amplvY), 2e-5, "projected vdf() f_y(v)==f_y(-v)");
    }
    // and z-VDF is always symmetric in the face-on orientation
    if(true)
        ok &= test(vdfp.interpolate(vel.vz, amplvZ), vdfp.interpolate(-vel.vz, amplvZ), 1e-13, "projected vdf() f_z(v)==f_z(-v)");

    // compare the projected VDF in a rotated frame with the projected moments
    galaxymodel::computeVelocityDistribution<N>(model, posproj, gridv, gridv, gridv,
        &densvdfprojrot, &amplvXrot, &amplvYrot, &amplvZrot,  false, ori);
    ok &= test(densvdfprojrot, densprojrot, 5e-4, "projected vdf() density vs. projected moments() density, rotated");

    // the mean velocity in VDF matches the first moment returned by computeMoments()
    ok &= test(vdfp.integrate(vmin, vmax, amplvXrot, 1) + velnorm, velprojrot.vx + velnorm, 1e-4, "projected vdf() mean vx vs. projected moments(), rotated");
    ok &= test(vdfp.integrate(vmin, vmax, amplvYrot, 1) + velnorm, velprojrot.vy + velnorm, 1e-4, "projected vdf() mean vy vs. projected moments(), rotated");
    ok &= test(vdfp.integrate(vmin, vmax, amplvZrot, 1) + velnorm, velprojrot.vz + velnorm, 1e-4, "projected vdf() mean vz vs. projected moments(), rotated");

    // the dispersion of VDF matches the second moment returned by computeMoments()
    ok &= test(vdfp.integrate(vmin, vmax, amplvXrot, 2), vel2projrot.vx2, 3e-4, "projected vdf() mean vx^2 vs. projected moments(), rotated");
    ok &= test(vdfp.integrate(vmin, vmax, amplvYrot, 2), vel2projrot.vy2, 3e-4, "projected vdf() mean vy^2 vs. projected moments(), rotated");
    ok &= test(vdfp.integrate(vmin, vmax, amplvZrot, 2), vel2projrot.vz2, 3e-4, "projected vdf() mean vz^2 vs. projected moments(), rotated");

    // values of interpolated VDF should match those computed by projectedDF
    // marginalized over vX,vY when considering f_z(v_Z), or over vY,vZ when considering f_x(v_X)
    ok &= test(vdfp.interpolate(vel.vx, amplvX) * densvdfproj, projdf_noerrx, 1e-3, "projected vdf() f_x(v) vs. projected df(), no error");
    ok &= test(vdfp.interpolate(vel.vz, amplvZ) * densvdfproj, projdf_noerrz, 1e-3, "projected vdf() f_z(v) vs. projected df(), no error");

    // VDF convolved with a narrow Gaussian and re-interpolated back onto the same B-spline basis
    math::FiniteElement1d<N> fem(gridv);
    math::BandMatrix<double> proj = fem.computeProjMatrix();
    math::Matrix<double>     conv = fem.computeConvMatrix(math::Gaussian(smallerrz.vz));
    std::vector<double> vecconv(amplvZ.size());
    blas_dgemv(math::CblasNoTrans, 1, conv, amplvZ, 0, vecconv);
    std::vector<double> amplvZconv = solveBand(proj, vecconv);  // amplitudes of Gaussian-convolved B-spline representation of f_z(v_Z)
    ok &= test(vdfp.interpolate(vel.vz, amplvZconv) * densvdfproj, projdf_smallerrz, 1e-3, "projected vdf() f_z(v) vs. projected df(), small error");

    // write out the interpolated VDF
    if(utils::verbosityLevel >= utils::VL_VERBOSE) {
        std::vector<double> gridvel = math::createUniformGrid(101, vmin, vmax);
        strm << "\n\n#projected\n#v\tf_x(v)  f_y(v)  f_z(v)\trotated:f_x f_y f_z\n";
        for(size_t i=0; i<gridvel.size(); i++) {
            strm << utils::pp(gridvel[i], 7) + '\t' +
            utils::pp(vdfi.interpolate(gridvel[i], amplvX), 7) + ' ' +
            utils::pp(vdfi.interpolate(gridvel[i], amplvY), 7) + ' ' +
            utils::pp(vdfi.interpolate(gridvel[i], amplvZ), 7) + '\t'+
            utils::pp(vdfi.interpolate(gridvel[i], amplvXrot), 7) + ' ' +
            utils::pp(vdfi.interpolate(gridvel[i], amplvYrot), 7) + ' ' +
            utils::pp(vdfi.interpolate(gridvel[i], amplvZrot), 7) + '\n';
        }
    }

    // in the spherical case, projected VDF should be the same in a rotated frame
    if(spherical) {
        math::blas_daxpy(-1, amplvX, amplvXrot);
        math::blas_daxpy(-1, amplvY, amplvYrot);
        math::blas_daxpy(-1, amplvZ, amplvZrot);
        double ampldiff = sqrt((math::blas_dnrm2(amplvXrot) + math::blas_dnrm2(amplvYrot) + math::blas_dnrm2(amplvZrot))/3);
        ok &= test(densvdfprojrot, densvdfproj, 1e-4, "projected vdf() density rotated vs. nonrotated");
        ok &= test(1+ampldiff, 1, 1e-3, "projected vdf() f_{x,y,z} rotated vs. non-rotated");
    }

    // in the spherical isotorpic case, the three projected VDFs should be identical
    if(isotropic) {
        math::blas_daxpy(-1, amplvX, amplvY);
        math::blas_daxpy(-1, amplvX, amplvZ);
        double ampldiff = sqrt((math::blas_dnrm2(amplvY) + math::blas_dnrm2(amplvZ))/2);
        ok &= test(1+ampldiff, 1, 5e-4, "projected vdf() isotropic f_x==f_y");
    }

    return ok;
}


int main()
{
    bool ok = true;
    potential::Plummer pot(1, 1);
    actions::ActionFinderSpherical af(pot);

    std::cout << "\033[1m  Spherical isotropic Plummer model  \033[0m\n";
    ok &= test(galaxymodel::GalaxyModel(pot, af,
        df::QuasiSphericalCOM(
            potential::Sphericalized<potential::BaseDensity>(pot),
            potential::Sphericalized<potential::BasePotential>(pot))),
        "SphIso", /*havetruedens*/ true, /*havetruevel*/ true, /*isotropic*/ true, /*spherical*/ true);

    std::cout << "\033[1m  Spherical anisotropic Plummer model  \033[0m\n";
    ok &= test(galaxymodel::GalaxyModel(pot, af,
        df::QuasiSphericalCOM(
            potential::Sphericalized<potential::BaseDensity>(pot),
            potential::Sphericalized<potential::BasePotential>(pot), -0.3, INFINITY*0.8)),
        "SphAniso", /*havetruedens*/ true, /*havetruevel*/ false, /*isotropic*/ false, /*spherical*/ true);

    df::DoublePowerLawParam param;
    param.J0       = 1.0;
    param.slopeIn  = 0.5;
    param.slopeOut = 7.0;
    param.steepness= 2.0;
    param.coefJrIn = 1.3;
    param.coefJzIn = 1.2;
    param.coefJrOut= 0.9;
    param.coefJzOut= 1.5;
    param.rotFrac  = 1.0;
    param.Jphi0    = 0.5;
    param.norm     = 5.0;
    //potential::PtrPotential pot1(new potential::MiyamotoNagai(1, 0.75, 0.25));
    //actions::ActionFinderAxisymFudge af1(pot1);
    std::cout << "\033[1m  Flattened rotating DoublePowerLaw model  \033[0m\n";
    ok &= test(galaxymodel::GalaxyModel(pot, af, df::DoublePowerLaw(param)),
        "NonSph", /*havetruedens*/ false, /*havetruevel*/ false, /*isotropic*/ false, /*spherical*/ false);

    if(ok)
        std::cout << "\033[1;32mALL TESTS PASSED\033[0m\n";
    else
        std::cout << "\033[1;31mSOME TESTS FAILED\033[0m\n";
    return 0;
}
