/** \file    test_df_interpolated.cpp
    \author  Eugene Vasiliev
    \date    April 2016

    Test the accuracy of linearly and cubically interpolated distribution functions,
    by comparing the total mass of the model and density values computed at several radii.
*/
#include "galaxymodel.h"
#include "df_halo.h"
#include "df_interpolated.h"
#include "potential_analytic.h"
#include "actions_spherical.h"
#include "math_core.h"
#include "utils.h"
#include <iostream>
#include <fstream>
#include <cmath>

int main()
{
    bool ok = true;
    df::DoublePowerLawParam paramDPL;
    paramDPL.slopeIn   = 0;
    paramDPL.slopeOut  = 7.5;
    paramDPL.J0        = 1;
    paramDPL.coefJrIn  = 0.1;
    paramDPL.coefJzIn  = (3-paramDPL.coefJrIn)/2;
    paramDPL.coefJrOut = 1.2;
    paramDPL.coefJzOut = 0.9;
    paramDPL.norm = 78.19;
    potential::Plummer pot(1., 1.);
    const actions::ActionFinderSpherical af(pot);
    const df::DoublePowerLaw dfO(paramDPL);    // original distribution function
    df::PtrActionSpaceScaling scaling(new df::ActionSpaceScalingTriangLog());
    unsigned int gridSize[3] = {40, 7, 5};
    std::vector<double> gridU(gridSize[0]),
    gridV(math::createUniformGrid(gridSize[1], 0, 1)),
    gridW(math::createUniformGrid(gridSize[2], 0, 1));
    double totalMass = pot.totalMass();
    for(unsigned int i=0; i<gridSize[0]; i++) {
        double r = getRadiusByMass(pot, totalMass * (1 - cos(M_PI * i / gridSize[0])) / 2);
        //std::cout << r << ' ';
        double J = r * v_circ(pot, r);  // r^2*Omega
        double v[3];
        scaling->toScaled(actions::Actions(0,0,J), v);
        gridU[i] = v[0];
    }
    std::vector<double> amplL = df::createInterpolatedDFAmplitudes<1>(dfO, *scaling, gridU, gridV, gridW);
    const df::InterpolatedDF<1> dfL(scaling, gridU, gridV, gridW, amplL); // linearly-interpolated DF
    std::vector<double> amplC = df::createInterpolatedDFAmplitudes<3>(dfO, *scaling, gridU, gridV, gridW);
    const df::InterpolatedDF<3> dfC(scaling, gridU, gridV, gridW, amplC); // cubic-interpolated DF
    std::cout << "Constructed interpolated DFs\n";
    std::ofstream strm, strmL, strmC;
    if(utils::verbosityLevel >= utils::VL_VERBOSE)
        strm.open("test_df_interpolated.dfval");
    for(unsigned int i=0; i<gridSize[0]; i++) {
        for(unsigned int j=0; j<gridSize[1]; j++) {
            for(unsigned int k=0; k<gridSize[2]; k++) {
                double v[3] = {gridU[i], gridV[j], gridW[k]};
                actions::Actions J = scaling->toActions(v);
                double valL = dfL.value(J), valC = dfC.value(J);
                ok &= math::fcmp(valL, valC, 1e-12) == 0;  // should be the same
                strm << v[0] << ' ' << v[1] << ' ' << v[2] << ' ' << valL << ' ' << valC << '\n';
            }
        }
        strm << '\n';
    }
    strm.close();

    double sumLin=0;
    if(utils::verbosityLevel >= utils::VL_VERBOSE)
        strm.open("test_df_interpolated.phasevolL");
    for(unsigned int i=0; i<amplL.size(); i++) {
        double phasevol = dfL.computePhaseVolume(i);
        strm << amplL[i] << ' ' << phasevol << '\n';
        sumLin += amplL[i] * phasevol;
    }
    strm.close();

    double sumCub=0;
    if(utils::verbosityLevel >= utils::VL_VERBOSE)
        strm.open("test_df_interpolated.phasevolC");
    for(unsigned int i=0; i<amplC.size(); i++) {
        double phasevol = dfC.computePhaseVolume(i);
        strm << amplC[i] << ' ' << phasevol << '\n';
        sumCub += amplC[i] * phasevol;
    }
    strm.close();

    double massOrig = dfO.totalMass();
    double massLin  = dfL.totalMass();
    double massCub  = dfC.totalMass();
    std::cout << "M=" << pot.totalMass() << ", dfOrig M=" << massOrig <<
        ", dfInterLin M=" << massLin << ", sum over components=" << sumLin << 
        ", dfInterCub M=" << massCub << ", sum over components=" << sumCub << '\n';
    ok &= math::fcmp(massOrig, massLin, 2e-2)==0 && math::fcmp(sumLin, massLin, 1e-3)==0 &&
          math::fcmp(massOrig, massCub, 2e-2)==0 && math::fcmp(sumCub, massCub, 1e-3)==0;

    if(utils::verbosityLevel >= utils::VL_VERBOSE) {
        strm.open ("test_df_interpolated.densval");
        strmL.open("test_df_interpolated.denscompL");
        strmC.open("test_df_interpolated.denscompC");
    }
    for(double r=0; r<20; r<1 ? r+=0.1 : r*=1.1) {
        coord::PosCyl point(r, 0, 0);
        std::vector<double> densArrL(dfL.size()), densArrC(dfC.size());
        double densOrig, densIntL, densIntC;
        computeMoments(galaxymodel::GalaxyModelMulticomponent(pot, af, dfL),
            point, &densArrL[0], NULL, NULL, NULL, NULL, NULL, 1e-3, 10000);
        computeMoments(galaxymodel::GalaxyModelMulticomponent(pot, af, dfC),
            point, &densArrC[0], NULL, NULL, NULL, NULL, NULL, 1e-3, 10000);
        computeMoments(galaxymodel::GalaxyModel(pot, af, dfO),
            point, &densOrig,    NULL, NULL, NULL, NULL, NULL, 1e-3, 100000);
        computeMoments(galaxymodel::GalaxyModel(pot, af, dfL),
            point, &densIntL,    NULL, NULL, NULL, NULL, NULL, 1e-3, 100000);
        computeMoments(galaxymodel::GalaxyModel(pot, af, dfC),
            point, &densIntC,    NULL, NULL, NULL, NULL, NULL, 1e-3, 100000);
        double densSumL=0;
        for(unsigned int i=0; i<densArrL.size(); i++) {
            densSumL += densArrL[i];
            strmL << densArrL[i] << '\n';
        }
        strmL << '\n';
        double densSumC=0;
        for(unsigned int i=0; i<densArrC.size(); i++) {
            densSumC += densArrC[i];
            strmC << densArrC[i] << '\n';
        }
        strmC << '\n';
        strm << r << ' ' << pot.density(point) << '\t' << densOrig << '\t' <<
            densIntL << ' ' << densSumL << '\t' << densIntC << ' ' << densSumC << '\n';
        ok &= math::fcmp(densOrig, densIntL, 2e-1)==0 && math::fcmp(densSumL, densIntL, 1e-2)==0 && 
              math::fcmp(densOrig, densIntC, 5e-2)==0 && math::fcmp(densSumC, densIntC, 1e-2)==0;
        if(!ok) std::cout << "failed at r="<<r<<"\n";
    }

    if(ok)
        std::cout << "\033[1;32mALL TESTS PASSED\033[0m\n";
    else
        std::cout << "\033[1;31mSOME TESTS FAILED\033[0m\n";
    return 0;
}
