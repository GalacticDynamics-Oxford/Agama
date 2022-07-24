// -------------------------------------------------------------------------------------
//
//  Fit2.cc
//
//  Created by Brian Jia Jiunn Khor on 16/07/2015.
//  Copyright (c) 2015 Brian Jia Jiunn Khor. All rights reserved.
//
//  This code implements method of computing frequencies and dS/dJ introduced by Mikko
//  Kaasalainen & Laakso Teemu (2014), supervised by James Binney, Payel Das, and
//  Eugene Vasilyev in 2015 (Oxford Theoretical Galactic Dynamics Research Group)
//
//  This code is built on previous torus machinery developed by Paul McMillan and
//  Walter Dehnen
//
//  Source code for Paul McMillan can be obtained from github
//  github: http://github.com/PaulMcMillan-Astro/Torus
//
// --------------------------------------------------------------------------------------
//
//  This code computes values of dS_k / dJT and Omega (Frequencies at each Toy Angle)
//
//  The numerical value of each dS_k / dJT will then be used in angle mapping (Th -> Tht)
//
// --------------------------------------------------------------------------------------

#include "Toy_Isochrone.h"
#include "Fit.h"
#include <cassert>
#include <cmath>
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>

/////////////////////////////////////////////////////////////////////////////////////////
// typedefs and constants
namespace torus{

/////////////////////////////////////////////////////////////////////////////////////////

// (q,p) = toy phase space coordinates, (x,v) = real phase space coordinates
// J = real action, j = toy action

inline double Heff(const PSPD& xv, double dHdxv[4], const Potential* Phi)
{
    dHdxv[2] = double(xv(2));
    dHdxv[3] = double(xv(3));
    return 0.5*(pow(dHdxv[2],2) +pow(dHdxv[3],2))
        + Phi->eff(double(xv(0)),double(xv(1)),dHdxv[0],dHdxv[1]);
}

int dHdj_toy(const ToyMap &TM, const PSPD& jt, const PoiTra &PT, const Potential *Phi,
             const double Jphi, double dHdj[3])
// dH/dj = dH/d(x,v)*d(x,v)/d(q,p)*d(q,p)/dj
{
    PSPD qp = jt >> TM;
    PSPD xv = qp >> PT;
    double dqpdj[4][2], dxvdqp[4][4], dHdxv[4], dHdqp[4];
    double dqpdA[4][4];
    double* dqpdAp[4] = {dqpdA[0], dqpdA[1], dqpdA[2], dqpdA[3]};
    for (int i=0; i<4; i++) {
        dHdqp [i] = 0.;
    }
    TM.Derivatives(dqpdj, dqpdAp);
    PT.Derivatives(dxvdqp);
    //double H;
    /*H =*/ Heff(xv, dHdxv, Phi);
    for(int j=0; j<4; j++){
        for(int k=0; k<4; k++){
            dHdqp[j] += dHdxv[k] * dxvdqp[k][j];
        }
    }
    dHdj[0] = dHdj[1] = 0.;
    dHdj[2] = Jphi / (xv(0) * xv(0));  // H = ... + Lz^2/(2R^2), so dH/dLz = ... + Lz/R^2
    for(int k=0; k<4; k++) {
        dHdj[0] += dHdqp[k] * dqpdj[k][0];
        dHdj[1] += dHdqp[k] * dqpdj[k][1];
        dHdj[2] += dHdqp[k] * dqpdAp[k][2] * (Jphi>0?1:-1);  // derivs w.r.t. Jphi (=Lz)
    }
    return 0;
}

int fill_row_dHdSn(const double dHdj [2], const GenPar &dj1dS, const GenPar &dj2dS,
                   gsl_matrix* X, int row)
// dH/dS_k = dH/dj*dj/dS_k
// dji/dS_k = 2* k_i * cos ( k_1*th_1 + k_2*th_2 )
// compute rows with first input 1 and the rest of terms -dH/dS_k
{
    assert(dj1dS.NumberofTerms() == dj2dS.NumberofTerms());
    int numberofterms = dj1dS.NumberofTerms();
    gsl_matrix_set(X, row, 0, 1.0);
    for (int j = 0; j < numberofterms; j++){
        double dHdSk = dHdj[0]*dj1dS(j) + dHdj[1]*dj2dS(j);
        gsl_matrix_set(X, row, j+1, -dHdSk);
        // First element in a row is filled as 1, the rest are -dH/dSk
        // Rows are inserted into a matrix
    }
    return 0;
}

int dSbySampling (            // return:     error flag (see below)
    const Actions& J,         // Input:      Actions of Torus to be fit
    Potential* Phi,           // Input:      pointer to Potential
    const int Nr,             // Input:      # of grid cells in Pi
    const GenPar& Sn,         // Input:      parameters of generating function
    const PoiTra& PT,         // Input:      canonical map with parameters
    const ToyMap& TM,         // Input:      toy-potential map with parameters
    const double  /*dO*/,     // Input:      delta Omega
    Frequencies   &Om,        // In/Output:  Omega_r, Omega_l
    Errors   &chi,            // Output:     chi_rms for fit of dSn/dJi
    AngPar& Ap,               // Output:     dSn/dJr & dSn/dJl
    const int /*IperCell*/,   // Input:      max tol steps on average per cell
    const int /*err*/)        // Input:      error output?
//==============================================================================
// meaning of return:  0       -> everything went ok
//                    -1       -> N too small, e.g., due to:
//                                - H>0 occured too often in backward ToyIsochrone
//                    -4       -> negative Omega => reduce Etol
//                    -5       -> Matrix M^tM not pos. def. => something wrong
//                    -6       -> max{(dS/dJ)_n} > 1
//==============================================================================
{
    gsl_vector *T1, *T2, *T3, *y1, *y2, *y3, *S, *work;
    gsl_matrix *X, *V, *Xprime;
    GenFncFit GF(Sn, Nr, Nr);
    Ap = AngPar(Sn,Sn,Sn);

    // Initialise and allocate memory
    int Rdim = Sn.NumberofTerms() + 1; //Dimension of each row
    int ydim = Nr*Nr; //Dimension of y = dH/dj (no of points in toy angle space)

    // Check to see the system is overdetermined set of linear eqn. If the
    // no of unknowns is more than half of no. of eqn (no of point in toy
    // angle space sampled), abort it.

    if (Rdim > 0.5*ydim ){
        return -1;
    }

    T1 = gsl_vector_alloc(Rdim); // solution vector of Omega_r and dSn/dJ_r
    T2 = gsl_vector_alloc(Rdim); // solution vector of Omega_l and dSn/dJ_l
    T3 = gsl_vector_alloc(Rdim); // solution vector of Omega_phi and dSn/dJ_phi
    y1 = gsl_vector_alloc(ydim); // RHS of equation, dHdj_r
    y2 = gsl_vector_alloc(ydim); // RHS of equation, dHdj_l
    y3 = gsl_vector_alloc(ydim); // RHS of equation, dHdj_phi
    X  = gsl_matrix_alloc(ydim, Rdim); // Matrix X is for r- and l-component
    Xprime = gsl_matrix_alloc(ydim, Rdim); // A copy of matrix X
    V = gsl_matrix_alloc(Rdim, Rdim); // Matrix needed to perform SVD on X
    work = gsl_vector_alloc(Rdim); // workspace vector for SVD
    S = gsl_vector_alloc(Rdim); // Vector needed for SVD of X

    // Create grid of points in toy angle space
    for (int i2 = 0; i2 < Nr; i2++){
        for (int i1 = 0; i1 < Nr; i1++) {
            // Make sure there's point everywhere on the grid, 0 < theta < pi
            GenPar dj1dS(GF.parameters()), dj2dS(GF.parameters());
            PSPD jt;

            // 1. Compute toy actions and angles from real actions and toy angles
            // by applying generating function
            jt = GF.MapWithDerivs(J(0),J(1),i1,i2,dj1dS,dj2dS);
            if(jt(0)<0.) jt[0]=0;  // ensure that toy actions are non-negative
            if(jt(1)<0.) jt[1]=0;

            // 2. Compute dHdj and dHdSk for each point in toy angle space
            double dHdj[3];
            dHdj_toy(TM, jt, PT, Phi, J(2), dHdj);
            int row = i2*Nr + i1;

            // 3. Fill a big matrix with rows of 1 and - dH/dSk
            //    each row correspond to a point on toy angle space
            fill_row_dHdSn(dHdj, dj1dS, dj2dS, X, row);

            // Filling RHS of equation
            gsl_vector_set (y1, row, dHdj[0]);
            gsl_vector_set (y2, row, dHdj[1]);
            gsl_vector_set (y3, row, dHdj[2]);
        }
    }

    // Xprime is a copy of X. Copying is necessary because X will be overwritten
    // in singular matrix decomposition. Xprime is used in | X*Ti - yi | to
    // calculate error chi
    gsl_matrix_memcpy (Xprime, X);
    
    // 4. Numerically solve for frequencies and dS/dJ, solutions assign to T1, T2
    //    and then assign Frequencies, AngPar, and Error to solution respectively
    gsl_linalg_SV_decomp (X, V, S, work);
    gsl_linalg_SV_solve (X, V, S, y1, T1); // X now is decomposed matrix
    gsl_linalg_SV_solve (X, V, S, y2, T2);
    gsl_linalg_SV_solve (X, V, S, y3, T3);

    // Assigning dS/dJ
    for (int i=0; i < Sn.NumberofTerms(); i++){
        Ap.dSdJ1(i, gsl_vector_get(T1, i + 1));
        Ap.dSdJ2(i, gsl_vector_get(T2, i + 1));
        Ap.dSdJ3(i, gsl_vector_get(T3, i + 1));
    }

    // Assigning Frequencies
    Om [0] = gsl_vector_get(T1,0);
    Om [1] = gsl_vector_get(T2,0);
    Om [2] = gsl_vector_get(T3,0);

    // Computing error, T1, T2, T3 will be overwriten here but it doesn't matter
    gsl_blas_dgemv (CblasNoTrans, 1., Xprime, T1, -1., y1); // X*T1 - y1
    gsl_blas_dgemv (CblasNoTrans, 1., Xprime, T2, -1., y2); // X*T2 - y2
    gsl_blas_dgemv (CblasNoTrans, 1., Xprime, T3, -1., y3); // X*T3 - y3
    // Then take dot product of residual, this results in sum of residual squared
    // which is chi_square
    gsl_blas_ddot (y1, y1, &chi[1]); // (X*T1 - y1)*(X*T1 - y1) = sum(chi^2) (for dSn/dJ1)
    gsl_blas_ddot (y2, y2, &chi[2]); // (X*T2 - y2)*(X*T2 - y2) = sum(chi^2) (for dSn/dJ2)
    gsl_blas_ddot (y3, y3, &chi[3]); // (X*T3 - y3)*(X*T3 - y3) = sum(chi^2) (for dSn/dJ3)
    // Then divide by total no. of equations and take square root
    chi[1] = sqrt(chi[1]/(Nr*Nr)); // real error estimate
    chi[2] = sqrt(chi[2]/(Nr*Nr));
    chi[3] = sqrt(chi[3]/(Nr*Nr));
    chi[0] = 0;  // not used

    // 5. clean up
    gsl_vector_free(T1);
    gsl_vector_free(T2);
    gsl_vector_free(T3);
    gsl_vector_free(y1);
    gsl_vector_free(y2);
    gsl_vector_free(y3);
    gsl_vector_free(S);
    gsl_vector_free(work);
    gsl_matrix_free(X);
    gsl_matrix_free(V);
    gsl_matrix_free(Xprime);

    // Check for sensible solutions
    if(Om(0)<0. || Om(1)<0.) {
        return -4;
    }

    if(Ap.dSdJ1().maxS()>1.f || Ap.dSdJ2().maxS()>1.f){
        return -6;
    }
    return 0;
}

}