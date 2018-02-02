/***************************************************************************//**
\file Fit.h
\brief Contains all the routines that do the Torus fitting.

									     */
/*
*                                                                              *
* Fit.h                                                                        *
*                                                                              *
* C++ code written by Walter Dehnen, 1995-96,                                  *
*                     Paul McMillan, 2007-10                                   *
* e-mail: paul@astro.lu.se                                                     *
* github: https://github.com/PaulMcMillan-Astro/Torus                          *
*                                                                              *
********************************************************************************
*                                                                              *
* routines and their purpose                                                   *
* SbyLevMar()   Fits parameters of canonical transformation and/or of toy map  *
*               and/or the coefficients S_n of the generating function.        *
*                                                                              *
* Omega()       given output of SbyLevMar(), computes frequencies              *
*                                                                              *
* dSbyInteg()   given both output of SbyLevMar() and an estimate of Omega with *
*               given uncertainty, computes dS/dJ and frequencies subject to   *
*               the constraint of the given Omega and its uncertainty          *
*                                                                              *
* FullFit()     Combines SbyLevMar(), Omega(), and dSbyInteg() in the proper   *
*	        way given below                                                *
*                                                                              *
*                                                                              *
********************************************************************************
*                                                                              *
* The proper way to establish a torus fit is the following                     *
*                                                                              *
* (i)     fit the parameters of can. transf. and toy map using SbyLevMar()     *
* (ii)    fit these parameters AND the S using SbyLevMar()                     *
* (iii)    fit dS/dJ and Omega by dSbyInteg()                                  *
*                                                                              *
*******************************************************************************/

#ifndef _TorusFit_
#define _TorusFit_ 1

#include "GeneratingFunction.h"
#include "Potential.h"

namespace torus{
/**
   \brief Fits parameters of canonical transformation and/or of toy map  
   and/or the coefficients S_n of the generating function. 
 */
int SbyLevMar(          // return:	error flag (see below)
    const Actions&,     // input:	Actions of Torus to be fit
    Potential*,         // input:	pointer to Potential
    const int,          // input:	option for fit (see below)
    const int,          // input:	# of theta_r for fit
    const int,          // input:	# of theta_th for fit
    const int,          // input:	max. # of iterations
    const double,       // input:	stop if   dH_rms     < tol1 ...
    const double,       // input:	AND  if   |dch^2/dp| < tol2
    GenPar&,            // in/output:	parameters of generating function
    PoiTra&,            // in/output:	canonical map with parameters
    ToyMap&,            // in/output:	toy-potential map with parameters
    double&,            // in/output:	lambda for the iterations
    double&,            // output:	mean H
    double&,            // output:	rms deviation from mean
    int&,               // output:	# of occurence of negative actions
    const double =0,    // input:	estimate of expected <H>
    const int=0);	// input:	error output ?
//  meaning of return: 	positive = 	number of iterations done
//                     	-1       	error in input (eg. option)
//                     	-2       	negativ actions already for input
//                     	-3       	singular matrix occuring (very strange)
//		       	-4       	dchi^2/da > 10^3 or <H> differs by more
//					than 100% from its estimated expection
//					if given or its first value.
//
//  meaning of option: 	0        	fit everthing
//                     	add 1    	don't fit GenPar
//                     	add 2    	don't fit ToyMap parameters
//                     	add 4    	don't fit PoiTra parameters


//------------------------------------------------------------------------------

/** 
    \brief Given output of SbyLevMar(), computes frequencies by orbit 
    integration 
*/ 
int Omega(          	// return:	error flag (see below)
    Potential*,         // input:	pointer to Potential
    const Actions&,     // input:	J
    const GenPar&,      // input:	parameters of generating function
    const PoiTra&,      // input:	canonical map with parameters
    const ToyMap&,      // input:	toy-potential map with parameters
    const double,       // input:	start toy angle th_r
    const double,       // input:	start toy angle th_l
    const double,       // input:	difference in th to integrate over
    Frequencies&,	// output:	estimates of Omega_r, Omega_l, Omega_phi
    double&,            // output:	delta Omega_rl
    double&);           // output:	delta Omega_phi
//  meaning of return:   0		everything seemed to be ok
//                      -1              too many time steps (never gets there)
//		        -2		error in backward maps

//------------------------------------------------------------------------------

/**  \brief given both output of SbyLevMar and an estimate of Omega with 
     given uncertainty, computes dS/dJ and frequencies subject to   
     the constraint of the given Omega and its uncertainty          
*/
int dSbyInteg(          // return:	error flag (see below)
    const Actions&,     // input:	Actions of Torus to be fit
    Potential*,         // input:	pointer to Potential
    const int,          // input:	# of grid cells in Pi
    const GenPar&,      // input:	parameters of generating function
    const PoiTra&,      // input:	canonical map with parameters
    const ToyMap&,      // input:	toy-potential map with parameters
    const double,       // input:	estimate of delta Omega
    Frequencies&,	// in/output:	estimate of / fitted (Omega_r, Omega_l)
    Errors&,	        // output:	0:empty, 1,2,3: chi_rms for fit dSn/dJi
    AngPar&,            // output:	dSn/dJr & dSn/dJl & dSn/dJphi
    const int   =200,   // input:	max. tolerated steps on average per cell
    const int   =0);    // input:	error output ?
//  meaning of return:   0		everything seemed to be ok
//                      -1		N too small
//                      -2		some error occured in the backward map,
//			 		most probably H>0 in ToyMap::Backward()
//                      -3		error in orbit integration (too long)
//		       	-4		neg. Omega => reduce Energy tolerance
//		       	-5		M^tM not pos.def. => something wrong
 
/**  \brief given both output of SbyLevMar and an estimate of Omega with 
 given uncertainty, computes dS/dJ and frequencies subject to   
 the constraint of the given Omega and its uncertainty.
 same interface as dSbyInteg, but use a different method
 */
int dSbySampling(       // return:	error flag (see below)
    const Actions&,     // input:	Actions of Torus to be fit
    Potential*,         // input:	pointer to Potential
    const int,          // input:	# of grid cells in Pi
    const GenPar&,      // input:	parameters of generating function
    const PoiTra&,      // input:	canonical map with parameters
    const ToyMap&,      // input:	toy-potential map with parameters
    const double,       // input:	estimate of delta Omega
    Frequencies&,	// in/output:	estimate of / fitted (Omega_r, Omega_l)
    Errors&,	        // output:	0:empty, 1,2,3: chi_rms for fit dSn/dJi
    AngPar&,            // output:	dSn/dJr & dSn/dJl & dSn/dJphi
    const int   =200,   // input:	max. tolerated steps on average per cell
    const int   =0);    // input:	error output ?
    
//------------------------------------------------------------------------------

/** \brief dSbyInteg for the special case of Jz=0 (where the problem
    is lower dimensional)
*/
int z0dSbyInteg(          // return:	error flag (see below)
    const Actions&,     // input:	Actions of Torus to be fit
    Potential*,         // input:	pointer to Potential
    const int,          // input:	# of grid cells in Pi
    const GenPar&,      // input:	parameters of generating function
    const PoiTra&,      // input:	canonical map with parameters
    const ToyMap&,      // input:	toy-potential map with parameters
    const double,       // input:	estimate of delta Omega
    Frequencies&,	// in/output:	estimate of / fitted (Omega_r, Omega_l)
    Errors&,    	// output:	0:empty, 1,2,3: chi_rms for fit dSn/dJi
    AngPar&,            // output:	dSn/dJr & dSn/dJl & dSn/dJphi
    const int   =200,   // input:	max. tolerated steps on average per cell
    const int   =0);    // input:	error output ?
//  meaning of return:   0		everything seemed to be ok
//                      -1		N too small
//                      -2		some error occured in the backward map,
//			 		most probably H>0 in ToyMap::Backward()
//                      -3		error in orbit integration (too long)
//		       	-4		neg. Omega => reduce Energy tolerance
//		       	-5		M^tM not pos.def. => something wrong




//------------------------------------------------------------------------------

/**
\brief Fit the working components of a Torus in the general case. Does everything.
 */

int AllFit(		// return:	error flag (see below)
    const Actions&,     // input:	Actions of Torus to be fit
    Potential*,	   	// input:	pointer to Potential
    const double,	// input:	goal for |dJ|/|J|
    const int,		// input:	max. number of S_n
    const int,		// input:	max. number of iterations
    const int,		// input:	overdetermination of eqs. for angle fit
    const int,    	// input:	min. number of cells for fit of dS/dJ
    PoiTra&,            // in/output:	canonical map with parameters
    ToyMap&,            // in/output:	toy-potential map with parameters
    GenPar&,            // in/output:	parameters of generating function
    AngPar&,            // output:	dSn/dJr & dSn/dJl
    Frequencies&,	// in/output:	estimates of O_r, O_l, O_phi
    double&,            // output:	mean H
    Errors&,	        // output:	|dJ|/|J|, chirms of angle fits
    const int,          // input:       Full (0) or Half (1) fit
    const bool,         // input:       Safe (vary one thing at a time) fit?
    const int,          // input:   Number of tailorings
    const int     =200, // input:	max. tolerated steps on average per cell
    const double  =0.,  // input:	estimate of expected <H>
    const int     =24,  // input:	min No of theta (per dim) for 1. fit
    const int     =0,   // input:	error output?
    const bool useNewAngMap=false);  // input:   whether to use new method for angle mapping

//  meaning of return:	 0		everything seemed to go well
//			-1		something wrong with input, usually
//					bad starting values for the parameters
//			-2		Fit failed the goal by a factor <= 2
//			-3		Fit failed the goal by more than that
//			-4 		Fit aborted: serious problems occured

//------------------------------------------------------------------------------

/**
\brief Fit the working components of a torus in the special case of low Jz

Basically this differs from the normal fit in that first the (J_R, 0,
J_phi) orbit is fit to high accuracy. Then these Sn are used as a
starting point for fitting the full orbit. The first attempt to fit
the full orbit uses only these terms and ones with n_z=0 (i.e. no
cross-terms), then these are added.
 */

int LowJzFit(		// return:	error flag (see below)
    const Actions&,     // input:	Actions of Torus to be fit
    Potential*,	   	// input:	pointer to Potential
    const double,	// input:	goal for |dJ|/|J|
    const int,		// input:	max. number of S_n
    const int,		// input:	max. number of iterations
    const int,		// input:	overdetermination of eqs. for angle fit
    const int,    	// input:	min. number of cells for fit of dS/dJ
    PoiTra&,            // in/output:	canonical map with parameters
    ToyMap&,            // in/output:	toy-potential map with parameters
    GenPar&,            // in/output:	parameters of generating function
    AngPar&,            // output:	dSn/dJr & dSn/dJl
    Frequencies&,	// in/output:	estimates of O_r, O_l, O_phi
    double&,            // output:	mean H
    Errors&,	        // output:	|dJ|/|J|, chirms of angle fits
    const int,          // input:       Full (0) or Half (1) fit
    const int,          // input:       Number of tailorings
    const int     =200, // input:	max. tolerated steps on average per cell
    const double  =0.,  // input:	estimate of expected <H>
    const int     =24,  // input:	min No of theta (per dim) for 1. fit
    const int     =0,   // input:	error output?
    const bool useNewAngMap=false);  // input:   whether to use new method for angle mapping

//  meaning of return:	 0		everything seemed to go well
//			-1		something wrong with input, usually
//					bad starting values for the parameters
//			-2		Fit failed the goal by a factor <= 2
//			-3		Fit failed the goal by more than that
//			-4 		Fit aborted: serious problems occured


/**
\brief Fit the working components of a torus in the case where a point
transform is required (i.e. because the orbit is nearly a shell
orbit).

 */

int PTFit(		// return:	error flag (see below)
    const Actions&,     // input:	Actions of Torus to be fit
    Potential*,	   	// input:	pointer to Potential
    const double,	// input:	goal for |dJ|/|J|
    const int,		// input:	max. number of S_n
    const int,		// input:	max. number of iterations
    const int,		// input:	overdetermination of eqs. for angle fit
    const int,    	// input:	min. number of cells for fit of dS/dJ
    PoiTra&,            // in/output:	canonical map with parameters
    ToyMap&,            // in/output:	toy-potential map with parameters
    GenPar&,            // in/output:	parameters of generating function
    AngPar&,            // output:	dSn/dJr & dSn/dJl
    Frequencies&,	// in/output:	estimates of O_r, O_l, O_phi
    double&,            // output:	mean H
    Errors&,	        // output:	|dJ|/|J|, chirms of angle fits
    const int,          // input:       Full (0) or Half (1) fit
    const int,          // input:       Number of tailorings
    const int     =200, // input:	max. tolerated steps on average per cell
    const double  =0.,  // input:	estimate of expected <H>
    const int     =24,  // input:	min No of theta (per dim) for 1. fit
    const int     =0,   // input:	error output?
    const bool useNewAngMap=false);  // input:   whether to use new method for angle mapping

//  meaning of return:	 0		everything seemed to go well
//			-1		something wrong with input, usually
//					bad starting values for the parameters
//			-2		Fit failed the goal by a factor <= 2
//			-3		Fit failed the goal by more than that
//			-4 		Fit aborted: serious problems occured

} // namespace
#endif
