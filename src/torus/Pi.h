/***************************************************************************//**
\file Pi.h
\brief Contains value of Pi and various related values (e.g. 1/sqrt(2 Pi)).
You'd be amazed how often this is useful.

*                                                                              *
* Pi.h                                                                         *
*                                                                              *
* C++ code written by Walter Dehnen, 1994/95,                                  *
* Oxford University, Department of Physics, Theoretical Physics.               *
* address: 1 Keble Road, Oxford, OX1 3NP, United Kingdom                       *
* e-mail:  dehnen@thphys.ox.ac.uk                                              *
*                                                                              *
*******************************************************************************/

#ifndef _Pi_def_
#define _Pi_def_ 1
namespace torus {

const double Pi   = 3.14159265358979323846264338328;
const double Pih  = 0.5  * Pi;
const double Piq  = 0.25 * Pi;
const double Pi3h = 3.   * Pih;
const double TPi  = 2.   * Pi;
const double FPi  = 4.   * Pi;
const double iTPi  = 1./TPi;

const double SPi   = 1.772453850905516027298167483341;	// Sqrt[Pi]
const double STPi  = 2.506628274631000502415765284811;  // Sqrt[2 Pi]
const double iSTPi = 1./STPi;                           // 1./Sqrt[2 Pi]

//const complex<double> IMAG = complex<double>(0,1);	// i
//const complex<double> ITPi = IMAG * TPi;		// 2 i Pi

} // namespace
#endif
