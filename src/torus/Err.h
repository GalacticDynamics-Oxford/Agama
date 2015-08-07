/***************************************************************************//**
\file Err.h
\brief Error handling code. Barely used. Very much antiquated. Should definitely be brought up to date.

*                                                                              *
*  Err.h                                                                       *
*                                                                              *
* C++ code written by Walter Dehnen, 1994/95,                                  *
* Oxford University, Department of Physics, Theoretical Physics.               *
* address: 1 Keble Road, Oxford OX1 3NP, United Kingdom                        *
* e-mail:  p.mcmillan1@physics.ox.ac.uk                                        *
*                                                                              *
*******************************************************************************/
/*

Barely used, in truth

*/

#ifndef _Torus_Err_def_
#define _Torus_Err_def_ 1

#include <cstdlib>
#include <string>
#include <stdexcept>

namespace Torus{

inline void TorusError(std::string m, int i)
{
    throw std::runtime_error("Torus error "+char(i+48)+m);
}

// global variable indicating the error occured (not very elegant but most
// C++ compilers do currently not support exception handling ...)
// meaning of values:  0  anything but the following
//                     1  error in auxiliary functions (in Aux.h)
//                     2  error in classes IM_par or IsoMap (in Iso.h)
//                     3  error in classes PT_par or PointTrans (in Poi.h)
//                     4  error in classes GF_par, GenFunc, GenFuncFit,
//                        or AngMap (in Gen.h)
//                     5  error in potential

} // namespace
#endif
