/***************************************************************************//**
\file  Constants.h 
\brief Contains namespace GalactoConstants. 
Gives various parameters of the Galaxy and observing systems.

*                                                                              *
*  C++ code written by Walter Dehnen, 1995/96.                                 *
*  Oxford University, Department of Physics, Theoretical Physics.              *
*  address: 1 Keble Road, Oxford OX1 3NP, United Kingdom.                      *
*  e-mail:  dehnen.physics.ox.ac.uk                                            *
*                                                                              *
*******************************************************************************/
/*

Various constants used all over the place. Probably unwise to store them 
here in truth.


 */

#ifndef _Galactic_Constants_def_
#define _Galactic_Constants_def_ 1

#include "Units.h"

namespace torus{
/**
\brief Contains default parameters of the Galaxy (e.g. R_0, v_0) and for conversions between coordinate systems

 */

namespace GalactoConstants {

  const double EtoPB1950_0[3] = {-0.0669887394, 0.4927284661, -0.8676008112};
  const double EtoPB1950_1[3] = {-0.8727557659, -0.4503469580, -0.1883746017};
  const double EtoPB1950_2[3] = {-0.4835389146, 0.7445846333, 0.4601997848};
  const double* const EtoPB1950[3] = {EtoPB1950_0,EtoPB1950_1,EtoPB1950_2};

  const double  EtoPJ1991_0[3] = {-0.0548755604, 0.4941094279, -0.8676661490};
  const double  EtoPJ1991_1[3] = {-0.8734370902,-0.4448296300, -0.1980763734};
  const double  EtoPJ1991_2[3] = {-0.4838350155, 0.7469822445,	0.4559837762};
  const double* const EtoPJ1991[3] = {EtoPJ1991_0,EtoPJ1991_1,EtoPJ1991_2};

  const double  EtoPJ2000_0[3] ={-0.0548761806632,0.4941094158461,
				 -0.867666116641649};
  const double  EtoPJ2000_1[3] ={-0.8734369590164, -0.4448300538949, -0.198076};
  const double  EtoPJ2000_2[3] ={-0.4838351820817,  0.746982,         0.455984};

  const double* const EtoPJ2000[3] = {EtoPJ2000_0,EtoPJ2000_1,EtoPJ2000_2};


  const double 	
    Rsun_in_kpc 	= 8.5   ,
    zsun_in_kpc 	= 0.014,
    vcsun_in_kms	= -244.5 , // McMillan 2011: convenient
    usun_in_kms 	= 11.1 ,   // Schonrich Binney Dehnen
    vsun_in_kms 	= 12.24,
    wsun_in_kms 	= 7.25 ,
    Zsun		= 0.014, //temporary!!
    Rsun	 	= Rsun_in_kpc  * Units::kpc,
    zsun 		= zsun_in_kpc  * Units::kpc,
    vcsun		= vcsun_in_kms * Units::kms,
    usun 		= usun_in_kms  * Units::kms,
    vsun 		= vsun_in_kms  * Units::kms,
    wsun 		= wsun_in_kms  * Units::kms;

}

//
// Note, that the galaxy rotates in negative mathematical sense, hence
// the negative circular velocity. The v-velocity in the local standard
// of rest, however, is defined in the direction of galactic rotation,
// and the sun's peculiar motion has positive v-component.
//

//GalactoConstants Con;

} // namespace
#endif
