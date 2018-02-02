/***************************************************************************//**
\file Units.h
\brief Contains namespace Units
Gives various units in terms of code units (which are kpc,Myr,Msun)


                                                                              
  Units.h                                                                     
                                                                              
 C++ code written by Walter Dehnen, 1995/96,                                  
 Oxford University, Department of Physics, Theoretical Physics.               
 address: 1 Keble Road, Oxford OX1 3NP, United Kingdom                        
 e-mail:  p.mcmillan1@physics.ox.ac.uk                                        
                                                               
                                             
                                                             
*******************************************************************************/

#ifndef _Units_set_
#define _Units_set_ 1

#include "Pi.h"
#include <string>
namespace torus{
using std::string;

/**
\brief Gives various units in terms of code units (which are kpc,Myr,Msun)

                                     
********************************************************************************
                                                                              
 The unit system employed has:                                                
                                                                              
 angle:      measured in radian,                                              
 length:     measured in kilo parsec,                                         
 time:       measured in Mega years,                                          
 mass:       measured in solar masses.                                        
                                                                              
 This implies the following dimensions:                                        
                                                                              
 quantity        dimension / size                using other units         

 angular vel.  1 Myr^-1	                         = 977.775320024919 km/s/kpc  

 velocity      1 kpc/Myr                         = 977.775320024919 km/s   
   
 action/mass   1 kpc^2/Myr                       = 977.775320024919 kpc*km/s  

 potential     1 kpc^2/Myr^2                     = 956044.576449833 (km/s)^2  

 acceleration  1 kpc/Myr^2                                                 
   
 G             4.49865897 E-12 kpc^3/Myr^2/Msun                   
            
 4 Pi G        5.65318158 E-11 kpc^3/Myr^2/Msun                               
                                                                              
 Note that this implies handy numbers for typical quantities in the Galaxy:   
                                                                              
 velocities 		are of the order of 0.1 to 0.4                         
 radii      		are of the order of 0.1 to 100                         
 dynamical times	are of the order of 1   to 1000                        
 G times total mass    is  of the order of 1            


So, for example if you want to input a value of 200 km/s, you write it
as 200. * Units::kms. If you want to know what a given velocity is in
km/s you can output it as velocity_name * Units::kms_i (or,
equivalently, velocity_name/Units::kms).
*/

namespace Units {
  // names of basic units
  const string 
    angle_unit    	= "radian",
    length_unit   	= "kpc",
    time_unit     	= "Myr",
    mass_unit     	= "M_sun";
  // basic units
  const double 
    radian	= 1.,
    kpc		= 1.,
    Myr		= 1.,
    Msun	= 1.,
    // angle
    rad         =  radian,			/*!<  radian */
    degree     	=  rad * TPi / 360.,		/*!<  degrees = 360=circle */
    arcmin	=  degree / 60.,		/*!<  arc minutes */
    arcsec     	=  arcmin / 60.,		/*!<  arc seconds */
    mas		=  0.001 * arcsec,		/*!<  milli arc seconds */
    anghr      	=  rad * TPi / 24.,		/*!<  angle hour(24=circle */
    angmin     	=  anghr / 60.,			/*!<  angle minutes */
    angsec     	=  angmin / 60.,		/*!<  angle seconds */
    //  length
    cm		=  kpc * 3.240778828894144e-22,	/*!<  centimeter  */
    meter      	=  1.e2 * cm,			/*!<  meter  */
    km		=  1.e5 * cm,			/*!<  kilo meter  */
    ly          =  3.0660669364447e-4 * kpc,	/*!<  light year */
    pc		=  1.e-3 * kpc,			/*!<  parsec  */
    Mpc		=  1.e3 * kpc,			/*!<  Mega parsec  */
    AU		=  arcsec * pc,			/*!<  astronomical unit  */
    // time
    sec		=  3.168753556551954e-14 * Myr,	/*!<  time second  */
    hour	=  3600  * sec,			/*!<  time hour */
    day		=  24    * sec,			/*!<  time day */
    yr		=  1.e-6 * Myr,			/*!<  year */
    hyr		=  1.e-4 * Myr,			/*!<  hundred years */
    century 	=  hyr,				/*!<  century */
    Gyr		=  1.e3  * Myr,			/*!<  giga year */
    //  mass
    gram       	=  Msun / 1.989e33,		/*!<  gram */
    kg		=  1.e3 * gram,			/*!<  kilogram  */
    //  velocity
    kpcMyr     	=  kpc / Myr,			/*!<  kpc per Myr */
    kpcGyr     	=  kpc / Gyr,			/*!<  kpc per Gyr */
    AUyr       	=  AU / yr,			/*!<  AU per yr */
    kms		=  km / sec,			/*!<  km per second */
    c_light    	=  ly / yr,			/*!<  speed of light */
    // angle velocity
    radMyr     	=  radian / Myr,		/*!<  radian per Myr */
    kmskpc     	=  kms / kpc,			/*!<  kms per kpc */
    masyr      	= mas / yr,			/*!<  milli arcsec per year */
    ashyr      	=  arcsec / hyr,		/*!<  arcsec per century */
    secyr      	=  angsec / yr,			/*!<  angsec per yr */
    asyr       	=  arcsec / yr,			/*!<  arcsec per yr */
    // area
    pc2    	=  pc * pc,			/*!<  square parsec */
    kpc2       	=  kpc * kpc,			/*!<  square kilo parsec */
    cm2		=  cm * cm,			/*!<  square centimeter */
    //  volume
    pc3		=  pc2 * pc,			/*!<  cubic parsec */
    kpc3       	=  kpc2 * kpc,			/*!<  cubic kilo parsec */
    cm3		=  cm2 * cm,			/*!<  cubic centimeter */
    // constant of gravity
    G          	=  4.498658966346282e-12,	/*!<  Newtons G */
    Grav       	=  G,				/*!<  Newtons G */
    fPiG       	=  5.653181583871732e-11,	/*!<  4 Pi G */
    // inverse quantities = useful for transformations
    // inverse angle
    rad_i   	=  1./rad,             /*!<  inverse  */
    radian_i   	=  1./radian,             /*!<  inverse  */
    degree_i 	=  1./degree,             /*!<  inverse  */
    arcmin_i  	=  1./arcmin,             /*!<  inverse  */
    arcsec_i  	=  1./arcsec,             /*!<  inverse  */
    mas_i      	=  1./mas,             /*!<  inverse  */
    anghr_i   	=  1./anghr,             /*!<  inverse  */
    angmin_i  	=  1./angmin,             /*!<  inverse  */
    angsec_i  	=  1./angsec,             /*!<  inverse  */
    // inverse length
    cm_i      	=  1./cm,             /*!<  inverse  */
    meter_i    	=  1./meter,             /*!<  inverse  */
    km_i      	=  1./km,             /*!<  inverse  */
    AU_i      	=  1./AU,             /*!<  inverse  */
    ly_i       	=  1./ly,             /*!<  inverse  */
    pc_i      	=  1./pc,             /*!<  inverse  */
    kpc_i     	=  1./kpc,             /*!<  inverse  */
    Mpc_i     	=  1./Mpc,             /*!<  inverse  */
    // inverse time
    sec_i     	=  1./sec,             /*!<  inverse  */
    hour_i     	=  1./hour,             /*!<  inverse  */
    day_i      	=  1./day,             /*!<  inverse  */
    yr_i      	=  1./yr,             /*!<  inverse  */
    hyr_i     	=  1./hyr,             /*!<  inverse  */
    century_i	=  1./century,             /*!<  inverse  */
    Myr_i     	=  1./Myr,             /*!<  inverse  */
    Gyr_i     	=  1./Gyr,             /*!<  inverse  */
    // inverse mass
    gram_i    	=  1./gram,             /*!<  inverse  */
    kg_i       	=  1./kg,             /*!<  inverse  */
    Msun_i    	=  1./Msun,             /*!<  inverse  */
    // inverse velocity
    kpcMyr_i	=  1./kpcMyr,             /*!<  inverse  */
    kpcGyr_i	=  1./kpcGyr,             /*!<  inverse  */
    AUyr_i     	=  1./AUyr,             /*!<  inverse  */
    kms_i    	=  1./kms,             /*!<  inverse  */
    c_light_i  	=  1./c_light,             /*!<  inverse  */
    // inverse angle velocity
    radMyr_i    =  1./radMyr,             /*!<  inverse  */
    kmskpc_i   	=  1./kmskpc,             /*!<  inverse  */
    masyr_i    	=  1./masyr,             /*!<  inverse  */
    ashyr_i    	=  1./ashyr,             /*!<  inverse  */
    secyr_i    	=  1./secyr,             /*!<  inverse  */
    asyr_i     	=  1./asyr,             /*!<  inverse  */
    // inverse area
    pc2_i      	=  1./pc2,             /*!<  inverse  */
    kpc2_i     	=  1./kpc2,             /*!<  inverse  */
    cm2_i      	=  1./cm2,             /*!<  inverse  */
    // inverse volume 
    pc3_i      	=  1./pc3,             /*!<  inverse  */
    kpc3_i     	=  1./kpc3,             /*!<  inverse  */
    cm3_i      	=  1./cm3,             /*!<  inverse  */
    // inverse of constant of gravity           
    G_i      	=  1./G,             /*!<  inverse  */
    Grav_i   	=  G_i,             /*!<  inverse  */
    fPiG_i   	=  1./fPiG;             /*!<  inverse  */
}


} // namespace
#endif
