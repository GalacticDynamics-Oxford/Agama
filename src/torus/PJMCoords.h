/***************************************************************************//**
\file PJMCoords.h
\brief Contains class OmniCoords.
Code that converts between various different Heliocentric & Galctocentric 
coordinate systems.

                                                                              
  PJMCoords.h (adapted from Coordinates.h                                     
                                                                              
  C++ code written by Paul McMillan 2010,                                     
                      Walter Dehnen, 1995-96,                                 
  Oxford University, Department of Physics, Theoretical Physics.              
  address: 1 Keble Road, Oxford OX1 34P, United Kingdom                       
  e-mail:  p.mcmillan1@physics.ox.ac.uk                                       
                                                                              
********************************************************************************
                                                                              
  GENERAL REMARKS                                                             
                                                                              
  There are six coordinate systems which can be used, three galactocentric,   
  and three heliocentric. Usually, modellers use a galactocentric coordinate  
  system, whereas the observer's natural coordinate systems are heliocentric  
  and polar, the latter since position on the sky and distance are completely 
  different observables. The so called equatorial coordinates (distance,      
  right ascension, declination and their time derivatives) are orientad at    
  the earth's polar axis. However, because of the earth's precession, that    
  axis is not at a fixed angle with respect to very distant objects.          
  Therefore, one often uses so-called galactic coordinates which are also     
  heliocentric but orientated with respect to the galactic centre and the     
  galactic poles, respectively. This coordinate system is changing as well    
  with the time scale being the sun's orbital time (2x10^8 yr) much longer    
  than the earth's precession time (10^4yr).	                               
                                                                              
********************************************************************************
                                                                              
  PRECISE DEFINITION OF THE COORDINATE SYSTEMS                                
                                                                              
  GCA Galactocentric CArtesian                                                
      position = (X,Y,Z),  velocity = (vX,vY,vZ)                              
      units: as defined in Units.h (kpc, kpc/Myr)			       
         These are inertial right-handed coordinates with e_X pointing in the 
      present direction of the sun and e_Z towards the north galactic pole.   
                                                                              
  GCY Galactocentric CYlindrical:                                             
      position = (R,Z,phi),  velocity = (vR,vZ,vphi)                          
      units: as defined in Units.h (kpc, radian, kpc/Myr)		       
         The cylindrical galactocentric coordinates are easily defined in     
      terms of the galactrocentric cartesian coordinates introduced above:    
		X = R cos(phi), Y = R sin(phi), Z = z			       
       	vR = dR/dt, vz = dz/dt, vphi = R dphi/dt.		       
                                                                              
                                                                              
  LSR Local Standard of Rest                                                  
      position = (x,y,z), velocity = (u,v,w)                                  
      units: as defined in Units.h (kpc, kpc/Myr)		   	       
         The local standard of rest (LSR) represents an inertial (non-rota-   
      ting) coordinate system with origin of position on the sun, e_x poin-   
      ting towards the Galactic centre (GC), e_y in direction of Galactic     
      rotation (GR), and e_z towards NGP, i.e.			     	       
		(x,y,z) = (R0-X, -Y, Z-Zsun)                                   
      where R0 is the distance sun-GC, and Zsun is the solar height above the 
      plane. The LSR's origin of velocity is on that of a (hypothetic) star   
      on a circular orbit at the solar position, i.e.	     		       
		(u,v,w) = (-v_X, v_circ-v_Y, v_Z)			       
      where v_circ is the circular rotation velocity at R=R0, for the Milky   
      Way it is negative (it's adopted value and those for R0, Zsun are taken 
      from the file Constants.h). The velocity (u,v,w) of a star w.r.t. LSR   
      are often called its `peculiar motion'.                                 
      In the literature the direction of the x or u vector is not unique,     
      some authors use GC others GAC direction. Here, GC direction is used,   
      in order to end up with a right-handed coordinate system.               
                                                                              
  HCA HelioCentric Cartesian                                                  
      position = (x,y,z), velocity = (vx,vy,vz)                               
      units: as defined in Units.h (kpc, kpc/Myr)                             
         This coordinate system differs from the above (LSR) only in the      
      origin of the velocity, which is the solar motion.                      
      A direct consequence is that transformations to the Polar coordinates   
      does not involve the peculiar motion of the sun.                        
                                                                              
  HGP Heliocentric Galactic Polar:                                            
      position = (r,l,b), velocity = (vr,ml,mb)			       
      units: as defined in Units.h (kpc,radian,kpc/Myr,radian/Myr)            
	  These usually called `galactic coordinates' are often used for       
      observations of galactic objects. They can be defined in terms of the   
      HCA as follows                                                          
		x   = r * cos(b) * cos(l)   e_x points to GC 		       
		y   = r * cos(b) * sin(l)                                      
		z   = r * sin(b)                                               
               vr  = dr/dt                                                    
               ml* = dl/dt * cos(b)                                           
               mb  = db/dt                                                    
      l and b are usually called galactic longitude and galactic latitude,    
      respectively. Note that the peculiar motion of the sun contributes to   
      the radial velocity and proper motion of an object at rest in the       
      local standard of rest                                                  
                                                                              
  HEQ Heliocentric Equatorial Polar:                                          
      position = (r,a,d), velocity = (vr,ma*,md)                              
      units: as defined in Units.h (kpc,radian,kpc/Myr,radian/Myr)            
      NOTE:  ma* = cos(d) * da/dt = cos(d) * ma                               
         The equatorial coordinates (distance, right ascension, declination)  
      are those to be used in the telescopes. They are fixed to the earth     
      polar axis, which changes rapidly enough to make these coordinates less 
      useful than the other. Hence, most observers give galactic coordinates. 
      The precise definition depends on epoch, by default J2000 is used, but  
      B1950 is also possible, the epoch may be given as an argument.          
                                                                              
  --------------                                                              
  @) GC, GAC, GR, GAR, and NGP denote Galactic centre, Galactic anticentre,   
  Galactic rotation, Galactic antirotation, and north Galactic pole,          
  respectively.                                                               
                                                                              
********************************************************************************
                                                                              
  Summary of Coordinate Systems:                                              
                                                                              
   centre   |                coordinate system                                
   x   v    | cartesian |  cylindrical | spherical polar                      
  ----------+--------------------------------------------                     
  GC   GC @)|	 GCA	      GCY	        -                              
  sun  LSR  |   LSR           -                -                              
  sun  sun  |   HCA           -             HGP, HEQ                          
                                                                              
  The Transformations between systems with different centres is always done   
  in the cartesian frames.                                                    
                                                                              
  --------------                                                              
  @) GC denotes the galactic centre                                           
                                                                              
********************************************************************************
*                                                                              *
*  Parameter dependences of the basic transformations between the coordinate   *
*  systems:                                                                    *
*                                                                              *
*  GCA  <-->  GCY      -                                                       *
*  GCA  <-->  LSR      depends on R0, zsun, Vcirc(R0)                          *
*  LSR  <-->  HCA      depends on (U,V,W)sun                                   *
*  HCA  <-->  HGP      -                                                       *
*  HCA  <-->  HEQ      depends on epoch (default is J2000)                     *
*                                                                              *
*                                                                              *
*******************************************************************************

Paul's notes

Eventually I decided that I should probably use some proper c++ for this 
(i.e. create a class that stores the transformation constants) 






*/

#ifndef _Galactic_Coordinates_PJM_def_
#define _Galactic_Coordinates_PJM_def_ 1
#include "Constants.h"
#include "WD_Vector.h"
#include "WD_Matrix.h"

namespace torus {
using namespace WD;

#ifndef _Galactic_Coordinates_def_
typedef Vector<double,6> GCA;
typedef Vector<double,6> GCY;
typedef Vector<double,6> GCR;
typedef Vector<double,6> LSR;
typedef Vector<double,6> HCA;
typedef Vector<double,6> HGP;
typedef Vector<double,6> HEQ;
#endif

typedef Vector<bool,6> bool6;
typedef Vector<double,6> vec6;
typedef Matrix<double,3,3> mat33;


/**
\brief Converts between various different Heliocentric & Galctocentric 
coordinate systems.

                                   
  GENERAL REMARKS                                                             
                                                                              
  There are six coordinate systems which can be used, three galactocentric,   
  and three heliocentric. Usually, modellers use a galactocentric coordinate  
  system, whereas the observer's natural coordinate systems are heliocentric  
  and polar, the latter since position on the sky and distance are completely 
  different observables. The so called equatorial coordinates (distance,      
  right ascension, declination and their time derivatives) are orientad at    
  the earth's polar axis. However, because of the earth's precession, that    
  axis is not at a fixed angle with respect to very distant objects.          
  Therefore, one often uses so-called galactic coordinates which are also     
  heliocentric but orientated with respect to the galactic centre and the     
  galactic poles, respectively. This coordinate system is changing as well    
  with the time scale being the sun's orbital time (2x10^8 yr) much longer    
  than the earth's precession time (10^4yr).	                               
                                                                              
********************************************************************************
                                                                              
  PRECISE DEFINITION OF THE COORDINATE SYSTEMS                                
                                                                              
  GCA Galactocentric CArtesian                                                
      position = (X,Y,Z),  velocity = (vX,vY,vZ)                              
      units: as defined in Units.h (kpc, kpc/Myr)			       
         These are inertial right-handed coordinates with e_X pointing in the 
      present direction of the sun and e_Z towards the north galactic pole.   
                                                                              
  GCY Galactocentric CYlindrical:                                             
      position = (R,Z,phi),  velocity = (vR,vZ,vphi)                          
      units: as defined in Units.h (kpc, radian, kpc/Myr)		       
         The cylindrical galactocentric coordinates are easily defined in     
      terms of the galactrocentric cartesian coordinates introduced above:    
		X = R cos(phi), Y = R sin(phi), Z = z			       
       	vR = dR/dt, vz = dz/dt, vphi = R dphi/dt.		       
                                                                              
                                                                              
  LSR Local Standard of Rest                                                  
      position = (x,y,z), velocity = (u,v,w)                                  
      units: as defined in Units.h (kpc, kpc/Myr)		   	       
         The local standard of rest (LSR) represents an inertial (non-rota-   
      ting) coordinate system with origin of position on the sun, e_x poin-   
      ting towards the Galactic centre (GC), e_y in direction of Galactic     
      rotation (GR), and e_z towards NGP, i.e.			     	       
		(x,y,z) = (R0-X, -Y, Z-Zsun)                                   
      where R0 is the distance sun-GC, and Zsun is the solar height above the 
      plane. The LSR's origin of velocity is on that of a (hypothetic) star   
      on a circular orbit at the solar position, i.e.	     		       
		(u,v,w) = (-v_X, v_circ-v_Y, v_Z)			       
      where v_circ is the circular rotation velocity at R=R0, for the Milky   
      Way it is negative (it's adopted value and those for R0, Zsun are taken 
      from the file Constants.h). The velocity (u,v,w) of a star w.r.t. LSR   
      are often called its `peculiar motion'.                                 
      In the literature the direction of the x or u vector is not unique,     
      some authors use GC others GAC direction. Here, GC direction is used,   
      in order to end up with a right-handed coordinate system.               
                                                                              
  HCA HelioCentric Cartesian                                                  
      position = (x,y,z), velocity = (vx,vy,vz)                               
      units: as defined in Units.h (kpc, kpc/Myr)                             
         This coordinate system differs from the above (LSR) only in the      
      origin of the velocity, which is the solar motion.                      
      A direct consequence is that transformations to the Polar coordinates   
      does not involve the peculiar motion of the sun.                        
                                                                              
  HGP Heliocentric Galactic Polar:                                            
      position = (r,l,b), velocity = (vr,ml,mb)			       
      units: as defined in Units.h (kpc,radian,kpc/Myr,radian/Myr)            
	  These usually called `galactic coordinates' are often used for       
      observations of galactic objects. They can be defined in terms of the   
      HCA as follows                                                          
		x   = r * cos(b) * cos(l)   e_x points to GC 		       
		y   = r * cos(b) * sin(l)                                      
		z   = r * sin(b)                                               
               vr  = dr/dt                                                    
               ml* = dl/dt * cos(b)                                           
               mb  = db/dt                                                    
      l and b are usually called galactic longitude and galactic latitude,    
      respectively. Note that the peculiar motion of the sun contributes to   
      the radial velocity and proper motion of an object at rest in the       
      local standard of rest                                                  
                                                                              
  HEQ Heliocentric Equatorial Polar:                                          
      position = (r,a,d), velocity = (vr,ma*,md)                              
      units: as defined in Units.h (kpc,radian,kpc/Myr,radian/Myr)            
      NOTE:  ma* = cos(d) * da/dt = cos(d) * ma                               
         The equatorial coordinates (distance, right ascension, declination)  
      are those to be used in the telescopes. They are fixed to the earth     
      polar axis, which changes rapidly enough to make these coordinates less 
      useful than the other. Hence, most observers give galactic coordinates. 
      The precise definition depends on epoch, by default J2000 is used, but  
      B1950 is also possible, the epoch may be given as an argument.          
                                                                              
  --------------                                                              
  @) GC, GAC, GR, GAR, and NGP denote Galactic centre, Galactic anticentre,   
  Galactic rotation, Galactic antirotation, and north Galactic pole,          
  respectively.                                                               
                                                                              
********************************************************************************
                                                                              
  Summary of Coordinate Systems:                                              
                                                                              
   centre   |                coordinate system                                
   x   v    | cartesian |  cylindrical | spherical polar                      
  ----------+--------------------------------------------                     
  GC   GC @)|	 GCA	      GCY	        -                              
  sun  LSR  |   LSR           -                -                              
  sun  sun  |   HCA           -             HGP, HEQ                          
                                                                              
  The Transformations between systems with different centres is always done   
  in the cartesian frames.                                                    
                                                                              
  --------------                                                              
  @) GC denotes the galactic centre                                           
                                                                              
********************************************************************************
*                                                                              *
*  Parameter dependences of the basic transformations between the coordinate   *
*  systems:                                                                    *
*                                                                              *
*  GCA  <-->  GCY      -                                                       *
*  GCA  <-->  LSR      depends on R0, zsun, Vcirc(R0)                          *
*  LSR  <-->  HCA      depends on (U,V,W)sun                                   *
*  HCA  <-->  HGP      -                                                       *
*  HCA  <-->  HEQ      depends on epoch (default is J2000)                     *
*                                                                              *
*                                                                              *
*******************************************************************************



*/


class OmniCoords {
  //bool knowHEQ,knowHGP,knowHCA,knowLSR,knowGCA,knowGCY;
  bool6 know;
  double Rsun,zsun,vcsun,Usun,Vsun,Wsun, epoch; 
  mat33   EtoP;
  void SetTrans();
  void HEQfromHCA();  void HGPfromHCA();  void HCAfromLSR();
  void LSRfromGCA();  void GCAfromGCY();
  void Backward (int);
  void HCAfromHEQ();  void HCAfromHGP();  void LSRfromHCA();
  void GCAfromLSR();  void GCYfromGCA();
  void Forward(int);
  vec6 rv[6];
 public:
  void change_sol_pos(double,double);
  void change_vc(double);
  void change_vsol(double,double,double);
  void set_SBD10();
  void set_DB98();
  void change_epoch(double);
  
  void give_sol_pos(double&,double&);
  double give_Rsun()  {return Rsun;}
  double give_zsun()  {return zsun;}
  void give_vc(double&);
  double give_vcsun() {return vcsun;}
  void give_vsol(double&,double&,double&);
  void give_epoch(double&);
  
  vec6  give_HEQ() {return give(0);}  vec6  give_HGP() {return give(1);}
  vec6  give_HCA() {return give(2);}  vec6  give_LSR() {return give(3);}  
  vec6  give_GCA() {return give(4);}  vec6  give_GCY() {return give(5);}
  vec6  give_HEQ_units() {return give_units(0);}  
  vec6  give_HGP_units() {return give_units(1);}
  vec6  give_HCA_units() {return give_units(2);}  
  vec6  give_LSR_units() {return give_units(3);}  
  vec6  give_GCA_units() {return give_units(4);}  
  vec6  give_GCY_units() {return give_units(5);}
  vec6  give(int);
  vec6  give_units(int);
  void  take_HEQ(vec6);  void  take_HGP(vec6);  void  take_HCA(vec6);
  void  take_LSR(vec6);  void  take_GCA(vec6);  void  take_GCY(vec6);
  void  take_HEQ_units(vec6);  void  take_HGP_units(vec6);  
  void  take_HCA_units(vec6);  void  take_LSR_units(vec6);  
  void  take_GCA_units(vec6);  void  take_GCY_units(vec6);

  // Just want to convert. No fannying about.
  vec6 HEQfromHGP(vec6 sHGP) {take_HGP(sHGP); return give_HEQ();}
  vec6 HEQfromHCA(vec6 sHCA) {take_HCA(sHCA); return give_HEQ();}
  vec6 HEQfromLSR(vec6 sLSR) {take_LSR(sLSR); return give_HEQ();}
  vec6 HEQfromGCA(vec6 sGCA) {take_GCA(sGCA); return give_HEQ();}
  vec6 HEQfromGCY(vec6 sGCY) {take_GCY(sGCY); return give_HEQ();}

  vec6 HGPfromHEQ(vec6 sHEQ) {take_HEQ(sHEQ); return give_HGP();}
  vec6 HGPfromHCA(vec6 sHCA) {take_HCA(sHCA); return give_HGP();}
  vec6 HGPfromLSR(vec6 sLSR) {take_LSR(sLSR); return give_HGP();}
  vec6 HGPfromGCA(vec6 sGCA) {take_GCA(sGCA); return give_HGP();}
  vec6 HGPfromGCY(vec6 sGCY) {take_GCY(sGCY); return give_HGP();}

  vec6 HCAfromHEQ(vec6 sHEQ) {take_HEQ(sHEQ); return give_HCA();}
  vec6 HCAfromHGP(vec6 sHGP) {take_HGP(sHGP); return give_HCA();}
  vec6 HCAfromLSR(vec6 sLSR) {take_LSR(sLSR); return give_HCA();}
  vec6 HCAfromGCA(vec6 sGCA) {take_GCA(sGCA); return give_HCA();}
  vec6 HCAfromGCY(vec6 sGCY) {take_GCY(sGCY); return give_HCA();}

  vec6 LSRfromHEQ(vec6 sHEQ) {take_HEQ(sHEQ); return give_LSR();}
  vec6 LSRfromHGP(vec6 sHGP) {take_HGP(sHGP); return give_LSR();}
  vec6 LSRfromHCA(vec6 sHCA) {take_HCA(sHCA); return give_LSR();}
  vec6 LSRfromGCA(vec6 sGCA) {take_GCA(sGCA); return give_LSR();}
  vec6 LSRfromGCY(vec6 sGCY) {take_GCY(sGCY); return give_LSR();}

  vec6 GCAfromHEQ(vec6 sHEQ) {take_HEQ(sHEQ); return give_GCA();}
  vec6 GCAfromHGP(vec6 sHGP) {take_HGP(sHGP); return give_GCA();}
  vec6 GCAfromHCA(vec6 sHCA) {take_HCA(sHCA); return give_GCA();}
  vec6 GCAfromLSR(vec6 sLSR) {take_LSR(sLSR); return give_GCA();}
  vec6 GCAfromGCY(vec6 sGCY) {take_GCY(sGCY); return give_GCA();}

  vec6 GCYfromHEQ(vec6 sHEQ) {take_HEQ(sHEQ); return give_GCY();}
  vec6 GCYfromHGP(vec6 sHGP) {take_HGP(sHGP); return give_GCY();}
  vec6 GCYfromHCA(vec6 sHCA) {take_HCA(sHCA); return give_GCY();}
  vec6 GCYfromLSR(vec6 sLSR) {take_LSR(sLSR); return give_GCY();}
  vec6 GCYfromGCA(vec6 sGCA) {take_GCA(sGCA); return give_GCY();}

  OmniCoords() ;
  ~OmniCoords() {};
};


inline OmniCoords::OmniCoords() {
  know = false; 
  Rsun = GalactoConstants::Rsun; zsun = GalactoConstants::zsun; vcsun = GalactoConstants::vcsun;
  Usun = GalactoConstants::usun; Vsun = GalactoConstants::vsun; Wsun  = GalactoConstants::wsun; 
  epoch = 2000.;
  SetTrans();
}

inline void OmniCoords::take_HEQ(vec6 tHEQ) {
  rv[0] = tHEQ; know = false; know[0] = true;
}
inline void OmniCoords::take_HEQ_units(vec6 tHEQ) {
  vec6 tmp = tHEQ;
  tmp[1] *=Units::degree;
  tmp[2] *=Units::degree;
  tmp[3] *=Units::kms;
  tmp[4] *=Units::masyr;
  tmp[5] *=Units::masyr;
  rv[0] = tmp; know = false; know[0] = true;
}
inline void OmniCoords::take_HGP(vec6 tHGP) {
  rv[1] = tHGP; know = false; know[1] = true;
}
inline void OmniCoords::take_HGP_units(vec6 tHGP) {
  vec6 tmp = tHGP;
  tmp[1] *=Units::degree;
  tmp[2] *=Units::degree;
  tmp[3] *=Units::kms;
  tmp[4] *=Units::masyr;
  tmp[5] *=Units::masyr;
  rv[1] = tmp; know = false; know[1] = true;
}

inline void OmniCoords::take_HCA(vec6 tHCA) {
  rv[2] = tHCA; know = false; know[2] = true;
}
inline void OmniCoords::take_HCA_units(vec6 tHCA) {
  vec6 tmp = tHCA;
  for(int i=3;i!=6;i++) tmp[i] *= Units::kms;
  rv[2] = tmp; know = false; know[2] = true;
}

inline void OmniCoords::take_LSR(vec6 tLSR) {
  rv[3] = tLSR; know = false; know[3] = true;
}
inline void OmniCoords::take_LSR_units(vec6 tLSR) {
  vec6 tmp = tLSR;
  for(int i=3;i!=6;i++) tmp[i] *= Units::kms;
  rv[3] = tmp; know = false; know[3] = true;
}

inline void OmniCoords::take_GCA(vec6 tGCA) {
  rv[4] = tGCA; know = false; know[4] = true;
}
inline void OmniCoords::take_GCA_units(vec6 tGCA) {
  vec6 tmp = tGCA;
  for(int i=3;i!=6;i++) tmp[i] *= Units::kms;
  rv[4] = tmp; know = false; know[4] = true;
}

inline void OmniCoords::take_GCY(vec6 tGCY) {
  rv[5] = tGCY; know = false; know[5] = true;
}
inline void OmniCoords::take_GCY_units(vec6 tGCY) {
  vec6 tmp = tGCY;
  tmp[2] *= Units::degree;
  for(int i=3;i!=6;i++) tmp[i] *= Units::kms;
  rv[5] = tmp; know = false; know[5] = true;
}

inline void OmniCoords::Backward(int n) {
  switch(n) {
  case 0 :
    HEQfromHCA(); break;
  case 1 : 
    HGPfromHCA(); break;
  case 2 : 
    HCAfromLSR(); break;
  case 3 :
    LSRfromGCA(); break;
  case 4 : 
    GCAfromGCY(); break;
  default :
    cerr << "dude what?\n";
  }
}

inline void OmniCoords::Forward(int n) {
  switch(n) {
  case 0 :
    HCAfromHEQ();break;
  case 1 : 
    HCAfromHGP(); break;
  case 2 : 
    LSRfromHCA();break;
  case 3 :
    GCAfromLSR(); break;
  case 4 : 
    GCYfromGCA();break;
  default :
    cerr << "dude what?\n";
  }
}

inline vec6 OmniCoords::give(int n) {
  if(know[n]) return rv[n]; // if known, give

  if(n==0) {                // special case, because HEQ found from HCA not 
    bool found = false;     //  from HGP.
    for(int i=2;i<=5 && !found;i++) 
      if(know[i]) {
	for(int j=i;j>1;j--) Backward(j);
	found = true;
      }
    Backward(0);
    return rv[n];
  }
  for(int i=n-1;i>=1;i--)    // if 'earlier' are known
    if(know[i]) {
      for(int j=i;j<n;j++) Forward(j);
      return rv[n];
    }
  
  for(int i=n+1;i<=5;i++)   // if 'later' are known
    if(know[i]) {
      for(int j=i-1;j>=n;j--) Backward(j);
      return rv[n];
    }

  if(know[0]) {
    Forward(0);
    if(n==1) Backward(1);
    else for(int i=2;i<n;i++) Forward(i);
    return rv[n];
  }
  cerr << "No input set\n";
  return vec6(0.);
}

inline vec6 OmniCoords::give_units(int n) {
  vec6 coords = give(n);
  if(n>1) { 
    for(int i=3;i!=6;i++) coords[i] *= Units::kms_i;
    if(n==5) coords[2] *= Units::degree_i;
    return coords;
  }
  coords[1] *=Units::degree_i;
  coords[2] *=Units::degree_i;
  coords[3] *=Units::kms_i;
  coords[4] *=Units::masyr_i;
  coords[5] *=Units::masyr_i;
  return coords;
}



inline void OmniCoords::change_vc(double vc) {
  vcsun = vc; know = false;
  if(vcsun>0.) 
    cerr << "Careful: You have set vc_sun>0, which is not the norm\n";
}
inline void OmniCoords::change_sol_pos(double R, double z) {
  Rsun = R; zsun = z; know = false;
}
inline void OmniCoords::change_vsol(double U, double V, double W) {
  Usun = U; Vsun=V; Wsun=W; know = false;
}

inline void OmniCoords::change_epoch(double ep) {
  epoch = ep; SetTrans(); know = false;
}

inline void OmniCoords::give_vc(double& vc) {
 vcsun = vc;
}
inline void OmniCoords::give_sol_pos(double& R, double& z) {
  R = Rsun; z = zsun;
}
inline void OmniCoords::give_vsol(double& U, double& V, double& W) {
  U=Usun; V=Vsun; W=Wsun;
}
inline void OmniCoords::give_epoch(double& ep) {
  ep=epoch;
}

inline void OmniCoords::set_SBD10() {
  change_vsol(11.1*Units::kms,12.24*Units::kms,7.25*Units::kms);
}

inline void OmniCoords::set_DB98() {
  change_vsol(10.*Units::kms,5.25*Units::kms,7.17*Units::kms);
}

}  //namespace
#endif
