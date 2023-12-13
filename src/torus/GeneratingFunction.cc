/*******************************************************************************
*                                                                              *
* GeneratingFunction.cc                                                        *
*                                                                              *
* C++ code written by Walter Dehnen, 1995-96,                                  *
*                     Paul McMillan, 2007                                      *
* e-mail: paul@astro.lu.se                                                     *
* github: https://github.com/PaulMcMillan-Astro/Torus                          *
*                                                                              *
*******************************************************************************/

#include <iostream>
#include <fstream>
#include <iomanip>
#include "GeneratingFunction.h"
#include "WD_Numerics.h"
#include <cmath>

namespace torus{

using std::setfill;
using std::setw;


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// class GenPar

int GenPar::NumberofTermsUsed() const 
{
  int output=0;
  for(int i=0;i!=ntot;i++) if(S[i]) output++;
  return output;
}

void GenPar::findNN()
{
    short i, n2_min=0, n2_max=0, n1_max=0;
    for(i=0; i<ntot; i++) {
        if(N1[i] > n1_max) n1_max = N1[i];
        if(N2[i] > n2_max) n2_max = N2[i];
        if(N2[i] < n2_min) n2_min = N2[i];
    }
    nn1 = n1_max + 1;
    nn2 = (n2_max - n2_min)/2 + 1;
}
//////////////////////////////////////////////////////////////////////////////
void GenPar::sortSn() // sort in order of n (order by n2, then n1 if needed)
{
  if(ntot>0) {
    short i, ascending=1;
    int   *M=new int[ntot];
    for(i=0; i<ntot; i++) {
      M[i] = 1000 * N2[i] + N1[i];
      if(ascending && i>0 && M[i] < M[i-1]) ascending = 0;
    }
    if(!ascending) {
      int *I = new int[ntot];
      HeapIndex(M, ntot, I);
      short *N1new = new short[ntot];
      short *N2new = new short[ntot];
      double *Snew  = new double[ntot];
      for(i=0; i<ntot; i++) {
        N1new[i] = N1[I[i]];   
        N2new[i] = N2[I[i]];   
        Snew[i]  = S[I[i]];   
      }
      delete[] I;
      delete[] N1;
      delete[] N2;
      delete[] S;
      N1 = N1new;
      N2 = N2new;
      S  = Snew;
    }
    delete[] M;
  }
}
//////////////////////////////////////////////////////////////////////////////
void GenPar::SanityCheck() // Check that nothing dumb is being done
{
  if(ntot>0) {
    int Mt,oldM=-1000000;
    for(int i=0; i<ntot; i++) {
      Mt = 1000 * N2[i] + N1[i];
      if(i && Mt<oldM) {sortSn(); i=0;} // if not sorted, sort and restart.
      if((i && Mt==oldM) || (Mt>=0 && !(Mt%1000)) || N1[i]<0) { // if illegal n
        short *N1new = new short[--ntot];
        short *N2new = new short[ntot];
        double *Snew  = new double[ntot];
        for(int j=0; j!=i;   j++) {
          N1new[j] = N1[j];   N2new[j] = N2[j];   Snew[j] = S[j]; }
        for(int j=i; j!=ntot;j++) {
          N1new[j] = N1[j+1]; N2new[j] = N2[j+1]; Snew[j] = S[j+1]; }
        delete[] N1; delete[] N2; delete[] S;
        N1 =  N1new; N2 =  N2new; S   = Snew;
        i--; // so we don't miss a value
      }
      oldM=Mt;
    }
  }
  findNN();
  //cout << ntot
}
////////////////////////////////////////////////////////////////////////////////
GenPar::GenPar(const int N)
{
    ntot = N;
    nn1  = 0;
    nn2  = 0;
    if(ntot>0) {
        N1 = new short[ntot];
        N2 = new short[ntot];
        S  = new double[ntot];
    } else {
        N1 = 0;
        N2 = 0;
        S  = 0;
    }
}
////////////////////////////////////////////////////////////////////////////////
GenPar::GenPar(const GenPar& G)
{
    ntot = G.ntot;
    nn1  = G.nn1;
    nn2  = G.nn2;
    N1   = new short[ntot];
    N2   = new short[ntot];
    S    = new double[ntot];
    for(short i=0; i<ntot; i++) {
        N1[i] = G.N1[i];
        N2[i] = G.N2[i];
        S[i]  = G.S[i];
    }
    SanityCheck();
}
////////////////////////////////////////////////////////////////////////////////
GenPar::~GenPar()
{
    if(ntot>0) {
        delete[] N1;
        delete[] N2;
        delete[] S;
    }
}
////////////////////////////////////////////////////////////////////////////////
// void GenPar::MakeGeneric()
// {
//   ntot = 37;
//   nn1  = 6;
//   nn2  = 7;
//   N1   = new short[ntot];
//   N2   = new short[ntot];
//   S    = new double[ntot];
  
//    for(int i=0;i!=ntot;i++) {
//     N1[i] = (i<8)? 1+i%4 : (i<20)? 1+(i-1)%5 : (i<27)? i%7 : (i-2)%5;
//     N2[i] = (i<4)? 6 : (i<8)? 4 : (i<14)? 2 : (i<20)? 0 : (i<27)? -2 : 
//             (i<32)? -4: -6; 
//    }

//   for(short i=0; i<ntot; i++) S[i] = 0.; 
//   sortSn();   // Shouldn't be needed, but 
//   findNN();   // better safe than sorry.
// }

void GenPar::MakeGeneric()
{
  ntot = 10;
  nn1  = 4;
  nn2  = 5;
  N1   = new short[ntot];
  N2   = new short[ntot];
  S    = new double[ntot];
  
  N2[0] = -4; N1[0] = 0; 
  N2[1] = -2; N1[1] = 0;  N2[2] = -2; N1[2] = 1;  N2[3] = -2; N1[3] = 2;
  N2[4] = 0;  N1[4] = 1;  N2[5] = 0;  N1[5] = 2;  N2[6] = 0;  N1[6] = 3; 
  N2[7] = 2;  N1[7] = 1;  N2[8] = 2;  N1[8] = 2; 
  N2[9] = 4;  N1[9] = 1;

  for(int i=0; i<ntot; i++) S[i] = 0.;
  sortSn();   // Shouldn't be needed, but 
  findNN();   // better safe than sorry.
  SanityCheck();
}



////////////////////////////////////////////////////////////////////////////////
int GenPar::same_terms_as(const GenPar& G) const
{
    if(ntot != G.ntot) return 0;
    if(nn1  != G.nn1)  return 0;
    if(nn2  != G.nn2)  return 0;
    for(short i=0; i<ntot; i++)
        if(N1[i]!=G.N1[i] || N2[i]!=G.N2[i]) return 0;
    return 1;
}
////////////////////////////////////////////////////////////////////////////////
void GenPar::write(ostream& to) const
{
    to << ntot;
    for(short i=0; i<ntot; i++)
        to << "\n " << N1[i] << ' ' << N2[i] << "  " << S[i];
}
////////////////////////////////////////////////////////////////////////////////
void GenPar::write_log(ostream& to) const
{
    int i, ils, n2act=1000, n1act=0,  n1max=0;
    to << ntot;
    for(i=0; i<ntot; i++) {
        if(N1[i]> n1max) n1max = N1[i];
        if(N2[i]!=n2act) {
            n1act = 0;
            n2act = N2[i];
            if(n2act < -9)              to << "\n  " << n2act << ' ';
            else if(n2act<0 || n2act>9) to << "\n   " << n2act << ' ';
            else                        to << "\n    " << n2act << ' ';
        }
        while (N1[i] > n1act) {
             to << " - ";
             n1act++;
        }
        ils = (S[i]==0.) ? 0 : int(-10.*log10(fabs(S[i])));
        to << setw(2) << setfill(' ') << ils << ' ';
        n1act++;
    }
    to << "\nn2/n1 ";
    for(i=0; i<=n1max; i++) to << setw(2) << setfill(' ') << i << ' ';
    to << '\n';
}
////////////////////////////////////////////////////////////////////////////////
void GenPar::read(istream& from)
{
    short newtot;
    from >> newtot;
    if(newtot <= 0) throw std::runtime_error("Torus Error -4: GenPar: number of terms <= 0");
    if(newtot != ntot) {
        if(ntot>0) {
            delete[] N1;
            delete[] N2;
            delete[] S;
        }
        ntot = newtot;
        N1   = new short[ntot];
        N2   = new short[ntot];
        S    = new double[ntot];
    } 
    short i;
    for(i=0; i<ntot; i++) {
        from >> N1[i] >> N2[i] >> S[i];
        if(N2[i]%2 != 0)
            throw std::runtime_error("Torus Error -4: GenPar: odd n2 in read()");
        if(N1[i]==0 && N2[i]>=0)
          throw std::runtime_error("Torus Error -4: GenPar: +ve n2 with n1==0 in read()");
    }
    sortSn();
    SanityCheck();
    //findNN();
}

void GenPar::read(int *N1in, int *N2in, double *Snin, short newtot)
{
    if(newtot <= 0) throw std::runtime_error("Torus Error -4: GenPar: number of terms <= 0");
    if(newtot != ntot) {
        if(ntot>0) {
            delete[] N1;
            delete[] N2;
            delete[] S;
        }
        ntot = newtot;
        N1   = new short[ntot];
        N2   = new short[ntot];
        S    = new double[ntot];
    } 
    short i;
    for(i=0; i<ntot; i++) {
      N1[i] = N1in[i];
      N2[i] = N2in[i];
      S[i]  = Snin[i];
      if(N2[i]%2 != 0)
        throw std::runtime_error("Torus Error -4: GenPar: odd n2 in read()");
      if(N1[i]==0 && N2[i]>=0)
        throw std::runtime_error("Torus Error -4: GenPar: +ve n2 with n1==0 in read()");
    }
    sortSn();
    SanityCheck();
    //findNN();
}



//////////////////////////////////////////////////////////////////////////////
void GenPar::tailor(const double a, const double b, const int Max)
// to tailor a GenPar:
// if |S_n1,n2| < a * max[|S_n1,n2|]  ==>  delete it.
// if |S_n1,n2| > b * max[|S_n1,n2|]  ==>  create new around it.
{
    short i,n, n1,n2, n2min=N2[0]-2,n2max=N2[ntot-1]+2, newtot=ntot,
                   maxtot=short(fmax(int(ntot),Max));
    double Smax=0.;
// make a 2D map and compute |S|max
    char **map = new char* [nn1+2];
    for(n1=0; n1<nn1+2; n1++) {
        map[n1] = (new char[n2max-n2min+2])-n2min;
        for(n2=n2min; n2<=n2max; n2++)
            map[n1][n2] = 0;
    }
    for(i=0; i<ntot; i++) {
        map[N1[i]][N2[i]] = 1;
        if(fabs(S[i])>Smax) Smax = fabs(S[i]);
    }
// delete S if |S| < a|S|max AND n1>3, n2<-4 or n2>4
    for(i=0; i<ntot; i++)
        if(fabs(S[i])<a*Smax && (N1[i]>3 || N2[i]<-4 || N2[i]>4)) {
            map[N1[i]][N2[i]] = 3;
            newtot--;
        }
// create up to 4 (usually 2) Sn if |S| > b|S|max
    for(i=0; i<ntot; i++)
        if(fabs(S[i]) > b * Smax) {
            if(newtot>=maxtot) break;
            if(map[N1[i]+1][N2[i]] == 0)             // One to the right
              { map[N1[i]+1][N2[i]] = 2; newtot++; }
            if(N1[i]>20 && map[N1[i]+2][N2[i]] == 0) // Maybe even two.
              { map[N1[i]+2][N2[i]] = 2; newtot++; }
            if(!(N1[i]==1 && N2[i]>=0) && N1[i]!=0)
              if(map[N1[i]-1][N2[i]] == 0 || map[N1[i]-1][N2[i]] == 3) // new
                {map[N1[i]-1][N2[i]] = 2; newtot++;} //One to the left (not 0,0)
            if(!(N1[i]==0 && N2[i]==-2) && map[N1[i]][N2[i]+2] == 0)
              { map[N1[i]][N2[i]+2] = 2; newtot++; } // One up (not 0,0)
            if(map[N1[i]][N2[i]-2] == 0)
              { map[N1[i]][N2[i]-2] = 2; newtot++; } // One down
            if(N2[i]<0 && map[N1[i]][N2[i]+2] == 3)  // One up if just del.
              { map[N1[i]][N2[i]+2] = 2; newtot++; } // and it's closer to 0
            if(N2[i]>0 && map[N1[i]][N2[i]-2] == 3)  // One down if just del.
              { map[N1[i]][N2[i]-2] = 2; newtot++; } // and it's closer to 0
        }
// put in new arrays N1, N2, S.
    short *N1new = new short[newtot];
    short *N2new = new short[newtot];
    double *Snew  = new double[newtot];
    for(i=0,n=0; i<ntot; i++)
        if(map[N1[i]][N2[i]] == 1) {
            N1new[n] = N1[i];
            N2new[n] = N2[i];
            Snew[n]  = S[i];
            n++;
        }
    for(n2=n2min; n2<=n2max; n2+=2)
        for(n1=0; n1<nn1+2; n1++)
              if(map[n1][n2] == 2) {
                N1new[n] = n1;
                N2new[n] = n2;
                Snew[n]  = 0.;
                n++;
                if(n==newtot) goto nomore;
            }
    nomore:
    for(n1=0; n1<nn1+2; n1++)
        delete[] (map[n1]+n2min);
    delete[] map;
    if(ntot>0) {
        delete[] N1;
        delete[] N2;
        delete[] S;
    }
    ntot = newtot;
    N1   = N1new;
    N2   = N2new;
    S    = Snew;
    sortSn();
    //SanityCheck(); // Uncomment if trying something new
    findNN();
}

//////////////////////////////////////////////////////////////////////////////
void GenPar::edgetailor(const double a, const int Max)
// to tailor a GenPar:
// if |S_n1,n2| < a * max[|S_n1,n2|]  ==>  delete it.
// if |S_n1,n2| > b * max[|S_n1,n2|]  ==>  create new around it.
{
  short i,n, n1,n2, n2min=N2[0]-2,n2max=N2[ntot-1]+2, newtot=ntot,
    maxtot=short(fmax(int(ntot),Max)),nedge=0;
  //double Smax=0.;
  // make a 2D map and compute |S|max
  char **map = new char* [nn1+2];
  bool *edge = new bool[ntot];
  for(n1=0; n1<nn1+2; n1++) {
    map[n1] = (new char[n2max-n2min+2])-n2min;
    for(n2=n2min; n2<=n2max; n2++)
      map[n1][n2] = 0;
  }
  for(i=0; i<ntot; i++)
    map[N1[i]][N2[i]] = 1;


  for(i=0; i<ntot; i++) {
    if(map[N1[i]+1][N2[i]] && (map[N1[i]][N2[i]-2]) 
       && (map[N1[i]][N2[i]+2] || (N1[i]==0 && N2[i]==-2))) {
      edge[i] = false;
    } else {
      edge[i] = true;
      nedge++;
    }
  }

  float *vals = new float[nedge];
  int *wheres = new int[nedge],
    *I = new int[nedge];
  for(i=0, n=0; i!=ntot;i++)
    if(edge[i]) {
      wheres[n] = i;
      vals[n++] = fabs(S[i]);
    }
  HeapIndex(vals, nedge, I);
  for(n=0;n!=int((1.-a)*nedge);n++)
    edge[wheres[I[n]]] = false;

  delete[] vals;
  delete[] wheres;
  delete[] I;

  for(i=0; i<ntot; i++) {
    if(edge[i] && newtot<maxtot) {
      if(map[N1[i]+1][N2[i]] == 0)             // One to the right
        { map[N1[i]+1][N2[i]] = 2; newtot++; }
      if(N1[i]>20 && map[N1[i]+2][N2[i]] == 0) // Maybe even two.
        { map[N1[i]+2][N2[i]] = 2; newtot++; }
      if(!(N1[i]==1 && N2[i]>=0) && N1[i]!=0) {
        if(map[N1[i]-1][N2[i]] == 0 || map[N1[i]-1][N2[i]] == 3) // new
          {map[N1[i]-1][N2[i]] = 2; newtot++;} //One to the left (not 0,0)
      }
      if(!(N1[i]==0 && N2[i]==-2) && map[N1[i]][N2[i]+2] == 0)
        { map[N1[i]][N2[i]+2] = 2; newtot++; } // One up (not 0,0)
      if(map[N1[i]][N2[i]-2] == 0)
        { map[N1[i]][N2[i]-2] = 2; newtot++; } // One down
    }
  }
  delete[] edge;

// put in new arrays N1, N2, S.
  short *N1new = new short[newtot];
  short *N2new = new short[newtot];
  double *Snew  = new double[newtot];
  for(i=0,n=0; i<ntot; i++)
    if(map[N1[i]][N2[i]] == 1) {
      N1new[n] = N1[i];
      N2new[n] = N2[i];
      Snew[n]  = S[i];
      n++;
    }
  bool done=false;
  for(n2=n2min; n2<=n2max && !done; n2+=2)
    for(n1=0; n1<nn1+2 && !done; n1++)
      if(map[n1][n2] == 2) {
        N1new[n] = n1;
        N2new[n] = n2;
        Snew[n]  = 0.;
        n++;
        if(n==newtot) done = true;
      }

  for(n1=0; n1<nn1+2; n1++)
    delete[] (map[n1]+n2min);
  delete[] map;
  if(ntot>0) {
    delete[] N1;
    delete[] N2;
    delete[] S;
  }
  ntot = newtot;
  N1   = N1new;
  N2   = N2new;
  S    = Snew;
  sortSn();
  //  SanityCheck(); // Uncomment if trying something new
  findNN();
}

//////////////////////////////////////////////////////////////////////////////
double GenPar::maxS() const
{
    double *Si, *Sn=S+ntot, Smax = fabs(S[0]);
    for(Si=S+1; Si<Sn; Si++)
        if(fabs(*Si) > Smax) Smax = fabs(*Si);
    return Smax;
}
//////////////////////////////////////////////////////////////////////////////
void GenPar::cut(const double f)
// sets all Sn=0 for which |Sn| < f * max(|Sn|)
{
    double *Si, *Sn=S+ntot, Smax = fabs(S[0]);
    for(Si=S+1; Si<Sn; Si++)
        if(fabs(*Si)  > Smax) Smax = fabs(*Si);
    Smax *= fabs(f);
    for(Si=S; Si<Sn; Si++)
        if(fabs(*Si) <= Smax) *Si = 0.f;
}

//////////////////////////////////////////////////////////////////////////////
int GenPar::Jz0()
// sets all Sn=0 for which n_z != 0
{
  bool no_orig=false;
  int newtot=0;
  for (int i=0; i!=ntot; i++) if(N2[i]==0) newtot++;
  if(!newtot) { newtot++; no_orig=true; }
  short *N1new = new short[newtot];
  short *N2new = new short[newtot];
  double *Snew  = new double[newtot];
  if(no_orig) {
    N1new[0] = 1;
    N2new[0] = 0;
    Snew[0]  = 0;
  } else {
    for(int i=0,n=0; i<ntot && n<newtot; i++)
      if(N2[i] == 0) {
        N1new[n] = N1[i];
        N2new[n] = N2[i];
        Snew[n]  = S[i];
        n++;
      }
  }
  if(ntot>0) {
    delete[] N1;
    delete[] N2;
    delete[] S;
  }
  ntot = newtot;
  N1   = N1new;
  N2   = N2new;
  S    = Snew;
  sortSn();
  findNN();
  if(no_orig) return -1;
  return 0;
}
int GenPar::JR0()
// sets all Sn=0 for which n_z != 0
{
  bool no_orig=false;
  int newtot=0;
  for (int i=0; i!=ntot; i++) if(N1[i]==0) newtot++;
  if(!newtot) { newtot++; no_orig=true; }
  short *N1new = new short[newtot];
  short *N2new = new short[newtot];
  double *Snew  = new double[newtot];
  if(no_orig) {
    N1new[0] = 1;
    N2new[0] = 0;
    Snew[0]  = 0;
  } else {
    for(int i=0,n=0; i<ntot && n<newtot; i++)
      if(N1[i] == 0) {
        N1new[n] = N1[i];
        N2new[n] = N2[i];
        Snew[n]  = S[i];
        n++;
      }
  }
  if(ntot>0) {
    delete[] N1;
    delete[] N2;
    delete[] S;
  }
  ntot = newtot;
  N1   = N1new;
  N2   = N2new;
  S    = Snew;
  sortSn();
  findNN();
  if(no_orig) return -1;
  return 0;
}


//////////////////////////////////////////////////////////////////////////////
int GenPar::NoMix()
// sets all Sn=0 for which (n_z != 0 && n_R !=0)
{
  bool no_orig=false;
  int newtot=0;
  for (int i=0; i!=ntot; i++) if(N1[i]==0 || N2[i]==0) newtot++;
  if(!newtot) { newtot++; no_orig=true; }
  short *N1new = new short[newtot];
  short *N2new = new short[newtot];
  double *Snew  = new double[newtot];
  if(no_orig) {
    N1new[0] = 1;
    N2new[0] = 0;
    Snew[0]  = 0;
  } else {
    for(int i=0,n=0; i<ntot && n<newtot; i++)
      if(N2[i] == 0 || N1[i] == 0) {
        N1new[n] = N1[i];
        N2new[n] = N2[i];
        Snew[n]  = S[i];
        n++;
      }
  }
  if(ntot>0) {
    delete[] N1;
    delete[] N2;
    delete[] S;
  }
  ntot = newtot;
  N1   = N1new;
  N2   = N2new;
  S    = Snew;
  sortSn();
  findNN();
  if(no_orig) return -1;
  return 0;
}

//////////////////////////////////////////////////////////////////////////////
void GenPar::addn1eq0(const int nadd)
// sets all Sn=0 for which n_z != 0
{
  if(nadd<1) {
    cerr << "GenPar.addn1eq0: number of points to add must be more than 0!\n";
    return;
  }
  int newtot=ntot+nadd;
  for (int i=0; i!=ntot; i++) {
    if(N2[i]!=0) cerr << "GenPar.addn1eq0 called on non Jz=0 GenPar\n";
    if(N1[i]==0) cerr << "GenPar.addn1eq0 called on GenPar with n1=0 terms\n";
  }
  short *N1new = new short[newtot];
  short *N2new = new short[newtot];
  double *Snew  = new double[newtot];
  for(int i=0,n=-2*nadd; i!=nadd; i++,n+=2) {
    N2new[i] = n;
    N1new[i] = 0;
    Snew[i]  = 0.;
  }
  for(int i=0; i!=ntot; i++) {
    N1new[i+nadd] = N1[i];
    N2new[i+nadd] = N2[i];
    Snew [i+nadd] = S[i];
  } 
  if(ntot>0) {
    delete[] N1;
    delete[] N2;
    delete[] S;
  }
  ntot = newtot;
  N1   = N1new;
  N2   = N2new;
  S    = Snew;
  sortSn();
  SanityCheck();
  findNN();
}

int GenPar::AddTerm(const int nadd1, const int nadd2) {
  for (int i=0; i!=ntot; i++) {
    if(N1[i]==nadd1 && N2[i]==nadd2) return 0;
  }
  int newtot=ntot+1;
  short *N1new = new short[newtot];
  short *N2new = new short[newtot];
  double *Snew  = new double[newtot];
  for(int i=0; i!=ntot; i++) {
    N1new[i] = N1[i];
    N2new[i] = N2[i];
    Snew [i] = S[i];
  } 
  N1new[ntot] = nadd1;
  N2new[ntot] = nadd2;
  Snew [ntot] = 0.;
  if(ntot>0) {
    delete[] N1;
    delete[] N2;
    delete[] S;
  }
  ntot = newtot;
  N1   = N1new;
  N2   = N2new;
  S    = Snew;
  sortSn();
  SanityCheck();
  findNN();
  return 1;
}

void GenPar::Build_JR0(const int type) {
  int newtot=0;
  short *N1new=0, *N2new=0;
  double *Snew=0;
  if(type==1){
    int newn2=0;
    newtot=ntot+1;
    JR0();
    N1new = new short[newtot];
    N2new = new short[newtot];
    Snew = new double[newtot];
    for(int i=0; i!=ntot; i++) {
      if(N2[i]<newn2) newn2 = N2[i];
      N1new[i] = N1[i];
      N2new[i] = N2[i];
      Snew [i] = S[i];
    } 
    newn2 -= 2;
    N1new[ntot] = 0;
    N2new[ntot] = newn2;
    Snew [ntot] = 0.; 
  }
  else if(type==2) {
    int  newn1=0;
    newtot=ntot+1;
    NoMix();
    N1new = new short[newtot];
    N2new = new short[newtot];
    Snew = new double[newtot];
    for(int i=0; i!=ntot; i++) {
      if(N1[i]>newn1) newn1 = N1[i];
      N1new[i] = N1[i];
      N2new[i] = N2[i];
      Snew [i] = S[i];
    } 
    newn1++;
    N1new[ntot] = newn1;
    N2new[ntot] = 0;
    Snew [ntot] = 0.; 
  }
  else if(type==3) {
    bool gotn2_0 = false;
    int minn2=0, maxn1mix=0;
    for(int i=0; i!=ntot; i++) {
      if(N2[i] < minn2) minn2 = N2[i];
      if(N2[i] && N1[i]>maxn1mix) maxn1mix = N1[i];
    }
    for(int i=0; i!=ntot; i++) if(!(N2[i]) && N1[i]==maxn1mix+1) gotn2_0 = true;
    newtot = (gotn2_0)? ntot-minn2 : ntot - minn2 + 1;
    N1new = new short[newtot];
    N2new = new short[newtot];
    Snew = new double[newtot];
    for(int i=0; i!=ntot; i++) {
      N1new[i] = N1[i];
      N2new[i] = N2[i];
      Snew [i] = S[i];
    }
    maxn1mix++;
    for(int i=ntot; i!=newtot; i++) {
      N1new[i] = maxn1mix;
      N2new[i] = minn2+2*(i-ntot);
      if(gotn2_0 && N2new[i] >=0) N2new[i] += 2;
      Snew [i] = 0.;
    }
  }
  if(ntot>0) {
    delete[] N1;
    delete[] N2;
    delete[] S;
  }
  ntot = newtot;
  N1   = N1new;
  N2   = N2new;
  S    = Snew;
  sortSn();
  SanityCheck();
  findNN();
}


//////////////////////////////////////////////////////////////////////////////
GenPar& GenPar::operator=(const GenPar& G)
{
    if(same_terms_as(G)) {
        for(short i=0; i<ntot; i++)
            S[i] = G.S[i];
    } else {
        if (ntot!=G.ntot) {
            if(ntot>0) {
                delete[] N1;
                delete[] N2;
                delete[] S;
            }
            ntot = G.ntot;
            N1   = new short[ntot];
            N2   = new short[ntot];
            S    = new double[ntot];
        }
        for(short i=0; i<ntot; i++) {
            N1[i] = G.N1[i];
            N2[i] = G.N2[i];
            S[i]  = G.S[i];
        }
    } 
    //SanityCheck();
    findNN();
    return *this;
}
////////////////////////////////////////////////////////////////////////////////
GenPar& GenPar::operator+=(const GenPar& G)
{
    if(ntot<=0)   return *this=G; 
    if(G.ntot<=0) return *this;
    if(same_terms_as(G)) {
        for(short i=0; i<ntot; i++) S[i] += G.S[i];
        return *this;
    }
    short* n1t = new short[ntot+G.ntot];
    short* n2t = new short[ntot+G.ntot];
    double* st  = new double[ntot+G.ntot];
    short i=0,j=0,n=0,mi,mj; 
    while(i<ntot || j<G.ntot) {
        if(j==G.ntot) {
            n1t[n] = N1[i];
            n2t[n] = N2[i];
            st[n]  = S[i];
            i++;
            n++;
        } else if(i==ntot) {
            n1t[n] = G.N1[j];
            n2t[n] = G.N2[j];
            st[n]  = G.S[j];
            j++;
            n++;
        } else {
            mi = 100 * N2[i] + N1[i];
            mj = 100 * G.N2[j] + G.N1[j];
            if(mi<mj && i<ntot) {
                n1t[n] = N1[i];
                n2t[n] = N2[i];
                st[n]  = S[i];
                i++;
                n++;
            } else if (mj<mi && j<G.ntot) {
                n1t[n] = G.N1[j];
                n2t[n] = G.N2[j];
                st[n]  = G.S[j];
                j++;
                n++;
            } else { // mi==mj
                n1t[n] = N1[i];
                n2t[n] = N2[i];
                st[n]  = S[i] + G.S[j];
                i++;
                j++;
                n++;
            }
        }
    }
    if(ntot>0) {
        delete[] S;
        delete[] N1;
        delete[] N2;
    }
    ntot = n;
    N1 = new short[ntot];
    N2 = new short[ntot];
    S  = new double[ntot];
    for(i=0; i<ntot; i++) {
        N1[i] = n1t[i];
        N2[i] = n2t[i];
        S[i]  = st[i];
    }
    delete[] st;
    delete[] n1t;
    delete[] n2t;
    //SanityCheck();
    findNN();
    return *this;
}
////////////////////////////////////////////////////////////////////////////////
GenPar& GenPar::operator-=(const GenPar& G)
{
    if(ntot==0)   return *this=-G;
    if(G.ntot==0) return *this;
    if(same_terms_as(G)) {
        for(short i=0; i<ntot; i++) S[i] -= G.S[i];
        return *this;
    }
    short* n1t = new short[ntot+G.ntot];
    short* n2t = new short[ntot+G.ntot];
    double* st  = new double[ntot+G.ntot];
    short i=0,j=0,n=0,mi,mj; 
    while(i<ntot || j<G.ntot) {
        if(j==G.ntot) {
            n1t[n] = N1[i];
            n2t[n] = N2[i];
            st[n]  = S[i];
            i++;
            n++;
        } else if(i==ntot) {
            n1t[n] = G.N1[j];
            n2t[n] = G.N2[j];
            st[n]  = G.S[j];
            j++;
            n++;
        } else {
            mi = 100 * N2[i] + N1[i];
            mj = 100 * G.N2[j] + G.N1[j];
            if(mi<mj && i<ntot) {
                n1t[n] = N1[i];
                n2t[n] = N2[i];
                st[n]  = S[i];
                i++;
                n++;
            } else if (mj<mi && j<G.ntot) {
                n1t[n] = G.N1[j];
                n2t[n] = G.N2[j];
                st[n]  =-G.S[j];
                j++;
                n++;
            } else { // mi==mj
                n1t[n] = N1[i];
                n2t[n] = N2[i];
                st[n]  = S[i] - G.S[j];
                i++;
                j++;
                n++;
            }
        }
    }
    if(ntot>0) {
        delete[] S;
        delete[] N1;
        delete[] N2;
    }
    ntot = n;
    N1 = new short[ntot];
    N2 = new short[ntot];
    S  = new double[ntot];
    for(i=0; i<ntot; i++) {
        N1[i] = n1t[i];
        N2[i] = n2t[i];
        S[i]  = st[i];
    }
    delete[] st;
    delete[] n1t;
    delete[] n2t;
    //SanityCheck();
    findNN();
    return *this;
}
////////////////////////////////////////////////////////////////////////////////
GenPar& GenPar::operator=(const double x)
{
    for(short i=0; i<ntot; i++) S[i] = x;
    return *this;
}
////////////////////////////////////////////////////////////////////////////////
GenPar& GenPar::operator*=(const double x)
{
    for(short i=0; i<ntot; i++) S[i] *= x;
    return *this;
}
////////////////////////////////////////////////////////////////////////////////
GenPar& GenPar::operator/=(const double x)
{
    if(x==0.) throw std::runtime_error("Torus Error -4: GenPar: division by zero");
    for(short i=0; i<ntot; i++) S[i] *= x;
    return *this;
}
////////////////////////////////////////////////////////////////////////////////
GenPar GenPar::operator-() const
{
    GenPar F(*this);
    for(short i=0; i<ntot; i++) F.S[i] = -F.S[i];
    return F;
}
////////////////////////////////////////////////////////////////////////////////
GenPar GenPar::operator-(const GenPar& GP) const
{
  GenPar F(*this); return F-=GP;
}
////////////////////////////////////////////////////////////////////////////////
GenPar GenPar::operator+(const GenPar& GP) const
{
  GenPar F(*this); return F+=GP;
}
////////////////////////////////////////////////////////////////////////////////
GenPar GenPar::operator*(const double& d) const
{
  GenPar F(*this); return F*=d;
}
////////////////////////////////////////////////////////////////////////////////
GenPar GenPar::operator/(const double& d) const
{
  GenPar F(*this); return F/=d;
}
////////////////////////////////////////////////////////////////////////////////
int GenPar::operator==(const GenPar& G) const
{
    if(ntot != G.ntot) return 0;
    if(nn1  != G.nn1 ) return 0;
    if(nn2  != G.nn2 ) return 0;
    for(short i=0; i<ntot; i++)
        if(N1[i]!=G.N1[i] || N2[i]!=G.N2[i] || S[i]!=G.S[i]) return 0;
    return 1;
}
////////////////////////////////////////////////////////////////////////////////
int GenPar::operator!=(const GenPar& G) const
{
    if(ntot != G.ntot) return 1;
    if(nn1  != G.nn1 ) return 1;
    if(nn2  != G.nn2 ) return 1;
    for(short i=0; i<ntot; i++)
        if(N1[i]!=G.N1[i] || N2[i]!=G.N2[i] || S[i]!=G.S[i]) return 1;
    return 0;
}
////////////////////////////////////////////////////////////////////////////////
void GenPar::put(ostream& to) const
{
    short i;
    to<<ntot<<' ';
    for(i=0; i<ntot; i++) to<<' '<<N1[i];
    to<<'\n';
    for(i=0; i<ntot; i++) to<<' '<<N2[i];
    to<<'\n';
    put_terms(to);
}


////////////////////////////////////////////////////////////////////////////////
void GenPar::put_terms(ostream& /*to*/) const
{
//    ::put(S,ntot,to);
}
////////////////////////////////////////////////////////////////////////////////
void GenPar::get(istream& from)
{
    short newtot;
    from >> newtot;
    if(newtot <= 0) throw std::runtime_error("Torus Error -4: GenPar: number of terms <= 0");
    if(newtot != ntot) {
        if(ntot>0) {
            delete[] N1;
            delete[] N2;
            delete[] S;
        }
        ntot = newtot;
        N1   = new short[ntot];
        N2   = new short[ntot];
        S    = new double[ntot];
    } 
    short i;
    for(i=0; i<ntot; i++) from >> N1[i];
    for(i=0; i<ntot; i++) {
        from >> N2[i];
        if(N2[i]%2 != 0) throw std::runtime_error("Torus Error -4: GenPar: odd n2 in get()");
    }
    get_terms(from);
    SanityCheck();
    //findNN();
}
////////////////////////////////////////////////////////////////////////////////
void GenPar::get_terms(istream& /*from*/)
{
//    ::get(S,ntot,from);
}
////////////////////////////////////////////////////////////////////////////////
int GenPar::skip(istream& from) const
{
    short n, n12;
    from>>n;                                // get number of terms
    if(n <= 0)
        throw std::runtime_error("Torus Error -4: GenPar: number of terms <= 0");
    short i;
    for(i=0; i<n; i++) from>>n12;        // skip n_1
    for(i=0; i<n; i++) from>>n12;        // skip n_2
    skip_terms(from,n);                  // skip S_n
    return n;                            // return number of terms
}
////////////////////////////////////////////////////////////////////////////////
void GenPar::skip_terms(istream& from, const int n) const
{
    int n5=5*n;
    ws(from);
    from.seekg(n5+n5/80,std::ios::cur);
}

////////////////////////////////////////////////////////////////////////////////
//##############################################################################
// class GenFnc

PSPD GenFnc::Forward(const PSPD& Jt) const
// if necessary, this routine can be speeded up by reducing the calls of
// cos() and using recursiv computations of cos(n1*th1 + n2*th2) exploiting
// the fact that the (n1,n2) are ordered;
{
    PSPD  jt=Jt;
    short i;
    double sncos;
    for(i=0; i<Sn.NumberofTerms(); i++) if(Sn(i)!=0.) {
        sncos  = Sn(i) * cos(jt(2)*Sn.n1(i) + jt(3) * Sn.n2(i));
        sncos += sncos;   // sncos = 2 * Sn * cos(n1*th1 + n2*th2)
        jt[0] += Sn.n1(i) * sncos;
        jt[1] += Sn.n2(i) * sncos;
    }
    return jt;
}
////////////////////////////////////////////////////////////////////////////////
PSPT GenFnc::Forward3D(const PSPT& Jt3) const
{
  PSPT jt3 = Jt3;
  PSPD Jt2=Jt3.Give_PSPD();
  jt3.Take_PSPD(Forward(Jt2));
  return jt3;
}
////////////////////////////////////////////////////////////////////////////////
PSPD GenFnc::ForwardWithDerivs(const PSPD& Jt, double djdt[2][2]) const
{
    PSPD   jt=Jt;
    short  i;
    double thn, sncos, snsin;
    djdt[0][0]=0;
    djdt[1][0]=0;
    djdt[1][1]=0;
    for(i=0; i<Sn.NumberofTerms(); i++) if(Sn(i)!=0.) {
        thn    = jt(2)*Sn.n1(i) + jt(3)*Sn.n2(i);
        sncos  = Sn(i) * cos(thn); 
        snsin  = Sn(i) * sin(thn); 
        sncos += sncos;   // sncos = 2 * Sn * cos(n1*th1 + n2*th2)
        snsin += snsin;   // snsin = 2 * Sn * sin(n1*th1 + n2*th2)
        jt[0] += Sn.n1(i) * sncos;
        jt[1] += Sn.n2(i) * sncos;
        djdt[0][0] -= Sn.n1(i) * Sn.n1(i) * snsin;
        djdt[1][0] -= Sn.n1(i) * Sn.n2(i) * snsin;
        djdt[1][1] -= Sn.n2(i) * Sn.n2(i) * snsin;
    }
    djdt[0][1] = djdt[1][0];
    return jt;
}

//##############################################################################
// class GenFncFit

void GenFncFit::FreeTrig()
{
    short i;
    for(i=0; i<Sn.NumberofTerms(); i++) {
        delete[] cc1[i];
        delete[] cc2[i];
        delete[] ss1[i];
        delete[] ss2[i];
    }
    delete[] cc1;
    delete[] cc2;
    delete[] ss1;
    delete[] ss2;
}
////////////////////////////////////////////////////////////////////////////////
void GenFncFit::SetTrig()
{
    short    i,t;
    double thn;
    Pin1 = Pi/double(Nth1);
    Pin2 = Pi/double(Nth2);
    for(i=0; i<Sn.NumberofTerms(); i++) {
        for(t=0; t<Nth1; t++) { 
            thn       = Sn.n1(i) * t * Pin1;
            cc1[i][t] = cos(thn);
            ss1[i][t] = sin(thn);
        }
        for(t=0; t<Nth2; t++) { 
            thn       = Sn.n2(i) * t * Pin2;
            cc2[i][t] = cos(thn);
            ss2[i][t] = sin(thn);
        }
    }
}
////////////////////////////////////////////////////////////////////////////////
void GenFncFit::AllocAndSetTrig()
{
    short    i,t;
    double thn;
    Pin1 = Pi/double(Nth1);
    Pin2 = Pi/double(Nth2);
    cc1  = new double* [Sn.NumberofTerms()];
    cc2  = new double* [Sn.NumberofTerms()];
    ss1  = new double* [Sn.NumberofTerms()];
    ss2  = new double* [Sn.NumberofTerms()];
    for(i=0; i<Sn.NumberofTerms(); i++) {
        cc1[i] = new double[Nth1];
        cc2[i] = new double[Nth2];
        ss1[i] = new double[Nth1];
        ss2[i] = new double[Nth2];
        for(t=0; t<Nth1; t++) { 
            thn       = Sn.n1(i) * t * Pin1;
            cc1[i][t] = cos(thn);
            ss1[i][t] = sin(thn);
        }
        for(t=0; t<Nth2; t++) { 
            thn       = Sn.n2(i) * t * Pin2;
            cc2[i][t] = cos(thn);
            ss2[i][t] = sin(thn);
        }
    }
}
////////////////////////////////////////////////////////////////////////////////
void GenFncFit::set_Nth(const int nt1, const int nt2)
{
    if(nt1==Nth1 && nt1==Nth2) return;
    FreeTrig();
    Nth1=nt1;
    Nth2=nt2;
    AllocAndSetTrig();
}
////////////////////////////////////////////////////////////////////////////////
void GenFncFit::set_parameters(const GenPar& S)
{
    if( S.same_terms_as(Sn)) {
        GenFnc::set_parameters(S);
    } else if(S.NumberofTerms() == Sn.NumberofTerms()) {
        GenFnc::set_parameters(S);
        SetTrig();
    } else {
        GenFnc::set_parameters(S);
        FreeTrig();
        AllocAndSetTrig();
    }
}
////////////////////////////////////////////////////////////////////////////////
PSPD GenFncFit::Map(const double J1,const double J2, const int t1,const int t2)
const
{
    const double acc = 1.e-10;
    if(t1<0 || t1>=Nth1 || t2<0 || t2>=Nth2)
        throw std::runtime_error("Torus Error -4: GenFncFit: (t1,t2) out of range");
    PSPD   jt=PSPD(J1, J2, t1*Pin1, t2*Pin2);
    double sncos;
    for(short i=0; i<Sn.NumberofTerms(); i++) if(Sn(i)!=0.) { 
                        // remember that Sn is the GenPar in the member GenFnc S
        sncos  = Sn(i) * (cc1[i][t1]*cc2[i][t2] - ss1[i][t1]*ss2[i][t2]);
        if(fabs(sncos) <= acc) sncos = 0.;
            else sncos += sncos;   // 2 * S_(n1,n2) * cos(n1*th1 + n2*th2)
        jt[0] += Sn.n1(i) * sncos;
        jt[1] += Sn.n2(i) * sncos;
    }
    return jt;
}
////////////////////////////////////////////////////////////////////////////////
PSPD GenFncFit::MapWithDerivs(const double J1, const double J2, const int t1,
                               const int t2, GenPar& dJ1dSn, GenPar& dJ2dSn)
const
{
    const double acc = 1.e-10;
    if(t1<0 || t1>=Nth1 || t2<0 || t2>=Nth2)
        throw std::runtime_error("Torus Error -4: GenFncFit: (t1,t2) out of range");
    PSPD   jt=PSPD(J1, J2, t1*Pin1, t2*Pin2);
    double costn;
    for(short i=0; i<Sn.NumberofTerms(); i++) {
        costn      = (cc1[i][t1]*cc2[i][t2] - ss1[i][t1]*ss2[i][t2]);
        if(fabs(costn) <= acc) costn = 0.;
            else costn += costn;   // 2 * cos(n1*th1 + n2*th2)
        dJ1dSn[i]  = Sn.n1(i) * costn;
        dJ2dSn[i]  = Sn.n2(i) * costn;
        jt[0]     += Sn(i) * dJ1dSn(i);
        jt[1]     += Sn(i) * dJ2dSn(i);
    }
    return jt;
}

////////////////////////////////////////////////////////////////////////////////
// class AngMap ************************************************************* //
////////////////////////////////////////////////////////////////////////////////

PSPD AngMap::Map(const PSPD& Jt) const
{
    short    i;
    PSPD   JT=Jt;
    double sinth;
    for(i=0; i<A.NumberofTerms(); i++) {
        sinth  = sin(A.dS1.n1(i)*Jt(2) + A.dS1.n2(i)*Jt(3));
        sinth += sinth;            // 2 * sin(n1*th1+n2*th2)
        JT[2] += A.dS1(i) * sinth;
        JT[3] += A.dS2(i) * sinth;
    }
    return JT;
}
////////////////////////////////////////////////////////////////////////////////
PSPD AngMap::BackwardWithDerivs(const PSPD& Jt, double dTdt[2][2]) const
{
    short    i;
    PSPD   JT=Jt;
    double temp, costh, sinth;
    dTdt[0][0] = dTdt[1][1] = 1.;
    dTdt[0][1] = dTdt[1][0] = 0.;
    for(i=0; i<A.NumberofTerms(); i++) {
        temp   = A.dS1.n1(i)*Jt(2) + A.dS1.n2(i)*Jt(3);
        sinth  = sin(temp); sinth += sinth;    // 2 * sin(n1*th1+n2*th2)
        costh  = cos(temp); costh += costh;    // 2 * cos(n1*th1+n2*th2)
        JT[2] += A.dS1(i) * sinth;
        JT[3] += A.dS2(i) * sinth;
        dTdt[0][0] += (temp = A.dS1(i)*costh) * A.dS1.n1(i);
        dTdt[0][1] +=  temp                   * A.dS1.n2(i);
        dTdt[1][0] += (temp = A.dS2(i)*costh) * A.dS2.n1(i);
        dTdt[1][1] +=  temp                   * A.dS2.n2(i);
    }
    return JT;
}
////////////////////////////////////////////////////////////////////////////////
PSPD AngMap::NewtonStep(double& F, double& dF1, double& dF2, const PSPD Jt,
                        const PSPD& JT) const
{
    short    i;
    PSPD   dJt=0.;
    double temp,thn,costh,sinth,f1,f2,f11=1.,f12=0.,f21=0.,f22=1.,det;
    f1 = Jt(2)-JT(2);
    f2 = Jt(3)-JT(3);
    for(i=0; i<A.NumberofTerms(); i++) {
        thn   = A.dS1.n1(i)*Jt(2) + A.dS1.n2(i)*Jt(3);
        costh = cos(thn);
        costh+= costh;
        sinth = sin(thn);
        sinth+= sinth;
        f1   += A.dS1(i) * sinth;
        f2   += A.dS2(i) * sinth;
        temp  = A.dS1(i) * costh;
        f11  += temp * A.dS1.n1(i);
        f12  += temp * A.dS1.n2(i);
        temp  = A.dS2(i) * costh;
        f21  += temp * A.dS1.n1(i);
        f22  += temp * A.dS1.n2(i);
    }
    if(std::isnan(f1) || std::isinf(f1) || fabs(f1)>INT_MAX)
      f1 = 0.001; // just in case
    if(std::isnan(f2) || std::isinf(f2) || fabs(f2)>INT_MAX)
      f2 = 0.001; // just in case
    f1 = math::wrapAngle(f1+Pi)-Pi;
    f2 = math::wrapAngle(f2+Pi)-Pi;
    det    = f11*f22 - f12*f21;
    dJt[2] = (f2*f12-f1*f22) / det;
    dJt[3] = (f1*f21-f2*f11) / det;
    F      = 0.5 * (f1*f1+f2*f2);
    dF1    = f1*f11+f2*f21;
    dF2    = f1*f12+f2*f22;
    return dJt;
}
////////////////////////////////////////////////////////////////////////////////
PSPD AngMap::Forward(const PSPD& Input) const
{
    const double    alpha=1.e-4,LIN=1.e-6;
    PSPD   JT=Input,Jt,Jtry,dJt,dJT;
    short  it=0,maxit=20;
    double          F0,Fn,dF1,dF2;
    double Fo,lam,laq,lom=1.,loq,lax,lin,slp,a,b,r1,r2,disc;
    AlignAngles(JT);         //set angles to be between 0 and 2pi
    Jt = JT;
    int numIter=0;
    do {
        lam = 1;
        dJt = NewtonStep(F0,dF1,dF2,Jt,JT); // see above
        slp = dF1*dJt(2)+dF2*dJt(3);
        Fo  = F0;
        for(;;) {
            lax = 0.5*lam;
            lin = 0.05*lam;
            if(lin<LIN) lin=LIN;
            Jtry= Jt+lam*dJt;
            AlignAngles(Jtry);
            dJT = Map(Jtry) - JT;
            if(std::isnan(dJT(2)) || std::isinf(dJT(2)) || fabs(dJT(2))>INT_MAX)
              dJT[2] = 0.001;
            if(std::isnan(dJT(3)) || std::isinf(dJT(3)) || fabs(dJT(3))>INT_MAX)
              dJT[3] = 0.001;
        dJT[2] = math::wrapAngle(dJT(2)+Pi)-Pi;
        dJT[3] = math::wrapAngle(dJT(3)+Pi)-Pi;
            Fn = 0.5*(dJT(2)*dJT(2)+dJT(3)*dJT(3));
            if( Fn<=F0+alpha*lam*slp || lam<=LIN || ++numIter>42) break;
            if(lam==1.) {
                lom = lam;
                lam =-0.5*slp/(Fn-F0-slp);
            }
            else {
                r1  = Fn-F0-lam*slp;
                r2  = Fo-F0-lom*slp;
                laq = lam*lam;
                loq = lom*lom;
                a   = r1/laq-r2/loq;
                b   =-lom*r1/laq+lam*r1/loq;
                laq = lam-lom;
                b  /= laq;
                lom = lam;
                if(a==0.) 
                    lam =-0.5*slp/b;
                else {
                    a /= laq;
                    disc=b*b-3.*a*slp;
                    if(disc<0.) disc=0.;
                    lam=(-b+sqrt(disc))/(3.*a);
                }
            }
            if(lam>lax) lam=lax;
            if(lam<lin) lam=lin;
            Fo  = Fn;
        }
        Jt = Jtry;
    } while ( Fn>1.e-15 && maxit>it++ );
    AlignAngles(Jt);
    return Jt;
}

////////////////////////////////////////////////////////////////////////////////
PSPT AngMap::Backward3D (const PSPT& Jt) const 
{ // from toy to real
  short  i;
  PSPT   JT = Jt;
  //Angles Ang = ang;
  double sinth;
  for(i=0; i<A.NumberofTerms(); i++) {
    sinth  = sin(A.dS1.n1(i)*Jt(3) + A.dS1.n2(i)*Jt(4));
    sinth += sinth;            // 2 * sin(n1*th1+n2*th2)
    JT[3] += A.dS1(i) * sinth;
    JT[4] += A.dS2(i) * sinth;
    JT[5] += A.dS3(i) * sinth;
  }
  return JT;
}
////////////////////////////////////////////////////////////////////////////////
PSPT AngMap::Backward3DWithDerivs(const PSPT& Jt, double dTdt[2][2]) const
{
    short    i;
    PSPT   JT=Jt;
    double temp, costh, sinth;
    dTdt[0][0] = dTdt[1][1] = 1.;
    dTdt[0][1] = dTdt[1][0] = 0.;
    for(i=0; i<A.NumberofTerms(); i++) {
        temp   = A.dS1.n1(i)*Jt(3) + A.dS1.n2(i)*Jt(4);
        sinth  = sin(temp); sinth += sinth;    // 2 * sin(n1*th1+n2*th2)
        costh  = cos(temp); costh += costh;    // 2 * cos(n1*th1+n2*th2)
        JT[3] += A.dS1(i) * sinth;
        JT[4] += A.dS2(i) * sinth;
        JT[5] += A.dS3(i) * sinth;
        dTdt[0][0] += (temp = A.dS1(i)*costh) * A.dS1.n1(i);
        dTdt[0][1] +=  temp                   * A.dS1.n2(i);
        dTdt[1][0] += (temp = A.dS2(i)*costh) * A.dS2.n1(i);
        dTdt[1][1] +=  temp                   * A.dS2.n2(i);
    }
    return JT;
}



////////////////////////////////////////////////////////////////////////////////
PSPT AngMap::Forward3D (const PSPT& JT3) const
{ // from real to toy
  double sinth;
  PSPT Jt3 = JT3;
  PSPD Jt2,JT2 = JT3.Give_PSPD();

  Jt2 = Forward(JT2); // find th_R, th_z (above)
  Jt3.Take_PSPD(Jt2);

  for(int i=0; i<A.NumberofTerms(); i++) {
    sinth  = sin(A.dS1.n1(i)*Jt3(3) + A.dS1.n2(i)*Jt3(4));
    sinth += sinth;             // 2 * sin(n1*th1+n2*th2)
    Jt3[5] -= A.dS3(i) * sinth; // no th_phi dependence in sinth, 
                                // so this step is the easiest
  }
  return Jt3;
}






} // namespace

//end of GenFnc.cc//////////////////////////////////////////////////////////////
