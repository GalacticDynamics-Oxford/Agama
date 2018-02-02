/***************************************************************************//**
\file PJM_utils.h 
\brief Various useful little utilities.

*                                                                              *
* PJM_utils.h                                                                  *
*                                                                              *
* C++ code written by Paul McMillan, 2011                                      *
* Oxford University, Department of Physics, Theoretical Physics.               *
* address: 1 Keble Road, Oxford OX1 3NP, United Kingdom                        *
* e-mail:  p.mcmillan1@physics.ox.ac.uk                                        *
*                                                                              *
*       Classes which output the value of a distribution function              *
*                                                                              *
*******************************************************************************/

#ifndef _PJMutils_
#define _PJMutils_ 1
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <cstdlib>
namespace torus {
using std::cerr;
using std::string;
using std::ifstream;
using std::ofstream;

void my_open(ifstream& from, const char* name) {
  from.open(name);
  if(!from) {
    cerr << "FILE "<< name <<" doesn't exist\n";
    exit(1);
  }
}

void my_open(ofstream& to, const char* name) {
  to.open(name);
  if(!to) {
    cerr << "FILE "<< name <<" can't be created\n";
    exit(1);
  }
}

void my_open(ifstream& from, string name) {
  from.open(name.c_str());
  if(!from) {
    cerr << "FILE "<< name <<" doesn't exist\n";
    exit(1);
  }
}

void my_open(ofstream& to, string name) {
  to.open(name.c_str());
  if(!to) {
    cerr << "FILE "<< name <<" can't be created\n";
    exit(1);
  }
}

int how_many_lines(ifstream &from) {
  string line;
  int nlines=0;
  from.clear();
  from.seekg(0);
  while(getline(from,line)) nlines++;
  from.clear();
  from.seekg(0);
  return nlines;
}

int entrys_in_line(ifstream &from) {
  string line;
  int nentrys=0,ini = from.tellg();
  if(!getline(from,line)) return 0;
  from.clear(); // just in case
  from.seekg(ini);
  bool gap=false, wasgap=true;
  for(unsigned int i=0;i!=line.size();i++) {
    if(line[i] == ' ' || line[i] == '\t') gap=true;
    else gap = false;
    if(gap && !wasgap) nentrys++;
    wasgap = gap;
  }
  if(!wasgap) nentrys++;
  return nentrys;
}

int entrys_in_line(string line) {
  int nentrys=0;
  bool gap=false, wasgap=true;
  for(unsigned int i=0;i!=line.size();i++) {
    if(line[i] == ' ' || line[i] == '\t') gap=true;
    else gap = false;
    if(gap && !wasgap) nentrys++;
    wasgap = gap;
  }
  if(!wasgap) nentrys++;
  return nentrys;
}

string int_to_string(int number)
{
  std::stringstream ss;//create a stringstream
   ss << number;//add number to the stream
   return ss.str();//return a string with the contents of the stream
}

/** \brief Contains various useful functions.

 */
namespace PJM {
  template <class T>
    T *matrix(int n){
    T *m1 = new T[n];
    for(int i=0;i<n;i++) m1[i]=0.;
    return m1;
  }

  template <class T>
    T **matrix(int n,int m){
    T **m1 = new T*[n];
    for(int i=0; i<n; i++) m1[i] = matrix<T>(m);
    return m1;
  }
  template <class T>
    T ***matrix(int n,int m,int l){
    T ***m1 = new T**[n];
    for(int i=0; i<n; i++) m1[i] = matrix<T>(m,l);
    return m1;
  }
  template <class T>
    T ****matrix(int n,int m,int l,int k){
    T ****m1 = new T***[n];
    for(int i=0; i<n; i++) m1[i] = matrix<T>(m,l,k);
    return m1;
  }
  template <class T>
    T *****matrix(int n,int m,int l,int k,int j){
    T *****m1 = new T****[n];
    for(int i=0; i<n; i++) m1[i] = matrix<T>(m,l,k,j);
    return m1;
  }


  template <class T>
    void del_matrix(T **m1,int n){
    for(int i=0;i<n;i++) delete[] m1[i];
    delete [] m1;
  }
  template <class T>
    void del_matrix(T ***m1,int n,int m){
    for(int i=0;i<n;i++) del_matrix<T>(m1[i],m);
    delete [] m1;
  }

  template <class T>
    void del_matrix(T ****m1,int n,int m,int l){
    for(int i=0;i<n;i++) del_matrix<T>(m1[i],m,l);
    delete [] m1;
  }

  template <class T>
    void del_matrix(T *****m1,int n,int m,int l, int k){
    for(int i=0;i<n;i++) del_matrix<T>(m1[i],m,l,k);
    delete [] m1;
  }



  template <class T>
    void copy_array(T *m2, T *m1, int n) {
    for(int i=0;i<n;i++) m2[i] = m1[i];
  }
  
  template <class T>
    void copy_array(T *m2, T *m1, int n, int m) {
    for(int i=0;i<n;i++) 
      for(int j=0;j<m;j++) 
	m2[i][j] = m1[i][j];
  }
  
  template <class T>
    void copy_array(T *m2, T *m1, int n, int m, int l) {
    for(int i=0;i<n;i++) 
      for(int j=0;j<m;j++) 
	for(int k=0;k<l;k++)
	  m2[i][j][k] = m1[i][j][k];
  }

  template <class T>
    void fill_array(T *m2, T m1, int n) {
    for(int i=0;i<n;i++) m2[i] = m1;
  }
  
  template <class T>
    void fill_array(T *m2, T m1, int n, int m) {
    for(int i=0;i<n;i++) 
      for(int j=0;j<m;j++) 
	m2[i][j] = m1;
  }
  
  template <class T>
    void fill_array(T *m2, T m1, int n, int m, int l) {
    for(int i=0;i<n;i++) 
      for(int j=0;j<m;j++) 
	for(int k=0;k<l;k++)
	  m2[i][j][k] = m1;
  }

  void ERROR(string out) {
    std::cerr << out << '\n'; exit(1);
  }


  double gaussian(double x, double x0, double sig) {
    double isig = 1./sig, tmp = (x-x0)*isig, tmp2 = -0.5*tmp*tmp;
    const double isrttpi = 0.39894228;
    return isrttpi*isig*exp(tmp2);
  }

}

} // namespace
#endif
