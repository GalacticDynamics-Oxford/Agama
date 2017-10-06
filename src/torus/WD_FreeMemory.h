/**
\file WD_FreeMemory.h 
\brief Templates AllocxD and FreexD (where 1<x<=4)
Allocates and deallocates x dimensional arrays.


FreeMemory.h 

C++ code written by Walter Dehnen, 1996, 
Oxford University, Department of Physics, Theoretical Physics 
address: 1 Keble Road, Oxford OX1 3NP, United Kingdom 
e-mail : w.dehnen@physics.ox.ac.uk 

*****************************************************************************

contains template functions for tha allocation and de-allocation of 1D,2D, 
3D, and 4D arrays of given size. The number of calls of the operator new 
is D, the dimensionality. The Syntax is as follows: 

allocation: 
    function names:  Alloc1D, Alloc2D, Alloc3D, Alloc4D 
    1st Argument:    pointer of corresponding depth, i.e. T** for Alloc2D 
    2nd Argument:    size of the array to be allocated in each dimension, 
                     for Alloc1D this is a const int, and a const int[D], 
                     where D denotes the dimensionality, for the rest. 
    3rd Argument [optional]: initialization value 

de-allocation: 
    function names:  Free1D, Free2D, Free3D, Free4D 
    1st Argument:    pointer of corresponding depth (see above), which has 
                     before been passed to the corresponding allocation 
                     routine above. 

****************************************************************************/

#ifndef FreeMemory
#define FreeMemory
namespace WD{

template <class ALLOCTYPE>
inline int Alloc1D(ALLOCTYPE* &A, const int N)
{
    A = new ALLOCTYPE[N]; 
    if(!A) return 1;
    return 0;
}

template <class ALLOCTYPE>
inline int Alloc1D(ALLOCTYPE* &A, const int N, const ALLOCTYPE X)
{
    A = new ALLOCTYPE[N]; 
    if(!A) return 1;
    ALLOCTYPE *a, *Au=A+N;
    for(a=A; a<Au; a++) *a = X;
    return 0;
}

template <class ALLOCTYPE>
inline int Alloc2D(ALLOCTYPE** &A, const int N[2])
{
    int i, iN1;
    A    = new ALLOCTYPE* [N[0]];        if(!A) return 1;
    A[0] = new ALLOCTYPE[N[0]*N[1]];     if(!A[0]) return 1;
    for(i=1, iN1=N[1]; i<N[0]; i++,iN1+=N[1])
        A[i] = A[0] + iN1;
    return 0;
}

template <class ALLOCTYPE>
inline int Alloc2D(ALLOCTYPE** &A, const int N[2], const ALLOCTYPE X)
{
    int i,iN1,j;
    A    = new ALLOCTYPE* [N[0]];        if(!A) return 1;
    A[0] = new ALLOCTYPE[N[0]*N[1]];     if(!A[0]) return 1;
    for(i=iN1=0; i<N[0]; i++,iN1+=N[1]) {
        A[i] = A[0]+iN1;
        for(j=0; j<N[1]; j++) A[i][j] = X;
    }
    return 0;
}

template <class ALLOCTYPE>
inline int Alloc3D(ALLOCTYPE*** &A, const int N[3])
{
    int i,j, iN1,iN12,jN2, N12=N[1]*N[2];
    A       = new ALLOCTYPE** [N[0]];       if(!A) return 1;
    A[0]    = new ALLOCTYPE* [N[0]*N[1]]; if(!A[0]) return 1;
    A[0][0] = new ALLOCTYPE    [N[0]*N12];  if(!A[0][0]) return 1;
    for(i=iN1=iN12=0; i<N[0]; i++,iN1+=N[1],iN12+=N12) {
        A[i]    = A[0]    + iN1;
        A[i][0] = A[0][0] + iN12;
        for(j=1,jN2=N[2]; j<N[1]; j++,jN2+=N[2])
            A[i][j] = A[i][0] + jN2;
    }
    return 0;
}

template <class ALLOCTYPE>
inline int Alloc3D(ALLOCTYPE*** &A, const int N[3], const ALLOCTYPE X)
{
    int i,j,k, iN1,iN12,jN2, N12=N[1]*N[2];
    A       = new ALLOCTYPE** [N[0]];       if(!A) return 1;
    A[0]    = new ALLOCTYPE* [N[0]*N[1]]; if(!A[0]) return 1;
    A[0][0] = new ALLOCTYPE    [N[0]*N12];  if(!A[0][0]) return 1;
    for(i=iN1=iN12=0; i<N[0]; i++,iN1+=N[1],iN12+=N12) {
        A[i]    = A[0]    + iN1;
        A[i][0] = A[0][0] + iN12;
        for(j=jN2=0; j<N[1]; j++,jN2+=N[2]) {
            A[i][j] = A[i][0] + jN2;
            for(k=0; k<N[2]; k++) A[i][j][k] = X;
        }
    }
    return 0;
}

template <class ALLOCTYPE>
inline int Alloc4D(ALLOCTYPE**** &A, const int N[4])
{
    int i,j,k, iN1,iN12,iN123, jN2,jN23, kN3,
                 N12=N[1]*N[2],N123=N12*N[3],N23=N[2]*N[3];
    A          = new ALLOCTYPE*** [N[0]];      if(!A) return 1;
    A[0]       = new ALLOCTYPE** [N[0]*N[1]]; if(!A[0]) return 1;
    A[0][0]    = new ALLOCTYPE* [N[0]*N12];  if(!A[0][0]) return 1;
    A[0][0][0] = new ALLOCTYPE[N[0]*N123]; if(!A[0][0][0]) return 1;
    for(i=iN1=iN12=iN123=0; i<N[0]; i++,iN1+=N[1],iN12+=N12,iN123+=N123) {
        A[i]       = A[0]       + iN1;
        A[i][0]    = A[0][0]    + iN12;
        A[i][0][0] = A[0][0][0] + iN123;
        for(j=jN2=jN23=0; j<N[1]; j++,jN2+=N[2],jN23+=N23) {
            A[i][j]    = A[i][0]    + jN2;
            A[i][j][0] = A[i][0][0] + jN23;
            for(k=1,kN3=N[3]; k<N[2]; k++,kN3+=N[3])
                A[i][j][k] = A[i][j][0] + kN3;
        }
    }
    return 0;
}

template <class ALLOCTYPE>
inline int Alloc4D(ALLOCTYPE**** &A, const int N[4], const ALLOCTYPE X)
{
    int i,j,k,l, iN1,iN12,iN123, jN2,jN23, kN3,
                 N12=N[1]*N[2],N123=N12*N[3],N23=N[2]*N[3];
    A          = new ALLOCTYPE*** [N[0]];      if(!A) return 1;
    A[0]       = new ALLOCTYPE** [N[0]*N[1]]; if(!A[0]) return 1;
    A[0][0]    = new ALLOCTYPE* [N[0]*N12];  if(!A[0][0]) return 1;
    A[0][0][0] = new ALLOCTYPE     [N[0]*N123]; if(!A[0][0][0]) return 1;
    for(i=iN1=iN12=iN123=0; i<N[0]; i++,iN1+=N[1],iN12+=N12,iN123+=N123) {
        A[i]       = A[0]       + iN1;
        A[i][0]    = A[0][0]    + iN12;
        A[i][0][0] = A[0][0][0] + iN123;
        for(j=jN2=jN23=0; j<N[1]; j++,jN2+=N[2],jN23+=N23) {
            A[i][j]    = A[i][0]    + jN2;
            A[i][j][0] = A[i][0][0] + jN23;
            for(k=kN3=0; k<N[2]; k++,kN3+=N[3]) {
                A[i][j][k] = A[i][j][0] + kN3;
                for(l=0; l<N[3]; l++) A[i][j][k][l] = X;
            }
        }
    }
    return 0;
}

template <class ALLOCTYPE>
inline void Free1D(ALLOCTYPE* A)
{
    delete[] A;
}

template <class ALLOCTYPE>
inline void Free2D(ALLOCTYPE** A)
{
    delete[] A[0];
    delete[] A;
}

template <class ALLOCTYPE>
inline void Free3D(ALLOCTYPE*** A)
{
    delete[] A[0][0];
    delete[] A[0];
    delete[] A;
}

template <class ALLOCTYPE>
inline void Free4D(ALLOCTYPE**** A)
{
    delete[] A[0][0][0];
    delete[] A[0][0];
    delete[] A[0];
    delete[] A;
}
} // namespace
#endif
