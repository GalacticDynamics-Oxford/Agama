/** \name   test_math_linalg.cpp
    \author Eugene Vasiliev
    \date   2015-2024

    Test the accuracy and performance of various linear algebra operations (matrix decompositions, linear equations, etc...)
*/
#include "math_core.h"
#include "math_random.h"
#include "math_linalg.h"
#include <iostream>
#include <cmath>
#include <ctime>

bool test(bool condition)
{
    std::cout << (condition ? "\n" : " \033[1;31m**\033[0m\n");
    return condition;
}

int main()
{
    bool ok=true;

    const unsigned int NR=600, NC=700, NV=10000;  // matrix size (Nrows<Ncols), # of nonzero values
    const unsigned int MIN_CLOCKS = CLOCKS_PER_SEC;
    std::vector<math::Triplet> spdata;
    // init matrix content
    for(unsigned int k=0; k<NV; k++) {
        unsigned int i = static_cast<unsigned int>(math::random()*NR);
        unsigned int j = static_cast<unsigned int>(math::random()*NC);
        double v = math::random();
        spdata.push_back(math::Triplet(i, j, v));
    }
    math::SparseMatrix<double> spmat(NR, NC, spdata);
    math::Matrix<double> mat(spmat);
    math::Matrix<double> tmat((math::TransposedMatrix<double>(spmat)));
    {   // not used, just check that it compiles
        math::SparseMatrix<float> spmat(NR, NC, spdata);
        math::SparseMatrix<float> spmat1 = spmat;
        math::Matrix<float> mat(NR, NC, spdata);
        math::Matrix<float> mat1 = mat;
    }

    // init rhs
    std::vector<double> rhs(NR), chs(NC), sol, mul(NR), cul(NC);
    for(unsigned int i=0; i<NR; i++)
        rhs[i] = math::random();
    for(unsigned int i=0; i<NC; i++)
        chs[i] = math::random();

    {   // band-matrix operations
        bool okband = true;
        int size=8, band=3;
        try{
            math::BandMatrix<double>(size, size);  // should fail: bandwidth must be smaller than size
            okband = false;
        }
        catch(...) {}  // clear the exception
        math::BandMatrix<double> bmat(size, band, NAN);
        for(int r=0; r<size; r++)
            for(int c=0; c<size; c++) {
                try{
                    bmat(r, c) = r*size + c;
                    if(r>c+band || c>r+band)  // element access should fail if |r-c|>band
                        okband = false;
                }
                catch(...) {
                    if(r<=c+band && c<=r+band)
                        okband = false;
                }
            }
        // iterate over nonzero elements
        int nelem=bmat.size();
        for(int k=0; k<=nelem; k++) {
            try{
                size_t r, c;
                double val = bmat.elem(k, r, c);  // should fail if i>=nelem
                if(k>=nelem)
                    okband = false;
                if(!(val == bmat(r, c)))
                    okband = false;
            }
            catch(...) {
                if(k<nelem)
                    okband = false;
            }
        }
        // convert to other matrix types
        std::vector<math::Triplet> values = bmat.values();
        math::Matrix<double> dmat(bmat);
        math::SparseMatrix<double> smat(bmat);
        if((int)values.size() != nelem)
            okband = false;
        for(int k=0; k<nelem; k++) {
            size_t i = values[k].i, j = values[k].j;
            double v = values[k].v;
            double b = bmat.at(i, j);
            double d = dmat.at(i, j);
            double s = smat.at(i, j);
            if(!(b==v && d==v && s==v))
                okband = false;
        }
        // matrix/vector multiplication
        std::vector<double> vec(rhs.begin(), rhs.begin()+size);  // random data
        std::vector<double> vb(vec), vd(vec), vs(vec), tb(vec), td(vec), ts(vec);
        math::blas_dgemv(math::CblasNoTrans, 2., bmat, vec, 3., vb);
        math::blas_dgemv(math::CblasNoTrans, 2., dmat, vec, 3., vd);
        math::blas_dgemv(math::CblasNoTrans, 2., smat, vec, 3., vs);
        math::blas_dgemv(math::CblasTrans,  -3., bmat, vec, 4., tb);
        math::blas_dgemv(math::CblasTrans,  -3., dmat, vec, 4., td);
        math::blas_dgemv(math::CblasTrans,  -3., smat, vec, 4., ts);
        for(int k=0; k<size; k++) {
            if( math::fcmp(vb[k], vd[k]) || math::fcmp(vb[k], vs[k]) ||
                math::fcmp(tb[k], td[k]) || math::fcmp(tb[k], ts[k]) )
                okband = false;
        }
        if(!okband) {
            std::cout << "Band matrix operations failed";
            ok &= test(false);
        }
    }

    {   // check size compatibility
        bool okdim = true;
        sol.resize(NC);
        math::Matrix<double> product(NR, NR);
        math::SparseMatrix<double> spproduct(NR, NR);
        try{
            // should fail: matrices are not aligned
            math::blas_dgemm(math::CblasNoTrans, math::CblasNoTrans, 1, mat, mat, 0, product);
            std::cout << "DD ";
            okdim = false;
        }
        catch(...) {}
        try{
            // should fail: matrices are not aligned
            math::blas_dgemm(math::CblasNoTrans, math::CblasNoTrans, 1, spmat, spmat, 0, spproduct);
            std::cout << "SS ";
            okdim = false;
        }
        catch(...) {}
        try{
            // should fail: matrices are aligned, but the result matrix has wrong size
            math::blas_dgemm(math::CblasTrans, math::CblasNoTrans, 1, spmat, spmat, 0, spproduct);
            std::cout << "StS ";
            okdim = false;
        }
        catch(...) {}
        if(!okdim) {
            std::cout << "matrix multiplication did not fail where it should.";
            ok &= test(false);
        }
    }

    // test matrix multiplication: construct positive-definite matrix M M^T + diag(1)
    math::SparseMatrix<double> spdmat(NR, NR);
    clock_t tbegin=std::clock();
    int niter = 0;
    for(; !niter || std::clock()-tbegin < MIN_CLOCKS; niter++)
        math::blas_dgemm(math::CblasNoTrans, math::CblasTrans, 1, spmat, spmat, 0, spdmat);
    std::cout << "Sparse MM: " << ((std::clock()-tbegin)*1.0/CLOCKS_PER_SEC/niter) << " s, " <<
    spdmat.size() << " elements\n";
    math::Matrix<double> dmat(NR, NR);
    for(unsigned int k=0; k<NR; k++)
        dmat(k, k) = 1.;
    tbegin=std::clock();
    for(niter=0; !niter || std::clock()-tbegin < MIN_CLOCKS; niter++)
        math::blas_dgemm(math::CblasNoTrans, math::CblasTrans, 1, mat, mat, 0, dmat);
    std::cout << "Dense  MM: " << ((std::clock()-tbegin)*1.0/CLOCKS_PER_SEC/niter) << " s\n";

    {   // sparse LU
        tbegin=std::clock();
        for(niter=0; !niter || std::clock()-tbegin < MIN_CLOCKS; niter++)
            sol = math::LUDecomp(spdmat).solve(rhs);
        std::cout << "Sparse LU: " << ((std::clock()-tbegin)*1.0/CLOCKS_PER_SEC/niter);
        math::blas_dgemv(math::CblasNoTrans, 1, spdmat, sol, 0, mul);
        math::blas_daxpy(-1, rhs, mul);
        double norm = sqrt(math::blas_dnrm2(mul) / NR);
        std::cout << " s, rmserr=" << norm;
        bool sizeok = true;
        try{
            math::LUDecomp lu(spmat);
            std::cout << " Nonsquare matrix incorrectly accepted";
            sizeok = false;
        }
        catch(...) {}
        ok &= test(sizeok && norm < 1e-10);
    }

    {   // dense LU
        tbegin=std::clock();
        for(niter=0; !niter || std::clock()-tbegin < MIN_CLOCKS; niter++) {
            math::LUDecomp lu(dmat), lu1=lu;
            sol = math::LUDecomp(lu1).solve(rhs);  // test copy constructor and assignment
        }
        std::cout << "Dense  LU: " << ((std::clock()-tbegin)*1.0/CLOCKS_PER_SEC/niter);
        math::blas_dgemv(math::CblasNoTrans, 1, dmat, sol, 0, mul);
        math::blas_daxpy(-1, rhs, mul);
        double norm = sqrt(math::blas_dnrm2(mul) / NR);
        std::cout << " s, rmserr=" << norm;
        bool sizeok = true;
        try{
            math::LUDecomp lu(mat);
            std::cout << " Nonsquare matrix incorrectly accepted";
            sizeok = false;
        }
        catch(...) {}
        ok &= test(sizeok && norm < 1e-10);
    }

    {   // Cholesky
        clock_t tbegin=std::clock();
        math::Matrix<double> L;
        for(niter=0; !niter || std::clock()-tbegin < MIN_CLOCKS; niter++) {
            math::CholeskyDecomp ch(dmat), ch1=ch;
            sol = math::CholeskyDecomp(ch1).solve(rhs);
            L = ch.L();
        }
        std::cout << "Cholesky : " << ((std::clock()-tbegin)*1.0/CLOCKS_PER_SEC/niter);
        math::blas_dgemv(math::CblasNoTrans, 1, dmat, sol, 0, mul);
        math::blas_daxpy(-1, rhs, mul);
        double norm = sqrt(math::blas_dnrm2(mul) / NR);
        std::cout << " s, rmserr=" << norm << std::flush;
        // check that L L^T = original matrix
        math::Matrix<double> M(NR, NR);
        math::blas_dgemm(math::CblasNoTrans, math::CblasTrans, 1, L, L, 0, M);
        math::blas_daxpy(-1, dmat, M);
        double norm2 = sqrt(math::blas_dnrm2(M)) / NR;
        std::cout << ", |M - L L^T| = " << norm2;
        bool sizeok = true;
        try{
            math::CholeskyDecomp ch(mat);
            std::cout << " Nonsquare matrix incorrectly accepted";
            sizeok = false;
        }
        catch(...) {}
        ok &= test(sizeok && norm < 1e-10 && norm2 < 1e-15);
    }

    {   // QR, two ways: "mat" is wide (rows < columns), "tmat" is tall (opposite)
        clock_t tbegin=std::clock();
        math::Matrix<double> Q, R;
        for(niter=0; !niter || std::clock()-tbegin < MIN_CLOCKS; niter++) {
            // underdetermined system: more variables (NC) than equations (NR)
            math::QRDecomp qr(mat), qr1=qr;
            sol = math::QRDecomp(qr1).solve(rhs);
            qr.QR(Q, R);
        }
        std::cout << "QR       : " << ((std::clock()-tbegin)*1.0/CLOCKS_PER_SEC/niter);
        math::blas_dgemv(math::CblasNoTrans, 1, mat, sol, 0, mul);
        math::blas_daxpy(-1, rhs, mul);
        double normu = sqrt(math::blas_dnrm2(mul) / NR);
        std::cout << " s, rmserr=" << normu << std::flush;
        // check that Q R = original matrix
        math::Matrix<double> M(NR, NC);
        math::blas_dgemm(math::CblasNoTrans, math::CblasNoTrans, 1, Q, R, 0, M);
        math::blas_daxpy(-1, mat, M);
        double norm2u = sqrt(math::blas_dnrm2(M)) / NR;
        std::cout << ", |M - Q R| = " << norm2u;
        ok &= Q.rows() == Q.cols() && R.rows() == mat.rows() && R.cols() == mat.cols();
        
        // overdetermined system: more equations (NC) than variables (NR)
        math::QRDecomp qr(tmat);
        sol = qr.solve(chs);
        qr.QR(Q, R);
        math::blas_dgemv(math::CblasNoTrans, 1, tmat, sol, 0, cul);
        math::blas_daxpy(-1, chs, cul);
        double normo = sqrt(math::blas_dnrm2(cul) / NC);  // expected to be macroscopically large
        // check that Q R = original matrix
        M = math::Matrix<double>(NC, NR);
        math::blas_dgemm(math::CblasNoTrans, math::CblasNoTrans, 1, Q, R, 0, M);
        math::blas_daxpy(-1, tmat, M);
        double norm2o = sqrt(math::blas_dnrm2(M)) / NR;
        std::cout << " and " << norm2o;
        ok &= Q.rows() == Q.cols() && R.rows() == tmat.rows() && R.cols() == tmat.cols();
        ok &= test(normu < 1e-12 && norm2u < 1e-15 && normo > 0.01 && norm2o < 1e-15);
    }

    {   // SVD, two ways: "tmat" is tall (rows > columns), "dmat" is square
        clock_t tbegin=std::clock();
        math::Matrix<double> U, V;
        std::vector <double> vS;
        for(niter=0; !niter || std::clock()-tbegin < MIN_CLOCKS; niter++) {
            // overdetermined system: more equations (NC) than variables (NR)
            math::SVDecomp sv(tmat), sv1=sv;
            sol = math::SVDecomp(sv1).solve(chs);
            U = sv.U();
            V = sv.V();
            vS= sv.S();
        }
        std::cout << "Sing.val.: " << ((std::clock()-tbegin)*1.0/CLOCKS_PER_SEC/niter);
        math::blas_dgemv(math::CblasNoTrans, 1, tmat, sol, 0, cul);
        math::blas_daxpy(-1, chs, cul);
        double normo = sqrt(math::blas_dnrm2(cul) / NC);
        std::cout << " s, rmserr=" << normo << std::flush;
        // check that U S V^T = original matrix
        math::Matrix<double> S(NR, NR, 0);
        for(unsigned int k=0; k<NR; k++)  S(k, k) = vS[k];
        math::Matrix<double> tmp(NC, NR);
        math::blas_dgemm(math::CblasNoTrans, math::CblasNoTrans, 1, U, S, 0, tmp);
        S = math::Matrix<double>(NC, NR);
        math::blas_dgemm(math::CblasNoTrans, math::CblasTrans, 1, tmp, V, 0, S);
        math::blas_daxpy(-1, tmat, S);
        double norm2o = sqrt(math::blas_dnrm2(S)) / NR;
        std::cout << ", |M - U S V^T| = " << norm2o << '\n';

        // square matrix
        math::SVDecomp sv(dmat);
        sol = math::SVDecomp(sv).solve(rhs);
        U = sv.U();
        V = sv.V();
        vS= sv.S();
        std::cout << "SV, again: " << ((std::clock()-tbegin)*1.0/CLOCKS_PER_SEC/niter);
        math::blas_dgemv(math::CblasNoTrans, 1, dmat, sol, 0, mul);
        math::blas_daxpy(-1, rhs, mul);
        double norms = sqrt(math::blas_dnrm2(mul) / NR);
        std::cout << " s, rmserr=" << norms << std::flush;
        // check that U S V^T = original matrix
        S = math::Matrix<double>(NR, NR, 0);
        for(unsigned int k=0; k<NR; k++)  S(k, k) = vS[k];
        tmp = math::Matrix<double>(NR, NR);
        math::blas_dgemm(math::CblasNoTrans, math::CblasNoTrans, 1, U, S, 0, tmp);
        math::blas_dgemm(math::CblasNoTrans, math::CblasTrans, 1, tmp, V, 0, S);
        math::blas_daxpy(-1, dmat, S);
        double norm2s = sqrt(math::blas_dnrm2(S)) / NR;
        std::cout << ", |M - U S V^T| = " << norm2s;
        ok &= test(norms < 1e-13 && norm2s < 1e-13 && normo > 0.01 && norm2o < 1e-13);
    }

    {   // tridiagonal systems
        math::BandMatrix<double> mat(NR, 1, NAN);
        for(unsigned int i=0; i<NR; i++) {
            if(i>0)    mat(i, i-1) = -0.3;
            if(true)   mat(i, i  ) =  1.1;
            if(i<NR-1) mat(i, i+1) = -0.4;
        }
        std::vector<double> sol;
        clock_t tbegin=std::clock();
        for(niter=0; !niter || std::clock()-tbegin < MIN_CLOCKS; niter++)
            sol = math::solveBand(mat, rhs);
        std::vector<double> prod(rhs);
        math::blas_dgemv(math::CblasNoTrans, 1., mat, sol, -1., prod);
        double norm = sqrt(math::blas_dnrm2(prod) / NR);
        std::cout << "Tridiag. : " << ((std::clock()-tbegin)*1.0/CLOCKS_PER_SEC/niter) <<
            " s, rmserr=" << norm;
        ok &= test(norm < 1e-15);
        /// same using a generic sparse LU solver
        math::SparseMatrix<double> spmat(mat);
        tbegin=std::clock();
        for(niter=0; !niter || std::clock()-tbegin < MIN_CLOCKS; niter++)
            sol = math::LUDecomp(spmat).solve(rhs);
        math::blas_dgemv(math::CblasNoTrans, 1., spmat, sol, 0., prod);
        math::blas_daxpy(-1, rhs, prod);
        norm = sqrt(math::blas_dnrm2(prod) / NR);
        std::cout << "Sparse LU: " << ((std::clock()-tbegin)*1.0/CLOCKS_PER_SEC/niter) <<
            " s, rmserr=" << norm;
        ok &= test(norm < 1e-15);
    }

    {   // more general band-diagonal systems
        math::BandMatrix<double> mat(NR, 3, NAN);
        for(unsigned int i=0; i<NR; i++) {
            if(i>2)    mat(i, i-3) = -0.05;
            if(i>1)    mat(i, i-2) =  0.15;
            if(i>0)    mat(i, i-1) = -0.3;
            if(true)   mat(i, i  ) =  1.2;  // diag
            if(i<NR-1) mat(i, i+1) = -0.4;
            if(i<NR-2) mat(i, i+2) =  0.2;
            if(i<NR-3) mat(i, i+3) = -0.1;
        }
        std::vector<double> sol;
        clock_t tbegin=std::clock();
        for(niter=0; !niter || std::clock()-tbegin < MIN_CLOCKS; niter++)
            sol = math::solveBand(mat, rhs);
        std::vector<double> prod(NR);
        math::blas_dgemv(math::CblasNoTrans, 1., mat, sol, 0., prod);
        math::blas_daxpy(-1, rhs, prod);
        double norm = sqrt(math::blas_dnrm2(prod) / NR);
        std::cout << "Band-diag: " << ((std::clock()-tbegin)*1.0/CLOCKS_PER_SEC/niter) <<
            " s, rmserr=" << norm;
        ok &= test(norm < 1e-15);
        /// same using generic sparse LU
        tbegin=std::clock();
        math::SparseMatrix<double> spmat(mat);
        for(niter=0; !niter || std::clock()-tbegin < MIN_CLOCKS; niter++)
            sol = math::LUDecomp(spmat).solve(rhs);
        prod = rhs;
        math::blas_dgemv(math::CblasNoTrans, 1., spmat, sol, -1., prod);
        norm = sqrt(math::blas_dnrm2(prod) / NR);
        std::cout << "Sparse LU: " << ((std::clock()-tbegin)*1.0/CLOCKS_PER_SEC/niter) <<
            " s, rmserr=" << norm;
        ok &= test(norm < 1e-15);
    }

    if(ok)
        std::cout << "\033[1;32mALL TESTS PASSED\033[0m\n";
    else
        std::cout << "\033[1;31mSOME TESTS FAILED\033[0m\n";
    return 0;
}
