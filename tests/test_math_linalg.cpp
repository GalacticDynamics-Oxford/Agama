#include "math_core.h"
#include "math_linalg.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <ctime>

bool test(bool condition)
{
    std::cout << (condition ? "\n" : " \033[1;31m**\033[0m\n");
    return condition;
}

int main()
{
    std::cout << std::setprecision(10);
    bool ok=true;

    const unsigned int NR=500, NV=5000;  // matrix size, # of nonzero values
    std::vector<math::Triplet> spdata;
    // init matrix content
    for(unsigned int k=0; k<NV; k++) {
        unsigned int i = static_cast<unsigned int>(math::random()*NR);
        unsigned int j = static_cast<unsigned int>(math::random()*NR);
        double v = math::random();
        spdata.push_back(math::Triplet(i, j, v));
    }
    math::SparseMatrix<double> spmat(NR, NR, spdata);
    math::Matrix<double> mat(spmat);
    {   // not used, just check that it compiles
        math::SparseMatrix<float> spmat(NR, NR, spdata);
        math::SparseMatrix<float> spmat1 = spmat;
        math::Matrix<float> mat(NR, NR, spdata);
        math::Matrix<float> mat1 = mat;
    }

    // init rhs
    std::vector<double> rhs(NR), sol, mul(NR);
    for(unsigned int i=0; i<NR; i++)
        rhs[i] = math::random();

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
            if( fabs(vb[k]-vd[k])>1e-15 || fabs(vb[k]-vs[k])>1e-15 ||
                fabs(tb[k]-td[k])>1e-15 || fabs(tb[k]-ts[k])>1e-15 )
                okband = false;
        }
        if(!okband) {
            std::cout << "Band matrix operations failed";
            ok &= test(false);
        }
    }

    // test matrix multiplication: construct positive-definite matrix M M^T + diag(1)
    math::SparseMatrix<double> spdmat(NR, NR);
    clock_t tbegin=std::clock();
    math::blas_dgemm(math::CblasNoTrans, math::CblasTrans, 1, spmat, spmat, 0, spdmat);
    std::cout << "Sparse MM: " << ((std::clock()-tbegin)*1.0/CLOCKS_PER_SEC) << " s, " <<
    spdmat.size() << " elements\n";
    math::Matrix<double> dmat(NR, NR);
    for(unsigned int k=0; k<NR; k++)
        dmat(k, k) = 1.;
    tbegin=std::clock();
    math::blas_dgemm(math::CblasNoTrans, math::CblasTrans, 1, mat, mat, 0, dmat);
    std::cout << "Dense  MM: " << ((std::clock()-tbegin)*1.0/CLOCKS_PER_SEC) << " s\n";

    {   // sparse LU
        tbegin=std::clock();
        sol = math::LUDecomp(spdmat).solve(rhs);
        math::blas_dgemv(math::CblasNoTrans, 1, spdmat, sol, 0, mul);
        math::blas_daxpy(-1, rhs, mul);
        double norm = sqrt(math::blas_dnrm2(mul) / NR);
        std::cout << "Sparse LU: " << ((std::clock()-tbegin)*1.0/CLOCKS_PER_SEC) <<
            " s, rmserr=" << norm;
        ok &= test(norm < 1e-10);
    }

    {   // dense LU
        tbegin=std::clock();
        math::LUDecomp lu(dmat), lu1=lu;
        sol = math::LUDecomp(lu1).solve(rhs);  // test copy constructor and assignment
        math::blas_dgemv(math::CblasNoTrans, 1, dmat, sol, 0, mul);
        math::blas_daxpy(-1, rhs, mul);
        double norm = sqrt(math::blas_dnrm2(mul) / NR);
        std::cout << "Dense  LU: " << ((std::clock()-tbegin)*1.0/CLOCKS_PER_SEC) <<
            " s, rmserr=" << norm;
        ok &= test(norm < 1e-10);
    }

    {   // Cholesky
        clock_t tbegin=std::clock();
        math::CholeskyDecomp ch(dmat), ch1=ch;
        sol = math::CholeskyDecomp(ch1).solve(rhs);
        math::blas_dgemv(math::CblasNoTrans, 1, dmat, sol, 0, mul);
        math::blas_daxpy(-1, rhs, mul);
        double norm = sqrt(math::blas_dnrm2(mul) / NR);
        std::cout << "Cholesky : " << ((std::clock()-tbegin)*1.0/CLOCKS_PER_SEC) <<
            " s, rmserr=" << norm << std::flush;
        // check that L L^T = original matrix
        math::Matrix<double> L(ch.L());
        math::Matrix<double> M(NR, NR);
        math::blas_dgemm(math::CblasNoTrans, math::CblasTrans, 1, L, L, 0, M);
        math::blas_daxpy(-1, dmat, M);
        double norm2 = sqrt(math::blas_dnrm2(M)) / NR;
        std::cout << ", |M - L L^T| = " << norm2;
        ok &= test(norm < 1e-10 && norm2 < 1e-15);
    }

    {   // SVD
        clock_t tbegin=std::clock();
        math::SVDecomp sv(dmat), sv1=sv;
        sol = math::SVDecomp(sv1).solve(rhs);
        math::blas_dgemv(math::CblasNoTrans, 1, dmat, sol, 0, mul);
        math::blas_daxpy(-1, rhs, mul);
        double norm = sqrt(math::blas_dnrm2(mul) / NR);
        std::cout << "Sing.val.: " << ((std::clock()-tbegin)*1.0/CLOCKS_PER_SEC) <<
            " s, rmserr=" << norm << std::flush;
        // check that U S V^T = original matrix
        math::Matrix<double> U(sv.U());
        math::Matrix<double> V(sv.V());
        std::vector<double> vS(sv.S());
        math::Matrix<double> S(NR, NR, 0);
        for(unsigned int k=0; k<NR; k++)  S(k, k) = vS[k];
        math::Matrix<double> tmp(NR, NR);
        math::blas_dgemm(math::CblasNoTrans, math::CblasNoTrans, 1, U, S, 0, tmp);
        math::blas_dgemm(math::CblasNoTrans, math::CblasTrans, 1, tmp, V, 0, S);
        math::blas_daxpy(-1, dmat, S);
        double norm2 = sqrt(math::blas_dnrm2(S)) / NR;
        std::cout << ", |M - U S V^T| = " << norm2 << ", cond.number = " <<
            vS.front() / vS.back();
        ok &= test(norm < 1e-10 && norm2 < 1e-12);
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
        for(int iter=0; iter<10000; iter++) {
            sol = math::solveBand(mat, rhs);
        }
        std::vector<double> prod(rhs);
        math::blas_dgemv(math::CblasNoTrans, 1., mat, sol, -1., prod);
        double norm = sqrt(math::blas_dnrm2(prod) / NR);
        std::cout << "Tridiag. : " << ((std::clock()-tbegin)*1.0/CLOCKS_PER_SEC/10000) <<
            " s, rmserr=" << norm;
        ok &= test(norm < 1e-15);
        /// same using a generic sparse LU solver
        math::SparseMatrix<double> spmat(mat);
        tbegin=std::clock();
        for(int iter=0; iter<100; iter++) {
            sol = math::LUDecomp(spmat).solve(rhs);
        }
        math::blas_dgemv(math::CblasNoTrans, 1., spmat, sol, 0., prod);
        math::blas_daxpy(-1, rhs, prod);
        norm = sqrt(math::blas_dnrm2(prod) / NR);
        std::cout << "SparseLU : " << ((std::clock()-tbegin)*1.0/CLOCKS_PER_SEC/100) <<
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
        for(int iter=0; iter<10000; iter++) {
            sol = math::solveBand(mat, rhs);
        }
        std::vector<double> prod(NR);
        math::blas_dgemv(math::CblasNoTrans, 1., mat, sol, 0., prod);
        math::blas_daxpy(-1, rhs, prod);
        double norm = sqrt(math::blas_dnrm2(prod) / NR);
        std::cout << "Band-diag: " << ((std::clock()-tbegin)*1.0/CLOCKS_PER_SEC/10000) <<
            " s, rmserr=" << norm;
        ok &= test(norm < 1e-15);
        /// same using generic sparse LU
        tbegin=std::clock();
        math::SparseMatrix<double> spmat(mat);
        for(int iter=0; iter<100; iter++) {
            sol = math::LUDecomp(spmat).solve(rhs);
        }
        prod = rhs;
        math::blas_dgemv(math::CblasNoTrans, 1., spmat, sol, -1., prod);
        norm = sqrt(math::blas_dnrm2(prod) / NR);
        std::cout << "SparseLU : " << ((std::clock()-tbegin)*1.0/CLOCKS_PER_SEC/100) <<
            " s, rmserr=" << norm;
        ok &= test(norm < 1e-15);
    }

    if(ok)
        std::cout << "\033[1;32mALL TESTS PASSED\033[0m\n";
    else
        std::cout << "\033[1;31mSOME TESTS FAILED\033[0m\n";
    return 0;
}
