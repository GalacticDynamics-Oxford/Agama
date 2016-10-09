#include "math_core.h"
#include "math_linalg.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <ctime>

double matrixNorm(const math::Matrix<double>& M)
{
    unsigned int size = M.rows() * M.cols();
    double sum = 0;
    for(unsigned int k=0; k<size; k++)
        sum += pow_2(M.data()[k]);
    return sqrt(sum/size);
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
    math::SpMatrix<double> spmat(NR, NR, spdata);
    math::Matrix<double> mat(spmat);
    {   // not used, just check that it compiles
        math::SpMatrix<float> spmat(NR, NR, spdata);
        math::SpMatrix<float> spmat1 = spmat;
        math::Matrix<float> mat(NR, NR, spdata);
        math::Matrix<float> mat1 = mat;
    }

    // init rhs
    std::vector<double> rhs(NR), sol, mul(NR);
    for(unsigned int i=0; i<NR; i++)
        rhs[i] = math::random();

    // test matrix multiplication: construct positive-definite matrix M M^T + diag(1)
    math::SpMatrix<double> spdmat(NR, NR);
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
        double norm = sqrt(math::blas_ddot(mul, mul) / NR);
        std::cout << "Sparse LU: " << ((std::clock()-tbegin)*1.0/CLOCKS_PER_SEC) <<
            " s, rmserr=" << norm << "\n";
        ok &= norm < 1e-10;
    }

    {   // dense LU
        tbegin=std::clock();
        math::LUDecomp lu(dmat), lu1=lu;
        sol = math::LUDecomp(lu1).solve(rhs);  // test copy constructor and assignment
        math::blas_dgemv(math::CblasNoTrans, 1, dmat, sol, 0, mul);
        math::blas_daxpy(-1, rhs, mul);
        double norm = sqrt(math::blas_ddot(mul, mul) / NR);
        std::cout << "Dense  LU: " << ((std::clock()-tbegin)*1.0/CLOCKS_PER_SEC) <<
            " s, rmserr=" << norm << "\n";
        ok &= norm < 1e-10;
    }

    {   // Cholesky
        clock_t tbegin=std::clock();
        math::CholeskyDecomp ch(dmat), ch1=ch;
        sol = math::CholeskyDecomp(ch1).solve(rhs);
        math::blas_dgemv(math::CblasNoTrans, 1, dmat, sol, 0, mul);
        math::blas_daxpy(-1, rhs, mul);
        double norm = sqrt(math::blas_ddot(mul, mul) / NR);
        std::cout << "Cholesky : " << ((std::clock()-tbegin)*1.0/CLOCKS_PER_SEC) <<
            " s, rmserr=" << norm << std::flush;
        ok &= norm < 1e-10;
        // check that L L^T = original matrix
        math::Matrix<double> L(ch.L());
        math::Matrix<double> M(NR, NR);
        math::blas_dgemm(math::CblasNoTrans, math::CblasTrans, 1, L, L, 0, M);
        math::blas_daxpy(-1, dmat, M);
        norm = matrixNorm(M);
        std::cout << ", |M - L L^T| = " << norm << "\n";
        ok &= norm < 1e-15;
    }
    {   // SVD
        clock_t tbegin=std::clock();
        math::SVDecomp sv(dmat), sv1=sv;
        sol = math::SVDecomp(sv1).solve(rhs);
        math::blas_dgemv(math::CblasNoTrans, 1, dmat, sol, 0, mul);
        math::blas_daxpy(-1, rhs, mul);
        double norm = sqrt(math::blas_ddot(mul, mul) / NR);
        std::cout << "Sing.val.: " << ((std::clock()-tbegin)*1.0/CLOCKS_PER_SEC) <<
            " s, rmserr=" << norm << std::flush;
        ok &= norm < 1e-10;
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
        norm = matrixNorm(S);
        std::cout << ", |M - U S V^T| = " << norm << ", cond.number = " <<
            vS.front() / vS.back() << "\n";
        ok &= norm < 1e-12;
    }

    if(ok)
        std::cout << "\033[1;32mALL TESTS PASSED\033[0m\n";
    else
        std::cout << "\033[1;31mSOME TESTS FAILED\033[0m\n";
    return 0;
}
