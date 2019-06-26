#include "KokkosKernels_config.h"


#if defined( KOKKOSKERNELS_ENABLE_TPL_BLAS ) && defined( KOKKOSKERNELS_ENABLE_TPL_LAPACK )

#include "KokkosBlas_Host_tpl.hpp"
#include "KokkosLapack_Host_tpl.hpp"
#include <lapacke.h>
#include <cblas.h>

namespace KokkosBlas {
    namespace Impl {
    
        //float 

        template<>
        void
        HostLapack<float>::geqp3(bool matrix_layout, 
                                 int m, int n,
                                 float* a, int lda, 
                                 int* jpvt, 
                                 float* tau){
            if (matrix_layout){
                LAPACKE_sgeqp3(LAPACK_ROW_MAJOR, m, n, a, lda, jpvt, tau);
            }
            else{
                LAPACKE_sgeqp3(LAPACK_COL_MAJOR, m, n, a, lda, jpvt, tau);
            }
        }

        template<>
        void 
        HostLapack<float>::unmqr(bool matrix_layout,
                                  char side, char trans,
                                  int m, int n, int k,
                                  const float* a, int lda,
                                  const float* tau, 
                                  float* c, int ldc){
            if(matrix_layout) {
                LAPACKE_sormqr(LAPACK_ROW_MAJOR, side, trans, m, n, k, a, lda, tau, c, ldc);
            }
            else {
                LAPACKE_sormqr(LAPACK_COL_MAJOR, side, trans, m, n, k, a, lda, tau, c, ldc);
            }
        }

        template<>
        void 
        HostLapack<float>::ormqr(bool matrix_layout,
                                  char side, char trans,
                                  int m, int n, int k,
                                  const float* a, int lda,
                                  const float* tau, 
                                  float* c, int ldc){
            if(matrix_layout) {
                LAPACKE_sormqr(LAPACK_ROW_MAJOR, side, trans, m, n, k, a, lda, tau, c, ldc);
            }
            else {
                LAPACKE_sormqr(LAPACK_COL_MAJOR, side, trans, m, n, k, a, lda, tau, c, ldc);
            }
        }
        
        template<>
        void
        HostLapack<float>::potrf(bool matrix_layout, 
                                 char uplo, int n, 
                                 float* a, int lda){
            if(matrix_layout) {
                LAPACKE_spotrf(LAPACK_ROW_MAJOR, uplo, n, a, lda);
            }
            else {
                LAPACKE_spotrf(LAPACK_COL_MAJOR, uplo, n, a, lda);
            }
        }


        //double
         
        template<>
        void
        HostLapack<double>::geqp3(bool matrix_layout, 
                                 int m, int n,
                                 double* a, int lda, 
                                 int* jpvt, 
                                 double* tau){
            if (matrix_layout){
                LAPACKE_dgeqp3(LAPACK_ROW_MAJOR, m, n, a, lda, jpvt, tau);
            }
            else{
                LAPACKE_dgeqp3(LAPACK_COL_MAJOR, m, n, a, lda, jpvt, tau);
            }
        }

        template<>
        void 
        HostLapack<double>::unmqr(bool matrix_layout,
                                  char side, char trans,
                                  int m, int n, int k,
                                  const double* a, int lda,
                                  const double* tau, 
                                  double* c, int ldc){
            if(matrix_layout) {
                LAPACKE_dormqr(LAPACK_ROW_MAJOR, side, trans, m, n, k, a, lda, tau, c, ldc);
            }
            else {
                LAPACKE_dormqr(LAPACK_COL_MAJOR, side, trans, m, n, k, a, lda, tau, c, ldc);
            }
        }

        template<>
        void 
        HostLapack<double>::ormqr(bool matrix_layout,
                                  char side, char trans,
                                  int m, int n, int k,
                                  const double* a, int lda,
                                  const double* tau, 
                                  double* c, int ldc){
            if(matrix_layout) {
                LAPACKE_dormqr(LAPACK_ROW_MAJOR, side, trans, m, n, k, a, lda, tau, c, ldc);
            }
            else {
                LAPACKE_dormqr(LAPACK_ROW_MAJOR, side, trans, m, n, k, a, lda, tau, c, ldc);
            }
        }


        template<>
        void
        HostLapack<double>::potrf(bool matrix_layout, 
                                 char uplo, int n, 
                                 double* a, int lda){
            if(matrix_layout) {
                LAPACKE_dpotrf(LAPACK_ROW_MAJOR, uplo, n, a, lda);
            }
            else {
                LAPACKE_dpotrf(LAPACK_COL_MAJOR, uplo, n, a, lda);
            }
        }


        //std::complex<float>


        template<>
        void
        HostLapack<std::complex<float>>::geqp3(bool matrix_layout, 
                                 int m, int n,
                                 std::complex<float>* a, int lda, 
                                 int* jpvt, 
                                 std::complex<float>* tau){
            if (matrix_layout){
                LAPACKE_cgeqp3(LAPACK_ROW_MAJOR, m, n, 
                        reinterpret_cast < __complex__ float*>(a), lda, jpvt, 
                        reinterpret_cast < __complex__ float*>(tau));
            }
            else{
                LAPACKE_cgeqp3(LAPACK_COL_MAJOR, m, n, 
                        reinterpret_cast < __complex__ float*>(a), lda, jpvt, 
                        reinterpret_cast < __complex__ float*>(tau));
            }
        }

        template<>
        void 
        HostLapack<std::complex<float>>::unmqr(bool matrix_layout,
                                  char side, char trans,
                                  int m, int n, int k,
                                  const std::complex<float>* a, int lda,
                                  const std::complex<float>* tau, 
                                  std::complex<float>* c, int ldc){
            if(matrix_layout) {
                LAPACKE_cunmqr(LAPACK_ROW_MAJOR, side, trans, m, n, k, 
                        reinterpret_cast <const __complex__ float*>(a), lda, 
                        reinterpret_cast <const __complex__ float*>(tau), 
                        reinterpret_cast < __complex__ float*>(c), ldc);
            }
            else {
                LAPACKE_cunmqr(LAPACK_COL_MAJOR, side, trans, m, n, k, 
                        reinterpret_cast <const __complex__ float*>(a), lda, 
                        reinterpret_cast <const __complex__ float*>(tau), 
                        reinterpret_cast < __complex__ float*>(c), ldc);
            }
        }

        template<>
        void 
        HostLapack<std::complex<float>>::ormqr(bool matrix_layout,
                                  char side, char trans,
                                  int m, int n, int k,
                                  const std::complex<float>* a, int lda,
                                  const std::complex<float>* tau, 
                                  std::complex<float>* c, int ldc){
            if(matrix_layout) {
                LAPACKE_cunmqr(LAPACK_ROW_MAJOR, side, trans, m, n, k, 
                        reinterpret_cast <const __complex__ float*>(a), lda, 
                        reinterpret_cast <const __complex__ float*>(tau), 
                        reinterpret_cast < __complex__ float*>(c), ldc);
            }
            else {
                LAPACKE_cunmqr(LAPACK_COL_MAJOR, side, trans, m, n, k, 
                        reinterpret_cast <const __complex__ float*>(a), lda, 
                        reinterpret_cast <const __complex__ float*>(tau), 
                        reinterpret_cast < __complex__ float*>(c), ldc);
            }
        }


        template<>
        void
        HostLapack<std::complex<float>>::potrf(bool matrix_layout, 
                                 char uplo, int n, 
                                 std::complex<float>* a, int lda){
            if(matrix_layout) {
                LAPACKE_cpotrf(LAPACK_ROW_MAJOR, uplo, n, 
                        reinterpret_cast<__complex__ float*>( a ), lda);
            }
            else {
                LAPACKE_cpotrf(LAPACK_COL_MAJOR, uplo, n, 
                        reinterpret_cast<__complex__ float*>( a ), lda);
            }
        }


        //std::complex<double>

        template<>
        void
        HostLapack<std::complex<double>>::geqp3(bool matrix_layout, 
                                 int m, int n,
                                 std::complex<double>* a, int lda, 
                                 int* jpvt, 
                                 std::complex<double>* tau){
            if (matrix_layout){
                LAPACKE_zgeqp3(LAPACK_ROW_MAJOR, m, n, 
                        reinterpret_cast < __complex__ double*>(a), lda, jpvt, 
                        reinterpret_cast < __complex__ double*>(tau));
            }
            else{
                LAPACKE_zgeqp3(LAPACK_COL_MAJOR, m, n, 
                        reinterpret_cast < __complex__ double*>(a), lda, jpvt, 
                        reinterpret_cast < __complex__ double*>(tau));
            }
        }

        template<>
        void 
        HostLapack<std::complex<double>>::unmqr(bool matrix_layout,
                                  char side, char trans,
                                  int m, int n, int k,
                                  const std::complex<double>* a, int lda,
                                  const std::complex<double>* tau, 
                                  std::complex<double>* c, int ldc){
            if(matrix_layout) {
                LAPACKE_zunmqr(LAPACK_ROW_MAJOR, side, trans, m, n, k, 
                        reinterpret_cast <const __complex__ double*>(a), lda, 
                        reinterpret_cast <const __complex__ double*>(tau), 
                        reinterpret_cast <__complex__ double*>(c), ldc);
            }
            else {
                LAPACKE_zunmqr(LAPACK_COL_MAJOR, side, trans, m, n, k, 
                        reinterpret_cast <const __complex__ double*>(a), lda, 
                        reinterpret_cast <const __complex__ double*>(tau), 
                        reinterpret_cast <__complex__ double*>(c), ldc);
            }
        }

        template<>
        void 
        HostLapack<std::complex<double>>::ormqr(bool matrix_layout,
                                  char side, char trans,
                                  int m, int n, int k,
                                  const std::complex<double>* a, int lda,
                                  const std::complex<double>* tau, 
                                  std::complex<double>* c, int ldc){
            if(matrix_layout) {
                LAPACKE_zunmqr(LAPACK_ROW_MAJOR, side, trans, m, n, k, 
                        reinterpret_cast <const __complex__ double*>(a), lda, 
                        reinterpret_cast <const __complex__ double*>(tau),
                        reinterpret_cast <__complex__ double*>(c), ldc);
            }
            else {
                LAPACKE_zunmqr(LAPACK_COL_MAJOR, side, trans, m, n, k, 
                        reinterpret_cast <const __complex__ double*>(a), lda, 
                        reinterpret_cast <const __complex__ double*>(tau), 
                        reinterpret_cast <__complex__ double*>(c), ldc);
            }
        }

    
        template<>
        void
        HostLapack<std::complex<double>>::potrf(bool matrix_layout, 
                                 char uplo, int n, 
                                 std::complex<double>* a, int lda){
            if(matrix_layout) {
                LAPACKE_cpotrf(LAPACK_ROW_MAJOR, uplo, n, 
                        reinterpret_cast<__complex__ double*>( a ), lda);
            }
            else {
                LAPACKE_cpotrf(LAPACK_COL_MAJOR, uplo, n, 
                        reinterpret_cast<__complex__ double*>( a ), lda);
            }
        }

    } //namespace Impl
} //namespace KokkosBlas

#endif //ENABLE BLAS/LAPACK 
