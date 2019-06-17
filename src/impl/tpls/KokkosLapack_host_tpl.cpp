#include "KokkosKernels_config.h"
#include "KokkosBlas_Host_tpl.hpp"


#if defined (KOKKOSKERNELS_ENABLE_TPL_BLAS)

namespace KokkosBlas {
    namespace Impl {

        //float
        template<>
        void
        HostLapack<float>::geqp3(int matrix_layout, 
                                 lapack_int m, lapack_int n,
                                 float* a, lapack_int lda, 
                                 lapack_int* jpvt, 
                                 float* tau){
            LAPACKE_sgeqp3(&matrix_layout, m, n, a, lda, jpvt, tau);
        }

        //double
        
        template<>
        void
        HostLapack<double>::geqp3(int matrix_layout, 
                                 lapack_int m, lapack_int n,
                                 double* a, lapack_int lda, 
                                 lapack_int* jpvt, 
                                 double* tau){
            LAPACKE_dgeqp3(&matrix_layout, m, n, a, lda, jpvt, tau);
        }

        //std::complex<float>

        template<>
        void
        HostLapack<std::complex<float>>::geqp3(int matrix_layout, 
                                 lapack_int m, lapack_int n,
                                 std::complex<float>* a, lapack_int lda, 
                                 lapack_int* jpvt, 
                                 std::complex<float>* tau){
            LAPACKE_cgeqp3(&matrix_layout, m, n, a, lda, jpvt, tau);
        }


        //std::complex<double>

        template<>
        void
        HostLapack<std::complex<double>>::geqp3(int matrix_layout, 
                                 lapack_int m, lapack_int n,
                                 std::complex<double>* a, lapack_int lda, 
                                 lapack_int* jpvt, 
                                 std::complex<double>* tau){
            LAPACKE_zgeqp3(&matrix_layout, m, n, a, lda, jpvt, tau);
        }
        

    } //namespace Impl
} //namespace KokkosBlas

#endif 
