#ifndef KOKKOSLAPACK_HOST_TPL_HPP_
#define KOKKOSLAPACK_HOST_TPL_HPP_

#include "KokkosKernels_config.h"
#include "Kokkos_ArithTraits.hpp"

#if defined( KOKKOSKERNELS_ENABLE_TPL_BLAS ) && defined( KOKKOSKERNELS_ENABLE_TPL_LAPACK )
#include <lapacke.h>
#include <cblas.h>

namespace KokkosBlas{
    namespace Impl{

        template<typename T>
        struct HostLapack {
            typedef Kokkos::ArithTraits<T> ats;
            typedef typename ats::mag_type mag_type;

            static 
            void geqp3(bool matrix_layout,
                        int m, int n, 
                        T* a, int lda, 
                        int* jpvt,
                        T* tau);

            static
            void unmqr(bool matrix_layout,
                       char side, char trans,
                       int m, int n, int k,
                       const T* a, int lda, 
                       const T* tau,
                       T* c, int ldc);

            static
            void ormqr(bool matrix_layout, 
                       char side, char trans,
                       int m, int n, int k,
                       const T* a, int lda,
                       const T* tau, 
                       T*c, int ldc);

        };

    } //namespace Impl
} //namespace KokkosBlas

#include "KokkosLapack_Host_tpl.cpp"

#endif //ENABLE BLAS/LAPACK

#endif //KOKKOSLAPACK_HOST_TPL_HPP_
