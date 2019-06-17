#ifndef KOKKOSLAPACK_HOST_TPL_HPP_
#define KOKKOSBLAS_HOST_TPL_HPP_

#include "KokkkosKernels_config.h"
#include "Kokkos_ArithTraits.hpp"

#if defined (KOKKOSKERNELS_ENABLE_TPL_BLAS)
#include <lapacke.h>
#include <cblas.h>

namespace KokkosBlas{
    namespace Impl{

        template<typename T>
        struct HostLapack {
            typedef Kokkos::ArithTraits<T> ats;
            typedef typename ats::mag_type mag_type;

            static 
            void dgeqp3(int matrix_layout,
                        lapack_int m, lapack_int n, 
                        T* a, lapack_int lda, 
                        lapack_int* jpvt,
                        T* tau);
        };


    } //namespace Impl
} //namespace KokkosBlas
