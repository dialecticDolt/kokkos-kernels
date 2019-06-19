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
        };

    } //namespace Impl
} //namespace KokkosBlas

#include "KokkosLapack_Host_tpl.cpp"

#endif //ENABLE BLAS/LAPACK

#endif //KOKKOSLAPACK_HOST_TPL_HPP_
