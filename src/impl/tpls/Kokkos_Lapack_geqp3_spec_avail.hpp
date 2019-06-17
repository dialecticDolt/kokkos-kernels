#ifndef KOKKOS_LAPACK_GEQP3_TPL_SPEC_AVAIL_HPP_
#define KOKKOS_LAPACK_GEQP3_TPL_SPEC_AVAIL_HPP_

namespace KokkosLapack {
    namespace Impl {

        template<class AVT, class PVT, class TVT>
        struct geqp3_tpl_spec_avail {
            enum : bool {value = false};
        };

        //Hostspace lapack(netlib) or MKL
        //TODO: Check if these have the same syntax

        #ifdef KOKKOSKERNELS_ENABLE_TPL_LAPACK

        #define KOKKOSBLAS3_GEMM_TPL_SPEC_AVAIL_BLAS(ORDINAL, SCALAR, LAYOUTA, LAYOUTB, LAYOUTC, MEMSPACE) \
        template<class ExecSpace> \
        struct geqp3_tpl_spec_avail< \
            Kokkos::View<SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                        Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
            Kokkos::View<ORDINAL*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                        Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
            Kokkos::View<SCALAR*, LAYOUTC, Kokkos::Device<ExecSpace, MEMSPACE>, \
                        Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
            > {enum : bool {value = true}; };

        #if defined (KOKKOSKERNELS_INST_DOUBLE)\
         && defined (KOKKOSKERNELS_INST_LAYOUTLEFT)
        KOKKOSLAPACK_GEQP3_TPL_SPEC_AVAIL_LAPACK(int, double, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace)
        #endif

        #if defined (KOKKOSKERNELS_INST_FLOAT) \
         && defined (KOKKOSKERNELS_INST_LAYOUTLEFT) 
        KOKKOSLAPACK_GEQP3_TPL_SPEC_AVAIL_LAPACK(int, float, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace)
        #endif

        #if defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_DOUBLE_) \
         && defined (KOKKOSKERNELS_INST_LAYOUTLEFT)
        KOKKOSLAPACK_GEQP3_TPL_SPEC_AVAIL_LAPACK(int, Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace)
        #endif

        #if defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_FLOAT_) \
         && defined (KOKKOSKERNELS_INST_LAYOUTLEFT)
        KOKKOSLAPACK_GEQP3_TPL_SPEC_AVAIL_LAPACK(int, Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace)
        #endif

        #if defined (KOKKOSKERNELS_INST_DOUBLE)\
         && defined (KOKKOSKERNELS_INST_LAYOUTRIGHT)
        KOKKOSLAPACK_GEQP3_TPL_SPEC_AVAIL_LAPACK(int, double, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace)
        #endif

        #if defined (KOKKOSKERNELS_INST_FLOAT) \
         && defined (KOKKOSKERNELS_INST_LAYOUTRIGHT) 
        KOKKOSLAPACK_GEQP3_TPL_SPEC_AVAIL_LAPACK(int, float, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace)
        #endif

        #if defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_DOUBLE_) \
         && defined (KOKKOSKERNELS_INST_LAYOUTRIGHT)
        KOKKOSLAPACK_GEQP3_TPL_SPEC_AVAIL_LAPACK(int, Kokkos::complex<double>, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace)
        #endif

        #if defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_FLOAT_) \
         && defined (KOKKOSKERNELS_INST_LAYOUTRIGHT)
        KOKKOSLAPACK_GEQP3_TPL_SPEC_AVAIL_LAPACK(int, Kokkos::complex<float>, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace)
        #endif
        
        #endif //if lapack


        //MAGMA
        /* 
        #ifdef KOKKOSKERNELS_ENABLE_TPL_MAGMA

        #define KOKKOSLAPACK_GEQP3_TPL_SPEC_AVAIL_MAGMA(ORDINAL, SCALAR, LAYOUTA, LAYOUTB, LAYOUTC, MEMSPACE) \
        template<class ExecSpace> \
        struct geqp3_tpl_spec_avail< \
            Kokkos::View<SCALAR**, LAYOUTA, KOKKOS::DEVICE<ExecSpace, MEMSPACE>, \
                        Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
            Kokkos::View<ORDINAL*, LAYOUTB, KOKKOS::DEVICE<ExecSpace, MEMSPACE>, \
                        Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
            Kokkos::View<SCALAR*, LAYOUTC, KOKKOS::DEVICE<ExecSppace, MEMSPACE>, \
                        Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
            > { enum : bool {value = true}; };

        #if defined (KOKKOSKERNELS_INST_DOUBLE)\
         && defined (KOKKOSKERNELS_INST_LAYOUTLEFT)
        KOKKOSLAPACK_GEQP3_TPL_SPEC_AVAIL_MAGMA(int, double, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaSpace)
        #endif

        #if defined (KOKKOSKERNELS_INST_FLOAT) \
         && defined (KOKKOSKERNELS_INST_LAYOUTLEFT) 
        KOKKOSLAPACK_GEQP3_TPL_SPEC_AVAIL_MAGMA(int, float, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaSpace)
        #endif

        #if defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_DOUBLE_) \
         && defined (KOKKOSKERNELS_INST_LAYOUTLEFT)
        KOKKOSLAPACK_GEQP3_TPL_SPEC_AVAIL_MAGMA(int, Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaSpace)
        #endif

        #if defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_FLOAT_) \
         && defined (KOKKOSKERNELS_INST_LAYOUTLEFT)
        KOKKOSLAPACK_GEQP3_TPL_SPEC_AVAIL_MAGMA(int, Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaSpace)
        #endif

        #if defined (KOKKOSKERNELS_INST_DOUBLE)\
         && defined (KOKKOSKERNELS_INST_LAYOUTRIGHT)
        KOKKOSLAPACK_GEQP3_TPL_SPEC_AVAIL_MAGMA(int, double, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace)
        #endif

        #if defined (KOKKOSKERNELS_INST_FLOAT) \
         && defined (KOKKOSKERNELS_INST_LAYOUTRIGHT) 
        KOKKOSLAPACK_GEQP3_TPL_SPEC_AVAIL_MAGMA(int, float, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace)
        #endif

        #if defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_DOUBLE_) \
         && defined (KOKKOSKERNELS_INST_LAYOUTRIGHT)
        KOKKOSLAPACK_GEQP3_TPL_SPEC_AVAIL_MAGMA(int, Kokkos::complex<double>, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace)
        #endif

        #if defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_FLOAT_) \
         && defined (KOKKOSKERNELS_INST_LAYOUTRIGHT)
        KOKKOSLAPACK_GEQP3_TPL_SPEC_AVAIL_MAGMA(int, Kokkos::complex<float>, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace)
        #endif

        #endif //if MAGMA
        */


    } //namespace Impl
} //namespace KokkosLapack

#endif
