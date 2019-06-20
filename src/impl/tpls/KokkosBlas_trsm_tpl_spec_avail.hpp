#ifndef KOKKOSBLAS_TRSM_TPL_SPEC_AVAIL_HPP_
#define KOKKOSBLAS_TRSM_TPL_SPEC_AVAIL_HPP_

namespace KokkosBlas {
    namespace Impl {

        template<class AVT, class BVT>
        struct trsm_tpl_spec_avail {
            enum : bool {value = false};
        };

        //Hostspace LAPACKE(netlib) or MKL
        //TODO: Check if these have the same syntax

        #ifdef KOKKOSKERNELS_ENABLE_TPL_BLAS

        #define KOKKOSBLAS_TRSM_TPL_SPEC_AVAIL_BLAS(SCALAR, LAYOUTA, LAYOUTB, LAYOUTC, MEMSPACE) \
        template<class ExecSpace> \
        struct trsm_tpl_spec_avail< \
            Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                        Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
            Kokkos::View<SCALAR**, LAYOUTC, Kokkos::Device<ExecSpace, MEMSPACE>, \
                        Kokkos::MemoryTraits<Kokkos::Unmanaged> > \
            > {enum : bool {value = true}; };

        #if defined (KOKKOSKERNELS_INST_DOUBLE)\
         && defined (KOKKOSKERNELS_INST_LAYOUTLEFT)
        KOKKOSBLAS_TRSM_TPL_SPEC_AVAIL_BLAS(double, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace)
        #endif

        #if defined (KOKKOSKERNELS_INST_FLOAT) \
         && defined (KOKKOSKERNELS_INST_LAYOUTLEFT) 
        KOKKOSBLAS_TRSM_TPL_SPEC_AVAIL_BLAS(float, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace)
        #endif

        #if defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_DOUBLE_) \
         && defined (KOKKOSKERNELS_INST_LAYOUTLEFT)
        KOKKOSBLAS_TRSM_TPL_SPEC_AVAIL_BLAS(Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace)
        #endif

        #if defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_FLOAT_) \
         && defined (KOKKOSKERNELS_INST_LAYOUTLEFT)
        KOKKOSBLAS_TRSM_TPL_SPEC_AVAIL_BLAS(Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace)
        #endif

        #if defined (KOKKOSKERNELS_INST_DOUBLE)\
         && defined (KOKKOSKERNELS_INST_LAYOUTRIGHT)
        KOKKOSBLAS_TRSM_TPL_SPEC_AVAIL_BLAS(double, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace)
        #endif

        #if defined (KOKKOSKERNELS_INST_FLOAT) \
         && defined (KOKKOSKERNELS_INST_LAYOUTRIGHT) 
        KOKKOSBLAS_TRSM_TPL_SPEC_AVAIL_BLAS(float, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace)
        #endif

        #if defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_DOUBLE_) \
         && defined (KOKKOSKERNELS_INST_LAYOUTRIGHT)
        KOKKOSBLAS_TRSM_TPL_SPEC_AVAIL_BLAS(Kokkos::complex<double>, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace)
        #endif

        #if defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_FLOAT_) \
         && defined (KOKKOSKERNELS_INST_LAYOUTRIGHT)
        KOKKOSBLAS_TRSM_TPL_SPEC_AVAIL_BLAS(Kokkos::complex<float>, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace)
        #endif
        
        #endif //if BLAS/LAPACK


        //MAGMA
        /* 
        #ifdef KOKKOSKERNELS_ENABLE_TPL_MAGMA

        #define KOKKOSBLAS_TRSM_TPL_SPEC_AVAIL_MAGMA(ORDINAL, SCALAR, LAYOUTA, LAYOUTB, LAYOUTC, MEMSPACE) \
        template<class ExecSpace> \
        struct trsm_tpl_spec_avail< \
            Kokkos::View<SCALAR**, LAYOUTA, KOKKOS::DEVICE<ExecSpace, MEMSPACE>, \
                        Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
            Kokkos::View<ORDINAL*, LAYOUTB, KOKKOS::DEVICE<ExecSpace, MEMSPACE>, \
                        Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
            Kokkos::View<SCALAR*, LAYOUTC, KOKKOS::DEVICE<ExecSppace, MEMSPACE>, \
                        Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
            > { enum : bool {value = true}; };

        #if defined (KOKKOSKERNELS_INST_DOUBLE)\
         && defined (KOKKOSKERNELS_INST_LAYOUTLEFT)
        KOKKOSBLAS_TRSM_TPL_SPEC_AVAIL_MAGMA(int, double, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaSpace)
        #endif

        #if defined (KOKKOSKERNELS_INST_FLOAT) \
         && defined (KOKKOSKERNELS_INST_LAYOUTLEFT) 
        KOKKOSBLAS_TRSM_TPL_SPEC_AVAIL_MAGMA(int, float, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaSpace)
        #endif

        #if defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_DOUBLE_) \
         && defined (KOKKOSKERNELS_INST_LAYOUTLEFT)
        KOKKOSBLAS_TRSM_TPL_SPEC_AVAIL_MAGMA(int, Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaSpace)
        #endif

        #if defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_FLOAT_) \
         && defined (KOKKOSKERNELS_INST_LAYOUTLEFT)
        KOKKOSBLAS_TRSM_TPL_SPEC_AVAIL_MAGMA(int, Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaSpace)
        #endif

        #if defined (KOKKOSKERNELS_INST_DOUBLE)\
         && defined (KOKKOSKERNELS_INST_LAYOUTRIGHT)
        KOKKOSBLAS_TRSM_TPL_SPEC_AVAIL_MAGMA(int, double, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace)
        #endif

        #if defined (KOKKOSKERNELS_INST_FLOAT) \
         && defined (KOKKOSKERNELS_INST_LAYOUTRIGHT) 
        KOKKOSBLAS_TRSM_TPL_SPEC_AVAIL_MAGMA(int, float, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace)
        #endif

        #if defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_DOUBLE_) \
         && defined (KOKKOSKERNELS_INST_LAYOUTRIGHT)
        KOKKOSBLAS_TRSM_TPL_SPEC_AVAIL_MAGMA(int, Kokkos::complex<double>, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace)
        #endif

        #if defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_FLOAT_) \
         && defined (KOKKOSKERNELS_INST_LAYOUTRIGHT)
        KOKKOSBLAS_TRSM_TPL_SPEC_AVAIL_MAGMA(int, Kokkos::complex<float>, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace)
        #endif

        #endif //if MAGMA
        */

    } //namespace Impl
} //namespace KokkosLapack

#endif // KOKKOSBLAS_TRSM_TPL_SPEC_AVAIL_HPP_
