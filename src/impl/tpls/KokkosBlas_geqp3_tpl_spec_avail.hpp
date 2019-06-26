#ifndef KOKKOSBLAS_GEQP3_TPL_SPEC_AVAIL_HPP_
#define KOKKOSBLAS_GEQP3_TPL_SPEC_AVAIL_HPP_

namespace KokkosBlas {
    namespace Impl {

        template<class AVT, class PVT, class TVT>
        struct geqp3_tpl_spec_avail {
            enum : bool {value = false};
        };

        //Hostspace LAPACKE(netlib) or MKL
        //TODO: Check if these have the same syntax

        #ifdef KOKKOSKERNELS_ENABLE_TPL_BLAS

        #define KOKKOSBLAS_GEQP3_TPL_SPEC_AVAIL_BLAS(SCALAR, LAYOUTA, LAYOUTB, LAYOUTC, MEMSPACE) \
        template<class ExecSpace> \
        struct geqp3_tpl_spec_avail< \
            Kokkos::View<SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                        Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
            Kokkos::View<int*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                        Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
            Kokkos::View<SCALAR*, LAYOUTC, Kokkos::Device<ExecSpace, MEMSPACE>, \
                        Kokkos::MemoryTraits<Kokkos::Unmanaged> > \
            > {enum : bool {value = true}; };

        #if defined (KOKKOSKERNELS_INST_DOUBLE)\
         && defined (KOKKOSKERNELS_INST_LAYOUTLEFT)
        KOKKOSBLAS_GEQP3_TPL_SPEC_AVAIL_BLAS(double, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace)
        #endif

        #if defined (KOKKOSKERNELS_INST_FLOAT) \
         && defined (KOKKOSKERNELS_INST_LAYOUTLEFT) 
        KOKKOSBLAS_GEQP3_TPL_SPEC_AVAIL_BLAS(float, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace)
        #endif

        #if defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_DOUBLE_) \
         && defined (KOKKOSKERNELS_INST_LAYOUTLEFT)
        KOKKOSBLAS_GEQP3_TPL_SPEC_AVAIL_BLAS(Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace)
        #endif

        #if defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_FLOAT_) \
         && defined (KOKKOSKERNELS_INST_LAYOUTLEFT)
        KOKKOSBLAS_GEQP3_TPL_SPEC_AVAIL_BLAS(Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace)
        #endif

        #if defined (KOKKOSKERNELS_INST_DOUBLE)\
         && defined (KOKKOSKERNELS_INST_LAYOUTRIGHT)
        KOKKOSBLAS_GEQP3_TPL_SPEC_AVAIL_BLAS(double, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace)
        #endif

        #if defined (KOKKOSKERNELS_INST_FLOAT) \
         && defined (KOKKOSKERNELS_INST_LAYOUTRIGHT) 
        KOKKOSBLAS_GEQP3_TPL_SPEC_AVAIL_BLAS(float, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace)
        #endif

        #if defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_DOUBLE_) \
         && defined (KOKKOSKERNELS_INST_LAYOUTRIGHT)
        KOKKOSBLAS_GEQP3_TPL_SPEC_AVAIL_BLAS(Kokkos::complex<double>, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace)
        #endif

        #if defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_FLOAT_) \
         && defined (KOKKOSKERNELS_INST_LAYOUTRIGHT)
        KOKKOSBLAS_GEQP3_TPL_SPEC_AVAIL_BLAS(Kokkos::complex<float>,Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace)
        #endif
        
        #endif //if BLAS/LAPACK


        //MAGMA
 
        #ifdef KOKKOSKERNELS_ENABLE_TPL_MAGMA

        #define KOKKOSBLAS_GEQP3_TPL_SPEC_AVAIL_MAGMA(SCALAR, LAYOUTA, LAYOUTB, LAYOUTC, MEMSPACE) \
        template<class ExecSpace> \
        struct geqp3_tpl_spec_avail< \
            Kokkos::View<SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                        Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
            Kokkos::View<int*, Kokkos::HostSpace, \
                        Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
            Kokkos::View<SCALAR*, LAYOUTC, Kokkos::Device<ExecSpace, MEMSPACE>, \
                        Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
            > { enum : bool {value = true}; };

        #if defined (KOKKOSKERNELS_INST_DOUBLE)\
         && defined (KOKKOSKERNELS_INST_LAYOUTLEFT)
        KOKKOSBLAS_GEQP3_TPL_SPEC_AVAIL_MAGMA(double, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaSpace)
        #endif

        #endif //if MAGMA


    } //namespace Impl
} //namespace KokkosLapack

#endif // KOKKOSBLAS_GEQP3_TPL_SPEC_AVAIL_HPP_
