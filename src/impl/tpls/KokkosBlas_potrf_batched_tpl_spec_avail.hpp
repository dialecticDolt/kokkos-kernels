#ifndef KOKKOSBLAS_POTRF_BATCHED_TPL_SPEC_AVAIL_HPP_
#define KOKKOSBLAS_POTRF_BATCHED_TPL_SPEC_AVAIL_HPP_

namespace KokkosBlas {
    namespace Impl {

        template<class AVT>
        struct potrf_batched_tpl_spec_avail {
            enum : bool {value = false};
        };
        
        //MAGMA
        //
        #ifdef KOKKOSKERNELS_ENABLE_TPL_MAGMA


        #define KOKKOSBLAS_POTRF_TPL_SPEC_AVAIL_MAGMA(SCALAR, LAYOUTA, MEMSPACE) \
        template<class ExecSpace> \
        struct potrf_batched_tpl_spec_avail< \
            Kokkos::View< \
                          Kokkos::View<SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                                       Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
                          LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                          Kokkos::MemoryTraits<Kokkos::Unmanaged> > \
            > {enum : bool {value = true}; };

        #if defined (KOKKOSKERNELS_INST_DOUBLE)\
         && defined (KOKKOSKERNELS_INST_LAYOUTLEFT)
        KOKKOSBLAS_POTRF_BATCHED_TPL_SPEC_AVAIL_MAGMA(double, Kokkos::LayoutLeft, Kokkos::CudaSpace)
        #endif

        #if defined (KOKKOSKERNELS_INST_FLOAT) \
         && defined (KOKKOSKERNELS_INST_LAYOUTLEFT) 
        KOKKOSBLAS_POTRF_BATCHED_TPL_SPEC_AVAIL_MAGMA(float, Kokkos::LayoutLeft, Kokkos::CudaSpace)
        #endif

        #if defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_DOUBLE_) \
         && defined (KOKKOSKERNELS_INST_LAYOUTLEFT)
        KOKKOSBLAS_POTRF_BATCHED_TPL_SPEC_AVAIL_MAGMA(Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::CudaSpace)
        #endif

        #if defined (KOKKOSKERNELS_INST_KOKKOS_COMPLEX_FLOAT_) \
         && defined (KOKKOSKERNELS_INST_LAYOUTLEFT)
        KOKKOSBLAS_POTRF_BATCHED_TPL_SPEC_AVAIL_MAGMA(Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::CudaSpace)
        #endif

        #endif //if MAGMA

    } //namespace Impl
} //namespace KokkosBlas

#endif // KOKKOSBLAS_POTRF_BATCHED_TPL_SPEC_AVAIL_HPP_
