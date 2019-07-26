#ifndef KOKKOSBLAS_BATCHED_POTRF_SPEC_HPP_
#define KOKKOSBLAS_BATCHED_POTRF_SPEC_HPP_

#include "KokkosKernels_config.h"
#include "Kokkos_Core.hpp"


#if !defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMILE_LIBRARY
#include "KokkosBlas_potrf_batched_impl.hpp"
#endif

namespace KokkosBlas {
    namespace Impl {
        
        template<class AVT>
        struct potrf_batched_eti_spec_avail {
            enum : bool {value = false };
        };

    } //namespace Impl
} //namespace KokkosBlas

#define KOKKOSBLAS_POTRF_BATCHED_ETI_SPEC_AVAIL(SCALAR_TYPE, LAYOUT_TYPE, EXEC_SPACE_TYPE, MEM_SPACE_TYPE)\
    template<> \
    struct potrf_batched_eti_spec_avail< \
        Kokkos::View<SCALAR_TYPE**, LAYOUT_TYPE, \
                    Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> > \
    > { enum : bool { value = true }; };

#include<KokkosBlas_potrf_batched_tpl_spec_avail.hpp>
#include<generated_specializations_hpp/KokkosBlas_potrf_batched_eti_spec_avail.hpp>

namespace KokkosBlas {
    namespace Impl {
        //Unification Layer

        template<
            class AVT, 
            bool tpl_spec_avail = potrf_tpl_spec_avail<AVT>::value,
            bool eti_spec_avail = potrf_eti_spec_avail<AVT>::value
        >
        struct POTRF_BATCHED{
            static void potrf_batched(const char uplo, AVT& A);
        };



        #if !defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
        //specialization layer for no TPL
        template<class AVT>
        struct POTRF_BATCHED<AVT, false, KOKKOSKERNELS_IMPL_COMPILE_LIBRARY>{
            static void potrf(const char uplo, AVT& A){
                execute_potrf_batched<AVT>(uplo, A);
            }
        };
        #endif

    } //namespace Impl
} //namespace KokkosBlas


#define KOKKOSBLAS_POTRF_BATCHED_ETI_SPEC_DECL(SCALAR_TYPE, LAYOUT_TYPE, EXEC_SPACE_TYPE, MEM_SPACE_TYPE) \
    extern template struct \
    POTRF_BATCHED< Kokkos::View< \
                        Kokkos::View<SCALAR_TYPE**, LAYOUT_TYPE, \
                                     Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>, \
                                     Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
                        LAYOUT_TYPE, Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,\
                        Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
            false, true>; \

#define KOKKOSBLAS_POTRF_BATCHED_ETI_SPEC_INST(SCALAR_TYPE, LAYOUT_TYPE, EXEC_SPACE_TYPE, MEM_SPACE_TYPE) \
    template struct \
    POTRF_BATCHED< Kokkos::View< \
                        Kokkos::View<SCALAR_TYPE**, LAYOUT_TYPE, \
                                     Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>, \
                                     Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
                        LAYOUT_TYPE, Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,\
                        Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
            false, true>;

#include<KokkosBlas_potrf_batched_tpl_spec_decl.hpp>
#include<generated_specializations_hpp/KokkosBlas_potrf_batched_eti_spec_decl.hpp>

#endif //KOKKOSBLAS_BATCHED_IMPL_POTRF_HPP_
