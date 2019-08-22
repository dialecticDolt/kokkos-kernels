#ifndef KOKKOSBLAS_TRMM_SPEC_HPP_
#define KOKKOSBLAS_TRMM_SPEC_HPP_

#include "KokkosKernels_config.h"
#include "Kokkos_Core.hpp"


#if !defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMILE_LIBRARY
#include "KokkosBlas_trmm_impl.hpp"
#endif

namespace KokkosBlas {
    namespace Impl {
        
        template<class AVT, class BVT>
        struct trmm_eti_spec_avail {
            enum : bool {value = false };
        };

    } //namespace Impl
} //namespace KokkosBlas

#define KOKKOSBLAS_TRMM_ETI_SPEC_AVAIL(SCALAR_TYPE, LAYOUT_TYPE, EXEC_SPACE_TYPE, MEM_SPACE_TYPE)\
    template<> \
    struct trmm_eti_spec_avail< \
        Kokkos::View<const SCALAR_TYPE**, LAYOUT_TYPE, \
                    Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<SCALAR_TYPE**, LAYOUT_TYPE, \
                    Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> > \
    > { enum : bool { value = true }; };

#include<KokkosBlas_trmm_tpl_spec_avail.hpp>
#include<generated_specializations_hpp/KokkosBlas_trmm_eti_spec_avail.hpp>

namespace KokkosBlas {
    namespace Impl {
        //Unification Layer

        template<
            class AVT, class BVT, 
            bool tpl_spec_avail = trmm_tpl_spec_avail<AVT, BVT>::value,
            bool eti_spec_avail = trmm_eti_spec_avail<AVT, BVT>::value
        >
        struct TRMM{
            static void trmm(const char side, const char uplo,
                             const char transa, const char diag,
                             typename AVT::const_value_type& alpha,
                             AVT& A, BVT& B);

        };



        #if !defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
        //specialization layer for no TPL
        template<class AVT, class BVT>
        struct TRMM<AVT, BVT, false, KOKKOSKERNELS_IMPL_COMPILE_LIBRARY>{
            static void trmm(const char side, const char uplo,
                             const char transa, const char diag,
                             typename AVT::const_value_type& alpha,
                             AVT& A, BVT& B){
                execute_trmm<AVT, BVT>(side, uplo, transa, diag, 
                                       alpha, A, B);
            }
        };
        #endif

    } //namespace Impl
} //namespace KokkosBlas


#define KOKKOSBLAS_TRMM_ETI_SPEC_DECL(SCALAR_TYPE, LAYOUT_TYPE, EXEC_SPACE_TYPE, MEM_SPACE_TYPE) \
    extern template struct \
    TRMM< Kokkos::View<const SCALAR_TYPE**, LAYOUT_TYPE, \
                        Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>, \
                        Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
           Kokkos::View<SCALAR_TYPE**, LAYOUT_TYPE, \
                        Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>, \
                        Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
            false, true>; \

#define KOKKOSBLAS_TRMM_ETI_SPEC_INST(SCALAR_TYPE, LAYOUT_TYPE, EXEC_SPACE_TYPE, MEM_SPACE_TYPE) \
    template struct \
    TRMM< Kokkos::View<const SCALAR_TYPE**, LAYOUT_TYPE, \
                        Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>, \
                        Kokkos::MemoryTraits<Kokkos::Unmanaged> > , \
           Kokkos::View<SCALAR_TYPE**, LAYOUT_TYPE, \
                        Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>, \
                        Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
           false, true>;

#include<KokkosBlas_trmm_tpl_spec_decl.hpp>
#include<generated_specializations_hpp/KokkosBlas_trmm_eti_spec_decl.hpp>

#endif //KOKKOSBLAS_IMPL_TRMM_HPP_
