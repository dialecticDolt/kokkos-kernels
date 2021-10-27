#ifndef KOKKOSBLAS_UNMQR_SPEC_HPP_
#define KOKKOSBLAS_UNMQR_SPEC_HPP_

#include "KokkosKernels_config.h"
#include "Kokkos_Core.hpp"


#if !defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
#include "KokkosBlas_unmqr_impl.hpp"
#endif

namespace KokkosBlas {
    namespace Impl {
        
        template<class AVT, class TVT, class CVT, class WVT>
        struct unmqr_eti_spec_avail {
            enum : bool {value = false };
        };

    } //namespace Impl
} //namespace KokkosBlas

#define KOKKOSBLAS_UNMQR_ETI_SPEC_AVAIL(SCALAR_TYPE, LAYOUT_TYPE, EXEC_SPACE_TYPE, MEM_SPACE_TYPE)\
    template<> \
    struct unmqr_eti_spec_avail< \
        Kokkos::View<const SCALAR_TYPE**, LAYOUT_TYPE, \
                    Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<const SCALAR_TYPE*, LAYOUT_TYPE, \
                    Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<SCALAR_TYPE**, LAYOUT_TYPE, \
                    Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<SCALAR_TYPE*, LAYOUT_TYPE, \
                    Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> > \
    > { enum : bool { value = true }; };

#include<KokkosBlas_unmqr_tpl_spec_avail.hpp>
#include<generated_specializations_hpp/KokkosBlas_unmqr_eti_spec_avail.hpp>
#include<generated_specializations_hpp/KokkosBlas_unmqr_workspace_eti_spec_avail.hpp>

namespace KokkosBlas {
    namespace Impl {
        //Unification Layer

        template<
            class AVT, class TVT, class CVT, class WVT,
            bool tpl_spec_avail = unmqr_tpl_spec_avail<AVT, TVT, CVT, WVT>::value,
            bool eti_spec_avail = unmqr_eti_spec_avail<AVT, TVT, CVT, WVT>::value
        >
        struct UNMQR{
            static void unmqr(const char side, const char trans, int k, AVT& A, TVT& tau, CVT& C, WVT& workspace);
        };



        #if !defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
        //specialization layer for no TPL
        template<class AVT, class TVT, class CVT, class WVT>
        struct UNMQR<AVT, TVT, CVT, WVT, false, KOKKOSKERNELS_IMPL_COMPILE_LIBRARY>{
            static void unmqr(const char side, const char trans, int k, AVT& A, TVT& tau, CVT& C, WVT& workspace){
                execute_unmqr<AVT, TVT, CVT, WVT>(side, trans, k, A, tau, C, workspace);
            }
        };
        #endif

    } //namespace Impl
} //namespace KokkosBlas


#define KOKKOSBLAS_UNMQR_ETI_SPEC_DECL(SCALAR_TYPE, LAYOUT_TYPE, EXEC_SPACE_TYPE, MEM_SPACE_TYPE) \
    extern template struct \
    UNMQR< Kokkos::View<const SCALAR_TYPE **, LAYOUT_TYPE, \
                        Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>, \
                        Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
           Kokkos::View<const SCALAR_TYPE*, LAYOUT_TYPE, \
                        Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>, \
                        Kokkos::MemoryTraits<Kokkos::Unmanaged> > , \
           Kokkos::View<SCALAR_TYPE**, LAYOUT_TYPE, \
                        Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>, \
                        Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
           Kokkos::View<SCALAR_TYPE*, LAYOUT_TYPE, \
                        Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>, \
                        Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
            false, true>; \

#define KOKKOSBLAS_UNMQR_ETI_SPEC_INST(SCALAR_TYPE, LAYOUT_TYPE, EXEC_SPACE_TYPE, MEM_SPACE_TYPE) \
    template struct \
    UNMQR< Kokkos::View<const SCALAR_TYPE**, LAYOUT_TYPE, \
                        Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>, \
                        Kokkos::MemoryTraits<Kokkos::Unmanaged> > , \
           Kokkos::View<const SCALAR_TYPE*, LAYOUT_TYPE, \
                        Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>, \
                        Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
           Kokkos::View<SCALAR_TYPE**, LAYOUT_TYPE, \
                        Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>, \
                        Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
           Kokkos::View<SCALAR_TYPE*, LAYOUT_TYPE, \
                        Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>, \
                        Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
           false, true>;

#include<KokkosBlas_unmqr_tpl_spec_decl.hpp>
#include<generated_specializations_hpp/KokkosBlas_unmqr_eti_spec_decl.hpp>
#include<generated_specializations_hpp/KokkosBlas_unmqr_workspace_eti_spec_decl.hpp>

#endif //KOKKOSBLAS_IMPL_UNMQR_HPP_
