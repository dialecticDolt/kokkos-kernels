#ifndef KOKKOSBLAS_GEQP3_SPEC_HPP_
#define KOKKOSBLAS_GEQP3_SPEC_HPP_

#include "KokkosKernels_config.h"
#include "Kokkos_Core.hpp"


//#if !defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMILE_LIBRARY
#include "KokkosBlas_geqp3_impl.hpp"
//#endif

namespace KokkosBlas {
    namespace Impl {
        
        template<class AVT, class PVT, class TVT>
        struct geqp3_eti_spec_avail{
            enum : bool {value = false };
        };

    } //namespace Impl
} //namespace KokkosBlas

#define KOKKOSBLAS_GEQP3_ETI_SPEC_AVAIL(ORDINAL_TYPE, SCALAR_TYPE, LAYOUT_TYPE, EXEC_SPACE_TYPE, MEM_SPACE_TYPE)\
template<> \
struct geqp3_eti_spec_avail< \
        Kokkos::View<SCALAR_TYPE**, LAYOUT_TYPE, \
                    Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<ORDINAL_TYPE *, LAYOUT_TYPE, \
                    Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> > \
        Kokkos::View<SCALAR_TYPE*, LAYOUT_TYPE, \
                    Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> > > \
{enum : bool {value = true}; };

#include<KokkosBlas_geqp3_tpl_spec_avail.hpp>
//#include<generated_specializations_hpp/KokkosBlas_geqp3_eti_spec_avail.hpp>

namespace KokkosBlas {
    namespace Impl {
        //Unification Layer

        template<
            class AVT, class PVT, class TVT, 
            bool tpl_spec_avail = geqp3_tpl_spec_avail<AVT, PVT, TVT>::value,
            bool eti_spec_avail = geqp3_eti_spec_avail<AVT, PVT, TVT>::value
        >
        struct GEQP3{
            static void geqp3(AVT& A, PVT& p, TVT& tau);
        };



        #if !defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
        //specialization layer for no TPL
        template<class AVT, class PVT, class TVT>
        struct GEQP3<AVT, PVT, TVT, false, KOKKOSKERNELS_IMPL_COMPILE_LIBRARY>{
            static void geqp3(AVT& A, PVT& p, TVT& tau){
                execute_geqp3<AVT, PVT, TVT>(A, p, tau);
            }
        };
        #endif

    } //namespace Impl
} //namespace KokkosBlas


#define KOKKOSBLAS_GEQP3_ETI_SPEC_DECL(ORDINAL_TYPE, SCALAR_TYPE, LAYOUT_TYPE, EXEC_SPACE_TYPE, MEM_SPACE_TYPE) \
    extern template struct \
    GEQP3< Kokkos::View<SCALAR_TYPE **, LAYOUT_TYPE, \
                        Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>, \
                        Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
           Kokkos::View<ORDINAL_TYPE*, LAYOUT_TYPE, \
                        Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>, \
                        Kokkos::MemoryTraits<Kokkos::Unmnaged> > , \
            false, true>;

#include<KokkosBlas_geqp3_tpl_spec_decl.hpp>
//#include<generated_specializations_hpp/KokkosBlas_geqp3_et_spec_decl.hpp>

#endif //KOKKOSBLAS_IMPL_GEQP3_HPP_
