#ifndef KOKKOSBLAS_GEQP3_SPEC_HPP_
#define KOKKOSBLAS_GEQP3_SPEC_HPP_

#include "KokkosKernels_config.h"
#include "Kokkos_Core.hpp"

#if !defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMILE_LIBRARY
#include <KokkosBlas_geqp3_impl.hpp>
#endif

namespace KokkosBlas {
    namespace Impl {
        
        template<class AVT, class PVT, class TVT>
        struct geqp3_eti_spec_avail{
            enum : bool {value = false };
        }


        template<
            class AVT, class PVT, class TVT, 
            bool tpl_spec_avail = geqp3_tpl_spec_avail<AVT, PVT, TVT>::value,
            bool eti_spec_avail = geqp3_eti_spec_avail<AVT, PVT, TVT>::value
        >
        struct GEQP3{
            static void geqp3(AVT& A, PVT& p, TVT& tau);
        }

        //specialization layer for no TPL
        template<class AVT, class PVT, class TVT>
        struct GEQP3<AVT, PVT, TVT, false, KOKKOSKERNELS_IMPL_COMPILE_LIBRARY>{
            static void geqp3(AVT& A, PVT& p, TVT& tau){
                execute_geqp3(A, p, tau);
            }
        }

        #if !KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
        #include<impl/tpls/geqp3_tpl_spec_decl.hpp>
        #include<impl/generated_specializations_hpp/geqp3_eti_spec_decl.hpp>
        #endif

    } //namespace Impl
} //namespace KokkosBlas


#endif
