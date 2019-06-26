#ifndef KOKKOSBLAS_POTRF_HPP_
#define KOKKOSBLAS_POTRF_HPP_

#include <KokkosKernels_Macros.hpp>
#include "KokkosBlas_potrf_spec.hpp"
#include <KokkosKernels_helpers.hpp>
#include <sstream>
#include <type_traits>

namespace KokkosBlas {
/* 
 * \brief Dense POTRF (Device Level) Top Looking Block Cholesky
 * \tparam uplo 'U' ( A= U' U ) or 'L' (A = LL')
 * \tparam A Input/Ouput matrix, as a 2-D Kokkos::View
 * 
 * \param A [in/out] Input/Output Matrix, as a 2-D Kokkos::View 
 * 
 */

    template<class AViewType>
    void potrf(const char uplo[], AViewType& A){

        #if (KOKKOSKERNELS_DEBUG_LEVEL >0)
        static_assert(Kokkos::Impl::is_view<AViewType>::value, "KokkosBlas::potrf: A must be a Kokkos::View");
        static_assert(static_cast<int> (AViewType::rank)==2, "KokkosBlas::potrf: A must have rank 2");
        
        int64_t A0 = A.extent(0);
        int64_t A1 = A.extent(1);

        //TODO: Add runtime checking of flags
        //TODO: Add runtime dimension checks

        #endif //KOKKOSKERNELS_DEBUG_LEVEL > 0 

        //return if degenerate matrix provided
        if((A.extent(0)==0) || (A.extent(1)==0))
            return;

        //particular View specializations
        typedef Kokkos::View<typename AViewType::non_const_value_type**,
                typename AViewType::array_layout,
                typename AViewType::device_type,
                Kokkos::MemoryTraits<Kokkos::Unmanaged> > AVT;

        AVT A_i = A;
        
        typedef KokkosBlas::Impl::POTRF<AVT> impl_type;
        impl_type::potrf(uplo[0], A_i);

    }

} //namespace KokkosBlas

#endif //KOKKOSBLAS_POTRF_HPP_
