#ifndef KOKKOSBLAS_TRSM_HPP_
#define KOKKOSBLAS_TRSM_HPP_

#include <KokkosKernels_Macros.hpp>
#include "KokkosBlas_trsm_spec.hpp"
#include <KokkosKernels_helpers.hpp>
#include <sstream>
#include <type_traits>

namespace KokkosBlas {
/* 
 * \brief Dense TRSM (Device Level) Triangular Solve Multiple RHS
 * \tparam side 'L' ( AX = alpha*B ) or 'R' (XA = alpha*B)
 * \tparam Uplo (Given as Upper or Lower portion of Matrix A)
 * \tparam Trans (Transpose A)
 * \tparam diag (Is A unit trangular)
 * \tparam A Input matrix, as a 2-D Kokkos::View
 * \tparam B Input matrix, as a 2-D Kokkos::View
 * 
 * \param A [in] Input matrix, as a 2-D Kokkos::View. 
 * \param B [in/out] Input/Output Matrix, as a 2-D Kokkos::View 
 * 
 */

    template<class AViewType, class BViewType>
    void trsm(const char side[], const char uplo[], const char transa[], const char diag[],
              typename AViewType::const_value_type& alpha, AViewType& A, BViewType& B){

        #if (KOKKOSKERNELS_DEBUG_LEVEL >0)
        static_assert(Kokkos::Impl::is_view<AViewType>::value, "KokkosBlas::trsm: A must be a Kokkos::View");
        static_assert(Kokkos::Impl::is_view<BViewType>::value, "KokkosBlas::trsm: B  must be a Kokkos::View");
        static_assert(static_cast<int> (AViewType::rank)==2, "KokkosBlas::trsm: A must have rank 2");
        
        static_assert(static_cast<int> (AViewType::rank)==2, "KokkosBlas::trsm: B must have rank 2");
        
        int64_t A0 = A.extent(0);
        int64_t A1 = A.extent(1);
        int64_t B0 = B.extent(0);
        int64_t B1 = B.extent(1);

        //TODO: Add runtime checking of flags
        //TODO: Add runtime dimension checks

        #endif //KOKKOSKERNELS_DEBUG_LEVEL > 0 

        //return if degenerate matrix provided
        if((A.extent(0)==0) || (A.extent(1)==0))
            return;

        //particular View specializations
        typedef Kokkos::View<typename AViewType::const_value_type**,
                typename AViewType::array_layout,
                typename AViewType::device_type,
                Kokkos::MemoryTraits<Kokkos::Unmanaged> > AVT;

        typedef Kokkos::View<typename BViewType::non_const_value_type**,
                typename AViewType::array_layout,
                typename AViewType::device_type,
                Kokkos::MemoryTraits<Kokkos::Unmanaged> > BVT;
       
        AVT A_i = A;
        BVT B_i = B;
        
        typedef KokkosBlas::Impl::TRSM<AVT, BVT> impl_type;
        impl_type::trsm(side[0], uplo[0], transa[0], diag[0], alpha, A_i, B_i);

    }

} //namespace KokkosBlas

#endif //KOKKOSBLAS_TRSM_HPP_
