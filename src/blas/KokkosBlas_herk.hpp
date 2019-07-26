#ifndef KOKKOSBLAS_HERK_HPP_
#define KOKKOSBLAS_HERK_HPP_

#include <KokkosKernels_Macros.hpp>
#include "KokkosBlas_herk_spec.hpp"
#include <KokkosKernels_helpers.hpp>
#include <sstream>
#include <type_traits>

namespace KokkosBlas {

/* 
 * \brief Dense HERK (Device Level) Symmetric(real) or Hermetian (complex) Rank-k Update
 * \tparam side 'L' ( C = a*A*A' + b*C ) or 'R' (C = a*A'*A + beta*C)
 * \tparam Uplo (Rewrite Upper or Lower portion of Matrix C)
 * \tparam A Input matrix, as a 2-D Kokkos::View (If 'L', n x k, If 'R', k x n)
 * \tparam C Input matrix, as a 2-D Kokkos::View (Hermitian/Symmetric)
 * 
 * \param A [in] Input matrix, as a 2-D Kokkos::View. 
 * \param C [in/out] Input/Output Matrix, as a 2-D Kokkos::View 
 * 
 */
    template<class AViewType, class CViewType>
    void herk(const char uplo[], const char trans[],
              typename AViewType::const_value_type& alpha, AViewType& A, 
              typename CViewType::const_value_type& beta, CViewType& C){

        #if (KOKKOSKERNELS_DEBUG_LEVEL >0)
        static_assert(Kokkos::Impl::is_view<AViewType>::value, "KokkosBlas::herk: A must be a Kokkos::View");
        static_assert(Kokkos::Impl::is_view<CViewType>::value, "KokkosBlas::herk: C  must be a Kokkos::View");
        static_assert(static_cast<int> (AViewType::rank)==2, "KokkosBlas::herk: A must have rank 2");
        
        static_assert(static_cast<int> (CViewType::rank)==2, "KokkosBlas::herk: B must have rank 2");
        
        int64_t A0 = A.extent(0);
        int64_t A1 = A.extent(1);
        int64_t C0 = C.extent(0);
        int64_t C1 = C.extent(1);

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

        typedef Kokkos::View<typename CViewType::non_const_value_type**,
                typename CViewType::array_layout,
                typename CViewType::device_type,
                Kokkos::MemoryTraits<Kokkos::Unmanaged> > CVT;
       
        AVT A_i = A;
        CVT C_i = C;
        
        typedef KokkosBlas::Impl::HERK<AVT, CVT> impl_type;
        impl_type::herk(uplo[0], trans[0], alpha, A_i, beta, C_i);

    }

} //namespace KokkosBlas

#endif //KOKKOSBLAS_HERK_HPP_
