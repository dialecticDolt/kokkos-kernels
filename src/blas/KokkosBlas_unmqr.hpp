#ifndef KOKKOSBLAS_UNMQR_HPP_
#define KOKKOSBLAS_UNMQR_HPP_

#include <KokkosKernels_Macros.hpp>
#include "KokkosBlas_unmqr_spec.hpp"
#include <KokkosKernels_helpers.hpp>
#include <sstream>
#include <type_traits>

namespace KokkosBlas {
/* 
 * \brief Multiply rectangular matrix C by Q or Q^H (where Q is output of QR by geqp3)
 * \tparam side: char: Apply to 'L' left of 'R' right
 * \tparam transpose (trans): char: Apply Q if trans='N', Apply Q^H if trans='R'
 * \tparam k: Number of elementary reflectors that define Q
 * \tparam CViewType Input/Output matrix, as a 2-D Kokkos::View
 * \tparam TauViewType Input vector     , as a 1-D Kokkos::View
 * \tparam AViewType Input matrix       , as a 1-D Kokkos::View
 * 
 * \param C [in/out] Input matrix, as a 2-D Kokkos::View. Overwritten by multiplication.
 * \param A [in] Input matrix, as a 1-D Kokkos::View. Output of qeqp3.
 * \param tau [in] Input vector, as a 1-D Kokkos::View. Scalar factors of reflectors.
 */

    template<class AViewType, class TauViewType, class CViewType>
    void unmqr(char side, char trans, int k, AViewType& A, TauViewType& tau, CViewType& C){

        #if (KOKKOSKERNELS_DEBUG_LEVEL >0)
        static_assert(Kokkos::Impl::is_view<AViewType>::value, "AViewType must be a Kokkos::View");
        static_assert(Kokkos::Impl::is_view<TauViewType>::value, "TauViewType must be a Kokkos::View");
        static_assert(Kokkos::Impl::is_view<CViewType>::value, "CViewType must be a Kokkos::View");
        static_assert(static_cast<int> (AViewType::rank)==2, "AViewType must have rank 2");
        static_assert(static_cast<int> (TauViewType::rank)==1, "TauViewType must have rank 1");
        static_assert(static_cast<int> (CViewType::rank)==2, "CViewType must have rank 2");
        int64_t A0 = A.extent(0);
        int64_t A1 = A.extent(1);
        int64_t C0 = C.extent(0);
        int64_t C1 = C.extent(1);
        int64_t tau0 = tau.extent(0); 

        /* TODO: How to use assert with error message in c++
        assert(p0>=A1, "Permutation vector is not long enough. Should be = A.cols");
        if(A0>A1){
            assert(tau0>=A0, "Tau vector must be longer than max(A.cols, A.rows)");
        }
        else{
            assert(tau0>=A1, "Tau vector must be longer than max(A.cols, A.rows)");
        }
        */
        #endif //KOKKOSKERNELS_DEBUG_LEVEL > 0 

        //return if degenerate matrix provided
        if((A.extent(0)==0) || (A.extent(1)==0))
            return;

        //stanardize of particulat View specializations
        typedef Kokkos::View<typename AViewType::const_value_type**,
                typename AViewType::array_layout,
                typename AViewType::device_type,
                Kokkos::MemoryTraits<Kokkos::Unmanaged> > AVT;

        typedef Kokkos::View<typename TauViewType::const_value_type*,
                typename TauViewType::array_layout,
                typename TauViewType::device_type,
                Kokkos::MemoryTraits<Kokkos::Unmanaged> > TVT;

        typedef Kokkos::View<typename CViewType::non_const_value_type**,
                typename CViewType::array_layout,
                typename CViewType::device_type,
                Kokkos::MemoryTraits<Kokkos::Unmanaged> > CVT;
       
        AVT A_i = A;
        TVT tau_i = tau;
        CVT C_i = C;
        typedef KokkosBlas::Impl::UNMQR<AVT, TVT, CVT> impl_type;
        impl_type::unmqr(side, trans, k, A_i, tau_i, C_i);


    }

} //namespace KokkosBlas

#endif //KOKKOSBLAS_UNMQR_HPP_
