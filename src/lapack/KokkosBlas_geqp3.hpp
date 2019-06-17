#ifndef KOKKOSBLAS_GEQP3_HPP_
#define KOKKOSBLAS_GEQP3_HPP_

#include <KokkosKernels_Macros.hpp>
#include <KokkosBlas_geqp3_spec.hpp>
#include <KokkosKernels_helpers.hpp>
#include <sstream>
#include <type_traits>

namespace KokkosBlas {
/* 
 * \brief Dense QR with column pivoting: AP = QR
 * \tparam AViewType Input matrix, as a 2-D Kokkos::View
 * \tparam PViewType Input vector, as a 1-D Kokkos::View
 * \tparam TauViewType Output,     as a 1-D Kokkos::View
 * 
 * \param A [in/out] Input matrix, as a 2-D Kokkos::View. Overwritten by factorization.
 * \param p [in/out] Input vector, as a 1-D Kokkos::View (Integer Type).
 *  On input if p[i-1] != 0, then column i is moved to beginning of AP before computation.
 *  Otherwise the column is able to be permuted freely. 
 *  p is overwritten to give the output permutation. 
 *
 * \param tau [out] Output vector, as a 1-D Kokkos::View
 *  Gives the scalar factors of the reflectors. 
 * 
 */

    template<class AViewType, class PViewType, class TauViewType>
    void geqp3(AViewType& A, PViewType& p, TauViewType& tau){

        #if (KOKKOSKERNELS_DEBUG_LEVEL >0)
        static_assert(Kokkos::Impl::is_view<AViewType>::value, "AViewType must be a Kokkos::View");
        static_assert(Kokkos::Impl::is_view<PViewType>::value, "PViewType must be a Kokkos::View");
        static_assert(Kokkos::Impl::is_view<TauViewType>::value, "TauViewType must be a Kokkos::View");
        static_assert(static_cast<int> (AViewType::rank)==2, "AViewType must have rank 2");
        static_assert(static_cast<int> (PViewType::rank)==1, "PViewType must have rank 1");
        static_assert(static_cast<int> (TauViewType::rank)==1, "TauViewType must have rank 1");
        int64_t A0 = A.extent(0);
        int64_t A1 = A.extent(1);
        int64_t p0 = p.extent(0);
        int64_t tau0 = tau.extent(0);
        assert(p0>=A1, "Permutation vector is not long enough. Should be = A.cols");
        if(A0>A1){
            assert(tau0>=A0, "Tau vector must be longer than max(A.cols, A.rows)");
        }
        else{
            assert(tau0>=A1, "Tau vector must be longer than max(A.cols, A.rows)");
        }
        #endif //KOKKOSKERNELS_DEBUG_LEVEL > 0 

        //return if degenerate matrix provided
        if((A.extent(0)==0) || (A.extent(1)==0))
            return;

        //stanardize of particulat View specializations
        typedef Kokkos::View<typename AViewType::non_const_value_type**,
                typename AViewType::array_layout,
                typename AViewType::device_type,
                Kokkos::MemoryTraits<Kokkos::Unmanaged> > AVT;
        typedef Kokkos::View<typename PViewType::non_const_value_type*,
                typename PViewType::array_layout,
                typename PViewType::device_type,
                Kokkos::MemoryTraits<Kokkos::Unmanaged> > PVT;
        typedef Kokkos::View<typename TauViewType::non_const_value_type*,
                typename TauViewType::array_layout,
                typename TauViewType::device_type,
                Kokkos::MemoryTraits<Kokkos::Unmanaged> > TVT;
        typedef Impl::GEQP3<AVT, PVT, TVT> impl_type;
        impl_type::geqp3(A, p, tau);


    }

} //namespace KokkosBlas

#endif //KOKKOSBLAS_GEQP3_HPP_
