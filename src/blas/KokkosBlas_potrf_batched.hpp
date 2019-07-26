#ifndef KOKKOSBLAS_BATCHED_POTRF_HPP_
#define KOKKOSBLAS_BATCHED_POTRF_HPP_

#include <KokkosKernels_Macros.hpp>
#include "KokkosBlas_potrf_spec.hpp"
#include <KokkosKernels_helpers.hpp>
#include <sstream>
#include <type_traits>

namespace KokkosBlas {

    namespace Batched {
    /* 
     * \brief Dense POTRF (Batched) Top Looking Block Cholesky
     * \tparam uplo 'U' ( A= U' U ) or 'L' (A = LL')
     * \tparam A Input/Ouput list of matrix, as a Kokkos::View of 2-D Kokkos::View
     * 
     * \param A [in/out] Input/Output Matrix, as a 2-D Kokkos::View 
     * 
     */

        template<class AViewType>
        void potrf(const char uplo[], AViewType& A){

            #if (KOKKOSKERNELS_DEBUG_LEVEL >0)
            static_assert(Kokkos::Impl::is_view<AViewType>::value, "KokkosBlas::Batched::potrf: A must be a Kokkos::View of Kokkos::Views");
            static_assert(static_cast<int> (AViewType::value_type::rank)==2, "KokkosBlas::Batched::potrf: Each element of A must have rank 2");
            static_assert(static_cast<int> (AViewType::rank) == 1, "KokkosBlas::Batched::potrf: A must have rank 1");
            int64_t A0 = A.extent(0);
            int64_t A1 = A.extent(1);

            //TODO: Add runtime checking of flags
            
            const char ul = uplo[0];
            bool valid_uplo = (ul == 'U' || ul == 'u') ||
                              (ul == 'L' || ul == 'l');

            if (!(valid_uplo)){
                std::ostringstream os;
                os << "KokkosBlas::Batched::potrf : side[0] = '" << uplo[0] << "'. " <<
                "Valid values include 'U' or 'u' (Upper), 'L' or 'l' (Lower).";
                Kokkos::Impl::throw_runtime_exception (os.str ());
            }

            //TODO: Add runtime dimension checks

            #endif //KOKKOSKERNELS_DEBUG_LEVEL > 0 

            //return if degenerate matrix provided
            if((A.extent(0)==0) || (A.extent(1)==0))
                return;

            //particular View specializations
            typedef Kokkos::View<
                                    Kokkos::View< typename AViewType::value_type::value_type**,
                                                  typename AViewType::array_layout,
                                                  typename AViewType::device_type,
                                                  Kokkos::MemoryTraits<Kokkos::Unmanaged> 
                                                >,
                                    typename AViewType::array_layout,
                                    typename AViewType::device_type,
                                    Kokkos::MemoryTraits<Kokkos::Unmanaged>
                                > AVT;

            AVT A_i = A;
            
            typedef KokkosBlas::Batched::Impl::POTRF_BATCHED<AVT> impl_type;
            impl_type::potrf_batched(ul, A_i);

        }

/*
    #if defined (KOKKOS_ENABLE_CUDA)
        template<class AViewType>
        void potrf(const char uplo[], AViewType& A, cudaStream_t stream){


            #if (KOKKOSKERNELS_DEBUG_LEVEL >0)
            static_assert(Kokkos::Impl::is_view<AViewType>::value, "KokkosBlas::Batched::potrf: A must be a Kokkos::View of Kokkos::Views");
            static_assert(static_cast<int> (AViewType::value_type::rank)==2, "KokkosBlas::Batched::potrf: Each element of A must have rank 2");
            
            int64_t A0 = A.extent(0);
            int64_t A1 = A.extent(1);

            //TODO: Add runtime checking of flags
            
            const char ul = uplo[0];
            bool valid_uplo = (ul == 'U' || ul == 'u') ||
                              (ul == 'L' || ul == 'l');

            if (!(valid_side)){
                std::ostringstream os;
                os << "KokkosBlas::Batched::potrf : side[0] = '" << uplo[0] << "'. " <<
                "Valid values include 'U' or 'u' (Upper), 'L' or 'l' (Lower).";
                Kokkos::Impl::throw_runtime_exception (os.str ());
            }

            //TODO: Add runtime dimension checks

            #endif //KOKKOSKERNELS_DEBUG_LEVEL > 0 

            //return if degenerate matrix provided
            if((A.extent(0)==0) || (A.extent(1)==0))
                return;

            //particular View specializations
            typedef Kokkos::View<typename AViewType::non_const_value_type*,
                    typename AViewType::array_layout,
                    typename AViewType::device_type,
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> > AVT;

            AVT A_i = A;
            
            typedef KokkosBlas::Impl::POTRF_BATCHED<AVT> impl_type;
            impl_type::potrf_batched(ul, A_i, stream);

        }

    #endif

*/


    } //namespace Batched
} //namespace KokkosBlas

#endif //KOKKOSBLAS_BATCHED_POTRF_HPP_
