#ifndef KOKKOSBLAS_BATCHED_HELPERS_
#define KOKKOSBLAS_BATCHED_HELPERS_


#include <KokkosKernels_Macros.hpp>
#include <KokkosKernels_helpers.hpp>
#include <type_traits>

namespace KokkosBlas{
    namespace Batched {

        template<class ViewType>
        typename ViewType::value_type::value_type** convertView(ViewType V){
            static_assert(Kokkos::Impl::is_view<ViewType>::value, "KokkosBlas::Batched::convertView :  V must be a Kokkos::View");
            static_assert(Kokkos::Impl::is_view<typename ViewType::value_type>::value, "KokkosBlas::Batched::convertView :  V must be a Kokkos:View of Kokkos::Views");
            static_assert(static_cast<int>(ViewType::rank)==1, "KokkosBlas::Batched::convertView : V must be a 1D View of Views");
            static_assert(static_cast<int>(ViewType::value_type::rank)==2, "Kokkos::Batched::convertView : V must store 2D Kokkos Views");
            
            const int N = V.extent(0);

            typedef typename ViewType::value_type::value_type numeric_t;
            typedef typename ViewType::execution_space ExecSpace; 
            numeric_t** outer;
            outer = (numeric_t**) Kokkos::kokkos_malloc(sizeof(numeric_t)*N);

            Kokkos::parallel_for(N, KOKKOS_LAMBDA(const int i){
                        outer[i] = V(i).data();
                    }); 

            return outer;
        }


    }//namespace Batched

} //namespace KokkosBlas



#endif //KOKKOSBLAS_BATCHED_HELPERS_
