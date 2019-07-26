#ifndef KOKKOSBLAS_IMPL_BATCHED_POTRF_HPP_
#define KOKKOSBLAS_IMPL_BATCHED_POTRF_HPP_

#include <Kokkos_Core.hpp>
#include <Kokkos_ArithTraits.hpp>
#include <KokkosKernels_config.h>
#include <type_traits>
#include <sstream>

namespace KokkosBlas {
    namespace Impl{
        //Put non TPL implementation here

        template<class AVT>
        void execute_potrf_batched(const char uplo, AVT& A){

            std::ostringstream os;
            os << "There is no kokkos implementation of Batched POTRF. \n \
                   Compile with TPL (MAGMA).\n";
            Kokkos::Impl::throw_runtime_exception(os.str());
        }

    } //namespace Impl
} //namespace KokkosBlas

#endif //KOKKOSBLAS_IMPL_BATCHED_POTRF_HPP_
