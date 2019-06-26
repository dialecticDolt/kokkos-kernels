#ifndef KOKKOSBLAS_IMPL_POTRF_HPP_
#define KOKKOSBLAS_IMPL_POTRF_HPP_

#include <Kokkos_Core.hpp>
#include <Kokkos_ArithTraits.hpp>
#include <KokkosKernels_config.h>
#include <type_traits>
#include <sstream>

namespace KokkosBlas {
    namespace Impl{
        //Put non TPL implementation here

        template<class AVT>
        void execute_potrf(const char uplo, AVT& A){

            std::ostringstream os;
            os << "There is no kokkos implementation of Device level POTRF. \n \
                   Use Team/Single Level or Compile with TPL (LAPACK or MAGMA).\n";
            Kokkos::Impl::throw_runtime_exception(os.str());
        }

    } //namespace Impl
} //namespace KokkosBlas

#endif //KOKKOSBLAS_IMPL_POTRF_HPP_
