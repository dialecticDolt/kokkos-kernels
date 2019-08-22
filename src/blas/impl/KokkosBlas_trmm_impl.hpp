#ifndef KOKKOSBLAS_IMPL_TRMM_HPP_
#define KOKKOSBLAS_IMPL_TRMM_HPP_

#include <Kokkos_Core.hpp>
#include <Kokkos_ArithTraits.hpp>
#include <KokkosKernels_config.h>
#include <type_traits>
#include <sstream>

namespace KokkosBlas {
    namespace Impl{
        //Put non TPL implementation here

        template<class AVT, class BVT>
        void execute_trmm(const char side, const char uplo,
                           const char transa, const char diag, 
                           typename AVT::const_value_type& a,
                           AVT& A, BVT& B){

            std::ostringstream os;
            os << "There is no kokkos implementation of Device level TRMM. \n \
                   Use Team/Single Level or Compile with TPL (LAPACK or MAGMA).\n";
            Kokkos::Impl::throw_runtime_exception(os.str());
        }

    } //namespace Impl
} //namespace KokkosBlas

#endif //KOKKOSBLAS_IMPL_TRMM_HPP_
