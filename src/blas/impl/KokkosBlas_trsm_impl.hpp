#ifndef KOKKOSBLAS_IMPL_TRSM_HPP_
#define KOKKOSBLAS_IMPL_TRSM_HPP_

#include <Kokkos_Core.hpp>
#include <Kokkos_ArithTraits.hpp>
#include <KokkosKernels_config.h>
#include <type_traits>
#include <sstream>

namespace KokkosBlas {
    namespace Impl{
        //Put non TPL implementation here

        template<class AVT, class BVT>
        void execute_unmqr(const char side, const char uplo,
                           const char transa, const char diag, 
                           AVT::non_const_value_type a,
                           AVT& A, BVT& B){

            std::ostringstream os;
            os << "There is no kokkos implementation of Device level TRSM. \n
                   Use Team/Single Level or Compile with TPL (LAPACK or MAGMA).\n";
            Kokkos::Impl::throw_runtime_exception(os.str());
        }

    } //namespace Impl
} //namespace KokkosBlas

#endif //KOKKOSBLAS_IMPL_TRSM_HPP_
