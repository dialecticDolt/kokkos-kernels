#ifndef KOKKOSBLAS_IMPL_HERK_HPP_
#define KOKKOSBLAS_IMPL_HERK_HPP_

#include <Kokkos_Core.hpp>
#include <Kokkos_ArithTraits.hpp>
#include <KokkosKernels_config.h>
#include <type_traits>
#include <sstream>

namespace KokkosBlas {
    namespace Impl{
        //Put non TPL implementation here

        template<class AVT, class CVT>
        void execute_herk(const char uplo, const char trans,
                          const int n, const int k, 
                          typename AVT::const_value_type& alpha,
                          AVT& A, 
                          typename CVT::const_value_type& beta, 
                          CVT& C){

            std::ostringstream os;
            os << "There is no kokkos implementation of Device level HERK. \n \
                   Use Team/Single Level or Compile with TPL (LAPACK or MAGMA).\n";
            Kokkos::Impl::throw_runtime_exception(os.str());
        }

    } //namespace Impl
} //namespace KokkosBlas

#endif //KOKKOSBLAS_IMPL_HERK_HPP_
