#ifndef KOKKOSBLAS_IMPL_GEQP3_HPP_
#define KOKKOSBLAS_IMPL_GEQP3_HPP_

#include <Kokkos_Core.hpp>
#include <Kokkos_ArithTraits.hpp>
#include <KokkosKernels_config.h>
#include <type_traits>
#include <sstream>

namespace KokkosBlas {
    namespace Impl{
        //Put non TPL implementation here
        
        template<class AVT, class PVT, class TVT>
        void execute_geqp3(AVT& A, PVT& p, TVT& t){
            std::ostringstream os;
            os << "There is no ETI implementation of GEQP3. Compile with TPL (LAPACK or MAGMA).\n";
            Kokkos::Impl::throw_runtime_exception(os.str());
        }

    } //namespace Impl
} //namespace KokkosBlas

#endif //KOKKOSBLAS_IMPL_GEQP3_HPP_
