#ifndef KOKKOS_LAPACK_GEQP3_HPP_
#define KOKKOS_LAPACK_GEQP3_HPP_

#include <Kokkos_Core.hpp>
#include <sstream>

namespace KokkosLapack {
    namespace Impl{
        //Put non TPL implementation here
        
        template<class AVT, class PVT, class TVT>
        void execute_geqp3(AVT& A, PVT& p, TVT& t){
            std::ostringstream os;
            os << "There is no ETI implemenation of GEQP3. Compile with TPL.\n";
            Kokkos::Impl::throw_runtime_exceptions(os.str());
        }

    }
}

#endif
