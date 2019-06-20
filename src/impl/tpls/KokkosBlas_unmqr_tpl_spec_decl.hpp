#ifndef KOKKOSBLAS_UNMQR_TPL_SPEC_DECL_HPP_
#define KOKKOSBLAS_UNMQR_TPL_SPEC_DECL_HPP_

#if defined( KOKKOSKERNELS_ENABLE_TPL_BLAS ) && defined( KOKKOSKERNELS_ENABLE_TPL_LAPACK )
#include "KokkosBlas_Host_tpl.hpp"
#include "KokkosLapack_Host_tpl.hpp"
#include <stdio.h>

namespace KokkosBlas {
    namespace Impl {

    #define KOKKOSBLAS_DUNMQR_BLAS(LAYOUTA, LAYOUTB, LAYOUTC, MEMSPACE, ETI_SPEC_AVAIL) \
    template<class ExecSpace> \
    struct UNMQR< \
        Kokkos::View<const double**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<const double*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<double**, LAYOUTC, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        true, ETI_SPEC_AVAIL> { \
        typedef double SCALAR; \
        typedef int ORDINAL; \
        typedef Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
        typedef Kokkos::View<const SCALAR*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > TauViewType; \
        typedef Kokkos::View<SCALAR**, LAYOUTC, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > CViewType; \
        \
        static void unmqr(char side, char trans, int k, AViewType& A, TauViewType& tau, CViewType& C){ \
        Kokkos::Profiling::pushRegion("KokkosLapack::unmqr[TPL_BLAS, double]");\
        int M = A.extent(0); \
        int N = A.extent(1); \
        bool A_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value; \
        bool C_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTC>::value; \
        const int AST = A_is_lr?A.stride(0):A.stride(1), LDA = AST == 0 ? 1:AST; \
        const int CST = C_is_lr?C.stride(0):C.stride(1), LDC = CST == 0 ? 1:CST; \
        HostLapack<double>::unmqr(A_is_lr, side, trans, M, N, k, A.data(), LDA, tau.data(), C.data(), LDC); \
        Kokkos::Profiling::popRegion(); \
        } \
    };

    #define KOKKOSBLAS_SUNMQR_BLAS(LAYOUTA, LAYOUTB, LAYOUTC, MEMSPACE, ETI_SPEC_AVAIL) \
    template<class ExecSpace> \
    struct UNMQR< \
        Kokkos::View<const float**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<const float*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<float**, LAYOUTC, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        true, ETI_SPEC_AVAIL> { \
        typedef float SCALAR; \
        typedef int ORDINAL; \
        typedef Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
        typedef Kokkos::View<const SCALAR*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > TauViewType; \
        typedef Kokkos::View<SCALAR**, LAYOUTC, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > CViewType; \
        \
        static void unmqr(char side, char trans, int k, AViewType& A, TauViewType& tau, CViewType& C){ \
        Kokkos::Profiling::pushRegion("KokkosLapack::unmqr[TPL_BLAS, float]");\
        int M = A.extent(0); \
        int N = A.extent(1); \
        bool A_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value; \
        bool C_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTC>::value; \
        const int AST = A_is_lr?A.stride(0):A.stride(1), LDA = AST == 0 ? 1:AST; \
        const int CST = C_is_lr?C.stride(0):C.stride(1), LDC = CST == 0 ? 1:CST; \
        HostLapack<float>::unmqr(A_is_lr, side, trans, M, N, k, A.data(), LDA, tau.data(), C.data(), LDC); \
        Kokkos::Profiling::popRegion(); \
        } \
    };

    #define KOKKOSBLAS_ZUNMQR_BLAS(LAYOUTA, LAYOUTB, LAYOUTC, MEMSPACE, ETI_SPEC_AVAIL) \
    template<class ExecSpace> \
    struct UNMQR< \
        Kokkos::View<const Kokkos::complex<double>**, \
                    LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<const Kokkos::complex<double>*, \
                    LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<Kokkos::complex<double>**, \
                    LAYOUTC, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        true, ETI_SPEC_AVAIL> { \
        typedef Kokkos::complex<double> SCALAR; \
        typedef int ORDINAL; \
        typedef Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
        typedef Kokkos::View<const SCALAR*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > TauViewType; \
        typedef Kokkos::View<SCALAR**, LAYOUTC, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > CViewType; \
        \
        static void unmqr(char side, char trans, int k, AViewType& A, TauViewType& tau, CViewType& C){ \
        Kokkos::Profiling::pushRegion("KokkosLapack::unmqr[TPL_BLAS, complex<double>]");\
        int M = A.extent(0); \
        int N = A.extent(1); \
        bool A_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value; \
        bool C_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTC>::value; \
        const int AST = A_is_lr?A.stride(0):A.stride(1), LDA = AST == 0 ? 1:AST; \
        const int CST = C_is_lr?C.stride(0):C.stride(1), LDC = CST == 0 ? 1:CST; \
        HostLapack<std::complex<double>>::unmqr(A_is_lr, side, trans, M, N, k, \
                reinterpret_cast<const std::complex<double>*>(A.data()), LDA, \
                reinterpret_cast<const std::complex<double>*>(tau.data()), \
                reinterpret_cast<std::complex<double>*>(C.data()), LDC); \
        Kokkos::Profiling::popRegion(); \
        } \
    };

    #define KOKKOSBLAS_CUNMQR_BLAS(LAYOUTA, LAYOUTB, LAYOUTC, MEMSPACE, ETI_SPEC_AVAIL) \
    template<class ExecSpace> \
    struct UNMQR< \
        Kokkos::View<const Kokkos::complex<float>**, \
                    LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<const Kokkos::complex<float>*, \
                    LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<Kokkos::complex<float>**, \
                    LAYOUTC, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        true, ETI_SPEC_AVAIL> { \
        typedef Kokkos::complex<float> SCALAR; \
        typedef int ORDINAL; \
        typedef Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
        typedef Kokkos::View<const SCALAR*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > TauViewType; \
        typedef Kokkos::View<SCALAR**, LAYOUTC, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > CViewType; \
        \
        static void unmqr(char side, char trans, int k, AViewType& A, TauViewType& tau, CViewType& C){ \
        Kokkos::Profiling::pushRegion("KokkosLapack::unmqr[TPL_BLAS, complex<float>]");\
        int M = A.extent(0); \
        int N = A.extent(1); \
        bool A_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value; \
        bool C_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTC>::value; \
        const int AST = A_is_lr?A.stride(0):A.stride(1), LDA = AST == 0 ? 1:AST; \
        const int CST = C_is_lr?C.stride(0):C.stride(1), LDC = CST == 0 ? 1:CST; \
        HostLapack<std::complex<float>>::unmqr(A_is_lr, side, trans, M, N, k, \
                reinterpret_cast<const std::complex<float>*>(A.data()), LDA, \
                reinterpret_cast<const std::complex<float>*>(tau.data()), \
                reinterpret_cast<std::complex<float>*>(C.data()), LDC); \
        Kokkos::Profiling::popRegion(); \
        } \
    };
 
    KOKKOSBLAS_DUNMQR_BLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, true)
    KOKKOSBLAS_DUNMQR_BLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, false)
    KOKKOSBLAS_DUNMQR_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true)
    KOKKOSBLAS_DUNMQR_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, false)

    KOKKOSBLAS_SUNMQR_BLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, true)
    KOKKOSBLAS_SUNMQR_BLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, false)
    KOKKOSBLAS_SUNMQR_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true)
    KOKKOSBLAS_SUNMQR_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, false)


    KOKKOSBLAS_ZUNMQR_BLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, true)
    KOKKOSBLAS_ZUNMQR_BLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, false)
    KOKKOSBLAS_ZUNMQR_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true)
    KOKKOSBLAS_ZUNMQR_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, false)


    KOKKOSBLAS_CUNMQR_BLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, true)
    KOKKOSBLAS_CUNMQR_BLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, false)
    KOKKOSBLAS_CUNMQR_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true)
    KOKKOSBLAS_CUNMQR_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, false)

    } // namespace Impl
} //namespace KokkosBlas

#endif //ENABLE BLAS/LAPACK


//Magma

/*
#ifdef KOKKOSKERNELS_ENABLE_TPL_MAGMA
#include<KokkosMagma_tpl_spec.hpp>

namespace KokkosBlas{
    namespace Impl{

    #define KOKKOSBLAS_DUNMQR_MAGMA(LAYOUTA, LAYOUTB, LAYOUTC, MEMSPACE, ETI_SPEC_AVAIL)


    } //namespace Impl
} //namespace KokkosBlas

*/

#endif //KOKKOSBLAS_UNMQR_TPL_SPEC_DECL_HPP_

