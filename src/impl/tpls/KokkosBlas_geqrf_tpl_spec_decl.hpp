#ifndef KOKKOSBLAS_GEQRF_TPL_SPEC_DECL_HPP_
#define KOKKOSBLAS_GEQRF_TPL_SPEC_DECL_HPP_

#if defined( KOKKOSKERNELS_ENABLE_TPL_BLAS ) && defined( KOKKOSKERNELS_ENABLE_TPL_LAPACK )
#include "KokkosBlas_Host_tpl.hpp"
#include "KokkosLapack_Host_tpl.hpp"
#include <stdio.h>

namespace KokkosBlas {
    namespace Impl {

    #define KOKKOSBLAS_DGEQRF_LAPACK(LAYOUTA, LAYOUTB, LAYOUTC, MEMSPACE, ETI_SPEC_AVAIL) \
    template<class ExecSpace> \
    struct GEQRF< \
        Kokkos::View<double**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<double*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<double*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        true, ETI_SPEC_AVAIL> { \
        typedef double SCALAR; \
        typedef int ORDINAL; \
        typedef Kokkos::View<SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
        typedef Kokkos::View<SCALAR*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > TauViewType; \
        typedef Kokkos::View<SCALAR*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > WViewType; \
        \
        static void geqrf(AViewType& A, TauViewType& tau, WViewType& workspace){ \
        Kokkos::Profiling::pushRegion("KokkosLapack::geqrf[TPL_LAPACK, double]");\
        int M = C.extent(0); \
        int N = C.extent(1); \
        bool A_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value; \
        bool C_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTC>::value; \
        const int AST = A_is_lr?A.stride(0):A.stride(1), LDA = AST == 0 ? 1:AST; \
        const int CST = C_is_lr?C.stride(0):C.stride(1), LDC = CST == 0 ? 1:CST; \
        const int lwork = workspace.extent(0); \
        HostLapack<double>::geqrf(M, N, A.data(), LDA, tau.data(), workspace.data(), lwork); \
        Kokkos::Profiling::popRegion(); \
        } \
    };

    #define KOKKOSBLAS_SGEQRF_LAPACK(LAYOUTA, LAYOUTB, LAYOUTC, MEMSPACE, ETI_SPEC_AVAIL) \
    template<class ExecSpace> \
    struct GEQRF< \
        Kokkos::View<float**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<float*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<float*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        true, ETI_SPEC_AVAIL> { \
        typedef float SCALAR; \
        typedef int ORDINAL; \
        typedef Kokkos::View<SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
        typedef Kokkos::View<SCALAR*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > TauViewType; \
        typedef Kokkos::View<SCALAR*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > WViewType; \
        \
        static void geqrf(AViewType& A, TauViewType& tau, WViewType& workspace){ \
        Kokkos::Profiling::pushRegion("KokkosLapack::geqrf[TPL_LAPACK, float]");\
        int M = C.extent(0); \
        int N = C.extent(1); \
        bool A_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value; \
        bool C_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTC>::value; \
        const int AST = A_is_lr?A.stride(0):A.stride(1), LDA = AST == 0 ? 1:AST; \
        const int CST = C_is_lr?C.stride(0):C.stride(1), LDC = CST == 0 ? 1:CST; \
        const int lwork = workspace.extent(0); \
        HostLapack<float>::geqrf(M, N, A.data(), LDA, tau.data(), workspace.data(), lwork); \
        Kokkos::Profiling::popRegion(); \
        } \
    };

    #define KOKKOSBLAS_ZGEQRF_LAPACK(LAYOUTA, LAYOUTB, LAYOUTC, MEMSPACE, ETI_SPEC_AVAIL) \
    template<class ExecSpace> \
    struct GEQRF< \
        Kokkos::View<Kokkos::complex<double>**, \
                    LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<Kokkos::complex<double>*, \
                    LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<Kokkos::complex<double>*, \
                    LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        true, ETI_SPEC_AVAIL> { \
        typedef Kokkos::complex<double> SCALAR; \
        typedef int ORDINAL; \
        typedef Kokkos::View<SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
        typedef Kokkos::View<SCALAR*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > TauViewType; \
        typedef Kokkos::View<SCALAR*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > WViewType; \
        static void geqrf(AViewType& A, TauViewType& tau, WViewType& workspace){ \
        Kokkos::Profiling::pushRegion("KokkosLapack::geqrf[TPL_LAPACK, complex<double>]");\
        int M = C.extent(0); \
        int N = C.extent(1); \
        bool A_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value; \
        bool C_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTC>::value; \
        const int AST = A_is_lr?A.stride(0):A.stride(1), LDA = AST == 0 ? 1:AST; \
        const int CST = C_is_lr?C.stride(0):C.stride(1), LDC = CST == 0 ? 1:CST; \
        const int lwork = workspace.extent(0); \
        HostLapack<std::complex<double>>::geqrf(M, N,  \
                reinterpret_cast<std::complex<double>*>(A.data()), LDA, \
                reinterpret_cast<std::complex<double>*>(tau.data()), \
                reinterpret_cast<std::complex<double>*>(workspace.data()), lwork); \
        Kokkos::Profiling::popRegion(); \
        } \
    };

    #define KOKKOSBLAS_CGEQRF_LAPACK(LAYOUTA, LAYOUTB, LAYOUTC, MEMSPACE, ETI_SPEC_AVAIL) \
    template<class ExecSpace> \
    struct GEQRF< \
        Kokkos::View<Kokkos::complex<float>**, \
                    LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<Kokkos::complex<float>*, \
                    LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<Kokkos::complex<float>*, \
                    LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        true, ETI_SPEC_AVAIL> { \
        typedef Kokkos::complex<float> SCALAR; \
        typedef int ORDINAL; \
        typedef Kokkos::View<SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
        typedef Kokkos::View<SCALAR*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > TauViewType; \
        typedef Kokkos::View<SCALAR*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > WViewType; \
        \
        static void geqrf(AViewType& A, TauViewType& tau, WViewType& workspace){ \
        Kokkos::Profiling::pushRegion("KokkosLapack::geqrf[TPL_LAPACK, complex<float>]");\
        int M = C.extent(0); \
        int N = C.extent(1); \
        bool A_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value; \
        bool C_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTC>::value; \
        const int AST = A_is_lr?A.stride(0):A.stride(1), LDA = AST == 0 ? 1:AST; \
        const int CST = C_is_lr?C.stride(0):C.stride(1), LDC = CST == 0 ? 1:CST; \
        const int lwork = workspace.extent(0); \
        HostLapack<std::complex<float>>::geqrf(M, N, \
                reinterpret_cast<std::complex<float>*>(A.data()), LDA, \
                reinterpret_cast<std::complex<float>*>(tau.data()), \
                reinterpret_cast<std::complex<float>*>(workspace.data()), lwork); \
        Kokkos::Profiling::popRegion(); \
        } \
    };
 
    KOKKOSBLAS_DGEQRF_LAPACK(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, true)
    KOKKOSBLAS_DGEQRF_LAPACK(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, false)
    KOKKOSBLAS_DGEQRF_LAPACK(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true)
    KOKKOSBLAS_DGEQRF_LAPACK(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, false)

    KOKKOSBLAS_SGEQRF_LAPACK(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, true)
    KOKKOSBLAS_SGEQRF_LAPACK(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, false)
    KOKKOSBLAS_SGEQRF_LAPACK(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true)
    KOKKOSBLAS_SGEQRF_LAPACK(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, false)


    KOKKOSBLAS_ZGEQRF_LAPACK(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, true)
    KOKKOSBLAS_ZGEQRF_LAPACK(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, false)
    KOKKOSBLAS_ZGEQRF_LAPACK(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true)
    KOKKOSBLAS_ZGEQRF_LAPACK(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, false)


    KOKKOSBLAS_CGEQRF_LAPACK(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, true)
    KOKKOSBLAS_CGEQRF_LAPACK(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, false)
    KOKKOSBLAS_CGEQRF_LAPACK(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true)
    KOKKOSBLAS_CGEQRF_LAPACK(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, false)

    } // namespace Impl
} //namespace KokkosBlas

#endif //ENABLE BLAS/LAPACK


//CUSOLVER

#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSOLVER
#include<KokkosBlas_tpl_spec.hpp>

namespace KokkosBlas{
    namespace Impl{

    #define KOKKOSBLAS_DGEQRF_CUSOLVER(LAYOUTA, LAYOUTB, LAYOUTC, MEMSPACE, ETI_SPEC_AVAIL) \
    template<class ExecSpace> \
    struct GEQRF< \
        Kokkos::View<double**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<double*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<double*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        true, ETI_SPEC_AVAIL> { \
        typedef double SCALAR; \
        typedef int ORDINAL; \
        typedef Kokkos::View<SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
        typedef Kokkos::View<SCALAR*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > TauViewType; \
        typedef Kokkos::View<SCALAR*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > WViewType; \
        \
        static void geqrf(AViewType& A, TauViewType& tau, WViewType& workspace){ \
        Kokkos::Profiling::pushRegion("KokkosBlas::geqrf[TPL_CUSOLVER, double]");\
        int devinfo = 0; \
        int M = C.extent(0);  \
        int N = C.extent(1);  \
        bool A_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value; \
        bool C_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTC>::value; \
        const int AST = A_is_lr?A.stride(0):A.stride(1), LDA = AST == 0 ? 1:AST; \
        const int CST = C_is_lr?C.stride(0):C.stride(1), LDC = CST == 0 ? 1:CST; \
        KokkosBlas::Impl::CudaSolverSingleton & s = KokkosBlas::Impl::CudaSolverSingleton::singleton(); \
        cusolverDnDgeqrf(s.handle, M, N, A.data(), LDA, tau.data(), workspace.data(), lwork, &devinfo); \
        Kokkos::Profiling::popRegion(); \
        } \
    };


    #define KOKKOSBLAS_SGEQRF_CUSOLVER(LAYOUTA, LAYOUTB, LAYOUTC, MEMSPACE, ETI_SPEC_AVAIL) \
    template<class ExecSpace> \
    struct GEQRF< \
        Kokkos::View<float**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<float*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<float*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        true, ETI_SPEC_AVAIL> { \
        typedef float SCALAR; \
        typedef int ORDINAL; \
        typedef Kokkos::View<SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
        typedef Kokkos::View<SCALAR*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > TauViewType; \
        typedef Kokkos::View<SCALAR*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > WViewType; \
        \
        static void geqrf(AViewType& A, TauViewType& tau, WViewType& workspace){ \
        Kokkos::Profiling::pushRegion("KokkosBlas::geqrf[TPL_CUSOLVER, double]");\
        int devinfo = 0; \
        int M = C.extent(0);  \
        int N = C.extent(1);  \
        bool A_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value; \
        bool C_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTC>::value; \
        KokkosBlas::Impl::CudaSolverSingleton & s = KokkosBlas::Impl::CudaSolverSingleton::singleton(); \
        cusolverDnSgeqrf(s.handle, M, N, A.data(), LDA, tau.data(), workspace.data(), lwork, &devinfo); \
        Kokkos::Profiling::popRegion(); \
        } \
    };

    #define KOKKOSBLAS_ZGEQRF_CUSOLVER(LAYOUTA, LAYOUTB, LAYOUTC, MEMSPACE, ETI_SPEC_AVAIL) \
    template<class ExecSpace> \
    struct GEQRF< \
        Kokkos::View<Kokkos::complex<double>**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<Kokkos::complex<double>*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<Kokkos::complex<double>*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        true, ETI_SPEC_AVAIL> { \
        typedef Kokkos::complex<double> SCALAR; \
        typedef double PRECISION; \
        typedef int ORDINAL; \
        typedef Kokkos::View<SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
        typedef Kokkos::View<SCALAR*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > TauViewType; \
        typedef Kokkos::View<SCALAR*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > WViewType; \
        \
        static void geqrf(AViewType& A, TauViewType& tau, WViewType& workspace){ \
        Kokkos::Profiling::pushRegion("KokkosBlas::geqrf[TPL_CUSOLVER, Kokkos::complex<double>]");\
        int devinfo = 0; \
        int M = C.extent(0);  \
        int N = C.extent(1);  \
        bool A_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value; \
        bool C_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTC>::value; \
        const int AST = A_is_lr?A.stride(0):A.stride(1), LDA = AST == 0 ? 1:AST; \
        const int CST = C_is_lr?C.stride(0):C.stride(1), LDC = CST == 0 ? 1:CST; \
        KokkosBlas::Impl::CudaSolverSingleton & s = KokkosBlas::Impl::CudaSolverSingleton::singleton(); \
        cusolverDnZgeqrf(s.handle, m_side, m_trans, M, N, k, \
            reinterpret_cast<cuDoubleComplex*>(A.data()), LDA, \
            reinterpret_cast<cuDoubleComplex*>(tau.data()), \
            reinterpret_cast<cuDoubleComplex*>(workspace.data()), \
            lwork, &devinfo); \
        Kokkos::Profiling::popRegion(); \
        } \
    };

    #define KOKKOSBLAS_ZGEQRF_CUSOLVER(LAYOUTA, LAYOUTB, LAYOUTC, MEMSPACE, ETI_SPEC_AVAIL) \
    template<class ExecSpace> \
    struct GEQRF< \
        Kokkos::View<Kokkos::complex<float>**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<Kokkos::complex<float>*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<Kokkos::complex<float>*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        true, ETI_SPEC_AVAIL> { \
        typedef Kokkos::complex<float> SCALAR; \
        typedef float PRECISION; \
        typedef int ORDINAL; \
        typedef Kokkos::View<SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
        typedef Kokkos::View<SCALAR*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > TauViewType; \
        typedef Kokkos::View<SCALAR*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > WViewType; \
        \
        static void geqrf(AViewType& A, TauViewType& tau, WViewType& workspace){ \
        Kokkos::Profiling::pushRegion("KokkosBlas::geqrf[TPL_CUSOLVER, Kokkos::complex<float>]");\
        int devinfo = 0; \
        int M = C.extent(0);  \
        int N = C.extent(1);  \
        bool A_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value; \
        bool C_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTC>::value; \
        const int AST = A_is_lr?A.stride(0):A.stride(1), LDA = AST == 0 ? 1:AST; \
        const int CST = C_is_lr?C.stride(0):C.stride(1), LDC = CST == 0 ? 1:CST; \
        KokkosBlas::Impl::CudaSolverSingleton & s = KokkosBlas::Impl::CudaSolverSingleton::singleton(); \
        cusolverDnCgeqrf(s.handle, m_side, m_trans, M, N, k, \
            reinterpret_cast<cuComplex*>(A.data()), LDA, \
            reinterpret_cast<cuComplex*>(tau.data()), \
            reinterpret_cast<cuComplex*>(workspace.data()), \
            lwork, &devinfo); \
        Kokkos::Profiling::popRegion(); \
        } \
    };

    KOKKOSBLAS_DGEQRF_CUSOLVER(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
    KOKKOSBLAS_DGEQRF_CUSOLVER(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaSpace, false)


    KOKKOSBLAS_SGEQRF_CUSOLVER(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
    KOKKOSBLAS_SGEQRF_CUSOLVER(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaSpace, false)

    KOKKOSBLAS_ZGEQRF_CUSOLVER(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
    KOKKOSBLAS_ZGEQRF_CUSOLVER(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaSpace, false)

    KOKKOSBLAS_CGEQRF_CUSOLVER(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
    KOKKOSBLAS_CGEQRF_CUSOLVER(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaSpace, false)

    } //namespace Impl
} //namespace KokkosBlas

#endif //IF CUSOLVER && CUBLAS

#endif //KOKKOSBLAS_GEQRF_TPL_SPEC_DECL_HPP_

