#ifndef KOKKOSBLAS_GEQP3_TPL_SPEC_DECL_HPP_
#define KOKKOSBLAS_GEQP3_TPL_SPEC_DECL_HPP_

#include <stdio.h>

#if defined( KOKKOSKERNELS_ENABLE_TPL_BLAS ) && defined( KOKKOSKERNELS_ENABLE_TPL_LAPACK )
#include "KokkosBlas_Host_tpl.hpp"
#include "KokkosLapack_Host_tpl.hpp"

namespace KokkosBlas {
    namespace Impl {


    #define KOKKOSBLAS_DGEQP3_BLAS(LAYOUTA, LAYOUTB, LAYOUTC, MEMSPACE, ETI_SPEC_AVAIL) \
    template<class ExecSpace> \
    struct GEQP3< \
        Kokkos::View<double**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<int*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<double*, LAYOUTC, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        true, ETI_SPEC_AVAIL> { \
        typedef double SCALAR; \
        typedef int ORDINAL; \
        typedef Kokkos::View<SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
        typedef Kokkos::View<int*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > PViewType; \
        typedef Kokkos::View<SCALAR*, LAYOUTC, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > TauViewType; \
        \
        static void geqp3(AViewType& A, PViewType& p, TauViewType& tau){ \
        Kokkos::Profiling::pushRegion("KokkosLapack::geqp3[TPL_BLAS, double]");\
        int M = A.extent(0); \
        int N = A.extent(1); \
        bool A_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value; \
        const int AST = A_is_lr?A.stride(0):A.stride(1), LDA = AST == 0 ? 1:AST; \
        HostLapack<double>::geqp3(A_is_lr, M, N, A.data(), LDA, p.data(), tau.data()); \
        Kokkos::Profiling::popRegion(); \
        } \
    };


    #define KOKKOSBLAS_SGEQP3_BLAS(LAYOUTA, LAYOUTB, LAYOUTC, MEMSPACE, ETI_SPEC_AVAIL) \
    template<class ExecSpace> \
    struct GEQP3< \
        Kokkos::View<float**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<int*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<float*, LAYOUTC, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        true, ETI_SPEC_AVAIL> { \
        typedef float SCALAR; \
        typedef int ORDINAL; \
        typedef Kokkos::View<SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
        typedef Kokkos::View<int*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > PViewType; \
        typedef Kokkos::View<SCALAR*, LAYOUTC, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > TauViewType; \
        \
        static void geqp3(AViewType& A, PViewType& p, TauViewType& tau){ \
        Kokkos::Profiling::pushRegion("KokkosLapack::geqp3[TPL_BLAS, float]");\
        const int M = A.extent(0); \
        const int N = A.extent(1); \
        bool A_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value; \
        \
        const int AST = A_is_lr?A.stride(0):A.stride(1), LDA = AST == 0 ? 1:AST; \
        HostLapack<float>::geqp3(A_is_lr, M, N,A.data(), LDA, p.data(), tau.data()); \
        Kokkos::Profiling::popRegion(); \
        } \
    };


    #define KOKKOSBLAS_ZGEQP3_BLAS(LAYOUTA, LAYOUTB, LAYOUTC, MEMSPACE, ETI_SPEC_AVAIL) \
    template<class ExecSpace> \
    struct GEQP3< \
        Kokkos::View<Kokkos::complex<double>**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<int*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<Kokkos::complex<double>*, LAYOUTC, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        true, ETI_SPEC_AVAIL> { \
        typedef Kokkos::complex<double> SCALAR; \
        typedef int ORDINAL; \
        typedef Kokkos::View<SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
        typedef Kokkos::View<int*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > PViewType; \
        typedef Kokkos::View<SCALAR*, LAYOUTC, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > TauViewType; \
        \
        static void geqp3(AViewType& A, PViewType& p, TauViewType& tau){ \
        Kokkos::Profiling::pushRegion("KokkosLapack::geqp3[TPL_BLAS, Complex<double>]");\
        const int M = A.extent(0); \
        const int N = A.extent(1); \
        bool A_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value; \
        \
        const int AST = A_is_lr?A.stride(0):A.stride(1), LDA = AST == 0 ? 1:AST; \
        HostLapack<std::complex<double>>::geqp3(A_is_lr, M, N, reinterpret_cast<std::complex<double>*>(A.data()), LDA,        \
                                                p.data(), reinterpret_cast<std::complex<double>*>(tau.data())                 \
                                                );                                                                            \
        Kokkos::Profiling::popRegion(); \
        } \
    };


    #define KOKKOSBLAS_CGEQP3_BLAS(LAYOUTA, LAYOUTB, LAYOUTC, MEMSPACE, ETI_SPEC_AVAIL) \
    template<class ExecSpace> \
    struct GEQP3< \
        Kokkos::View<Kokkos::complex<float>**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<int*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<Kokkos::complex<float>*, LAYOUTC, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        true, ETI_SPEC_AVAIL> { \
        typedef Kokkos::complex<float> SCALAR; \
        typedef int ORDINAL; \
        typedef Kokkos::View<SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
        typedef Kokkos::View<int*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > PViewType; \
        typedef Kokkos::View<SCALAR*, LAYOUTC, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > TauViewType; \
        \
        static void geqp3(AViewType& A, PViewType& p, TauViewType& tau){ \
        Kokkos::Profiling::pushRegion("KokkosLapack::geqp3[TPL_BLAS, Complex<float>]");\
        const int M = A.extent(0); \
        const int N = A.extent(1); \
        bool A_is_lr = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value; \
        \
        const int AST = A_is_lr?A.stride(0):A.stride(1), LDA = AST == 0 ? 1:AST; \
        HostLapack<std::complex<float>>::geqp3(A_is_lr, M, N, reinterpret_cast<std::complex<float>*>(A.data()), LDA,        \
                                                p.data(), reinterpret_cast<std::complex<float>*>(tau.data())                 \
                                                );                                                                            \
        Kokkos::Profiling::popRegion(); \
        } \
    };

    KOKKOSBLAS_DGEQP3_BLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, true)
    KOKKOSBLAS_DGEQP3_BLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, false)
    KOKKOSBLAS_DGEQP3_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true)
    KOKKOSBLAS_DGEQP3_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, false)

    KOKKOSBLAS_SGEQP3_BLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, true)
    KOKKOSBLAS_SGEQP3_BLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, false)
    KOKKOSBLAS_SGEQP3_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true)
    KOKKOSBLAS_SGEQP3_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, false)


    KOKKOSBLAS_ZGEQP3_BLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, true)
    KOKKOSBLAS_ZGEQP3_BLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, false)
    KOKKOSBLAS_ZGEQP3_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true)
    KOKKOSBLAS_ZGEQP3_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, false)


    KOKKOSBLAS_CGEQP3_BLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, true)
    KOKKOSBLAS_CGEQP3_BLAS(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, false)
    KOKKOSBLAS_CGEQP3_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true)
    KOKKOSBLAS_CGEQP3_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, false)

    } // namespace Impl
} //namespace KokkosBlas

#endif //ENABLE BLAS/LAPACK

//Magma

#ifdef KOKKOSKERNELS_ENABLE_TPL_MAGMA
#include<KokkosBlas_tpl_spec.hpp>

namespace KokkosBlas{
    namespace Impl{

    #define KOKKOSBLAS_DGEQP3_MAGMA(LAYOUTA, LAYOUTB, LAYOUTC, MEMSPACE, ETI_SPEC_AVAIL) \
    template<class ExecSpace> \
    struct GEQP3< \
        Kokkos::View<double**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<int*, Kokkos::HostSpace, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<double*, LAYOUTC, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        true, ETI_SPEC_AVAIL> { \
        typedef double SCALAR; \
        typedef Kokkos::View<SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
        typedef Kokkos::View<int*, Kokkos::HostSpace, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > PViewType; \
        typedef Kokkos::View<SCALAR*, LAYOUTC, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > TauViewType; \
        static void geqp3(AViewType& A, PViewType& p, TauViewType& tau){ \
            Kokkos::Profiling::pushRegion("KokkosLapack::geqp3[TPL_MAGMA, double]"); \
            KokkosBlas::Impl::MagmaSingleton & s = KokkosBlas::Impl::MagmaSingleton::singleton();\
            magma_int_t N = static_cast<magma_int_t>( A.extent(0) ); \
            magma_int_t M = static_cast<magma_int_t>( A.extent(1) ); \
            magma_int_t AST = static_cast<magma_int_t>( A.stride(1) ); \
            magma_int_t LDA = (AST == 0) ? 1: AST; \
            magma_int_t nb = magma_get_dgeqp3_nb(M, N); \
            magma_int_t lwork = (N+1)*nb + 2*N; \
            Kokkos::View<SCALAR*, Kokkos::Device<ExecSpace, MEMSPACE> > dwork("dwork", lwork); \
            magma_int_t info = 0; \
            magma_dgeqp3_gpu(M, N, \
                             reinterpret_cast<magmaDouble_ptr>( A.data() ), LDA,\
                             reinterpret_cast<magma_int_t*>( p.data() ), \
                             reinterpret_cast<magmaDouble_ptr>( tau.data() ), \
                             reinterpret_cast<magmaDouble_ptr>( dwork.data() ), lwork, \
                             &info);\
            Kokkos::Profiling::popRegion(); \
        } \
    };

    #define KOKKOSBLAS_SGEQP3_MAGMA(LAYOUTA, LAYOUTB, LAYOUTC, MEMSPACE, ETI_SPEC_AVAIL) \
    template<class ExecSpace> \
    struct GEQP3< \
        Kokkos::View<float**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<int*, Kokkos::HostSpace, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<float*, LAYOUTC, Kokkos::Device<ExecSpace, MEMSPACE>, \
                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        true, ETI_SPEC_AVAIL> { \
        typedef float SCALAR; \
        typedef Kokkos::View<SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
        typedef Kokkos::View<int*, Kokkos::HostSpace, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > PViewType; \
        typedef Kokkos::View<SCALAR*, LAYOUTC, Kokkos::Device<ExecSpace, MEMSPACE>, \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> > TauViewType; \
        static void geqp3(AViewType& A, PViewType& p, TauViewType& tau){ \
            Kokkos::Profiling::pushRegion("KokkosLapack::geqp3[TPL_MAGMA, float]"); \
            KokkosBlas::Impl::MagmaSingleton & s = KokkosBlas::Impl::MagmaSingleton::singleton();\
            magma_int_t N = static_cast<magma_int_t>( A.extent(0) ); \
            magma_int_t M = static_cast<magma_int_t>( A.extent(1) ); \
            magma_int_t AST = static_cast<magma_int_t>( A.stride(1) ); \
            magma_int_t LDA = (AST == 0) ? 1: AST; \
            magma_int_t nb = magma_get_sgeqp3_nb(M, N); \
            magma_int_t lwork = (N+1)*nb + 2*N; \
            Kokkos::View<SCALAR*, Kokkos::Device<ExecSpace, MEMSPACE> > dwork("dwork", lwork); \
            magma_int_t info = 0; \
            magma_sgeqp3_gpu(M, N, \
                             reinterpret_cast<magmaFloat_ptr>( A.data() ), LDA,\
                             reinterpret_cast<magma_int_t*>( p.data() ), \
                             reinterpret_cast<magmaFloat_ptr>( tau.data() ), \
                             reinterpret_cast<magmaFloat_ptr>( dwork.data() ), lwork, \
                             &info);\
            Kokkos::Profiling::popRegion(); \
        } \
    };

    KOKKOSBLAS_DGEQP3_MAGMA(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
    KOKKOSBLAS_DGEQP3_MAGMA(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaSpace, false)

    KOKKOSBLAS_SGEQP3_MAGMA(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
    KOKKOSBLAS_SGEQP3_MAGMA(Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaSpace, false)
    
    } //namespace Impl
} //namespace KokkosBlas

#endif //If MAGMA

#endif //KOKKOSBLAS_GEQP3_TPL_SPEC_DECL_HPP_

